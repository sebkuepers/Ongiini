"""SQLite store for the learn.ongiini.ai progress tracker.

Standalone sqlite at /data/learning.sqlite — separate from contributions
and chat memory so the learning surface can ship independently and the
schema can evolve without touching anything else.

The pattern is intentionally identical to contributions.py:
  - per-call connections via the ``_conn()`` context manager
  - schema warmup called from FastAPI lifespan, idempotent
  - bind-mounted file path so progress survives container restarts

The schema has six tables; see the warmup() docstring for details.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ..config import settings

log = logging.getLogger("ongiini.learning.db")


# Identity types — distinguishes a cold-visit learner from one bound to
# a WhatsApp number via the (future) /v1/learn/upgrade flow. Anonymous
# learners hold a UUID v4 as their learner_id; WhatsApp-bound learners
# hold a "wa:<salted-hash>" prefix so the two namespaces don't collide.
IDENTITY_ANONYMOUS = "anonymous"
IDENTITY_WHATSAPP = "whatsapp"

# Intake step values, in order. Used by intake.py to advance the state
# machine and by the API to surface which question to ask next.
INTAKE_START = "start"
INTAKE_NAME = "name"
INTAKE_AGE = "age"
INTAKE_LEVEL = "level"
INTAKE_OBJECTIVE = "objective"
INTAKE_DONE = "done"
INTAKE_STEPS = (
    INTAKE_START, INTAKE_NAME, INTAKE_AGE, INTAKE_LEVEL,
    INTAKE_OBJECTIVE, INTAKE_DONE,
)

# Card types — the three the model can generate. See cards.py.
CARD_VOCAB = "vocab"
CARD_TRANSLATION = "translation"
CARD_PRODUCTION = "production"
CARD_TYPES = (CARD_VOCAB, CARD_TRANSLATION, CARD_PRODUCTION)

# Attempt ratings — set by the grading layer (model output). 'partial'
# is what the model returns when the answer captures the right idea
# but with significant errors; we treat it as box-promoting for SRS
# purposes but the UI labels it differently.
RATING_CORRECT = "correct"
RATING_PARTIAL = "partial"
RATING_WRONG = "wrong"
RATINGS = (RATING_CORRECT, RATING_PARTIAL, RATING_WRONG)


def _db_path() -> Path:
    """The sqlite path. Lives next to other /data files; bind-mounted
    from the host so progress survives container restarts."""
    return settings.data_dir / "learning.sqlite"


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    """Per-call connection. sqlite handles concurrent readers fine and
    we only ever have one writer (the webhook), so no pool is needed.
    Foreign-key enforcement is OFF by default in sqlite — we turn it
    on here so the schema's REFERENCES clauses actually do their job."""
    conn = sqlite3.connect(_db_path(), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def warmup() -> None:
    """Create the six tables if not present. Called from the FastAPI
    lifespan on startup. Idempotent — safe to call repeatedly.

    Tables:
      * learners            — identity + intake-step pointer
      * learner_profiles    — name / age / level / objective
      * learning_goals      — one active language goal per learner (MVP)
      * learning_cards      — model-generated cards, cached by goal
      * card_attempts       — every grading event
      * card_review_state   — Leitner box + next_due_at, per (learner, card)
    """
    _db_path().parent.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS learners (
            learner_id      TEXT PRIMARY KEY,
            identity_type   TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            last_active_at  TEXT NOT NULL,
            intake_step     TEXT NOT NULL DEFAULT 'start'
        );

        CREATE TABLE IF NOT EXISTS learner_profiles (
            learner_id            TEXT PRIMARY KEY
                                  REFERENCES learners(learner_id) ON DELETE CASCADE,
            name                  TEXT,
            age                   INTEGER,
            current_level         TEXT,
            objective             TEXT,
            intake_completed_at   TEXT
        );

        CREATE TABLE IF NOT EXISTS learning_goals (
            goal_id      TEXT PRIMARY KEY,
            learner_id   TEXT NOT NULL
                         REFERENCES learners(learner_id) ON DELETE CASCADE,
            language     TEXT NOT NULL DEFAULT 'afrikaans',
            context      TEXT,
            status       TEXT NOT NULL DEFAULT 'active',
            created_at   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_goals_learner
            ON learning_goals(learner_id);

        CREATE TABLE IF NOT EXISTS learning_cards (
            card_id           TEXT PRIMARY KEY,
            goal_id           TEXT NOT NULL
                              REFERENCES learning_goals(goal_id) ON DELETE CASCADE,
            card_type         TEXT NOT NULL,
            prompt_text       TEXT NOT NULL,
            reference_answer  TEXT,
            hint_text         TEXT,
            difficulty        INTEGER,
            created_at        TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cards_goal
            ON learning_cards(goal_id);

        CREATE TABLE IF NOT EXISTS card_attempts (
            attempt_id     TEXT PRIMARY KEY,
            card_id        TEXT NOT NULL
                           REFERENCES learning_cards(card_id) ON DELETE CASCADE,
            learner_id     TEXT NOT NULL
                           REFERENCES learners(learner_id) ON DELETE CASCADE,
            user_answer    TEXT,
            ai_feedback    TEXT,
            rating         TEXT NOT NULL,
            hint_used      INTEGER NOT NULL DEFAULT 0,
            attempted_at   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_attempts_learner
            ON card_attempts(learner_id);
        CREATE INDEX IF NOT EXISTS idx_attempts_card
            ON card_attempts(card_id);

        CREATE TABLE IF NOT EXISTS card_review_state (
            learner_id     TEXT NOT NULL
                           REFERENCES learners(learner_id) ON DELETE CASCADE,
            card_id        TEXT NOT NULL
                           REFERENCES learning_cards(card_id) ON DELETE CASCADE,
            box            INTEGER NOT NULL DEFAULT 1,
            next_due_at    TEXT NOT NULL,
            last_seen_at   TEXT,
            total_seen     INTEGER NOT NULL DEFAULT 0,
            total_correct  INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (learner_id, card_id)
        );
        CREATE INDEX IF NOT EXISTS idx_review_due
            ON card_review_state(learner_id, next_due_at);
        -- Fronts the cascade when a card is deleted (review_state has
        -- to be walked by card_id; without this index it's a full
        -- table scan on every card delete).
        CREATE INDEX IF NOT EXISTS idx_review_card
            ON card_review_state(card_id);
        """)
    log.info("learning sqlite warmed at %s", _db_path())


# ──────────────────────────────────────────────────────────────────
# PII contract (enforced by store.py callers, NOT here)
#
# The ``card_attempts.user_answer`` column MUST be sanitised through
# ``ongiini.pii.sanitize`` before INSERT — same project-wide rule the
# WhatsApp and chat surfaces follow ("LLM sees raw text; disk + mem0
# see redacted text"). This module deliberately does not own the
# write path — see ``store.record_attempt`` for the enforced shape.
# ──────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    """Single source of truth for stored timestamps in this module.
    ISO-8601 UTC seconds. Matches the format ``contributions.py`` uses
    so timestamp parsing patterns stay consistent across the codebase.

    Module-private: callers in this package should import this name
    explicitly (``from .db import _now_iso``); external callers should
    not depend on its format directly."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
