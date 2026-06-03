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

# Card types. vocab / translation / production are EXERCISE cards
# (the learner answers, the model grades, SRS tracks). ``lesson`` is
# a TEACHING card — the model authors instructional content with
# examples; the learner reads + clicks "Got it →". Lessons appear at
# the start of a new module so the learner is taught a concept before
# being asked to apply it.
CARD_VOCAB = "vocab"
CARD_TRANSLATION = "translation"
CARD_PRODUCTION = "production"
CARD_LESSON = "lesson"
# Phase 2 card-variety expansion. Each is an EXERCISE type (goes
# through grading + SRS + Leitner). All share `prompt_text` +
# `reference_answer`; the per-type payload extras live alongside.
# Frontend has a dedicated renderer per type; backend grading reads
# card_type to pick the right rubric.
CARD_CLOZE = "cloze"                # fill-in-the-blank
CARD_REORDER = "reorder"            # arrange shuffled tokens
CARD_MULTIPLE_CHOICE = "multiple_choice"   # pick option with explanations
CARD_GRAMMAR = "grammar"            # transformation drill
CARD_PROVERB = "proverb"            # idiom / saying with cultural note
CARD_DIALOGUE = "dialogue"          # role-play completion
# Comprehensible-input track. A story is 4-8 short
# <<TARGET_LANGUAGE>> paragraphs the learner reads (with inline glosses)
# followed by 1-3 lenient comprehension questions. Stories are graded
# but DO NOT enter the SRS queue — they're input exposure, not retrieval
# practice. Always brand-new content; never recycled. The selector emits
# one per module after the first lesson and before the drills, so every
# subsequent drill in the module has the story's vocabulary + structures
# as fresh prior input.
CARD_STORY = "story"

CARD_TYPES = (
    CARD_VOCAB, CARD_TRANSLATION, CARD_PRODUCTION, CARD_LESSON,
    CARD_CLOZE, CARD_REORDER, CARD_MULTIPLE_CHOICE,
    CARD_GRAMMAR, CARD_PROVERB, CARD_DIALOGUE, CARD_STORY,
)
# Cards that go through the normal grading + SRS Leitner pipeline.
# Stories are graded but excluded — see SRS_EXCLUDED_CARD_TYPES below.
EXERCISE_CARD_TYPES = (
    CARD_VOCAB, CARD_TRANSLATION, CARD_PRODUCTION,
    CARD_CLOZE, CARD_REORDER, CARD_MULTIPLE_CHOICE,
    CARD_GRAMMAR, CARD_PROVERB, CARD_DIALOGUE, CARD_STORY,
)
# Card types whose attempts MUST NOT enter the SRS replay queue.
# Stories: always-new content; re-reading a story isn't retrieval
# practice. The grader still runs (so the learner gets feedback), but
# ``store.next_due_cards`` skips them.
SRS_EXCLUDED_CARD_TYPES = (CARD_STORY,)


# Message kinds for the learner_messages chat thread. The frontend
# renders each kind differently — text bubble vs rich lesson card vs
# coloured feedback callout. See coach.py for what writes them.
MSG_COACH_TEXT = "coach_text"
MSG_LEARNER_TEXT = "learner_text"
MSG_LESSON = "lesson"
MSG_EXERCISE = "exercise"
MSG_FEEDBACK = "feedback"
MSG_PROGRESS = "progress"
# Track C — Conversation mode. A second surface on the same goal:
# the learner chats with the coach IN TARGET LANGUAGE at their
# level; the coach replies in target language and surfaces a small
# notes block (corrections + new vocabulary) at end-of-turn. These
# message kinds live on the same learner_messages table so the chat
# rehydrates from the goal, but the cards renderer ignores them
# (filters MSG_LESSON / MSG_EXERCISE / etc.).
MSG_CHAT_LEARNER = "chat_learner"   # learner turn in target language
MSG_CHAT_COACH = "chat_coach"       # coach reply in target language
MSG_CHAT_NOTES = "chat_notes"       # end-of-turn corrections + new words
MESSAGE_KINDS = (
    MSG_COACH_TEXT, MSG_LEARNER_TEXT, MSG_LESSON, MSG_EXERCISE,
    MSG_FEEDBACK, MSG_PROGRESS,
    MSG_CHAT_LEARNER, MSG_CHAT_COACH, MSG_CHAT_NOTES,
)

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
        -- learner_profiles.intake_completed_at IS NULL is the canonical
        -- "intake in progress" signal — no separate cursor on this table.
        -- The LLM conducts the conversation; the API checks completeness
        -- by looking at the profile fields.
        CREATE TABLE IF NOT EXISTS learners (
            learner_id      TEXT PRIMARY KEY,
            identity_type   TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            last_active_at  TEXT NOT NULL
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

        -- One active goal per learner for MVP. The curriculum outline
        -- is a JSON document the LLM produces after intake and revises
        -- as the learner progresses. We do NOT validate its inner
        -- shape — that's the LLM's design space, not ours. Outline is
        -- nullable so a goal can exist before the LLM has had its
        -- first turn to write one.
        CREATE TABLE IF NOT EXISTS learning_goals (
            goal_id              TEXT PRIMARY KEY,
            learner_id           TEXT NOT NULL
                                 REFERENCES learners(learner_id) ON DELETE CASCADE,
            language             TEXT NOT NULL DEFAULT 'afrikaans',
            context              TEXT,
            status               TEXT NOT NULL DEFAULT 'active',
            curriculum_outline   TEXT,                    -- JSON, LLM-authored
            outline_updated_at   TEXT,                    -- ISO ts of last outline write
            created_at           TEXT NOT NULL
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

        -- Phase 2: per-(learner, goal) chat thread. The whole UI is a
        -- chat conversation; cards pop up inside it as rich messages.
        -- payload_json is the rich content (shape depends on kind);
        -- card_id is set for exercise/feedback/progress messages so the
        -- frontend can link them.
        CREATE TABLE IF NOT EXISTS learner_messages (
            message_id    TEXT PRIMARY KEY,
            learner_id    TEXT NOT NULL
                          REFERENCES learners(learner_id) ON DELETE CASCADE,
            goal_id       TEXT NOT NULL
                          REFERENCES learning_goals(goal_id) ON DELETE CASCADE,
            kind          TEXT NOT NULL,
            payload_json  TEXT NOT NULL,
            card_id       TEXT,
            answered      INTEGER NOT NULL DEFAULT 0,
            created_at    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_messages_goal_created
            ON learner_messages(goal_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_messages_unanswered
            ON learner_messages(learner_id, goal_id, answered);
        """)

        # Idempotent ALTERs on learning_goals for the title + archived_at
        # columns. SQLite has no IF NOT EXISTS for ALTER COLUMN — we
        # introspect PRAGMA table_info and add only when missing.
        cols = {r["name"] for r in c.execute("PRAGMA table_info(learning_goals)").fetchall()}
        if "title" not in cols:
            c.execute("ALTER TABLE learning_goals ADD COLUMN title TEXT")
        if "archived_at" not in cols:
            c.execute("ALTER TABLE learning_goals ADD COLUMN archived_at TEXT")

        # Multi-language overhaul: learning_goals gains source_language
        # (the learner's strongest language) + current_level (per-goal
        # override of the profile's intake-time level). Existing rows
        # had only the legacy `language` column (target). Backfill
        # source_language='english' for existing rows so they keep
        # making sense post-deploy.
        if "source_language" not in cols:
            c.execute("ALTER TABLE learning_goals ADD COLUMN source_language TEXT")
            c.execute(
                "UPDATE learning_goals SET source_language = 'english' "
                "WHERE source_language IS NULL"
            )
        if "current_level" not in cols:
            c.execute("ALTER TABLE learning_goals ADD COLUMN current_level TEXT")

        # module_id on learning_cards. Nullable so prior cards (which
        # the LLM authored without module tagging) keep working — they
        # just don't count toward any module's progress, which is the
        # honest behaviour. New cards must include it (cards.py prompts
        # for it; soft-required at the validator).
        card_cols = {r["name"] for r in c.execute("PRAGMA table_info(learning_cards)").fetchall()}
        if "module_id" not in card_cols:
            c.execute("ALTER TABLE learning_cards ADD COLUMN module_id TEXT")
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_cards_goal_module "
            "ON learning_cards(goal_id, module_id)"
        )

        # topic_id on learning_cards — drives topic-aware teach-then-
        # test pacing. Each new card is tagged with the topic_id it
        # teaches (lesson) or drills (exercise); the module digest
        # rolls these into per-topic counts so the prompt can enforce
        # "no exercises on untaught topics". Nullable for the same
        # back-compat reason as module_id.
        card_cols = {r["name"] for r in c.execute("PRAGMA table_info(learning_cards)").fetchall()}
        if "topic_id" not in card_cols:
            c.execute("ALTER TABLE learning_cards ADD COLUMN topic_id TEXT")
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_cards_goal_module_topic "
            "ON learning_cards(goal_id, module_id, topic_id)"
        )

        # extras_json on learning_cards — stores the per-card-type
        # structural extras (multiple_choice options, reorder tokens,
        # dialogue turns, grammar source_sentence, proverb
        # cultural_note) as a JSON blob. Without this, SRS replay of
        # a previously-failed card surfaces with only prompt_text +
        # reference_answer — the renderer would draw a reorder card
        # with no token chips, an MC card with no options, etc. The
        # extras are LLM-authored at first emission and never change,
        # so storing them on the card row (not in the volatile message
        # payload) survives replays.
        card_cols = {r["name"] for r in c.execute("PRAGMA table_info(learning_cards)").fetchall()}
        if "extras_json" not in card_cols:
            c.execute("ALTER TABLE learning_cards ADD COLUMN extras_json TEXT")

        # error_tags_json on card_attempts — Track D: error-pattern
        # tracking. The grader tags each non-correct attempt with 0-2
        # short category labels (gender_error, verb_conjugation,
        # word_order, etc.); store.error_pattern_summary aggregates
        # them per (learner, goal) so the curriculum designer + card
        # author can target the learner's actual weaknesses rather
        # than a generic plan. Nullable for the same back-compat
        # reason as the card columns above.
        attempt_cols = {
            r["name"]
            for r in c.execute("PRAGMA table_info(card_attempts)").fetchall()
        }
        if "error_tags_json" not in attempt_cols:
            c.execute(
                "ALTER TABLE card_attempts ADD COLUMN error_tags_json TEXT"
            )

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
