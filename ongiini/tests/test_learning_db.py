"""Schema warmup + cascade-delete behaviour for learning.sqlite.

We patch ``settings.data_dir`` so each test gets a clean tempdir DB.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ongiini.learning import db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point the learning DB at a tempdir so warmup creates a fresh
    SQLite for each test. Returns the path so the test can introspect."""
    # The module reads settings.data_dir lazily inside _db_path, so a
    # monkeypatch on settings is enough — no reload needed.
    from ongiini import config
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    db.warmup()
    return tmp_path / "learning.sqlite"


# ---------- warmup ----------

def test_warmup_creates_all_six_tables(temp_db):
    with db._conn() as c:
        names = {
            r["name"]
            for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert names >= {
        "learners",
        "learner_profiles",
        "learning_goals",
        "learning_cards",
        "card_attempts",
        "card_review_state",
    }


def test_warmup_is_idempotent(temp_db):
    # Calling warmup again should not blow up or duplicate rows.
    db.warmup()
    db.warmup()
    with db._conn() as c:
        rows = list(c.execute("SELECT name FROM sqlite_master WHERE type='table'"))
    # No duplicate table names.
    names = [r["name"] for r in rows]
    assert len(names) == len(set(names))


def test_learning_goals_has_outline_columns(temp_db):
    """The LLM-authored curriculum outline lives on learning_goals as
    JSON. Lock the columns down so a future schema refactor can't drop
    them without breaking this test."""
    with db._conn() as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(learning_goals)")}
    assert "curriculum_outline" in cols
    assert "outline_updated_at" in cols


def test_indexes_created(temp_db):
    with db._conn() as c:
        idx = {
            r["name"]
            for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
    # The two performance-critical ones from the review feedback.
    assert "idx_review_due" in idx
    assert "idx_review_card" in idx


def test_foreign_keys_enabled(temp_db):
    # Without PRAGMA foreign_keys = ON, all the REFERENCES clauses are
    # documentation-only. Cascade-delete depends on this being on.
    with db._conn() as c:
        row = c.execute("PRAGMA foreign_keys").fetchone()
    assert row[0] == 1


# ---------- cascade delete (the GDPR right-to-erasure path) ----------

def _insert_full_learner_tree(c, learner_id="learner-a"):
    """Insert a learner with one goal, one card, one attempt, one
    review-state row. Returns the learner_id."""
    now = db._now_iso()
    c.execute(
        "INSERT INTO learners (learner_id, identity_type, created_at, "
        "last_active_at) VALUES (?, ?, ?, ?)",
        (learner_id, db.IDENTITY_ANONYMOUS, now, now),
    )
    c.execute(
        "INSERT INTO learner_profiles (learner_id, name, age, current_level, "
        "objective, intake_completed_at) VALUES (?, ?, ?, ?, ?, ?)",
        (learner_id, "test", 25, "beginner", "job interview", now),
    )
    c.execute(
        "INSERT INTO learning_goals (goal_id, learner_id, language, context, "
        "status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("goal-1", learner_id, "afrikaans", "job interview", "active", now),
    )
    c.execute(
        "INSERT INTO learning_cards (card_id, goal_id, card_type, "
        "prompt_text, reference_answer, hint_text, difficulty, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("card-1", "goal-1", db.CARD_VOCAB, "hello in Afrikaans?",
         "hallo", "common greeting", 1, now),
    )
    c.execute(
        "INSERT INTO card_attempts (attempt_id, card_id, learner_id, "
        "user_answer, ai_feedback, rating, hint_used, attempted_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("att-1", "card-1", learner_id, "hallo", "correct!",
         db.RATING_CORRECT, 0, now),
    )
    c.execute(
        "INSERT INTO card_review_state (learner_id, card_id, box, "
        "next_due_at, last_seen_at, total_seen, total_correct) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (learner_id, "card-1", 2, now, now, 1, 1),
    )
    return learner_id


def test_delete_learner_cascades_to_all_child_tables(temp_db):
    """The 'delete my data' contract — deleting the learner row must
    wipe profile, goals, cards, attempts, and review state."""
    with db._conn() as c:
        _insert_full_learner_tree(c, "learner-a")
        # Sanity: rows exist before delete
        assert c.execute(
            "SELECT COUNT(*) FROM card_review_state"
        ).fetchone()[0] == 1
        assert c.execute(
            "SELECT COUNT(*) FROM card_attempts"
        ).fetchone()[0] == 1

        c.execute("DELETE FROM learners WHERE learner_id = ?", ("learner-a",))

        # Every child table is empty now (cascade fired).
        for table in (
            "learner_profiles", "learning_goals", "learning_cards",
            "card_attempts", "card_review_state",
        ):
            n = c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert n == 0, f"{table} should be empty after learner delete"


def test_delete_card_cascades_to_attempts_and_review_state(temp_db):
    """Deleting a single card should remove its attempts + review state
    but leave the learner and goal intact."""
    with db._conn() as c:
        _insert_full_learner_tree(c, "learner-b")
        c.execute("DELETE FROM learning_cards WHERE card_id = ?", ("card-1",))
        # Card-scoped rows gone…
        assert c.execute(
            "SELECT COUNT(*) FROM card_attempts WHERE card_id = ?",
            ("card-1",),
        ).fetchone()[0] == 0
        assert c.execute(
            "SELECT COUNT(*) FROM card_review_state WHERE card_id = ?",
            ("card-1",),
        ).fetchone()[0] == 0
        # …but the learner + goal survive.
        assert c.execute(
            "SELECT COUNT(*) FROM learners WHERE learner_id = ?",
            ("learner-b",),
        ).fetchone()[0] == 1
        assert c.execute(
            "SELECT COUNT(*) FROM learning_goals WHERE goal_id = ?",
            ("goal-1",),
        ).fetchone()[0] == 1


# ---------- helper sanity ----------

def test_now_iso_returns_seconds_resolution_utc_string(temp_db):
    ts = db._now_iso()
    # Format: "YYYY-MM-DDTHH:MM:SS+00:00"
    assert ts.endswith("+00:00")
    assert "T" in ts
    # No microseconds.
    assert "." not in ts
