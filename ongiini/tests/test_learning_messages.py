"""Tests for the learner_messages persistence layer."""
from __future__ import annotations

import pytest

from ongiini.learning import db, messages, store


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    from ongiini import config
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    db.warmup()
    return tmp_path / "learning.sqlite"


def _setup(temp_db):
    """Create a learner + active goal and return ids."""
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(learner_id)
    return learner_id, goal["goal_id"]


# ──────────────────────────────────────────────────────────────────
# append
# ──────────────────────────────────────────────────────────────────

def test_append_returns_row_with_generated_id(temp_db):
    learner_id, goal_id = _setup(temp_db)
    row = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_COACH_TEXT, payload={"text": "Welcome."},
    )
    assert len(row["message_id"]) == 36
    assert row["kind"] == db.MSG_COACH_TEXT
    assert row["payload"]["text"] == "Welcome."
    assert row["answered"] is False
    assert row["created_at"]


def test_append_rejects_unknown_kind(temp_db):
    learner_id, goal_id = _setup(temp_db)
    with pytest.raises(ValueError, match="unknown message kind"):
        messages.append(
            learner_id=learner_id, goal_id=goal_id,
            kind="not_a_kind", payload={"text": "x"},
        )


def test_append_raises_value_error_on_unserialisable_payload(temp_db):
    """Code-review #7: a non-JSON value (set, bytes, datetime) must
    surface as ValueError matching the docstring contract — NOT as an
    uncaught TypeError that becomes a 500 in the API layer."""
    learner_id, goal_id = _setup(temp_db)
    with pytest.raises(ValueError, match="JSON-serialisable"):
        messages.append(
            learner_id=learner_id, goal_id=goal_id,
            kind=db.MSG_PROGRESS,
            payload={"by_box": {1, 2, 3}},   # set isn't JSON-serialisable
        )


def test_claim_exercise_returns_false_for_nonexistent_id(temp_db):
    """Code-review #2: a row that doesn't exist must not crash and must
    return False — same as 'already claimed'. The coach treats this as
    'someone else handled it' and bails out cleanly."""
    assert messages.claim_exercise("does-not-exist") is False


def test_append_requires_ids(temp_db):
    with pytest.raises(ValueError, match="learner_id"):
        messages.append(
            learner_id="", goal_id="x",
            kind=db.MSG_COACH_TEXT, payload={"text": "x"},
        )
    with pytest.raises(ValueError, match="goal_id"):
        messages.append(
            learner_id="x", goal_id="",
            kind=db.MSG_COACH_TEXT, payload={"text": "x"},
        )


def test_append_pii_scrub_on_learner_text(temp_db):
    """The reviewer flagged that messages are a write path; lock down
    the PII scrub on learner-typed text."""
    learner_id, goal_id = _setup(temp_db)
    row = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_LEARNER_TEXT,
        payload={"text": "I emailed admin@spar.co.za about it"},
    )
    assert "admin@spar.co.za" not in row["payload"]["text"]
    assert "[REDACTED:email]" in row["payload"]["text"]


def test_append_pii_scrub_on_coach_text_defence_in_depth(temp_db):
    """Coach text comes from the model but we scrub it too, in case the
    model echoes content the learner pasted earlier."""
    learner_id, goal_id = _setup(temp_db)
    row = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_COACH_TEXT,
        payload={"text": "You said your email was alice@example.com — got it."},
    )
    assert "alice@example.com" not in row["payload"]["text"]
    assert "[REDACTED:email]" in row["payload"]["text"]


def test_append_scrub_lesson_title_and_body(temp_db):
    """Reviewer correction: lesson + feedback + exercise payloads ARE
    scrubbed (defence in depth — the model can echo back learner-
    pasted PII into 'example' fields). Locks in extended scrub."""
    learner_id, goal_id = _setup(temp_db)
    row = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_LESSON,
        payload={
            "title": "Contact ceo@example.com",
            "body": "Example: jan@example.com",
        },
    )
    assert "ceo@example.com" not in row["payload"]["title"]
    assert "jan@example.com" not in row["payload"]["body"]


def test_append_scrub_lesson_examples_list(temp_db):
    """The ``examples`` field on a lesson is a list[str]; every entry
    must be scrubbed individually."""
    learner_id, goal_id = _setup(temp_db)
    row = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_LESSON,
        payload={
            "title": "x", "body": "y",
            "examples": ["email alice@example.com", "plain text"],
        },
    )
    assert "alice@example.com" not in row["payload"]["examples"][0]
    assert row["payload"]["examples"][1] == "plain text"


def test_append_scrub_feedback_payload(temp_db):
    learner_id, goal_id = _setup(temp_db)
    row = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_FEEDBACK,
        payload={
            "rating": "correct",
            "feedback": "Yes — and your email maria@example.com was a great touch.",
        },
    )
    assert "maria@example.com" not in row["payload"]["feedback"]
    assert "[REDACTED:email]" in row["payload"]["feedback"]


def test_append_no_text_fields_on_progress(temp_db):
    """Progress payloads have no free-text fields; the helper should
    not crash on them."""
    learner_id, goal_id = _setup(temp_db)
    row = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_PROGRESS,
        payload={"box": 2, "total_seen": 5, "total_correct": 4, "by_box": {1: 1, 2: 4}},
    )
    assert row["payload"]["total_seen"] == 5


# ──────────────────────────────────────────────────────────────────
# list_for_goal
# ──────────────────────────────────────────────────────────────────

def test_list_for_goal_returns_in_chronological_order(temp_db):
    learner_id, goal_id = _setup(temp_db)
    for i in range(3):
        messages.append(
            learner_id=learner_id, goal_id=goal_id,
            kind=db.MSG_COACH_TEXT, payload={"text": f"step {i}"},
        )
    thread = messages.list_for_goal(learner_id=learner_id, goal_id=goal_id)
    assert len(thread) == 3
    assert [m["payload"]["text"] for m in thread] == ["step 0", "step 1", "step 2"]


def test_list_for_goal_empty_when_no_messages(temp_db):
    learner_id, goal_id = _setup(temp_db)
    assert messages.list_for_goal(learner_id=learner_id, goal_id=goal_id) == []


def test_list_for_goal_respects_limit(temp_db):
    learner_id, goal_id = _setup(temp_db)
    for i in range(5):
        messages.append(
            learner_id=learner_id, goal_id=goal_id,
            kind=db.MSG_COACH_TEXT, payload={"text": f"step {i}"},
        )
    out = messages.list_for_goal(
        learner_id=learner_id, goal_id=goal_id, limit=3,
    )
    assert len(out) == 3


def test_list_for_goal_returns_newest_messages_when_over_limit(temp_db):
    """Code-review CRITICAL: previously this selected oldest N, so
    once a thread crossed the cap, rehydration showed ancient history
    while recent turns silently disappeared. Lock in: the slice is
    the NEWEST N messages, returned in chronological order so the
    frontend can append-render."""
    learner_id, goal_id = _setup(temp_db)
    for i in range(10):
        messages.append(
            learner_id=learner_id, goal_id=goal_id,
            kind=db.MSG_COACH_TEXT, payload={"text": f"step {i}"},
        )
    out = messages.list_for_goal(
        learner_id=learner_id, goal_id=goal_id, limit=3,
    )
    # The 3 most-recent messages, oldest-first within the slice.
    texts = [m["payload"]["text"] for m in out]
    assert texts == ["step 7", "step 8", "step 9"]


def test_list_for_goal_scoped_to_goal(temp_db):
    """Messages for goal A must not show up when querying goal B."""
    learner_id, goal_a = _setup(temp_db)
    # Create a second goal for the same learner.
    from ongiini.learning import db as _db
    from ongiini.learning.db import _conn, _now_iso
    from uuid import uuid4
    goal_b = str(uuid4())
    with _conn() as c:
        c.execute(
            "INSERT INTO learning_goals (goal_id, learner_id, language, "
            "context, status, created_at) VALUES (?, ?, 'afrikaans', "
            "NULL, 'active', ?)",
            (goal_b, learner_id, _now_iso()),
        )

    messages.append(
        learner_id=learner_id, goal_id=goal_a,
        kind=db.MSG_COACH_TEXT, payload={"text": "goal A msg"},
    )
    messages.append(
        learner_id=learner_id, goal_id=goal_b,
        kind=db.MSG_COACH_TEXT, payload={"text": "goal B msg"},
    )

    a = messages.list_for_goal(learner_id=learner_id, goal_id=goal_a)
    b = messages.list_for_goal(learner_id=learner_id, goal_id=goal_b)
    assert len(a) == 1 and a[0]["payload"]["text"] == "goal A msg"
    assert len(b) == 1 and b[0]["payload"]["text"] == "goal B msg"


# ──────────────────────────────────────────────────────────────────
# latest_unanswered_exercise + mark_answered
# ──────────────────────────────────────────────────────────────────

def test_latest_unanswered_exercise_returns_most_recent(temp_db):
    learner_id, goal_id = _setup(temp_db)
    # Append two exercises.
    messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_EXERCISE,
        payload={"card_type": "vocab", "prompt_text": "first"},
        card_id="card-1",
    )
    second = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_EXERCISE,
        payload={"card_type": "vocab", "prompt_text": "second"},
        card_id="card-2",
    )
    out = messages.latest_unanswered_exercise(
        learner_id=learner_id, goal_id=goal_id,
    )
    assert out is not None
    assert out["message_id"] == second["message_id"]


def test_latest_unanswered_exercise_skips_answered(temp_db):
    learner_id, goal_id = _setup(temp_db)
    first = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_EXERCISE,
        payload={"card_type": "vocab", "prompt_text": "first"},
        card_id="card-1",
    )
    messages.mark_answered(first["message_id"])
    assert messages.latest_unanswered_exercise(
        learner_id=learner_id, goal_id=goal_id,
    ) is None


def test_latest_unanswered_exercise_returns_none_for_empty(temp_db):
    learner_id, goal_id = _setup(temp_db)
    assert messages.latest_unanswered_exercise(
        learner_id=learner_id, goal_id=goal_id,
    ) is None


def test_mark_answered_no_op_on_missing(temp_db):
    # Soft-fail — should not raise.
    messages.mark_answered("nonexistent-id")


# ──────────────────────────────────────────────────────────────────
# clear_for_goal
# ──────────────────────────────────────────────────────────────────

def test_clear_for_goal_removes_all_thread_rows(temp_db):
    learner_id, goal_id = _setup(temp_db)
    for i in range(4):
        messages.append(
            learner_id=learner_id, goal_id=goal_id,
            kind=db.MSG_COACH_TEXT, payload={"text": f"x {i}"},
        )
    n = messages.clear_for_goal(learner_id=learner_id, goal_id=goal_id)
    assert n == 4
    assert messages.list_for_goal(
        learner_id=learner_id, goal_id=goal_id,
    ) == []


# ──────────────────────────────────────────────────────────────────
# recent_text_pairs
# ──────────────────────────────────────────────────────────────────

def test_recent_text_pairs_excludes_non_text(temp_db):
    """Lessons + exercises + feedback + progress are NOT in the text
    pair list — they're surfaced separately via curriculum + last card."""
    learner_id, goal_id = _setup(temp_db)
    messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_COACH_TEXT, payload={"text": "Hi!"},
    )
    messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_LESSON, payload={"title": "x", "body": "y"},
    )
    messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_LEARNER_TEXT, payload={"text": "Hello."},
    )
    out = messages.recent_text_pairs(
        learner_id=learner_id, goal_id=goal_id,
    )
    kinds = [m["kind"] for m in out]
    assert db.MSG_LESSON not in kinds
    assert kinds == [db.MSG_COACH_TEXT, db.MSG_LEARNER_TEXT]


def test_append_scrub_chat_learner_text(temp_db):
    """Track C PII regression — the conversation-mode learner turn
    payload's ``text`` field must be scrubbed before INSERT, same
    rule as the cards-mode MSG_LEARNER_TEXT."""
    learner_id, goal_id = _setup(temp_db)
    row = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_CHAT_LEARNER,
        payload={"text": "Hallo, meine E-Mail ist test@example.com"},
    )
    text = row["payload"]["text"]
    assert "test@example.com" not in text
    assert "[REDACTED:email]" in text


def test_append_scrub_chat_coach_reply(temp_db):
    """Defence-in-depth: the coach's TARGET-language reply can echo
    back learner-pasted PII (the system prompt literally tells the
    coach to quote the learner). Scrubbed before INSERT."""
    learner_id, goal_id = _setup(temp_db)
    row = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_CHAT_COACH,
        payload={"reply": "Deine E-Mail (test@example.com) ist notiert."},
    )
    assert "test@example.com" not in row["payload"]["reply"]
    assert "[REDACTED:email]" in row["payload"]["reply"]


def test_append_scrub_chat_notes_corrections_and_new_words(temp_db):
    """The notes block contains the LLM's verbatim echo of the
    learner's phrase in corrections[].learner — exactly the surface
    where PII would leak through. Walks both list-of-dict fields."""
    learner_id, goal_id = _setup(temp_db)
    row = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_CHAT_NOTES,
        payload={
            "corrections": [
                {"learner": "ich heisse jane@example.com",
                 "correct": "ich heiße Jane",
                 "note":    "the email isn't a name"},
            ],
            "new_words": [
                {"word": "der Name", "meaning": "the name"},
            ],
        },
    )
    corr = row["payload"]["corrections"][0]
    assert "jane@example.com" not in corr["learner"]
    assert "[REDACTED:email]" in corr["learner"]
    # Other fields scrubbed too even though they're PII-free here.
    assert corr["correct"] == "ich heiße Jane"
    assert corr["note"] == "the email isn't a name"
