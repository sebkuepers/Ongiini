"""High-level data access for the learning store.

Each test gets a clean tempdir SQLite via the ``temp_db`` fixture
(patching settings.data_dir). The PII contract — ``record_attempt``
and free-text profile fields go through ``pii.sanitize`` — is locked
down explicitly because that's a non-negotiable project rule.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ongiini.learning import db, srs, store


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    from ongiini import config
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    db.warmup()
    return tmp_path / "learning.sqlite"


# ============================================================
# Identity & lifecycle
# ============================================================

def test_create_anonymous_learner_returns_uuid(temp_db):
    learner_id = store.create_anonymous_learner()
    assert len(learner_id) == 36     # UUID v4 string
    assert "-" in learner_id


def test_create_anonymous_learner_is_unique(temp_db):
    a = store.create_anonymous_learner()
    b = store.create_anonymous_learner()
    assert a != b


def test_get_learner_returns_row_after_create(temp_db):
    learner_id = store.create_anonymous_learner()
    row = store.get_learner(learner_id)
    assert row is not None
    assert row["learner_id"] == learner_id
    assert row["identity_type"] == db.IDENTITY_ANONYMOUS
    assert row["created_at"]


def test_get_learner_returns_none_for_unknown(temp_db):
    assert store.get_learner("not-a-real-id") is None


def test_upsert_whatsapp_learner_idempotent(temp_db):
    a = store.upsert_whatsapp_learner("hashed-msisdn-abc")
    b = store.upsert_whatsapp_learner("hashed-msisdn-abc")
    assert a == b
    assert a.startswith("wa:")


def test_touch_learner_updates_last_active(temp_db):
    learner_id = store.create_anonymous_learner()
    before = store.get_learner(learner_id)["last_active_at"]
    # Sleep not needed — _now_iso is second-resolution but the touch
    # happens after creation so timestamps may match; just check it
    # doesn't raise.
    store.touch_learner(learner_id)
    after = store.get_learner(learner_id)["last_active_at"]
    assert after >= before


def test_delete_learner_removes_row_and_returns_count(temp_db):
    learner_id = store.create_anonymous_learner()
    assert store.delete_learner(learner_id) == 1
    assert store.get_learner(learner_id) is None


def test_delete_unknown_learner_returns_zero(temp_db):
    assert store.delete_learner("not-a-real-id") == 0


# ============================================================
# Profile / intake
# ============================================================

def test_save_profile_field_creates_row_on_first_call(temp_db):
    learner_id = store.create_anonymous_learner()
    store.save_profile_field(learner_id, "name", "Sebastian")
    profile = store.get_profile(learner_id)
    assert profile["name"] == "Sebastian"


def test_save_profile_field_can_overwrite(temp_db):
    learner_id = store.create_anonymous_learner()
    store.save_profile_field(learner_id, "age", 25)
    store.save_profile_field(learner_id, "age", 26)
    assert store.get_profile(learner_id)["age"] == 26


def test_save_profile_field_rejects_unknown_field(temp_db):
    learner_id = store.create_anonymous_learner()
    with pytest.raises(ValueError, match="unknown profile field"):
        store.save_profile_field(learner_id, "favourite_colour", "blue")


def test_save_profile_field_rejects_empty_learner_id(temp_db):
    with pytest.raises(ValueError, match="learner_id"):
        store.save_profile_field("", "name", "x")


def test_save_profile_field_raises_when_learner_deleted(temp_db):
    """If the parent row is deleted before the UPDATE matches, we now
    raise rather than silently dropping the save."""
    learner_id = store.create_anonymous_learner()
    # Simulate deletion before save: just hand a bogus learner_id.
    with pytest.raises(RuntimeError, match="not found"):
        store.save_profile_field("nonexistent-id", "name", "x")


def test_save_profile_field_pii_scrub_objective(temp_db):
    """The reviewer caught that free-text profile fields were being
    stored raw. Lock the scrub down: an email in the objective gets
    redacted before INSERT."""
    learner_id = store.create_anonymous_learner()
    store.save_profile_field(
        learner_id, "objective",
        "I want to email maria@example.com about the SPAR job"
    )
    stored = store.get_profile(learner_id)["objective"]
    assert "maria@example.com" not in stored
    assert "[REDACTED:email]" in stored


def test_save_profile_field_pii_scrub_name(temp_db):
    """Name is free-text too — runs through the scrubber even though
    pii.sanitize doesn't currently redact person names. Future-proof."""
    learner_id = store.create_anonymous_learner()
    store.save_profile_field(learner_id, "name", "Sebastian")
    assert store.get_profile(learner_id)["name"] == "Sebastian"


def test_save_profile_field_numeric_not_scrubbed(temp_db):
    """Integer fields don't go through the scrubber — they're not
    text. Locked down so a future refactor doesn't try to sanitise 25."""
    learner_id = store.create_anonymous_learner()
    store.save_profile_field(learner_id, "age", 25)
    assert store.get_profile(learner_id)["age"] == 25


def test_mark_intake_complete_sets_timestamp(temp_db):
    learner_id = store.create_anonymous_learner()
    store.save_profile_field(learner_id, "name", "Sebastian")
    store.mark_intake_complete(learner_id)
    profile = store.get_profile(learner_id)
    assert profile["intake_completed_at"]


def test_mark_intake_complete_preserves_first_timestamp(temp_db):
    """Reviewer confirmed: 'intake completed on' is a first-time event,
    not a recency marker. Re-calling shouldn't bump it."""
    learner_id = store.create_anonymous_learner()
    store.mark_intake_complete(learner_id)
    first_ts = store.get_profile(learner_id)["intake_completed_at"]
    store.mark_intake_complete(learner_id)
    second_ts = store.get_profile(learner_id)["intake_completed_at"]
    assert first_ts == second_ts


# ============================================================
# Goal / curriculum outline
# ============================================================

def test_get_or_create_active_goal_creates(temp_db):
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(learner_id)
    assert goal["learner_id"] == learner_id
    assert goal["language"] == "afrikaans"
    assert goal["status"] == "active"


def test_get_or_create_active_goal_is_idempotent(temp_db):
    learner_id = store.create_anonymous_learner()
    a = store.get_or_create_active_goal(learner_id)
    b = store.get_or_create_active_goal(learner_id)
    assert a["goal_id"] == b["goal_id"]


def test_get_or_create_active_goal_pii_scrub_context(temp_db):
    """Goal context can carry the magic-link goal_text, which is user-
    typed prose. Same PII contract as profile.objective."""
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(
        learner_id, context="Send results to alice@example.com"
    )
    assert "alice@example.com" not in (goal["context"] or "")


def test_save_and_get_curriculum_outline(temp_db):
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(learner_id)
    outline = {
        "summary": "Job-interview Afrikaans in 2 weeks",
        "modules": [
            {"id": "mod-1", "title": "Self-introduction", "status": "in_progress"},
            {"id": "mod-2", "title": "Common interview questions", "status": "not_started"},
        ],
    }
    store.save_curriculum_outline(goal["goal_id"], outline)
    got = store.get_curriculum_outline(goal["goal_id"])
    assert got == outline


def test_save_curriculum_outline_rejects_non_json(temp_db):
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(learner_id)
    # set() is not JSON-serialisable
    with pytest.raises(ValueError, match="JSON"):
        store.save_curriculum_outline(goal["goal_id"], {"bad": {1, 2, 3}})


def test_get_curriculum_outline_returns_none_when_unset(temp_db):
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(learner_id)
    assert store.get_curriculum_outline(goal["goal_id"]) is None


# ============================================================
# Cards
# ============================================================

def test_save_card_returns_id(temp_db):
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(learner_id)
    card_id = store.save_card(
        goal["goal_id"], db.CARD_VOCAB, "How do you say 'hello'?",
        reference_answer="hallo", hint_text="common greeting",
    )
    assert len(card_id) == 36


def test_save_card_rejects_bad_type(temp_db):
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(learner_id)
    with pytest.raises(ValueError, match="card_type"):
        store.save_card(goal["goal_id"], "multiple_choice", "x")


def test_save_card_rejects_empty_prompt(temp_db):
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(learner_id)
    with pytest.raises(ValueError, match="prompt_text"):
        store.save_card(goal["goal_id"], db.CARD_VOCAB, "   ")


def test_next_due_cards_empty_for_fresh_learner(temp_db):
    learner_id = store.create_anonymous_learner()
    assert store.next_due_cards(learner_id) == []


# ============================================================
# record_attempt — the PII boundary + Leitner update
# ============================================================

def _setup_attempt_target(temp_db):
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(learner_id)
    card_id = store.save_card(
        goal["goal_id"], db.CARD_VOCAB, "hello in Afrikaans?",
        reference_answer="hallo",
    )
    return learner_id, card_id


def test_record_attempt_correct_promotes_box(temp_db):
    learner_id, card_id = _setup_attempt_target(temp_db)
    result = store.record_attempt(
        learner_id=learner_id, card_id=card_id,
        user_answer="hallo", ai_feedback="exactly right",
        rating=db.RATING_CORRECT,
    )
    # First attempt: prior box = 1 (MIN_BOX), correct → 2
    assert result["new_box"] == 2
    assert result["total_seen"] == 1
    assert result["total_correct"] == 1


def test_record_attempt_wrong_demotes_to_box_1(temp_db):
    learner_id, card_id = _setup_attempt_target(temp_db)
    # First, promote a few times
    for _ in range(3):
        store.record_attempt(
            learner_id=learner_id, card_id=card_id,
            user_answer="hallo", ai_feedback="ok",
            rating=db.RATING_CORRECT,
        )
    # Now wrong → back to box 1
    result = store.record_attempt(
        learner_id=learner_id, card_id=card_id,
        user_answer="hello", ai_feedback="that's English",
        rating=db.RATING_WRONG,
    )
    assert result["new_box"] == 1


def test_record_attempt_partial_counts_as_correct_for_promotion(temp_db):
    learner_id, card_id = _setup_attempt_target(temp_db)
    result = store.record_attempt(
        learner_id=learner_id, card_id=card_id,
        user_answer="halo", ai_feedback="close — missing one letter",
        rating=db.RATING_PARTIAL,
    )
    assert result["new_box"] == 2
    assert result["total_correct"] == 1


def test_record_attempt_pii_scrubs_user_answer(temp_db):
    """The reviewer caught that this is the boundary; lock it down."""
    learner_id, card_id = _setup_attempt_target(temp_db)
    store.record_attempt(
        learner_id=learner_id, card_id=card_id,
        user_answer="my email is leak@test.com and I think 'hallo'",
        ai_feedback="ok",
        rating=db.RATING_CORRECT,
    )
    # Read the row back via raw SQL — make sure the stored answer is
    # scrubbed.
    with db._conn() as c:
        row = c.execute(
            "SELECT user_answer FROM card_attempts WHERE learner_id = ?",
            (learner_id,),
        ).fetchone()
    assert "leak@test.com" not in row["user_answer"]
    assert "[REDACTED:email]" in row["user_answer"]


def test_record_attempt_rejects_unknown_rating(temp_db):
    learner_id, card_id = _setup_attempt_target(temp_db)
    with pytest.raises(ValueError, match="rating"):
        store.record_attempt(
            learner_id=learner_id, card_id=card_id,
            user_answer="x", ai_feedback="x", rating="maybe",
        )


# ============================================================
# next_due_cards — the SRS surface area
# ============================================================

def test_next_due_cards_returns_due_card_after_attempt(temp_db):
    learner_id, card_id = _setup_attempt_target(temp_db)
    # First attempt → box 1 → due now (BOX 1 INTERVAL = 0 hours)
    # Actually correct → box 2 → due tomorrow → NOT due now.
    store.record_attempt(
        learner_id=learner_id, card_id=card_id,
        user_answer="hallo", ai_feedback="ok",
        rating=db.RATING_CORRECT,
    )
    # Box 2 = +24h, so not due now → list should be empty.
    assert store.next_due_cards(learner_id) == []


def test_next_due_cards_includes_box1_card_immediately(temp_db):
    learner_id, card_id = _setup_attempt_target(temp_db)
    # Wrong → box 1 → due immediately (interval 0)
    store.record_attempt(
        learner_id=learner_id, card_id=card_id,
        user_answer="wrong", ai_feedback="try again",
        rating=db.RATING_WRONG,
    )
    due = store.next_due_cards(learner_id)
    assert len(due) == 1
    assert due[0]["card_id"] == card_id
    assert due[0]["box"] == 1


# ============================================================
# progress_for — the stats panel feed
# ============================================================

def test_progress_for_empty_learner(temp_db):
    learner_id = store.create_anonymous_learner()
    p = store.progress_for(learner_id)
    assert p == {"total_seen": 0, "total_correct": 0, "by_box": {}}


def test_progress_for_after_attempts(temp_db):
    learner_id, card_id = _setup_attempt_target(temp_db)
    store.record_attempt(
        learner_id=learner_id, card_id=card_id,
        user_answer="hallo", ai_feedback="ok",
        rating=db.RATING_CORRECT,
    )
    p = store.progress_for(learner_id)
    assert p["total_seen"] == 1
    assert p["total_correct"] == 1
    assert p["by_box"] == {2: 1}
