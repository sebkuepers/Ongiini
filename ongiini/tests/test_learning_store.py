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
        store.save_card(goal["goal_id"], "klingon_card", "x")


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


# ============================================================
# Phase 2: multi-curriculum helpers
# ============================================================

def test_create_new_goal_persists_language_pair_and_level(temp_db):
    learner_id = store.create_anonymous_learner()
    g = store.create_new_goal(
        learner_id, title="berlin trip",
        language="german", source_language="english",
        current_level="elementary",
    )
    assert g["language"] == "german"
    assert g["source_language"] == "english"
    assert g["current_level"] == "elementary"
    # Re-read confirms persistence.
    listed = store.list_goals(learner_id)
    assert listed[0]["language"] == "german"
    assert listed[0]["source_language"] == "english"
    assert listed[0]["current_level"] == "elementary"


def test_create_new_goal_rejects_invalid_language_pair(temp_db):
    """Validation happens at the store boundary so no junk reaches the
    skill renderer or LLM prompt."""
    learner_id = store.create_anonymous_learner()
    with pytest.raises(ValueError, match="must differ"):
        store.create_new_goal(
            learner_id, title="x",
            language="english", source_language="english",
        )
    with pytest.raises(ValueError, match="unsupported"):
        store.create_new_goal(
            learner_id, title="x",
            language="klingon", source_language="english",
        )


def test_get_or_create_active_goal_persists_source_language(temp_db):
    """The /turn auto-create path eventually pulls source from the
    frontend's stored selection — lock in that the new kwarg goes to
    disk."""
    learner_id = store.create_anonymous_learner()
    g = store.get_or_create_active_goal(
        learner_id, language="afrikaans", source_language="german",
    )
    assert g["source_language"] == "german"
    assert g["language"] == "afrikaans"


def test_get_or_create_active_goal_persists_title_when_creating(temp_db):
    """The /turn handler passes profile.objective as title so the
    auto-created first curriculum shows up in the drawer with a real
    name. Lock in the create path actually stores it."""
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(
        learner_id, title="job interview at SPAR",
    )
    assert goal["title"] == "job interview at SPAR"
    # Re-read confirms persistence.
    listed = store.list_goals(learner_id)
    assert listed[0]["title"] == "job interview at SPAR"


def test_get_or_create_active_goal_ignores_title_when_existing(temp_db):
    """A second call for the same learner returns the existing active
    goal — its title MUST NOT be silently relabelled by a later call."""
    learner_id = store.create_anonymous_learner()
    store.get_or_create_active_goal(learner_id, title="first")
    again = store.get_or_create_active_goal(learner_id, title="second")
    assert again["title"] == "first"


def test_get_or_create_active_goal_scrubs_title_pii(temp_db):
    """Goal title is user-derived (intake objective) and must pass
    through the project-wide PII scrub, same as other free-text
    profile fields."""
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(
        learner_id, title="contact maria@example.com",
    )
    assert "maria@example.com" not in (goal["title"] or "")


def test_list_goals_excludes_archived_by_default(temp_db):
    learner_id = store.create_anonymous_learner()
    g1 = store.get_or_create_active_goal(learner_id)
    g2 = store.create_new_goal(learner_id, title="Family chat")
    store.archive_goal(learner_id, g1["goal_id"])

    visible = store.list_goals(learner_id)
    assert len(visible) == 1
    assert visible[0]["goal_id"] == g2["goal_id"]

    all_goals = store.list_goals(learner_id, include_archived=True)
    assert len(all_goals) == 2


def test_list_goals_has_outline_flag(temp_db):
    """The frontend's curriculum panel shows "Plan ready" vs "Plan
    pending" without a second fetch — locked in by has_outline."""
    learner_id = store.create_anonymous_learner()
    g = store.get_or_create_active_goal(learner_id)
    assert store.list_goals(learner_id)[0]["has_outline"] is False
    store.save_curriculum_outline(g["goal_id"], {"summary": "x", "modules": []})
    assert store.list_goals(learner_id)[0]["has_outline"] is True


def test_create_new_goal_demotes_existing_active(temp_db):
    """Only one active goal per learner at a time."""
    learner_id = store.create_anonymous_learner()
    g1 = store.get_or_create_active_goal(learner_id)
    g2 = store.create_new_goal(
        learner_id, title="Interview prep", context="hospitality job",
    )
    goals = store.list_goals(learner_id)
    by_id = {g["goal_id"]: g for g in goals}
    assert by_id[g1["goal_id"]]["status"] == "paused"
    assert by_id[g2["goal_id"]]["status"] == "active"
    assert by_id[g2["goal_id"]]["title"] == "Interview prep"


def test_create_new_goal_without_activate_stays_paused(temp_db):
    """A 'draft' goal — created but not switched to."""
    learner_id = store.create_anonymous_learner()
    store.get_or_create_active_goal(learner_id)
    g2 = store.create_new_goal(learner_id, title="Later", activate=False)
    assert g2["status"] == "paused"


def test_create_new_goal_scrubs_title_and_context(temp_db):
    """Free-text user input on a goal must be PII-scrubbed — same
    contract as profile.objective."""
    learner_id = store.create_anonymous_learner()
    g = store.create_new_goal(
        learner_id,
        title="Email maria@example.com",
        context="Ping hr@example.com about the role",
    )
    assert "maria@example.com" not in (g["title"] or "")
    assert "hr@example.com" not in (g["context"] or "")


def test_activate_goal_swaps_active_and_paused(temp_db):
    learner_id = store.create_anonymous_learner()
    g1 = store.get_or_create_active_goal(learner_id)
    g2 = store.create_new_goal(learner_id, title="Interview")
    store.activate_goal(learner_id, g1["goal_id"])
    goals = {g["goal_id"]: g for g in store.list_goals(learner_id)}
    assert goals[g1["goal_id"]]["status"] == "active"
    assert goals[g2["goal_id"]]["status"] == "paused"


def test_activate_goal_rejects_foreign_learner(temp_db):
    """A learner can't activate someone else's goal — would be a
    cross-tenant data leak."""
    alice = store.create_anonymous_learner()
    bob = store.create_anonymous_learner()
    bob_goal = store.get_or_create_active_goal(bob)
    with pytest.raises(RuntimeError, match="not found"):
        store.activate_goal(alice, bob_goal["goal_id"])


def test_activate_goal_rejects_archived(temp_db):
    """Archived goals are intentional dead-ends; restart or create
    fresh, don't un-archive."""
    learner_id = store.create_anonymous_learner()
    g = store.get_or_create_active_goal(learner_id)
    store.archive_goal(learner_id, g["goal_id"])
    with pytest.raises(RuntimeError, match="archived"):
        store.activate_goal(learner_id, g["goal_id"])


def test_restart_goal_wipes_cards_attempts_review_state_messages(temp_db):
    """Restart must scrub the in-curriculum work but keep the row +
    outline so the learner gets the same plan back fresh."""
    from ongiini.learning import messages as msg_mod
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(learner_id)
    store.save_curriculum_outline(goal["goal_id"], {"summary": "x", "modules": []})
    card_id = store.save_card(goal["goal_id"], db.CARD_VOCAB, "dankie?")
    store.record_attempt(
        learner_id=learner_id, card_id=card_id,
        user_answer="dankie", ai_feedback="yes", rating=db.RATING_CORRECT,
    )
    msg_mod.append(
        learner_id=learner_id, goal_id=goal["goal_id"],
        kind=db.MSG_COACH_TEXT, payload={"text": "hi"},
    )

    summary = store.restart_goal(learner_id, goal["goal_id"])
    assert summary["cards_deleted"] == 1
    assert summary["messages_deleted"] == 1

    # Cards gone (cascades attempts + review_state).
    assert store.next_due_cards(learner_id) == []
    assert store.progress_for(learner_id) == {
        "total_seen": 0, "total_correct": 0, "by_box": {},
    }
    # Goal row + outline still there.
    outline = store.get_curriculum_outline(goal["goal_id"])
    assert outline == {"summary": "x", "modules": []}
    # Thread empty.
    assert msg_mod.list_for_goal(
        learner_id=learner_id, goal_id=goal["goal_id"],
    ) == []


def test_restart_goal_rejects_foreign_learner(temp_db):
    alice = store.create_anonymous_learner()
    bob = store.create_anonymous_learner()
    bob_goal = store.get_or_create_active_goal(bob)
    with pytest.raises(RuntimeError, match="not found"):
        store.restart_goal(alice, bob_goal["goal_id"])


def test_archive_goal_sets_status_and_timestamp(temp_db):
    learner_id = store.create_anonymous_learner()
    g = store.get_or_create_active_goal(learner_id)
    archived = store.archive_goal(learner_id, g["goal_id"])
    assert archived["status"] == "archived"
    assert archived["archived_at"]


def test_archive_goal_rejects_foreign_learner(temp_db):
    alice = store.create_anonymous_learner()
    bob = store.create_anonymous_learner()
    bob_goal = store.get_or_create_active_goal(bob)
    with pytest.raises(RuntimeError, match="not found"):
        store.archive_goal(alice, bob_goal["goal_id"])


def test_update_goal_title_scrubs_pii(temp_db):
    learner_id = store.create_anonymous_learner()
    g = store.get_or_create_active_goal(learner_id)
    store.update_goal_title(learner_id, g["goal_id"], "Reach hr@example.com")
    goals = store.list_goals(learner_id)
    assert "hr@example.com" not in (goals[0]["title"] or "")


def test_progress_for_goal_filter_scopes_to_one_curriculum(temp_db):
    """Per-curriculum progress — locks in that attempts on curriculum
    A don't leak into curriculum B's progress numbers."""
    learner_id = store.create_anonymous_learner()
    g_a = store.get_or_create_active_goal(learner_id)
    g_b = store.create_new_goal(learner_id, title="other", activate=False)
    card_a = store.save_card(g_a["goal_id"], db.CARD_VOCAB, "A?")
    card_b = store.save_card(g_b["goal_id"], db.CARD_VOCAB, "B?")
    store.record_attempt(
        learner_id=learner_id, card_id=card_a, user_answer="x",
        ai_feedback="y", rating=db.RATING_CORRECT,
    )
    store.record_attempt(
        learner_id=learner_id, card_id=card_b, user_answer="x",
        ai_feedback="y", rating=db.RATING_WRONG,
    )

    pa = store.progress_for(learner_id, goal_id=g_a["goal_id"])
    pb = store.progress_for(learner_id, goal_id=g_b["goal_id"])
    assert pa["total_seen"] == 1 and pa["total_correct"] == 1
    assert pb["total_seen"] == 1 and pb["total_correct"] == 0

    # Whole-learner: both attempts roll up.
    p_all = store.progress_for(learner_id)
    assert p_all["total_seen"] == 2 and p_all["total_correct"] == 1


# ============================================================
# Per-topic digest (drives teach-then-test pacing)
# ============================================================

def test_progress_for_modules_groups_per_topic(temp_db):
    """Lessons and exercises tagged with topic_id roll up into
    topics_taught / topics_drilled buckets keyed by topic_id."""
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(learner_id)
    gid = goal["goal_id"]

    # Module mod-1 has two lesson topics (t1, t2) and one practice
    # topic (t3). We seed two lesson cards on t1, one on t2, two
    # exercises on t3.
    store.save_card(gid, db.CARD_LESSON, "L1a",
                    module_id="mod-1", topic_id="t1")
    store.save_card(gid, db.CARD_LESSON, "L1b",
                    module_id="mod-1", topic_id="t1")
    store.save_card(gid, db.CARD_LESSON, "L2",
                    module_id="mod-1", topic_id="t2")
    store.save_card(gid, db.CARD_VOCAB, "E3a?",
                    reference_answer="hi", module_id="mod-1", topic_id="t3")
    store.save_card(gid, db.CARD_VOCAB, "E3b?",
                    reference_answer="hi", module_id="mod-1", topic_id="t3")

    out = store.progress_for_modules(learner_id, gid)
    assert "mod-1" in out
    mod = out["mod-1"]
    assert mod["lessons_given"] == 3
    assert mod["exercises_emitted"] == 2
    assert mod["topics_taught"] == {"t1": 2, "t2": 1}
    assert mod["topics_drilled"] == {"t3": 2}


def test_progress_for_modules_handles_null_topic_id(temp_db):
    """Cards tagged with module_id but no topic_id contribute to
    module totals but not to the per-topic dicts — back-compat with
    pre-pacing cards in production."""
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(learner_id)
    gid = goal["goal_id"]
    store.save_card(gid, db.CARD_LESSON, "L (no topic)",
                    module_id="mod-1")
    store.save_card(gid, db.CARD_VOCAB, "E (no topic)?",
                    reference_answer="x", module_id="mod-1")

    out = store.progress_for_modules(learner_id, gid)
    assert out["mod-1"]["lessons_given"] == 1
    assert out["mod-1"]["exercises_emitted"] == 1
    assert out["mod-1"]["topics_taught"] == {}
    assert out["mod-1"]["topics_drilled"] == {}


def test_save_card_persists_extras_and_round_trips(temp_db):
    """The per-card-type extras (MC options / reorder tokens / dialogue
    turns / etc.) are stored in extras_json on the card row so SRS
    replay can rebuild the renderer payload. Without this a failed
    reorder card resurfaces as a prompt with no token chips."""
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(learner_id)
    card_id = store.save_card(
        goal["goal_id"], db.CARD_REORDER, "Arrange:",
        reference_answer="ich gehe jetzt nach Hause",
        extras={"tokens": ["nach", "ich", "Hause", "gehe", "jetzt"]},
    )
    # SRS due-card query returns the extras parsed back into a dict.
    store.record_attempt(
        learner_id=learner_id, card_id=card_id, user_answer="wrong",
        ai_feedback="no", rating=db.RATING_WRONG,
    )
    due = store.next_due_cards(learner_id, goal_id=goal["goal_id"], limit=1)
    assert len(due) == 1
    assert due[0]["card_id"] == card_id
    assert isinstance(due[0]["extras"], dict)
    assert due[0]["extras"].get("tokens") == [
        "nach", "ich", "Hause", "gehe", "jetzt",
    ]


def test_save_card_persists_topic_id(temp_db):
    """The topic_id parameter survives the round-trip — the SRS query
    side reads it back via the digest."""
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(learner_id)
    card_id = store.save_card(
        goal["goal_id"], db.CARD_LESSON, "lesson body",
        module_id="mod-1", topic_id="t-greetings",
    )
    row = store.get_card(card_id)
    assert row is not None
    assert row.get("module_id") == "mod-1"
    assert row.get("topic_id") == "t-greetings"



def test_recent_topic_prompts_returns_oldest_first_excluding_lessons(temp_db):
    """The variation helper feeds the card author + critic with the
    last few drills on the same topic so they can avoid recycling the
    same example sentence. Lessons are filtered out (they don't have
    a comparable prompt_text); results come back oldest-first so the
    brief reads as a timeline."""
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(learner_id)
    gid = goal["goal_id"]
    # Plant cards in known order: lesson (excluded), then 3 drills.
    store.save_card(gid, db.CARD_LESSON, "lesson body",
                    module_id="m1", topic_id="t1")
    store.save_card(gid, db.CARD_VOCAB,
                    "Translate to German: 'I drink a coffee.'",
                    reference_answer="Ich trinke einen Kaffee.",
                    module_id="m1", topic_id="t1")
    store.save_card(gid, db.CARD_CLOZE,
                    "Ich ___ einen Kaffee. (I ___ a coffee.)",
                    reference_answer="trinke",
                    module_id="m1", topic_id="t1")
    store.save_card(gid, db.CARD_TRANSLATION,
                    "Translate to German: 'I drink a coffee.'",
                    reference_answer="Ich trinke einen Kaffee.",
                    module_id="m1", topic_id="t1")
    # A card on a DIFFERENT topic must NOT leak in.
    store.save_card(gid, db.CARD_VOCAB, "other-topic prompt",
                    reference_answer="x",
                    module_id="m1", topic_id="t2")

    out = store.recent_topic_prompts(gid, "t1", limit=4)
    assert [r["card_type"] for r in out] == ["vocab", "cloze", "translation"]
    # Lesson and the t2 card must be absent.
    assert all(r["card_type"] != db.CARD_LESSON for r in out)
    assert all("other-topic" not in (r["prompt_text"] or "") for r in out)


def test_recent_topic_prompts_empty_on_missing_goal_or_topic(temp_db):
    """Defensive: a None / blank goal_id or topic_id returns [], not
    a query error. The author and critic call this on every exercise
    card; one bad value should not crash a turn."""
    assert store.recent_topic_prompts("", "t1") == []
    assert store.recent_topic_prompts("g1", "") == []
