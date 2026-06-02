"""LearnerContext assembler tests."""
from __future__ import annotations

import pytest

from ongiini.learning import context, db, store


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    from ongiini import config
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    db.warmup()
    return tmp_path / "learning.sqlite"


def test_context_for_cold_visitor_minimal(temp_db):
    """No profile yet, no goal yet. Should not crash; should return
    a coherent shell so the LLM can drive the intake conversation."""
    learner_id = store.create_anonymous_learner()
    ctx = context.build_learner_context(learner_id)
    assert ctx.learner_id == learner_id
    assert ctx.profile is None
    assert ctx.goal_id is None
    assert ctx.goal_context is None
    assert ctx.recent_excerpts == []
    assert ctx.mem0_facts == []
    assert ctx.curriculum_outline is None
    assert ctx.progress == {"total_seen": 0, "total_correct": 0, "by_box": {}}


def test_context_picks_up_profile_after_intake(temp_db):
    learner_id = store.create_anonymous_learner()
    store.save_profile_field(learner_id, "name", "Sebastian")
    store.save_profile_field(learner_id, "age", 35)
    store.save_profile_field(learner_id, "current_level", "beginner")
    store.save_profile_field(learner_id, "objective", "job interview at SPAR")

    ctx = context.build_learner_context(learner_id)
    assert ctx.profile is not None
    assert ctx.profile["name"] == "Sebastian"
    assert ctx.profile["age"] == 35
    # profile.objective is exposed via ctx.profile['objective'] for
    # use as a fallback when no goal title is set. It is NO LONGER
    # silently copied into ctx.goal_context (see
    # test_goal_context_no_longer_silently_falls_back_to_profile_objective).
    assert ctx.profile["objective"] == "job interview at SPAR"
    assert ctx.goal_context is None


def test_context_goal_context_overrides_profile(temp_db):
    """Magic-link arrival carries a fresh objective — should win over
    the profile objective if both are set."""
    learner_id = store.create_anonymous_learner()
    store.save_profile_field(learner_id, "objective", "old goal")
    ctx = context.build_learner_context(
        learner_id, goal_context="brand new goal from magic link"
    )
    assert ctx.goal_context == "brand new goal from magic link"


def test_context_picks_up_curriculum_outline(temp_db):
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(learner_id)
    store.save_curriculum_outline(
        goal["goal_id"],
        {"summary": "test plan", "modules": [{"id": "m1", "title": "x"}]},
    )
    ctx = context.build_learner_context(learner_id, goal_id=goal["goal_id"])
    assert ctx.goal_id == goal["goal_id"]
    assert ctx.curriculum_outline is not None
    assert ctx.curriculum_outline["summary"] == "test plan"


def test_context_progress_reflects_attempts(temp_db):
    learner_id = store.create_anonymous_learner()
    goal = store.get_or_create_active_goal(learner_id)
    card_id = store.save_card(goal["goal_id"], db.CARD_VOCAB, "hello?")
    store.record_attempt(
        learner_id=learner_id, card_id=card_id,
        user_answer="hallo", ai_feedback="ok",
        rating=db.RATING_CORRECT,
    )
    ctx = context.build_learner_context(learner_id, goal_id=goal["goal_id"])
    assert ctx.progress["total_seen"] == 1
    assert ctx.progress["total_correct"] == 1


def test_context_phase_2_slots_present_but_empty(temp_db):
    """recent_excerpts and mem0_facts are part of the shape from day
    one so prompts can reference them; populated in Phase 2."""
    learner_id = store.create_anonymous_learner()
    ctx = context.build_learner_context(learner_id)
    assert ctx.recent_excerpts == []
    assert ctx.mem0_facts == []


def test_context_is_frozen(temp_db):
    """Prompt code shouldn't be able to mutate the assembled view."""
    learner_id = store.create_anonymous_learner()
    ctx = context.build_learner_context(learner_id)
    with pytest.raises(Exception):
        ctx.learner_id = "different"     # type: ignore[misc]


def test_goal_title_overrides_stale_profile_objective(temp_db):
    """Regression for the "restaurant when title says Job Interview at
    Serviceplan" bug. When a learner picks a fresh focus via
    /goals/new, the goal's TITLE must drive the curriculum prompt
    instead of an older intake objective."""
    learner_id = store.create_anonymous_learner()
    # Older intake objective on the profile.
    store.save_profile_field(learner_id, "objective", "ordering food at restaurant")
    goal = store.create_new_goal(
        learner_id, title="Job Interview at Serviceplan",
        language="german", source_language="english",
    )
    ctx = context.build_learner_context(
        learner_id, goal_id=goal["goal_id"],
    )
    assert ctx.goal_title == "Job Interview at Serviceplan"
    # profile.objective is still on the profile (kept for fallback /
    # history); the prompt's "focus" line will read goal_title first.
    assert (ctx.profile or {}).get("objective") == "ordering food at restaurant"


def test_goal_context_no_longer_silently_falls_back_to_profile_objective(temp_db):
    """When a goal has no context column set, ctx.goal_context must
    be None — not stealthily populated with profile.objective. Prior
    behaviour caused stale objectives to override fresh goal titles."""
    learner_id = store.create_anonymous_learner()
    store.save_profile_field(learner_id, "objective", "old intake answer")
    goal = store.create_new_goal(
        learner_id, title="fresh focus",
        language="german", source_language="english",
    )
    ctx = context.build_learner_context(
        learner_id, goal_id=goal["goal_id"],
    )
    assert ctx.goal_context is None
