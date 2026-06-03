"""Coach orchestrator tests — routing, race-safety, PII, error meta."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from owela import ModelRequest, ModelResponse

from ongiini.learning import (
    coach, context as ctx_mod, db, messages, store, turn_classifier,
)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    from ongiini import config
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    db.warmup()
    return tmp_path / "learning.sqlite"


@dataclass
class FakeModel:
    """Returns the next response in ``responses`` for each successive
    call, or ``response`` on every call when ``responses`` is empty.
    Captures every request so tests can assert."""
    response: str = ""
    responses: list[str] = field(default_factory=list)
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, req: ModelRequest) -> ModelResponse:
        self.requests.append(req)
        if self.responses:
            body = self.responses.pop(0)
        else:
            body = self.response
        return ModelResponse(
            content=body, tool_calls=[],
            finish_reason="stop", tokens_in=5, tokens_out=5,
            cached_tokens=0, raw=None,
        )


# Stock JSON responses for common steps. With the selector-driven
# architecture, outlines MUST include a ``topics`` list for the
# selector to find an in-progress topic to author. A bare module
# (no topics) would graduate the learner immediately.
_OUTLINE = json.dumps({
    "summary": "Job-interview Afrikaans in 2 weeks.",
    "modules": [
        {"id": "mod-1", "title": "Greetings", "status": "in_progress",
         "estimated_cards": 8,
         "topics": [
             {"id": "t1", "title": "Hello + hi", "kind": "lesson"},
             {"id": "t2", "title": "Goodbye", "kind": "lesson"},
             {"id": "p1", "title": "Drill greetings", "kind": "practice"},
         ]},
    ],
})
# Content payload for an exercise card — no card_type / module_id /
# topic_id, those are attached by the coach after generate_card_content.
_EXERCISE = json.dumps({
    "prompt_text": "How do you say 'thank you'?",
    "reference_answer": "dankie",
})
# Content payload for a lesson card — steps[] only, no scaffolding.
_LESSON = json.dumps({
    "title": "Time-of-day greetings",
    "steps": [
        {"kind": "concept", "body": "Greetings vary by time of day."},
        {"kind": "example", "body": "Examples:",
         "examples": ["goeie môre", "goeie naand"]},
    ],
})
# The design-review loop calls the critic immediately after the
# designer. Tests that just want the curriculum to ship can queue
# this "ready on iter 1" response right after the outline.
#
# Same payload is reused for the per-card critic that fires after
# every ``generate_card_content`` call — also a {"ready": ...}
# shape, so each card emission needs ONE critic response queued
# right after the design response.
_CRITIC_READY = json.dumps({"ready": True, "score": 9, "issues": []})


def _setup(temp_db):
    learner_id = store.create_anonymous_learner()
    store.save_profile_field(learner_id, "name", "Sebastian")
    store.save_profile_field(learner_id, "age", 35)
    store.save_profile_field(learner_id, "current_level", "beginner")
    store.save_profile_field(learner_id, "objective", "job interview at SPAR")
    store.mark_intake_complete(learner_id)
    goal = store.get_or_create_active_goal(learner_id)
    return learner_id, goal["goal_id"]


# ============================================================
# run_turn — empty text + no active card → produce next thing
# ============================================================

@pytest.mark.asyncio
async def test_run_turn_no_text_no_active_card_designs_outline_and_emits_lesson(temp_db):
    """Fresh outline with lesson topics → the selector picks LESSON
    for the first lesson topic. (This was MSG_EXERCISE pre-selector
    when the LLM picked the card_type; now the selector picks and
    teach-first is the rule.)"""
    learner_id, goal_id = _setup(temp_db)
    fm = FakeModel(responses=[
        _OUTLINE, _CRITIC_READY,    # outline + curriculum critic
        _LESSON, _CRITIC_READY,     # card + card critic
    ])
    out = await coach.run_turn(
        learner_id=learner_id, goal_id=goal_id,
        user_text=None, model=fm, skill_content="SKILL",
    )
    assert len(out) == 1
    assert out[0]["kind"] == db.MSG_LESSON
    # Outline got persisted as a side-effect.
    assert store.get_curriculum_outline(goal_id) is not None


@pytest.mark.asyncio
async def test_run_turn_no_text_with_active_exercise_returns_nothing(temp_db):
    """If the learner is mid-card and asks for 'what's next' (a
    button press without text), we don't push another card."""
    learner_id, goal_id = _setup(temp_db)
    # Manually plant an unanswered exercise.
    card_id = store.save_card(goal_id, db.CARD_VOCAB, "x?")
    messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_EXERCISE,
        payload={"card_type": "vocab", "prompt_text": "x?"},
        card_id=card_id,
    )
    fm = FakeModel()
    out = await coach.run_turn(
        learner_id=learner_id, goal_id=goal_id,
        user_text=None, model=fm, skill_content="SKILL",
    )
    assert out == []
    assert fm.requests == []     # no model calls wasted


# ============================================================
# Verdict routing
# ============================================================

@pytest.mark.asyncio
async def test_answer_verdict_grades_and_advances(temp_db):
    learner_id, goal_id = _setup(temp_db)
    card_id = store.save_card(
        goal_id, db.CARD_VOCAB, "How do you say 'thank you'?",
        reference_answer="dankie",
    )
    messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_EXERCISE,
        payload={"card_type": "vocab", "prompt_text": "x?"},
        card_id=card_id,
    )
    fm = FakeModel(responses=[
        # classifier → answer
        '{"verdict": "answer"}',
        # grading → correct
        '{"rating": "correct", "feedback": "Yes, dankie."}',
        # next thing: outline (designed lazily on first call)
        _OUTLINE,
        # curriculum critic approves on iter 1
        _CRITIC_READY,
        # ...then the next card. The pre-planted exercise wasn't
        # selector-anchored, so the selector still sees no taught
        # lessons → picks LESSON for the first lesson topic.
        _LESSON,
        # card critic approves on iter 1
        _CRITIC_READY,
    ])
    out = await coach.run_turn(
        learner_id=learner_id, goal_id=goal_id,
        user_text="dankie", model=fm, skill_content="SKILL",
    )
    kinds = [m["kind"] for m in out]
    # learner_text → feedback → progress → next card (a lesson now)
    assert kinds[0] == db.MSG_LEARNER_TEXT
    assert db.MSG_FEEDBACK in kinds
    assert db.MSG_PROGRESS in kinds
    assert kinds[-1] == db.MSG_LESSON


@pytest.mark.asyncio
async def test_question_verdict_invokes_coach_response(temp_db):
    learner_id, goal_id = _setup(temp_db)
    fm = FakeModel(responses=[
        # classifier → question
        '{"verdict": "question"}',
        # coach answer
        '{"text": "Good question — \'ek het\' means \'I have\'."}',
    ])
    out = await coach.run_turn(
        learner_id=learner_id, goal_id=goal_id,
        user_text="what does ek het mean?",
        model=fm, skill_content="SKILL",
    )
    assert len(out) == 2
    assert out[0]["kind"] == db.MSG_LEARNER_TEXT
    assert out[1]["kind"] == db.MSG_COACH_TEXT
    assert "ek het" in out[1]["payload"]["text"].lower()


@pytest.mark.asyncio
async def test_selector_drives_lesson_lesson_exercise_sequence(temp_db):
    """End-to-end: the selector enforces teach-then-test across
    multiple turns. With 2 lesson topics + 1 practice topic in the
    outline, the first two turns are lessons (t1, then t2), and
    the third turn is an exercise on the practice topic.

    This is the canonical sequence Sebastian asked for: lessons
    until lessons are done, THEN drills. No LLM-as-planner; pure
    deterministic flow."""
    learner_id, goal_id = _setup(temp_db)
    # Plant the outline directly so we don't burn responses on
    # design+critic for this sequence test.
    store.save_curriculum_outline(goal_id, json.loads(_OUTLINE))

    # Turn 1 — selector picks LESSON for t1. Card critic approves
    # immediately, so the queue is [design, critic].
    fm = FakeModel(responses=[_LESSON, _CRITIC_READY])
    out = await coach.run_turn(
        learner_id=learner_id, goal_id=goal_id,
        user_text=None, model=fm, skill_content="SKILL",
    )
    assert out[-1]["kind"] == db.MSG_LESSON
    assert out[-1]["payload"].get("title")

    # Turn 2 — t1 is taught (lessons_given[t1]=1). Selector picks
    # LESSON for t2 (NOT a repeat of t1, which was Sebastian's bug).
    fm = FakeModel(responses=[_LESSON, _CRITIC_READY])
    out = await coach.run_turn(
        learner_id=learner_id, goal_id=goal_id,
        user_text=None, model=fm, skill_content="SKILL",
    )
    assert out[-1]["kind"] == db.MSG_LESSON

    # Turn 3 — both lesson topics taught. Selector picks EXERCISE
    # for the practice topic p1.
    fm = FakeModel(responses=[_EXERCISE, _CRITIC_READY])
    out = await coach.run_turn(
        learner_id=learner_id, goal_id=goal_id,
        user_text=None, model=fm, skill_content="SKILL",
    )
    assert out[-1]["kind"] == db.MSG_EXERCISE
    assert out[-1]["payload"].get("card_type") == "vocab"   # first in rotation


@pytest.mark.asyncio
async def test_off_topic_verdict_redirects_without_model_call(temp_db):
    """Off-topic redirect is templated — no LLM call needed beyond
    the classifier."""
    learner_id, goal_id = _setup(temp_db)
    fm = FakeModel(responses=[
        # classifier → off_topic
        '{"verdict": "off_topic"}',
    ])
    out = await coach.run_turn(
        learner_id=learner_id, goal_id=goal_id,
        user_text="what's the weather like?",
        model=fm, skill_content="SKILL",
    )
    assert len(out) == 2
    assert out[1]["kind"] == db.MSG_COACH_TEXT
    txt = out[1]["payload"]["text"]
    assert "Afrikaans" in txt
    assert "chat.ongiini.ai" in txt or "WhatsApp" in txt
    # Only the classifier was called — no extra LLM call for the
    # redirect text.
    assert len(fm.requests) == 1


# ============================================================
# Race safety — the critical fix
# ============================================================

def test_claim_exercise_atomic_one_winner(temp_db):
    """The atomic claim is the load-bearing race fix. Two sequential
    callers attempting to claim the same exercise: only the first wins.
    This is the scenario the reviewer flagged (double-tap, browser
    retry) — the loser must bail out without grading."""
    learner_id, goal_id = _setup(temp_db)
    card_id = store.save_card(goal_id, db.CARD_VOCAB, "thank you?")
    msg = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_EXERCISE,
        payload={"card_type": "vocab", "prompt_text": "thank you?"},
        card_id=card_id,
    )

    # First caller claims successfully.
    assert messages.claim_exercise(msg["message_id"]) is True

    # Second caller can no longer claim — the WHERE answered=0 clause
    # filters the row out. This is what guards grade_answer +
    # record_attempt against double-execution.
    assert messages.claim_exercise(msg["message_id"]) is False

    # Sanity: the row is now marked answered.
    listed = messages.list_for_goal(learner_id=learner_id, goal_id=goal_id)
    assert any(m["message_id"] == msg["message_id"] and m["answered"]
               for m in listed)


def test_claim_exercise_empty_id_returns_false_no_raise(temp_db):
    """Empty message_id is treated as 'someone else already claimed' —
    safer than silently proceeding with grading on a malformed row."""
    assert messages.claim_exercise("") is False
    assert messages.claim_exercise(None) is False    # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_run_turn_with_no_active_exercise_routes_to_question_fallback(temp_db):
    """Realistic state: a prior request already graded the only
    exercise on the thread. When the next request arrives, the active
    exercise is gone but the classifier may still say "answer" (the
    input *looks* like one). We MUST NOT silently drop the input — we
    route it through the question handler so the coach responds and
    NEVER call grade_answer or record_attempt for a stale card."""
    learner_id, goal_id = _setup(temp_db)
    card_id = store.save_card(goal_id, db.CARD_VOCAB, "thank you?")
    exercise_msg = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_EXERCISE,
        payload={"card_type": "vocab", "prompt_text": "thank you?"},
        card_id=card_id,
    )
    # The previous request already graded this one.
    messages.claim_exercise(exercise_msg["message_id"])

    fm = FakeModel(responses=[
        # classifier → "answer" but no active exercise to bind to
        '{"verdict": "answer"}',
        # defensive fallback re-routes to the question handler
        '{"text": "I think you meant the previous card — let me '
        'set up the next one."}',
    ])
    out = await coach.run_turn(
        learner_id=learner_id, goal_id=goal_id,
        user_text="dankie", model=fm, skill_content="SKILL",
    )
    kinds = [m["kind"] for m in out]
    # learner_text + coach_text. No feedback, no progress — we did NOT
    # grade against a stale card; the attempt log stays clean.
    assert kinds == [db.MSG_LEARNER_TEXT, db.MSG_COACH_TEXT]
    assert store.progress_for(learner_id)["total_seen"] == 0


# ============================================================
# PII contract on messages — extended scrub
# ============================================================

def test_feedback_payload_scrubbed(temp_db):
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


def test_exercise_payload_scrubbed(temp_db):
    learner_id, goal_id = _setup(temp_db)
    row = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_EXERCISE,
        payload={
            "card_type": "production",
            "prompt_text": "Email admin@spar.co.za to set up the interview",
            "hint_text": "Start with 'Beste, mariaoa@example.com'",
        },
    )
    assert "admin@spar.co.za" not in row["payload"]["prompt_text"]
    assert "mariaoa@example.com" not in row["payload"]["hint_text"]


def test_lesson_payload_scrubs_title_body_and_examples(temp_db):
    learner_id, goal_id = _setup(temp_db)
    row = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_LESSON,
        payload={
            "title": "Contact: ceo@example.com",
            "body": "Reach out to hr@example.com when ready.",
            "examples": ["mail jan@example.com", "ordinary text"],
        },
    )
    assert "ceo@example.com" not in row["payload"]["title"]
    assert "hr@example.com" not in row["payload"]["body"]
    assert "jan@example.com" not in row["payload"]["examples"][0]
    # Non-email example preserved.
    assert row["payload"]["examples"][1] == "ordinary text"


# ============================================================
# Error meta on failure paths
# ============================================================

@pytest.mark.asyncio
async def test_grading_failure_includes_error_meta(temp_db):
    learner_id, goal_id = _setup(temp_db)
    card_id = store.save_card(goal_id, db.CARD_VOCAB, "x?")
    messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_EXERCISE,
        payload={"card_type": "vocab", "prompt_text": "x?"},
        card_id=card_id,
    )
    # Classifier says answer, then grader returns malformed JSON.
    fm = FakeModel(responses=[
        '{"verdict": "answer"}',
        'not valid json',
    ])
    out = await coach.run_turn(
        learner_id=learner_id, goal_id=goal_id,
        user_text="dankie", model=fm, skill_content="SKILL",
    )
    # Last message is a coach_text with error meta.
    err_msg = out[-1]
    assert err_msg["kind"] == db.MSG_COACH_TEXT
    assert err_msg["payload"]["meta"]["error"] == "grading_failed"


@pytest.mark.asyncio
async def test_outline_design_failure_includes_error_meta(temp_db):
    learner_id, goal_id = _setup(temp_db)
    fm = FakeModel(response="garbage that won't parse as JSON")
    out = await coach.run_turn(
        learner_id=learner_id, goal_id=goal_id,
        user_text=None, model=fm, skill_content="SKILL",
    )
    err_msg = out[-1]
    assert err_msg["kind"] == db.MSG_COACH_TEXT
    assert err_msg["payload"]["meta"]["error"] == "design_outline_failed"


# ============================================================
# Classifier order — recent_pairs snapshotted BEFORE learner append
# ============================================================

def test_advance_module_when_attempted_hits_estimate(temp_db):
    """The "10 / 6" bug: a module's status stayed in_progress forever
    because nothing flipped it to completed once the learner finished
    its estimated_cards. Lock in the auto-advance: when attempted >=
    estimated_cards, the next not_started module gets promoted."""
    outline = {
        "summary": "x",
        "modules": [
            {"id": "m1", "title": "First", "status": "in_progress",
             "estimated_cards": 6},
            {"id": "m2", "title": "Second", "status": "not_started",
             "estimated_cards": 5},
        ],
    }
    digest = {
        "m1": {"exercises_attempted": 6, "exercises_emitted": 6, "lessons_given": 0,
               "exercises_correct": 5, "cards_in_module": 6},
    }
    result = coach._advance_module_if_complete(
        goal_id="goal-1", outline=outline, digest=digest,
    )
    assert result is not None
    assert result.outline["modules"][0]["status"] == "completed"
    assert result.outline["modules"][1]["status"] == "in_progress"
    assert result.previous_title == "First"
    assert result.new_title == "Second"


def test_advance_module_when_emitted_overshoots(temp_db):
    """Emit-overshoot path: model kept emitting cards without the
    learner answering — at 1.5x estimate we promote anyway so the
    learner isn't stuck drilling forever."""
    outline = {
        "summary": "x",
        "modules": [
            {"id": "m1", "title": "First", "status": "in_progress",
             "estimated_cards": 6},
            {"id": "m2", "title": "Second", "status": "not_started",
             "estimated_cards": 5},
        ],
    }
    digest = {
        "m1": {"exercises_attempted": 4, "exercises_emitted": 9, "lessons_given": 0},
    }
    result = coach._advance_module_if_complete(
        goal_id="goal-1", outline=outline, digest=digest,
    )
    assert result is not None
    assert result.outline["modules"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_srs_replay_surfaces_due_card_instead_of_generating_new(temp_db):
    """The previously-missing piece: a card the learner got wrong
    earlier and is now due for re-review must come back via the SRS
    queue, not be silently replaced by a brand-new model-authored
    card."""
    learner_id, goal_id = _setup(temp_db)
    # Plant the outline + an existing wrong-answer card whose box-1
    # next_due_at is now (the default for box 1).
    store.save_curriculum_outline(goal_id, {
        "summary": "x",
        "modules": [{"id": "m1", "title": "M", "status": "in_progress",
                     "estimated_cards": 6}],
    })
    card_id = store.save_card(
        goal_id, db.CARD_VOCAB, "How do you say 'thanks'?",
        reference_answer="dankie", module_id="m1",
    )
    store.record_attempt(
        learner_id=learner_id, card_id=card_id,
        user_answer="garbage", ai_feedback="that's not right",
        rating=db.RATING_WRONG,
    )
    # Verify the card is now in box 1 and due for review.
    due = store.next_due_cards(learner_id, goal_id=goal_id)
    assert len(due) == 1
    assert due[0]["card_id"] == card_id

    # Trigger a "what's next" turn.
    fm = FakeModel()
    out = await coach.run_turn(
        learner_id=learner_id, goal_id=goal_id,
        user_text=None, model=fm, skill_content="SKILL",
    )
    # The SRS-due card should have surfaced — no model call needed.
    exercise_msgs = [m for m in out if m["kind"] == db.MSG_EXERCISE]
    assert len(exercise_msgs) == 1
    assert exercise_msgs[0]["card_id"] == card_id
    assert exercise_msgs[0]["payload"].get("review_box") == 1
    assert fm.requests == []   # no LLM call for the card


@pytest.mark.asyncio
async def test_srs_replay_skips_just_answered_card(temp_db):
    """Anki-style: the card the learner JUST answered (which may be
    immediately due again if they got it wrong) does not come back
    back-to-back. It surfaces on the turn AFTER one new card has been
    served."""
    learner_id, goal_id = _setup(temp_db)
    # Plant an outline WITH topics so the selector can author. The
    # SRS-replay-skip rule is what's being tested; the kind of card
    # emitted afterwards isn't load-bearing.
    store.save_curriculum_outline(goal_id, {
        "summary": "x",
        "modules": [{
            "id": "m1", "title": "M", "status": "in_progress",
            "estimated_cards": 6,
            "topics": [
                {"id": "t1", "title": "Hellos", "kind": "lesson"},
                {"id": "p1", "title": "Drill", "kind": "practice"},
            ],
        }],
    })
    card_id = store.save_card(
        goal_id, db.CARD_VOCAB, "thanks?",
        reference_answer="dankie", module_id="m1",
    )
    # Active exercise message for the card.
    messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_EXERCISE,
        payload={"card_type": "vocab", "prompt_text": "thanks?"},
        card_id=card_id,
    )
    # Wrong answer → card is due-now in box 1.
    store.record_attempt(
        learner_id=learner_id, card_id=card_id,
        user_answer="x", ai_feedback="no", rating=db.RATING_WRONG,
    )

    # User sends an answer → grading + next-card flow. The selector
    # picks LESSON for the first untaught lesson topic (t1).
    fm = FakeModel(responses=[
        '{"verdict": "answer"}',
        '{"rating": "wrong", "feedback": "no, dankie"}',
        _LESSON,
        _CRITIC_READY,
    ])
    out = await coach.run_turn(
        learner_id=learner_id, goal_id=goal_id,
        user_text="x",
        model=fm, skill_content="SKILL",
    )
    # The new card emitted at the end has a NEW card_id — not the
    # just-answered card_id re-surfaced via SRS replay.
    new_card_msgs = [
        m for m in out
        if m["kind"] in (db.MSG_LESSON, db.MSG_EXERCISE)
    ]
    assert len(new_card_msgs) == 1
    assert new_card_msgs[0]["card_id"] != card_id


def test_advance_module_no_op_when_under_target(temp_db):
    outline = {
        "summary": "x",
        "modules": [
            {"id": "m1", "title": "First", "status": "in_progress",
             "estimated_cards": 6},
        ],
    }
    digest = {"m1": {"exercises_attempted": 3, "exercises_emitted": 3}}
    assert coach._advance_module_if_complete(
        goal_id="goal-1", outline=outline, digest=digest,
    ) is None


def test_advance_module_last_module_has_no_new_title(temp_db):
    """When the just-completed module was the last one, the result
    still records the previous title (for logging) but ``new_title``
    is None — the coach uses that signal to skip the transition text
    (the graduation message will handle it on the next turn)."""
    outline = {
        "summary": "x",
        "modules": [
            {"id": "m1", "title": "Only One", "status": "in_progress",
             "estimated_cards": 4},
        ],
    }
    digest = {
        "m1": {"exercises_attempted": 4, "exercises_emitted": 4, "lessons_given": 0},
    }
    result = coach._advance_module_if_complete(
        goal_id="goal-1", outline=outline, digest=digest,
    )
    assert result is not None
    assert result.previous_title == "Only One"
    assert result.new_title is None
    assert result.outline["modules"][0]["status"] == "completed"


def test_emit_module_advance_text_skips_when_no_next_module(temp_db):
    """Helper sanity: the transition text helper returns None on the
    final-module case so the caller doesn't emit a "Next up: " bubble
    with no next title."""
    learner_id, goal_id = _setup(temp_db)
    advanced = coach._AdvanceResult(
        outline={"summary": "x", "modules": []},
        previous_title="Done",
        new_title=None,
    )
    assert coach._emit_module_advance_text(
        learner_id=learner_id, goal_id=goal_id, advanced=advanced,
    ) is None


def test_force_advance_module_promotes_regardless_of_estimated_cards(temp_db):
    """The selector-driven advance must NOT consult estimated_cards.
    Sebastian's bug: module 1 in_progress with est=10, digest shows
    only 7 emitted / 0 attempted (the digest snapshot at the moment
    advance_first fires can lag the real on-screen count). The gated
    helper refuses to advance here — but the force helper is the one
    called from the advance_first branch, and it MUST promote
    anyway because the selector has already concluded the module's
    drill load is empty."""
    outline = {
        "summary": "x",
        "modules": [
            {"id": "m1", "title": "Foundations", "status": "in_progress",
             "estimated_cards": 10},
            {"id": "m2", "title": "Perfekt", "status": "not_started",
             "estimated_cards": 12},
        ],
    }
    result = coach._force_advance_module(
        goal_id="goal-1", outline=outline,
    )
    assert result is not None
    assert result.outline["modules"][0]["status"] == "completed"
    assert result.outline["modules"][1]["status"] == "in_progress"
    assert result.previous_title == "Foundations"
    assert result.new_title == "Perfekt"


def test_force_advance_module_last_module_returns_no_next_title(temp_db):
    """When the just-completed module was the last one, force-advance
    still records previous_title (logs) but returns ``new_title=None``
    so the caller falls through to a real graduation message."""
    outline = {
        "summary": "x",
        "modules": [
            {"id": "m1", "title": "Only One", "status": "in_progress",
             "estimated_cards": 6},
        ],
    }
    result = coach._force_advance_module(
        goal_id="goal-1", outline=outline,
    )
    assert result is not None
    assert result.previous_title == "Only One"
    assert result.new_title is None
    assert result.outline["modules"][0]["status"] == "completed"


def test_force_advance_module_no_in_progress_returns_none(temp_db):
    """Defensive: an outline with no in_progress module (everything
    already completed) returns None — the caller's re-selection will
    then land on real graduation."""
    outline = {
        "summary": "x",
        "modules": [
            {"id": "m1", "title": "Done", "status": "completed",
             "estimated_cards": 6},
        ],
    }
    assert coach._force_advance_module(
        goal_id="goal-1", outline=outline,
    ) is None


@pytest.mark.asyncio
async def test_advance_first_branch_force_advances_under_estimate(temp_db):
    """The full integration of the bug: plant an outline + just enough
    cards to drive the selector to advance_first while leaving
    module 1's estimated_cards target unmet. Before the fix the coach
    refused to advance and emitted a false graduation; after the fix
    the next module promotes and the learner sees the transition
    bubble + a card from module 2."""
    learner_id, goal_id = _setup(temp_db)
    # Outline mirrors the screenshot: module 1 has the canonical
    # 2-lesson + 1-practice topic layout that the selector clears in
    # ~6 cards, while estimated_cards is set to 10 (the LLM's optimistic
    # planning number). Module 2 has a real lesson topic so the
    # selector can author a follow-up card after the advance.
    outline = {
        "summary": "x",
        "modules": [
            {"id": "m1", "title": "Foundations", "status": "in_progress",
             "estimated_cards": 10,
             "topics": [
                 {"id": "m1-l1", "title": "SVO", "kind": "lesson"},
                 {"id": "m1-l2", "title": "Verbs", "kind": "lesson"},
                 {"id": "m1-p1", "title": "Practice basics", "kind": "practice"},
             ]},
            {"id": "m2", "title": "Perfekt", "status": "not_started",
             "estimated_cards": 12,
             "topics": [
                 {"id": "m2-l1", "title": "Perfekt structure", "kind": "lesson"},
                 {"id": "m2-p1", "title": "Drill Perfekt", "kind": "practice"},
             ]},
        ],
    }
    store.save_curriculum_outline(goal_id, outline)
    # Plant cards under module 1 so the digest reports every topic
    # quota AND the recycle cap met. Selector won't advance_first
    # until each topic is drilled to TARGET_DRILLS_PER_PRACTICE_TOPIC
    # (the recycle cap is `< TARGET`, so we need >= TARGET on every
    # lesson topic too). Total: 2 lessons + 2 drills on p1 + 2 recycle
    # drills on each of l1, l2 = 8 cards. We DO NOT call record_attempt
    # — leaving exercises_attempted at 0 reproduces Sebastian's
    # under-estimate digest that the OLD code refused to advance from.
    store.save_card(goal_id, db.CARD_LESSON, "lesson body 1",
                    module_id="m1", topic_id="m1-l1")
    store.save_card(goal_id, db.CARD_LESSON, "lesson body 2",
                    module_id="m1", topic_id="m1-l2")
    store.save_card(goal_id, db.CARD_VOCAB, "drill p1 1",
                    reference_answer="x", module_id="m1", topic_id="m1-p1")
    store.save_card(goal_id, db.CARD_CLOZE, "p1 ___ drill 2",
                    reference_answer="x", module_id="m1", topic_id="m1-p1")
    store.save_card(goal_id, db.CARD_TRANSLATION, "recycle l1 a",
                    reference_answer="x", module_id="m1", topic_id="m1-l1")
    store.save_card(goal_id, db.CARD_TRANSLATION, "recycle l1 b",
                    reference_answer="x", module_id="m1", topic_id="m1-l1")
    store.save_card(goal_id, db.CARD_VOCAB, "recycle l2 a",
                    reference_answer="x", module_id="m1", topic_id="m1-l2")
    store.save_card(goal_id, db.CARD_VOCAB, "recycle l2 b",
                    reference_answer="x", module_id="m1", topic_id="m1-l2")
    # Queue the FakeModel for the lesson card (the selector will pick
    # LESSON for m2's first lesson topic after the advance) + critic.
    fm = FakeModel(responses=[_LESSON, _CRITIC_READY])
    out = await coach.run_turn(
        learner_id=learner_id, goal_id=goal_id,
        user_text=None, model=fm, skill_content="SKILL",
    )

    # NO graduation message; YES a module-advance bubble.
    coach_texts = [m for m in out if m["kind"] == db.MSG_COACH_TEXT]
    advance_msgs = [m for m in coach_texts
                    if (m.get("payload") or {}).get("kind") == "module_advance"]
    assert len(advance_msgs) == 1, (
        f"expected exactly one module_advance bubble, got coach_texts={coach_texts}"
    )
    advance_payload = advance_msgs[0]["payload"]
    assert advance_payload["previous_title"] == "Foundations"
    assert advance_payload["new_title"] == "Perfekt"
    # No graduation copy slipped through.
    for m in coach_texts:
        text = (m.get("payload") or {}).get("text") or ""
        assert "finished every module" not in text

    # The outline persisted with m1 → completed, m2 → in_progress.
    saved = store.get_curriculum_outline(goal_id)
    statuses = {m["id"]: m["status"] for m in saved["modules"]}
    assert statuses == {"m1": "completed", "m2": "in_progress"}

    # A lesson card from module 2 was emitted as the next thing.
    new_card_msgs = [m for m in out if m["kind"] == db.MSG_LESSON]
    assert len(new_card_msgs) == 1


def test_emit_module_advance_text_persists_structured_payload(temp_db):
    """Helper happy path: returns a MSG_COACH_TEXT carrying both a
    plain-English fallback ``text`` (older clients) AND the structured
    ``kind`` / ``previous_title`` / ``new_title`` fields the frontend
    uses to render via i18n."""
    learner_id, goal_id = _setup(temp_db)
    advanced = coach._AdvanceResult(
        outline={"summary": "x", "modules": []},
        previous_title="Greetings",
        new_title="Ordering food",
    )
    msg = coach._emit_module_advance_text(
        learner_id=learner_id, goal_id=goal_id, advanced=advanced,
    )
    assert msg is not None
    assert msg["kind"] == db.MSG_COACH_TEXT
    p = msg["payload"]
    assert p["kind"] == "module_advance"
    assert p["previous_title"] == "Greetings"
    assert p["new_title"] == "Ordering food"
    assert "Greetings" in p["text"]
    assert "Ordering food" in p["text"]


@pytest.mark.asyncio
async def test_classifier_does_not_see_just_appended_learner_message(temp_db):
    """The just-appended learner_text would otherwise show up in
    recent_pairs and the classifier would see the same text twice."""
    learner_id, goal_id = _setup(temp_db)
    # Seed one prior pair.
    messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=db.MSG_COACH_TEXT, payload={"text": "Welcome."},
    )
    fm = FakeModel(responses=[
        '{"verdict": "question"}',
        '{"text": "OK."}',
    ])
    await coach.run_turn(
        learner_id=learner_id, goal_id=goal_id,
        user_text="DUPLICATE_MARKER", model=fm, skill_content="SKILL",
    )
    # First request was to the classifier — assert that
    # DUPLICATE_MARKER appears in LEARNER'S MESSAGE block exactly once
    # (not also in RECENT CONVERSATION).
    classifier_msg = fm.requests[0].messages[1]["content"]
    assert classifier_msg.count("DUPLICATE_MARKER") == 1
