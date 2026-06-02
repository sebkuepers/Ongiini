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


# Stock JSON responses for common steps.
_OUTLINE = json.dumps({
    "summary": "Job-interview Afrikaans in 2 weeks.",
    "modules": [
        {"id": "mod-1", "title": "Greetings", "status": "in_progress"},
    ],
})
_EXERCISE = json.dumps({
    "card_type": "vocab",
    "prompt_text": "How do you say 'thank you'?",
    "reference_answer": "dankie",
})
_LESSON = json.dumps({
    "card_type": "lesson",
    "prompt_text": "In Afrikaans, common greetings include 'goeie môre'.",
    "title": "Time-of-day greetings",
})


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
async def test_run_turn_no_text_no_active_card_designs_outline_and_emits_card(temp_db):
    learner_id, goal_id = _setup(temp_db)
    fm = FakeModel(responses=[_OUTLINE, _EXERCISE])
    out = await coach.run_turn(
        learner_id=learner_id, goal_id=goal_id,
        user_text=None, model=fm, skill_content="SKILL",
    )
    assert len(out) == 1
    assert out[0]["kind"] == db.MSG_EXERCISE
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
        # ...then the next card
        _EXERCISE,
    ])
    out = await coach.run_turn(
        learner_id=learner_id, goal_id=goal_id,
        user_text="dankie", model=fm, skill_content="SKILL",
    )
    kinds = [m["kind"] for m in out]
    # learner_text → feedback → progress → next exercise
    assert kinds[0] == db.MSG_LEARNER_TEXT
    assert db.MSG_FEEDBACK in kinds
    assert db.MSG_PROGRESS in kinds
    assert kinds[-1] == db.MSG_EXERCISE


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
