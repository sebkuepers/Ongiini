"""Tests for the curriculum critic (LLM-as-judge for outline quality)."""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from owela import ModelRequest, ModelResponse

from ongiini.learning import curriculum_critic as cc
from ongiini.learning import context as ctx_mod


@dataclass
class FakeModel:
    response: str = ""
    raise_exc: Exception | None = None
    last_request: ModelRequest | None = None

    async def complete(self, req: ModelRequest) -> ModelResponse:
        self.last_request = req
        if self.raise_exc:
            raise self.raise_exc
        return ModelResponse(
            content=self.response, tool_calls=[],
            finish_reason="stop",
            tokens_in=5, tokens_out=5, cached_tokens=0, raw=None,
        )


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    from ongiini.learning import db
    from ongiini import config
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    db.warmup()
    return tmp_path / "learning.sqlite"


_DUMMY_OUTLINE = {
    "summary": "x",
    "modules": [
        {"id": "m1", "title": "Greetings", "status": "in_progress",
         "estimated_cards": 6,
         "topics": [
             {"id": "t1", "title": "Hello", "kind": "lesson"},
             {"id": "t2", "title": "Drill", "kind": "practice"},
         ]},
    ],
}


def _ctx(temp_db):
    from ongiini.learning import store
    learner_id = store.create_anonymous_learner()
    return ctx_mod.build_learner_context(learner_id)


# ──────────────────────────────────────────────────────────────────
# Happy + structured paths
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ready_true_short_circuits_loop(temp_db):
    fm = FakeModel(response=json.dumps({
        "ready": True, "score": 8,
        "issues": [],
        "strengths": ["modules are scoped to the goal"],
    }))
    out = await cc.critique_outline(
        _ctx(temp_db), _DUMMY_OUTLINE, model=fm, skill_content="SKILL",
    )
    assert out.ready is True
    assert out.score == 8
    assert out.strengths == ["modules are scoped to the goal"]


@pytest.mark.asyncio
async def test_ready_false_returns_issues(temp_db):
    fm = FakeModel(response=json.dumps({
        "ready": False, "score": 4,
        "issues": [
            "Module 1 has only practice topics; needs a lesson topic first.",
            "Module count of 1 is too thin for a 2-week interview goal.",
        ],
        "strengths": ["Topic titles are concrete."],
    }))
    out = await cc.critique_outline(
        _ctx(temp_db), _DUMMY_OUTLINE, model=fm, skill_content="SKILL",
    )
    assert out.ready is False
    assert out.score == 4
    assert len(out.issues) == 2
    assert "lesson topic" in out.issues[0]


@pytest.mark.asyncio
async def test_issues_capped_at_5_and_truncated(temp_db):
    """Don't let a runaway critic blow up the next revise prompt."""
    fm = FakeModel(response=json.dumps({
        "ready": False, "score": 3,
        "issues": ["a"*400, "b", "c", "d", "e", "f", "g"],
    }))
    out = await cc.critique_outline(
        _ctx(temp_db), _DUMMY_OUTLINE, model=fm, skill_content="SKILL",
    )
    assert len(out.issues) == 5
    assert len(out.issues[0]) <= 240


# ──────────────────────────────────────────────────────────────────
# Soft-fail paths — must NEVER raise; ready defaults to True so the
# orchestrator ships the outline + the warning surfaces the problem.
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_model_raises_degrades_to_ready_true_score_zero(temp_db):
    fm = FakeModel(raise_exc=RuntimeError("connection refused"))
    out = await cc.critique_outline(
        _ctx(temp_db), _DUMMY_OUTLINE, model=fm, skill_content="SKILL",
    )
    assert out.ready is True
    assert out.score == 0
    assert any("critic_failed" in i for i in out.issues)


@pytest.mark.asyncio
async def test_model_returns_garbage_json_degrades(temp_db):
    fm = FakeModel(response="here's my best guess, sorry")
    out = await cc.critique_outline(
        _ctx(temp_db), _DUMMY_OUTLINE, model=fm, skill_content="SKILL",
    )
    assert out.ready is True
    assert out.score == 0


@pytest.mark.asyncio
async def test_model_returns_error_field_degrades(temp_db):
    fm = FakeModel(response=json.dumps({"error": "couldn't decide"}))
    out = await cc.critique_outline(
        _ctx(temp_db), _DUMMY_OUTLINE, model=fm, skill_content="SKILL",
    )
    assert out.ready is True
    assert any("model error" in i for i in out.issues)


@pytest.mark.asyncio
async def test_missing_ready_field_defaults_true_no_block(temp_db):
    """If the model forgot the ready field, don't block the learner.
    Default to ready=True; the score will be 0 if missing, surfacing
    the issue in logs."""
    fm = FakeModel(response=json.dumps({"score": 6, "issues": []}))
    out = await cc.critique_outline(
        _ctx(temp_db), _DUMMY_OUTLINE, model=fm, skill_content="SKILL",
    )
    assert out.ready is True
    assert out.score == 6


@pytest.mark.asyncio
async def test_non_bool_ready_defaults_true(temp_db):
    fm = FakeModel(response=json.dumps({"ready": "yes", "score": 7}))
    out = await cc.critique_outline(
        _ctx(temp_db), _DUMMY_OUTLINE, model=fm, skill_content="SKILL",
    )
    assert out.ready is True


@pytest.mark.asyncio
async def test_score_clamped_to_valid_range(temp_db):
    fm = FakeModel(response=json.dumps({"ready": True, "score": 99}))
    out = await cc.critique_outline(
        _ctx(temp_db), _DUMMY_OUTLINE, model=fm, skill_content="SKILL",
    )
    assert out.score == 10

    # Negative scores clamp to 0 (NOT 1) so they're distinguishable
    # from the "missing field" default of 0. The reviewer flagged the
    # prior `max(1, ...)` clamp as ambiguous.
    fm2 = FakeModel(response=json.dumps({"ready": True, "score": -5}))
    out2 = await cc.critique_outline(
        _ctx(temp_db), _DUMMY_OUTLINE, model=fm2, skill_content="SKILL",
    )
    assert out2.score == 0


@pytest.mark.asyncio
async def test_score_zero_preserved_not_clamped_up(temp_db):
    """A legitimate `score: 0` from the model means "unusable plan" —
    must reach the downstream as 0, not get nudged to 1."""
    fm = FakeModel(response=json.dumps({"ready": False, "score": 0}))
    out = await cc.critique_outline(
        _ctx(temp_db), _DUMMY_OUTLINE, model=fm, skill_content="SKILL",
    )
    assert out.score == 0
    assert out.ready is False


@pytest.mark.asyncio
async def test_score_bool_rejected_as_missing(temp_db):
    """`{"score": true}` is a model parse mistake — Python's
    isinstance(_, int) would silently coerce True → 1, which is
    wrong. Confirm we treat it as missing (defaults to 0)."""
    fm = FakeModel(response=json.dumps({"ready": True, "score": True}))
    out = await cc.critique_outline(
        _ctx(temp_db), _DUMMY_OUTLINE, model=fm, skill_content="SKILL",
    )
    assert out.score == 0


@pytest.mark.asyncio
async def test_score_string_treated_as_missing(temp_db):
    """`"score": "7"` is not silently parsed — we want explicit ints
    from the model. Defaults to 0 (no signal)."""
    fm = FakeModel(response=json.dumps({"ready": True, "score": "7"}))
    out = await cc.critique_outline(
        _ctx(temp_db), _DUMMY_OUTLINE, model=fm, skill_content="SKILL",
    )
    assert out.score == 0


@pytest.mark.asyncio
async def test_issues_as_single_string_coerced_to_list(temp_db):
    """Tolerant of a model that forgot the list — wrap a single
    string into a 1-element list rather than dropping it silently."""
    fm = FakeModel(response=json.dumps({
        "ready": False, "score": 5,
        "issues": "Module 1 needs a lesson topic",
    }))
    out = await cc.critique_outline(
        _ctx(temp_db), _DUMMY_OUTLINE, model=fm, skill_content="SKILL",
    )
    assert out.issues == ["Module 1 needs a lesson topic"]


@pytest.mark.asyncio
async def test_empty_outline_does_not_crash(temp_db):
    """Defensive: a caller-side bug shouldn't be able to wedge the
    critic. Outline shape validation lives upstream in
    curriculum._validate_outline."""
    fm = FakeModel(response=json.dumps({"ready": False, "score": 1,
                                        "issues": ["empty plan"]}))
    out = await cc.critique_outline(
        _ctx(temp_db), {}, model=fm, skill_content="SKILL",
    )
    assert out.ready is False
    assert out.score == 1


# ──────────────────────────────────────────────────────────────────
# Prompt shape — locks in injection guard + that the outline reaches
# the model
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_prompt_carries_outline_and_focus_block(temp_db):
    fm = FakeModel(response=json.dumps({"ready": True, "score": 8}))
    await cc.critique_outline(
        _ctx(temp_db), _DUMMY_OUTLINE, model=fm, skill_content="SKILL",
    )
    user_msg = fm.last_request.messages[1]["content"]
    # The outline JSON shape reaches the model.
    assert '"summary"' in user_msg
    assert "Greetings" in user_msg
    # The "FOCUS" header is present so the critic knows what to grade
    # the outline against.
    assert "FOCUS FOR THIS CURRICULUM" in user_msg


@pytest.mark.asyncio
async def test_system_prompt_carries_skill_content(temp_db):
    fm = FakeModel(response=json.dumps({"ready": True, "score": 9}))
    await cc.critique_outline(
        _ctx(temp_db), _DUMMY_OUTLINE,
        model=fm, skill_content="SKILL-MARKER",
    )
    sys_msg = fm.last_request.messages[0]["content"]
    assert "SKILL-MARKER" in sys_msg
    assert "REVIEW CHECKLIST" in sys_msg or "checklist" in sys_msg.lower()
