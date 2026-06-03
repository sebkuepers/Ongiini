"""Tests for the card critic (LLM-as-judge for a single card).

Mirror of ``test_learning_curriculum_critic.py`` — same posture,
same soft-fail discipline, same tolerant JSON parser. Locks in the
shape so a Gemma quirk in production can't silently break the
ship-or-revise gate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from owela import ModelRequest, ModelResponse

from ongiini.learning import card_critic as cc
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


_DUMMY_PAYLOAD = {
    "card_type": "lesson",
    "title": "Greetings — guten Tag",
    "steps": [
        {"kind": "rule",
         "body": "'Guten Tag' is the all-purpose greeting (Good day).",
         "example": "Guten Tag! (Good day!)"},
        {"kind": "example",
         "body": "Use it walking into a shop or office.",
         "example": "Guten Tag, ich hätte gern einen Kaffee. "
                    "(Good day, I would like a coffee.)"},
    ],
    "quick_check": {
        "prompt": "How do you say 'Good day' in German?",
        "answer": "Guten Tag",
    },
}


def _ctx(temp_db):
    from ongiini.learning import store
    learner_id = store.create_anonymous_learner()
    return ctx_mod.build_learner_context(learner_id)


def _critique(ctx, fm, **overrides):
    """Default arg-pack so each test only writes what matters."""
    kwargs = {
        "model": fm,
        "skill_content": "SKILL",
        "card_type": "lesson",
        "module_title": "Greetings",
        "topic_id": "t1",
        "topic_title": "Hello",
    }
    kwargs.update(overrides)
    return cc.critique_card(ctx, _DUMMY_PAYLOAD, **kwargs)


# ──────────────────────────────────────────────────────────────────
# Happy + structured paths
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ready_true_short_circuits_loop(temp_db):
    fm = FakeModel(response=json.dumps({
        "ready": True, "score": 9,
        "issues": [],
        "strengths": ["gloss present", "level fits A1"],
    }))
    out = await _critique(_ctx(temp_db), fm)
    assert out.ready is True
    assert out.score == 9
    assert out.strengths == ["gloss present", "level fits A1"]


@pytest.mark.asyncio
async def test_ready_false_returns_issues(temp_db):
    fm = FakeModel(response=json.dumps({
        "ready": False, "score": 4,
        "issues": [
            "Step 2 'example' German has no English gloss.",
            "quick_check answer mismatch.",
        ],
        "strengths": ["title is clear"],
    }))
    out = await _critique(_ctx(temp_db), fm)
    assert out.ready is False
    assert out.score == 4
    assert len(out.issues) == 2
    assert "gloss" in out.issues[0]


@pytest.mark.asyncio
async def test_issues_capped_at_4_and_truncated(temp_db):
    """Don't let a runaway critic bloat the revise prompt."""
    fm = FakeModel(response=json.dumps({
        "ready": False, "score": 3,
        "issues": ["x" * 400, "b", "c", "d", "e", "f"],
    }))
    out = await _critique(_ctx(temp_db), fm)
    assert len(out.issues) == 4
    assert len(out.issues[0]) <= 240


@pytest.mark.asyncio
async def test_strengths_capped_at_2(temp_db):
    fm = FakeModel(response=json.dumps({
        "ready": True, "score": 8,
        "strengths": ["a", "b", "c", "d"],
    }))
    out = await _critique(_ctx(temp_db), fm)
    assert len(out.strengths) == 2


# ──────────────────────────────────────────────────────────────────
# Soft-fail paths — must NEVER raise; ready defaults to True so the
# orchestrator ships the card.
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_model_raises_degrades_to_ready_true_score_zero(temp_db):
    fm = FakeModel(raise_exc=RuntimeError("connection refused"))
    out = await _critique(_ctx(temp_db), fm)
    assert out.ready is True
    assert out.score == 0
    assert any("critic_failed" in i for i in out.issues)


@pytest.mark.asyncio
async def test_model_returns_garbage_json_degrades(temp_db):
    fm = FakeModel(response="here is the verdict: looks ok")
    out = await _critique(_ctx(temp_db), fm)
    assert out.ready is True
    assert out.score == 0


@pytest.mark.asyncio
async def test_model_returns_error_field_degrades(temp_db):
    fm = FakeModel(response=json.dumps({"error": "could not decide"}))
    out = await _critique(_ctx(temp_db), fm)
    assert out.ready is True
    assert any("model error" in i for i in out.issues)


@pytest.mark.asyncio
async def test_missing_ready_field_defaults_true_no_block(temp_db):
    fm = FakeModel(response=json.dumps({"score": 6, "issues": []}))
    out = await _critique(_ctx(temp_db), fm)
    assert out.ready is True
    assert out.score == 6


@pytest.mark.asyncio
async def test_non_bool_ready_defaults_true(temp_db):
    fm = FakeModel(response=json.dumps({"ready": "yes", "score": 7}))
    out = await _critique(_ctx(temp_db), fm)
    assert out.ready is True


@pytest.mark.asyncio
async def test_score_clamped_to_valid_range(temp_db):
    fm = FakeModel(response=json.dumps({"ready": True, "score": 99}))
    out = await _critique(_ctx(temp_db), fm)
    assert out.score == 10

    fm2 = FakeModel(response=json.dumps({"ready": True, "score": -3}))
    out2 = await _critique(_ctx(temp_db), fm2)
    assert out2.score == 0


@pytest.mark.asyncio
async def test_score_bool_rejected_as_missing(temp_db):
    """`{"score": true}` would silently coerce to 1 under bare
    isinstance(_, int). Confirm the critic rejects it back to 0."""
    fm = FakeModel(response=json.dumps({"ready": True, "score": True}))
    out = await _critique(_ctx(temp_db), fm)
    assert out.score == 0


@pytest.mark.asyncio
async def test_score_string_treated_as_missing(temp_db):
    fm = FakeModel(response=json.dumps({"ready": True, "score": "7"}))
    out = await _critique(_ctx(temp_db), fm)
    assert out.score == 0


@pytest.mark.asyncio
async def test_issues_as_single_string_coerced_to_list(temp_db):
    fm = FakeModel(response=json.dumps({
        "ready": False, "score": 5,
        "issues": "Step 1 has no gloss",
    }))
    out = await _critique(_ctx(temp_db), fm)
    assert out.issues == ["Step 1 has no gloss"]


# ──────────────────────────────────────────────────────────────────
# Prompt shape — locks in card + checklist routing
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_prompt_carries_card_payload(temp_db):
    fm = FakeModel(response=json.dumps({"ready": True, "score": 8}))
    await _critique(_ctx(temp_db), fm)
    user_msg = fm.last_request.messages[1]["content"]
    assert "card_type" in user_msg
    assert "Guten Tag" in user_msg
    assert "CARD UNDER REVIEW" in user_msg


@pytest.mark.asyncio
async def test_system_prompt_carries_skill_content(temp_db):
    fm = FakeModel(response=json.dumps({"ready": True, "score": 9}))
    await _critique(_ctx(temp_db), fm, skill_content="SKILL-MARKER")
    sys_msg = fm.last_request.messages[0]["content"]
    assert "SKILL-MARKER" in sys_msg
    assert "CHECKLIST" in sys_msg.upper()
