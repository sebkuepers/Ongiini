"""Tests for the curriculum module's design-review loop orchestrator
(`design_outline_with_review`).

Coverage targets:
  * Critic-approves-on-iter-1 short-circuits — minimum LLM calls.
  * Critic disapproves once → revise → critic approves → done.
  * Critic never approves → loop caps at max_iterations with WARNING.
  * Revise failure mid-loop falls back to the prior valid outline
    rather than crashing.
  * The change_reason passed to revise carries the critic's issues.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import pytest

from owela import ModelRequest, ModelResponse

from ongiini.learning import context as ctx_mod
from ongiini.learning import curriculum


@dataclass
class FakeModel:
    """Returns responses from a queue. The orchestrator drives:
    design → critique → (maybe revise → critique × N) so the queue
    order matters."""
    responses: list[str] = field(default_factory=list)
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, req: ModelRequest) -> ModelResponse:
        self.requests.append(req)
        body = self.responses.pop(0) if self.responses else "{}"
        return ModelResponse(
            content=body, tool_calls=[],
            finish_reason="stop", tokens_in=5, tokens_out=5,
            cached_tokens=0, raw=None,
        )


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    from ongiini.learning import db
    from ongiini import config
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    db.warmup()
    return tmp_path / "learning.sqlite"


_GOOD_OUTLINE = json.dumps({
    "summary": "Job-interview Afrikaans in 2 weeks.",
    "modules": [
        {"id": "m1", "title": "Greetings", "status": "in_progress",
         "estimated_cards": 6,
         "topics": [
             {"id": "t1", "title": "Hello", "kind": "lesson"},
             {"id": "t2", "title": "Drill", "kind": "practice"},
         ]},
    ],
})

_BETTER_OUTLINE = json.dumps({
    "summary": "Stronger plan.",
    "modules": [
        {"id": "m1", "title": "Greetings", "status": "in_progress",
         "estimated_cards": 8,
         "topics": [
             {"id": "t1", "title": "Hello", "kind": "lesson"},
             {"id": "t2", "title": "Self-intro", "kind": "lesson"},
             {"id": "t3", "title": "Drill", "kind": "practice"},
         ]},
        {"id": "m2", "title": "Interview answers",
         "status": "not_started", "estimated_cards": 6,
         "topics": [
             {"id": "t4", "title": "About me", "kind": "lesson"},
             {"id": "t5", "title": "Drill", "kind": "practice"},
         ]},
    ],
})


def _ctx(temp_db):
    from ongiini.learning import store
    learner_id = store.create_anonymous_learner()
    return ctx_mod.build_learner_context(learner_id)


# ──────────────────────────────────────────────────────────────────
# Happy paths
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_critic_approves_on_iter_1_uses_2_calls(temp_db):
    """Best case: designer + critic = 2 calls, no revise."""
    fm = FakeModel(responses=[
        _GOOD_OUTLINE,
        json.dumps({"ready": True, "score": 8, "issues": []}),
    ])
    out = await curriculum.design_outline_with_review(
        _ctx(temp_db), model=fm, skill_content="SKILL",
    )
    assert out["modules"][0]["title"] == "Greetings"
    assert len(fm.requests) == 2


@pytest.mark.asyncio
async def test_critic_disapproves_then_approves_uses_4_calls(temp_db):
    """designer → critic(no) → revise → critic(yes) = 4 calls."""
    fm = FakeModel(responses=[
        _GOOD_OUTLINE,
        json.dumps({"ready": False, "score": 4,
                    "issues": ["Module 1 needs a self-intro lesson topic"]}),
        _BETTER_OUTLINE,
        json.dumps({"ready": True, "score": 8, "issues": []}),
    ])
    out = await curriculum.design_outline_with_review(
        _ctx(temp_db), model=fm, skill_content="SKILL",
    )
    assert out["summary"] == "Stronger plan."
    assert len(fm.requests) == 4


@pytest.mark.asyncio
async def test_critic_change_reason_carries_issues(temp_db):
    """The revise call must receive the critic's specific issues so
    the next pass actually addresses them."""
    fm = FakeModel(responses=[
        _GOOD_OUTLINE,
        json.dumps({"ready": False, "score": 4,
                    "issues": ["Module 1 needs lesson topic"]}),
        _BETTER_OUTLINE,
        json.dumps({"ready": True, "score": 9, "issues": []}),
    ])
    await curriculum.design_outline_with_review(
        _ctx(temp_db), model=fm, skill_content="SKILL",
    )
    # The 3rd request (revise) carries the critic's issue in the
    # user prompt as the change_reason.
    revise_prompt = fm.requests[2].messages[1]["content"]
    assert "Module 1 needs lesson topic" in revise_prompt
    assert "REASON TO REVISE" in revise_prompt


# ──────────────────────────────────────────────────────────────────
# Cap path — never block the learner
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_max_iterations_ships_last_outline_with_warning(temp_db, caplog):
    """Stubborn critic that never says ready: after 3 iterations we
    ship the last revised version with a WARNING. The learner gets
    a curriculum either way."""
    fm = FakeModel(responses=[
        _GOOD_OUTLINE,                                          # design
        json.dumps({"ready": False, "score": 3, "issues": ["x"]}),
        _BETTER_OUTLINE,                                        # revise 1
        json.dumps({"ready": False, "score": 4, "issues": ["y"]}),
        _BETTER_OUTLINE,                                        # revise 2
        json.dumps({"ready": False, "score": 5, "issues": ["z"]}),
    ])
    with caplog.at_level(logging.WARNING, logger="ongiini.learning.curriculum"):
        out = await curriculum.design_outline_with_review(
            _ctx(temp_db), model=fm, skill_content="SKILL",
            max_iterations=3,
        )
    assert out["modules"][0]["title"] == "Greetings"
    # 1 designer + 3 critics + 2 revises = 6 calls
    assert len(fm.requests) == 6
    assert any("max iterations" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_max_iterations_2_caps_at_3_calls(temp_db):
    """Tighter cap for callers that want lower cost."""
    fm = FakeModel(responses=[
        _GOOD_OUTLINE,
        json.dumps({"ready": False, "score": 3, "issues": ["x"]}),
        _BETTER_OUTLINE,
        json.dumps({"ready": False, "score": 4, "issues": ["y"]}),
    ])
    out = await curriculum.design_outline_with_review(
        _ctx(temp_db), model=fm, skill_content="SKILL",
        max_iterations=2,
    )
    assert out is not None
    # 1 designer + 2 critics + 1 revise = 4 calls
    assert len(fm.requests) == 4


# ──────────────────────────────────────────────────────────────────
# Failure paths — degrade gracefully
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_critic_failure_ships_via_degraded_path(temp_db):
    """Critic returns garbage → degraded result has ready=True →
    orchestrator ships the designed outline. No infinite-retry."""
    fm = FakeModel(responses=[
        _GOOD_OUTLINE,
        "definitely not json",
    ])
    out = await curriculum.design_outline_with_review(
        _ctx(temp_db), model=fm, skill_content="SKILL",
    )
    assert out["modules"][0]["title"] == "Greetings"
    # Designer + one failed critic call = 2.
    assert len(fm.requests) == 2


@pytest.mark.asyncio
async def test_zero_iterations_skips_critic_entirely(temp_db, caplog):
    """Code-review follow-up: a caller asking for max_iterations=0
    or negative would otherwise silently ship an un-reviewed outline.
    Confirm the orchestrator short-circuits cleanly (1 designer call,
    no critic) and emits a debug log."""
    fm = FakeModel(responses=[_GOOD_OUTLINE])
    with caplog.at_level(logging.DEBUG, logger="ongiini.learning.curriculum"):
        out = await curriculum.design_outline_with_review(
            _ctx(temp_db), model=fm, skill_content="SKILL",
            max_iterations=0,
        )
    assert out["modules"][0]["title"] == "Greetings"
    assert len(fm.requests) == 1
    assert any("skipping critic loop" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_critic_not_ready_with_empty_issues_uses_synthesized_reason(temp_db):
    """Code-review follow-up: an explicit `{"ready": false, "issues": []}`
    would otherwise leave revise_outline with a meaningless change
    reason. Confirm we synthesize a useful one from the score."""
    fm = FakeModel(responses=[
        _GOOD_OUTLINE,
        json.dumps({"ready": False, "score": 4, "issues": []}),
        _BETTER_OUTLINE,
        json.dumps({"ready": True, "score": 9, "issues": []}),
    ])
    await curriculum.design_outline_with_review(
        _ctx(temp_db), model=fm, skill_content="SKILL",
    )
    revise_prompt = fm.requests[2].messages[1]["content"]
    # The synthesised text references the score so a degenerate
    # critic still steers the revise model.
    assert "4/10" in revise_prompt or "scored this 4" in revise_prompt


@pytest.mark.asyncio
async def test_revise_failure_falls_back_to_prior_outline(temp_db, caplog):
    """If revise returns invalid JSON mid-loop, we keep the most
    recent valid outline rather than crashing or blocking."""
    fm = FakeModel(responses=[
        _GOOD_OUTLINE,
        json.dumps({"ready": False, "score": 3, "issues": ["x"]}),
        "garbage that won't parse",
    ])
    with caplog.at_level(logging.WARNING, logger="ongiini.learning.curriculum"):
        out = await curriculum.design_outline_with_review(
            _ctx(temp_db), model=fm, skill_content="SKILL",
        )
    # We get the original designed outline (revise failed).
    assert out["modules"][0]["title"] == "Greetings"
    assert any("revise_outline failed" in rec.message for rec in caplog.records)
