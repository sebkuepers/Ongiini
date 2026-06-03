"""Tests for ``cards.generate_card_content_with_review`` — the
per-card design → critic → maybe-revise orchestrator.

We monkeypatch the underlying ``generate_card_content`` and
``card_critic.critique_card`` so the tests focus on orchestration
(call counts, steering-note plumbing, soft-fail composition) rather
than re-asserting validator behaviour, which is covered in
``test_learning_cards_validator.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ongiini.learning import card_critic
from ongiini.learning import cards as cards_mod
from ongiini.learning import context as ctx_mod
from ongiini.learning.llm import ModelOutputError


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    from ongiini.learning import db
    from ongiini import config
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    db.warmup()
    return tmp_path / "learning.sqlite"


def _ctx(temp_db):
    from ongiini.learning import store
    learner_id = store.create_anonymous_learner()
    return ctx_mod.build_learner_context(learner_id)


def _orchestrate(ctx):
    """Call args boilerplate."""
    return cards_mod.generate_card_content_with_review(
        ctx,
        model=object(),   # never reached; design fn is patched
        skill_content="SKILL",
        card_type="lesson",
        module_id="m1",
        module_title="Greetings",
        topic_id="t1",
        topic_title="Hello",
    )


@dataclass
class _DesignCall:
    steering_note: str | None
    payload: dict[str, Any]


def _make_design_stub(monkeypatch, *, returns: list[dict[str, Any]],
                     raises_on: dict[int, Exception] | None = None):
    """Patch ``cards.generate_card_content`` with a programmable stub.

    ``returns[i]`` is the payload for the i-th call.
    ``raises_on[i]`` (if provided) raises on the i-th call before the
    payload is consulted. Captured calls land in ``calls``."""
    calls: list[_DesignCall] = []
    raises_on = raises_on or {}

    async def stub(_ctx, **kwargs):
        idx = len(calls)
        calls.append(_DesignCall(
            steering_note=kwargs.get("steering_note"),
            payload={},
        ))
        if idx in raises_on:
            raise raises_on[idx]
        payload = returns[idx]
        calls[-1].payload = payload
        return payload

    monkeypatch.setattr(cards_mod, "generate_card_content", stub)
    return calls


def _make_critic_stub(monkeypatch, result):
    """Patch ``card_critic.critique_card`` to return a fixed result."""
    captured: list[dict[str, Any]] = []

    async def stub(_ctx, payload, **kwargs):
        captured.append({"payload": payload, "kwargs": kwargs})
        return result

    monkeypatch.setattr(card_critic, "critique_card", stub)
    return captured


# ──────────────────────────────────────────────────────────────────
# Critic approves → ship original; ONE revise NOT triggered
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_critic_approves_short_circuits_no_revise(temp_db, monkeypatch):
    original = {"card_type": "lesson", "title": "First"}
    design_calls = _make_design_stub(monkeypatch, returns=[original])
    _make_critic_stub(monkeypatch, card_critic.CardCritiqueResult(
        ready=True, score=9, issues=[], strengths=["clean"],
    ))

    out = await _orchestrate(_ctx(temp_db))

    assert out is original
    assert len(design_calls) == 1
    assert design_calls[0].steering_note is None


# ──────────────────────────────────────────────────────────────────
# Critic rejects → revise pass fires with issues as steering note
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_critic_rejects_revise_pass_uses_issues_as_steering(temp_db, monkeypatch):
    original = {"card_type": "lesson", "title": "first draft"}
    revised = {"card_type": "lesson", "title": "second draft"}
    design_calls = _make_design_stub(monkeypatch, returns=[original, revised])
    _make_critic_stub(monkeypatch, card_critic.CardCritiqueResult(
        ready=False, score=4,
        issues=["Step 2 missing English gloss", "quick_check answer wrong"],
    ))

    out = await _orchestrate(_ctx(temp_db))

    assert out is revised
    assert len(design_calls) == 2
    # First call: no steering. Second call: critic's issues included.
    assert design_calls[0].steering_note is None
    assert design_calls[1].steering_note is not None
    assert "missing English gloss" in design_calls[1].steering_note
    assert "quick_check answer wrong" in design_calls[1].steering_note


@pytest.mark.asyncio
async def test_critic_rejects_with_no_issues_falls_back_to_generic_steering(temp_db, monkeypatch):
    """Degenerate case: critic says ready=False but lists no issues.
    The orchestrator still asks for a revise rather than shipping a
    rejected card — but with a generic prompt addendum so the revise
    pass has something to act on."""
    original = {"card_type": "lesson", "title": "first"}
    revised = {"card_type": "lesson", "title": "second"}
    design_calls = _make_design_stub(monkeypatch, returns=[original, revised])
    _make_critic_stub(monkeypatch, card_critic.CardCritiqueResult(
        ready=False, score=3, issues=[],
    ))

    out = await _orchestrate(_ctx(temp_db))

    assert out is revised
    assert design_calls[1].steering_note is not None
    # The fallback steering must point the revise pass at the actual
    # checklist items the critic SHOULD have caught — assert on the
    # load-bearing checklist anchors (gloss + level) rather than just
    # the score digit (which appears in the prose anyway).
    steering = design_calls[1].steering_note
    assert "gloss" in steering
    assert "level" in steering


# ──────────────────────────────────────────────────────────────────
# Soft-fail composition — never block the learner
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_critic_crash_via_degraded_ships_original(temp_db, monkeypatch):
    """The critic itself uses .degraded() on crashes — that returns
    ready=True so the orchestrator ships the original. Confirm
    the orchestrator honours that and does NOT revise."""
    original = {"card_type": "lesson", "title": "first"}
    design_calls = _make_design_stub(monkeypatch, returns=[original])
    _make_critic_stub(
        monkeypatch,
        card_critic.CardCritiqueResult.degraded("model crashed"),
    )

    out = await _orchestrate(_ctx(temp_db))

    assert out is original
    assert len(design_calls) == 1


@pytest.mark.asyncio
async def test_revise_failure_ships_original(temp_db, monkeypatch):
    """If the revise call raises ModelOutputError, we ship the
    original card rather than crashing the turn — the learner gets a
    slightly-worse-but-shippable card, the warning lands in logs."""
    original = {"card_type": "lesson", "title": "first"}
    design_calls = _make_design_stub(
        monkeypatch,
        returns=[original, {"unused": True}],
        raises_on={1: ModelOutputError("invalid JSON on revise")},
    )
    _make_critic_stub(monkeypatch, card_critic.CardCritiqueResult(
        ready=False, score=2, issues=["lots of problems"],
    ))

    out = await _orchestrate(_ctx(temp_db))

    assert out is original
    assert len(design_calls) == 2


# ──────────────────────────────────────────────────────────────────
# Critic invocation routing
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_critic_receives_payload_and_selector_picks(temp_db, monkeypatch):
    """The orchestrator must forward the just-generated payload AND
    the selector's card_type / titles to the critic so it can score
    against the right type-specific rules."""
    original = {"card_type": "lesson", "title": "first"}
    _make_design_stub(monkeypatch, returns=[original])
    captured = _make_critic_stub(monkeypatch, card_critic.CardCritiqueResult(
        ready=True, score=8,
    ))

    await _orchestrate(_ctx(temp_db))

    assert len(captured) == 1
    assert captured[0]["payload"] is original
    kw = captured[0]["kwargs"]
    assert kw["card_type"] == "lesson"
    assert kw["module_title"] == "Greetings"
    assert kw["topic_title"] == "Hello"
    assert kw["skill_content"] == "SKILL"
