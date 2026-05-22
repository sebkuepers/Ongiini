"""Step dataclass tests."""

from __future__ import annotations

import time

from owela.step import (
    CritiqueStep, ModelCallStep, PlanStep, ReplyStep, ReviseStep,
    RouterStep, Step, ToolStep,
)


def test_step_defaults():
    s = Step()
    assert s.kind == "step"
    assert s.tokens_in == 0 and s.tokens_out == 0
    assert s.ended_at is None
    assert s.latency_ms() == 0


def test_step_latency():
    s = Step(started_at=1000.0, ended_at=1000.123)
    assert s.latency_ms() == 123


def test_router_step():
    rs = RouterStep(verdict="SEARCH", depth="DEEP")
    assert rs.kind == "router"
    assert rs.verdict == "SEARCH"
    assert rs.depth == "DEEP"


def test_model_call_step():
    mcs = ModelCallStep(turn=2, tokens_in=100, tokens_out=50, enable_thinking=True, reasoning_budget=500)
    assert mcs.kind == "model_call"
    assert mcs.turn == 2
    assert mcs.enable_thinking is True
    assert mcs.reasoning_budget == 500
    assert mcs.tool_calls == []


def test_tool_step():
    ts = ToolStep(tool_name="web_search", result_len=1200, args_len=20)
    assert ts.kind == "tool"
    assert ts.tool_name == "web_search"
    assert ts.result_len == 1200
    assert ts.error is None


def test_reply_step():
    rs = ReplyStep(reply_len=200, sent=True, dead_urls_stripped=1)
    assert rs.kind == "reply"
    assert rs.sent is True
    assert rs.dead_urls_stripped == 1


def test_v1_steps_exist():
    # PlanStep, CritiqueStep, ReviseStep are slots reserved for v1.
    # They should be constructible in v0 even if no executor branch
    # produces them yet — this protects against the v1 add being more
    # than a flag flip.
    PlanStep(plan_text="plan body")
    CritiqueStep(verdict="REVISE", reasons=["missing citation"])
    ReviseStep()


def test_step_attrs_is_isolated_per_instance():
    # The dataclass uses default_factory=dict for attrs — verifies a
    # subtle bug class where a shared mutable default would have two
    # Steps point at the same dict.
    a = Step()
    b = Step()
    a.attrs["x"] = 1
    assert "x" not in b.attrs
