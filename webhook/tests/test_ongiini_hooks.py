"""Tests for the Ongiini billing + tracing hooks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from owela import (
    HookRegistry, InboundMessage, ModelCallStep, Policy, ReplyStep, RouterStep,
    Step, ToolStep, TurnContext,
)
from owela.hooks import HookRegistry as Registry
from webhook.app.hooks_app.billing_hook import BillingHook
from webhook.app.hooks_app.tracing_hook import TracingHook


def _ctx(msisdn: str = "+264u") -> TurnContext:
    msg = InboundMessage(
        user_id=msisdn, msg_id="wamid.1", text="hello world",
        content_parts=[],
        history=[{"role": "user", "content": "earlier"}],
    )
    return TurnContext(msg=msg, policy=Policy(name="test"), runtime=MagicMock())


# ============================================================
# BillingHook
# ============================================================

@pytest.mark.asyncio
async def test_billing_aggregates_chat_across_turns():
    """Two ModelCallSteps in a turn → ONE 'chat' line summing both."""
    recorder = MagicMock()
    hook = BillingHook(recorder=recorder)

    steps = [
        RouterStep(tokens_in=20, tokens_out=5, verdict="SEARCH", depth="SHALLOW"),
        ModelCallStep(turn=1, tokens_in=100, tokens_out=10),
        ToolStep(tool_name="web_search", result_len=2000),
        ModelCallStep(turn=2, tokens_in=50, tokens_out=200),
        ReplyStep(reply_len=400, sent=True),
    ]
    await hook.on_turn_complete(steps, _ctx())

    # One 'router' line + one 'chat' line.
    calls = recorder.record.call_args_list
    assert len(calls) == 2

    # Router line — non-billable kind.
    router_call = next(c for c in calls if c.kwargs.get("kind") == "router")
    assert router_call.args[1] == 20    # tokens_in
    assert router_call.args[2] == 5     # tokens_out

    # Chat line — billable aggregate.
    chat_call = next(c for c in calls if c.kwargs.get("kind") == "chat")
    assert chat_call.args[1] == 150     # 100 + 50
    assert chat_call.args[2] == 210     # 10 + 200
    assert chat_call.kwargs["used_search"] is True


@pytest.mark.asyncio
async def test_billing_used_search_false_when_no_search_tool():
    recorder = MagicMock()
    hook = BillingHook(recorder=recorder)
    steps = [
        ModelCallStep(turn=1, tokens_in=10, tokens_out=20),
    ]
    await hook.on_turn_complete(steps, _ctx())
    chat_call = recorder.record.call_args_list[0]
    assert chat_call.kwargs["used_search"] is False


@pytest.mark.asyncio
async def test_billing_no_router_step_no_router_line():
    recorder = MagicMock()
    hook = BillingHook(recorder=recorder)
    steps = [ModelCallStep(turn=1, tokens_in=10, tokens_out=5)]
    await hook.on_turn_complete(steps, _ctx())
    # Only the chat line (no router step → no router record).
    kinds = [c.kwargs.get("kind") for c in recorder.record.call_args_list]
    assert "router" not in kinds


@pytest.mark.asyncio
async def test_billing_skips_zero_token_calls():
    """A ModelCallStep with no tokens (e.g. cached entirely) shouldn't
    spam usage.log with empty lines."""
    recorder = MagicMock()
    hook = BillingHook(recorder=recorder)
    steps = [ModelCallStep(turn=1, tokens_in=0, tokens_out=0)]
    await hook.on_turn_complete(steps, _ctx())
    recorder.record.assert_not_called()


@pytest.mark.asyncio
async def test_billing_recorder_failure_does_not_raise():
    """A broken billing log must NEVER break a reply."""
    recorder = MagicMock()
    recorder.record.side_effect = OSError("disk full")
    hook = BillingHook(recorder=recorder)
    steps = [
        ModelCallStep(turn=1, tokens_in=10, tokens_out=20),
        ReplyStep(sent=True),
    ]
    # Should not raise.
    await hook.on_turn_complete(steps, _ctx())


# ============================================================
# TracingHook
# ============================================================

@pytest.mark.asyncio
async def test_tracing_writes_one_jsonl_per_turn(tmp_path: Path):
    trace_path = tmp_path / "trace.jsonl"
    hook = TracingHook(trace_path=trace_path)

    start = time.monotonic()
    end = start + 0.05

    steps = [
        RouterStep(
            started_at=start, ended_at=start + 0.01,
            verdict="SEARCH", depth="SHALLOW",
            tokens_in=20, tokens_out=5,
        ),
        ModelCallStep(
            started_at=start + 0.01, ended_at=end,
            turn=1, tokens_in=100, tokens_out=50,
            cached_tokens=200, enable_thinking=False,
            finish_reason="stop", tool_calls=[],
        ),
        ReplyStep(reply_len=120, sent=True),
    ]
    await hook.on_turn_complete(steps, _ctx())

    lines = trace_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["msisdn"] == "+264u"
    assert entry["policy"] == "test"
    assert entry["router"]["verdict"] == "SEARCH"
    assert entry["router"]["depth"] == "SHALLOW"
    assert entry["total_tokens_in"] == 100
    assert entry["total_tokens_out"] == 50
    assert entry["reply_len"] == 120
    assert entry["sent"] is True
    assert entry["used_search"] is False    # no ToolStep with search name


@pytest.mark.asyncio
async def test_tracing_attributes_tool_results_to_parent_call(tmp_path: Path):
    """Each ToolStep gets attached to the most recent preceding
    ModelCallStep — makes the trace easy to read by turn."""
    hook = TracingHook(trace_path=tmp_path / "t.jsonl")
    steps = [
        ModelCallStep(turn=1, tokens_in=10, tokens_out=5),
        ToolStep(tool_name="web_search", result_len=2000),
        ModelCallStep(turn=2, tokens_in=20, tokens_out=30),
        ReplyStep(sent=True),
    ]
    await hook.on_turn_complete(steps, _ctx())

    entry = json.loads((tmp_path / "t.jsonl").read_text().strip())
    # First call gets the tool, second call does not.
    assert entry["calls"][0].get("tool_results") == [
        {"name": "web_search", "args_len": 0, "result_len": 2000,
         "error": None, "latency_ms": 0},
    ]
    assert "tool_results" not in entry["calls"][1]
    assert entry["used_search"] is True


@pytest.mark.asyncio
async def test_tracing_truncated_when_no_reply_step(tmp_path: Path):
    """Loop fell through max_steps → no ReplyStep. Trace marks truncated."""
    hook = TracingHook(trace_path=tmp_path / "t.jsonl")
    steps = [ModelCallStep(turn=1, tokens_in=5, tokens_out=5)]
    await hook.on_turn_complete(steps, _ctx())
    entry = json.loads((tmp_path / "t.jsonl").read_text().strip())
    assert entry["truncated"] is True
    assert entry["sent"] is False


@pytest.mark.asyncio
async def test_tracing_does_not_log_user_or_reply_content(tmp_path: Path):
    """Privacy contract: only lengths + names + counts, never content."""
    hook = TracingHook(trace_path=tmp_path / "t.jsonl")
    ctx = _ctx()
    # Sensitive content lives in ctx.msg.text; only its LENGTH should hit the trace.
    steps = [ModelCallStep(turn=1, tokens_in=10, tokens_out=5), ReplyStep(reply_len=42, sent=True)]
    # Stash a "reply" attribute on the ReplyStep that should NOT appear.
    steps[1].attrs["reply_text"] = "Sebastian was here"
    await hook.on_turn_complete(steps, ctx)
    body = (tmp_path / "t.jsonl").read_text()
    assert "hello world" not in body          # user text
    assert "Sebastian" not in body            # reply text
    assert "user_msg_len" in body              # but the length is logged


@pytest.mark.asyncio
async def test_tracing_write_failure_does_not_raise(tmp_path: Path):
    """Hook failures are soft."""
    hook = TracingHook(trace_path=tmp_path / "nonexistent" / "trace.jsonl")
    steps = [ModelCallStep(turn=1, tokens_in=10, tokens_out=5)]
    # Should not raise.
    await hook.on_turn_complete(steps, _ctx())


# ============================================================
# Through HookRegistry
# ============================================================

@pytest.mark.asyncio
async def test_hooks_fire_through_registry(tmp_path: Path):
    """Wiring sanity: both hooks fire when the registry sees a turn-complete."""
    recorder = MagicMock()
    trace_path = tmp_path / "t.jsonl"

    reg = Registry([BillingHook(recorder=recorder), TracingHook(trace_path=trace_path)])
    steps = [
        ModelCallStep(turn=1, tokens_in=10, tokens_out=5),
        ReplyStep(sent=True),
    ]
    await reg.on_turn_complete(steps, _ctx())

    recorder.record.assert_called_once()
    assert trace_path.exists()
