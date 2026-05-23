"""Tests for the Ongiini billing + tracing hooks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from owela import (
    CritiqueStep, HookRegistry, InboundMessage, ModelCallStep, PlanStep, Policy,
    ReplyStep, ReviseStep, RouterStep, Step, ToolStep, TurnContext,
)
from owela.hooks import HookRegistry as Registry
from ongiini.hooks.billing_hook import BillingHook
from ongiini.hooks.tracing_hook import TracingHook


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
async def test_billing_post_search_compose_input_excluded():
    """v1.3.1 fairness: a compose ModelCallStep AFTER a search ToolStep
    has its INPUT tokens excluded from the user's bill (the input is
    system-added search context, not user payload). Output is still
    billed."""
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

    # Chat line — billable aggregate, with v1.3.1 fairness rules:
    chat_call = next(c for c in calls if c.kwargs.get("kind") == "chat")
    # tokens_in: turn-1 compose (100, before any search) + turn-2 compose
    # input EXCLUDED (search context already in scope) = 100.
    assert chat_call.args[1] == 100
    # tokens_out: both compose outputs counted = 10 + 200 = 210.
    assert chat_call.args[2] == 210
    assert chat_call.kwargs["used_search"] is True


@pytest.mark.asyncio
async def test_billing_v1_phases_apply_per_phase_rules():
    """v1.3.1 per-phase billing rules:
      - PlanStep: full bill (planner thinks about the user's question).
      - Compose BEFORE search: full bill.
      - Compose AFTER search: input excluded, output billed.
      - Critique: NOT billed (internal quality control).
      - Revise: input NOT billed, output billed (revise output replaces
        the draft and becomes the user-visible reply).
    """
    recorder = MagicMock()
    hook = BillingHook(recorder=recorder)
    steps = [
        RouterStep(tokens_in=20, tokens_out=5, verdict="SEARCH", depth="DEEP"),
        PlanStep(tokens_in=30, tokens_out=180),                  # +30 in, +180 out
        ModelCallStep(turn=1, tokens_in=100, tokens_out=10),     # +100 in, +10 out
        ToolStep(tool_name="web_search", result_len=2000),       # flips search_ctx_active
        ModelCallStep(turn=2, tokens_in=80, tokens_out=400),     # +0 in, +400 out
        CritiqueStep(tokens_in=50, tokens_out=80, verdict="REVISE"),  # +0, +0 (free)
        ReviseStep(tokens_in=200, tokens_out=600),               # +0 in, +600 out
        ReplyStep(reply_len=400, sent=True),
    ]
    await hook.on_turn_complete(steps, _ctx())

    chat_call = next(c for c in recorder.record.call_args_list if c.kwargs.get("kind") == "chat")
    # tokens_in = 30 + 100 = 130
    assert chat_call.args[1] == 130
    # tokens_out = 180 + 10 + 400 + 600 = 1190
    assert chat_call.args[2] == 1190
    assert chat_call.kwargs["used_search"] is True


@pytest.mark.asyncio
async def test_billing_synthesized_model_call_steps_are_free():
    """v1.3.1: ModelCallSteps with synthesized=True (multi-query fan-out
    or auto-followup synth dispatches) have no real LLM call and must
    cost 0 in + 0 out, regardless of token fields."""
    recorder = MagicMock()
    hook = BillingHook(recorder=recorder)
    steps = [
        # A non-synth model call to establish the baseline.
        ModelCallStep(turn=1, tokens_in=50, tokens_out=30),
        # Synth steps — even with non-zero token fields (defensive),
        # the hook must NOT bill them.
        ModelCallStep(turn=0, synthesized=True,
                      tokens_in=999, tokens_out=999),
    ]
    await hook.on_turn_complete(steps, _ctx())
    chat = next(c for c in recorder.record.call_args_list if c.kwargs.get("kind") == "chat")
    assert chat.args[1] == 50    # only the non-synth call's input
    assert chat.args[2] == 30    # only the non-synth call's output


@pytest.mark.asyncio
async def test_billing_cached_tokens_subtracted_from_input():
    """v1.3.1: cached prompt tokens (prefix-cache hits) cost no real
    GPU work — they should NOT count against the user's monthly bill."""
    recorder = MagicMock()
    hook = BillingHook(recorder=recorder)
    steps = [
        # 1000 prompt tokens but 800 were cached → billable input is 200.
        ModelCallStep(turn=1, tokens_in=1000, tokens_out=50, cached_tokens=800),
    ]
    await hook.on_turn_complete(steps, _ctx())
    chat = next(c for c in recorder.record.call_args_list if c.kwargs.get("kind") == "chat")
    assert chat.args[1] == 200    # 1000 - 800 cached
    assert chat.args[2] == 50


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
async def test_tracing_captures_v1_phases_in_separate_block(tmp_path: Path):
    """v1: PlanStep / CritiqueStep / ReviseStep land in their own
    `phases` block alongside `calls`, with structural fields only
    (verdict, plan_len, revised_len, latency, error). Tokens roll into
    the totals."""
    trace_path = tmp_path / "trace.jsonl"
    hook = TracingHook(trace_path=trace_path)

    plan = PlanStep(plan_text="FACTS TO LOOK UP:\n- rate", tokens_in=30, tokens_out=180)
    plan.ended_at = plan.started_at + 0.05
    critique = CritiqueStep(verdict="REVISE", reasons=["x", "y"], tokens_in=50, tokens_out=80)
    critique.ended_at = critique.started_at + 0.02
    revise = ReviseStep(tokens_in=200, tokens_out=600)
    revise.attrs["revised_reply"] = "revised reply text"
    revise.ended_at = revise.started_at + 0.10

    steps = [
        RouterStep(verdict="SEARCH", depth="DEEP"),
        plan,
        ModelCallStep(turn=1, tokens_in=100, tokens_out=10),
        critique,
        revise,
        ReplyStep(reply_len=400, sent=True),
    ]
    await hook.on_turn_complete(steps, _ctx())

    entry = json.loads(trace_path.read_text().strip())
    assert "phases" in entry
    phase_kinds = [p["kind"] for p in entry["phases"]]
    assert phase_kinds == ["plan", "critique", "revise"]

    critique_entry = next(p for p in entry["phases"] if p["kind"] == "critique")
    assert critique_entry["verdict"] == "REVISE"
    assert critique_entry["reasons_count"] == 2

    revise_entry = next(p for p in entry["phases"] if p["kind"] == "revise")
    assert revise_entry["revised_len"] == len("revised reply text")

    # Totals MUST include the v1 phases.
    assert entry["total_tokens_in"] == 30 + 100 + 50 + 200      # plan+call+critique+revise
    assert entry["total_tokens_out"] == 180 + 10 + 80 + 600


@pytest.mark.asyncio
async def test_tracing_omits_critique_detail_by_default(tmp_path: Path):
    """Default mode: no raw critique text, no plan body. Privacy."""
    trace_path = tmp_path / "t.jsonl"
    hook = TracingHook(trace_path=trace_path)  # detail OFF by default

    plan = PlanStep(plan_text="FACTS TO LOOK UP:\n- specific data\n- another fact")
    crit = CritiqueStep(verdict="REVISE", reasons=["claim X not grounded"])
    crit.attrs["raw_critique"] = "VERY SECRET reviewer narrative including user content"
    steps = [
        RouterStep(verdict="SEARCH", depth="SHALLOW"),
        plan,
        ModelCallStep(turn=1),
        crit,
        ReplyStep(sent=True),
    ]
    await hook.on_turn_complete(steps, _ctx())

    entry = json.loads(trace_path.read_text().strip())
    plan_entry = next(p for p in entry["phases"] if p["kind"] == "plan")
    critique_entry = next(p for p in entry["phases"] if p["kind"] == "critique")
    # Structural fields stay.
    assert plan_entry["plan_len"] > 0
    assert critique_entry["verdict"] == "REVISE"
    assert critique_entry["reasons_count"] == 1
    # Detail fields must NOT be present.
    assert "plan_text" not in plan_entry
    assert "raw_critique" not in critique_entry
    assert "reasons" not in critique_entry
    # And the secret content must not have leaked anywhere into the line.
    body = trace_path.read_text()
    assert "VERY SECRET" not in body
    assert "specific data" not in body


@pytest.mark.asyncio
async def test_tracing_includes_critique_detail_when_flag_on(tmp_path: Path):
    """Detail mode: raw critique text + extracted reasons + plan
    snippet land in the per-phase entries. Used for experimentation."""
    trace_path = tmp_path / "t.jsonl"
    hook = TracingHook(trace_path=trace_path, include_critique_detail=True)

    plan = PlanStep(plan_text="FACTS TO LOOK UP:\n- price of X\n- price of Y" * 50)
    crit = CritiqueStep(verdict="REVISE", reasons=["claim X not grounded"])
    crit.attrs["raw_critique"] = "1. OK\n2. FAIL: claim X not grounded\nVERDICT: REVISE"
    steps = [
        RouterStep(verdict="SEARCH", depth="DEEP"),
        plan,
        ModelCallStep(turn=1),
        crit,
        ReplyStep(sent=True),
    ]
    await hook.on_turn_complete(steps, _ctx())

    entry = json.loads(trace_path.read_text().strip())
    plan_entry = next(p for p in entry["phases"] if p["kind"] == "plan")
    critique_entry = next(p for p in entry["phases"] if p["kind"] == "critique")
    # plan_text is included, capped at 400 chars.
    assert "plan_text" in plan_entry
    assert len(plan_entry["plan_text"]) <= 400
    assert "price of X" in plan_entry["plan_text"]
    # critique detail: extracted reasons AND raw body.
    assert critique_entry["reasons"] == ["claim X not grounded"]
    assert "FAIL: claim X not grounded" in critique_entry["raw_critique"]


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
    # v1.3 added audit keys (synthesized_by_policy, decision_source,
    # query_variant_index) — all None for model-chosen tools.
    assert entry["calls"][0].get("tool_results") == [
        {"name": "web_search", "args_len": 0, "result_len": 2000,
         "error": None, "latency_ms": 0,
         "synthesized_by_policy": None, "decision_source": None,
         "query_variant_index": None},
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
