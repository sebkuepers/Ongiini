"""The Owela executor — one function, one place where the loop lives.

``execute_turn`` runs a single user turn end-to-end. Behaviour is
driven entirely by the Policy returned from the Classifier; the
executor itself contains no conditional logic about WhatsApp, Gemma,
mem0, or any other product specifics. Anti-trap principle #1: no
special cases in here.

Shape of one turn (each numbered phase corresponds to a Step):

  1. Router       → RouterStep
  2. Interstitial → (transport side-effect; no step yet — v1 emits one)
  3. Plan         → PlanStep                 (v1, gated by policy)
  4. Act loop     → ModelCallStep + ToolStep (×N)
  5. Critique     → CritiqueStep + ReviseStep (v1, gated by policy)
  6. Reply        → ReplyStep

Each step is appended to ``steps`` and broadcast via the hook registry
both as it lands (``on_step``) and again at the end (``on_turn_complete``).

The returned ``list[Step]`` is the canonical record of the turn — the
Application (or hooks) can summarise it however they want.

Persistence is NOT performed by the executor; the standard pattern is
to register ``owela.hooks_builtin.MemoryRecordingHook`` in the
HookRegistry. See that module for the contract.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from .hooks import TurnContext
from .model import ModelRequest
from .policy import AUTO
from .step import ModelCallStep, ReplyStep, RouterStep, Step, ToolStep
from .tools import ToolContext
from .transport import InboundMessage

if TYPE_CHECKING:
    from .runtime import Runtime

log = logging.getLogger("owela.executor")


async def execute_turn(runtime: "Runtime", msg: InboundMessage) -> list[Step]:
    """Run one turn. See module docstring for phase ordering."""
    steps: list[Step] = []
    tool_ctx = ToolContext(user_id=msg.user_id, runtime=runtime, msg=msg)

    # Surface "we got it" UX immediately — read receipt + typing indicator
    # for transports that support it.
    try:
        await runtime.transport.acknowledge(msg)
    except Exception as exc:                          # noqa: BLE001 — UX is soft-fail
        log.warning("transport.acknowledge failed: %s", exc)

    # 1. Router. The classifier returns a small value object; the executor
    # owns timing and wraps it into a RouterStep. Adapters never need to
    # know about the Step contract.
    rstart = time.monotonic()
    result = await runtime.classifier.classify(msg)
    rstep = RouterStep(
        started_at=rstart,
        ended_at=time.monotonic(),
        verdict=result.verdict,
        depth=result.depth,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cached_tokens=result.cached_tokens,
    )
    rstep.attrs.update(result.attrs)
    steps.append(rstep)
    policy = runtime.policies.lookup(rstep.verdict, rstep.depth)
    ctx = TurnContext(msg=msg, policy=policy, runtime=runtime)
    await runtime.hooks.on_step(rstep, ctx)

    # 2. Interstitial (v1 — only if policy says so AND transport supports it)
    if policy.enable_interstitial:
        try:
            await runtime.transport.send_interstitial(msg.user_id, policy)
        except Exception as exc:                       # noqa: BLE001
            log.warning("transport.send_interstitial failed: %s", exc)

    # 3. Planner (v1 — only if policy says so AND a planner is wired in)
    if policy.enable_planner and runtime.planner is not None:
        plan_step = await runtime.planner.plan(msg, policy, steps)
        steps.append(plan_step)
        await runtime.hooks.on_step(plan_step, ctx)

    # 4. Act loop — model + parallel tool dispatch
    messages = await runtime.memory.assemble_messages(msg, policy, steps)
    draft = ""
    # ``prev_long_result`` is consumed by the NEXT iteration's reasoning
    # decision; on the final iteration it's computed and discarded, which
    # is fine.
    prev_long_result = False

    for turn_no in range(1, policy.max_steps + 1):
        first_turn = (turn_no == 1)
        tc = policy.first_tool if first_turn else AUTO
        enable_thinking = (
            policy.enable_thinking_after_long_results and prev_long_result
        )

        req = ModelRequest(
            messages=messages,
            tools=runtime.tools.schemas(expose=policy.expose_tools),
            tool_choice=tc,
            policy=policy,
            enable_thinking=enable_thinking,
        )
        call_start = time.monotonic()
        resp = await runtime.model.complete(req)

        call_step = ModelCallStep(
            started_at=call_start,
            turn=turn_no,
            finish_reason=resp.finish_reason,
            enable_thinking=enable_thinking,
            reasoning_budget=policy.reasoning_budget if enable_thinking else None,
            tool_calls=list(resp.tool_calls),
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            cached_tokens=resp.cached_tokens,
        )
        call_step.ended_at = time.monotonic()
        call_step.attrs["content"] = resp.content
        steps.append(call_step)
        await runtime.hooks.on_step(call_step, ctx)

        # No tool calls → we have our draft
        if not resp.tool_calls:
            draft = resp.content
            break

        # PARALLEL tool dispatch
        tool_steps = await runtime.tools.execute_parallel(resp.tool_calls, tool_ctx)
        steps.extend(tool_steps)
        for ts in tool_steps:
            await runtime.hooks.on_step(ts, ctx)

        # Append model's assistant turn + tool results so the next iteration
        # sees the loop history.
        messages.append({
            "role": "assistant",
            "content": resp.content,
            "tool_calls": resp.tool_calls,
        })
        for ts in tool_steps:
            messages.append({
                "role": "tool",
                "tool_call_id": ts.tool_call_id,
                "content": ts.attrs.get("result", ""),
            })

        # Decide whether the NEXT turn should enable reasoning (long
        # tool result → model needs deliberation to digest + cite).
        prev_long_result = any(
            ts.result_len >= policy.long_result_threshold_chars
            for ts in tool_steps
        )
    else:
        # Loop fell through max_steps without a final answer.
        draft = "Sorry, I'm having trouble answering that right now."

    # 5. Critique + revise (v1 — gated)
    if policy.enable_critique and runtime.reviewer is not None:
        crit = await runtime.reviewer.critique(msg, draft, steps, policy)
        steps.append(crit)
        await runtime.hooks.on_step(crit, ctx)
        if crit.verdict == "REVISE":
            revised = await runtime.reviewer.revise(msg, draft, crit, steps, policy)
            steps.append(revised)
            await runtime.hooks.on_step(revised, ctx)
            draft = revised.attrs.get("revised_reply", draft)

    # 6. Reply — transport handles its own hygiene (dead-URL strip, format,
    #    char cap). Hooks observe the ReplyStep but don't transform.
    #
    # used_search hint: True iff any search-shaped tool fired. The
    # WhatsApp transport uses this to gate the dead-URL HEAD-check —
    # cheap when no search was used, valuable when the model cited
    # tool output that may have stale links.
    _SEARCH_TOOLS = ("web_search", "fetch_url", "fetch_urls")
    used_search = any(
        isinstance(s, ToolStep) and s.tool_name in _SEARCH_TOOLS for s in steps
    )

    reply_step = ReplyStep(started_at=time.monotonic(), reply_len=len(draft))
    try:
        reply_step.sent = await runtime.transport.send(
            msg.user_id, draft, policy, used_search=used_search,
        )
    except Exception as exc:                           # noqa: BLE001 — transport errors logged but turn completes
        log.exception("transport.send failed: %s", exc)
        reply_step.sent = False
    reply_step.ended_at = time.monotonic()
    reply_step.attrs["reply_text"] = draft             # full text for hook visibility
    steps.append(reply_step)
    await runtime.hooks.on_step(reply_step, ctx)

    # Final fan-out — billing, tracing, eval recording, memory persistence
    # (via MemoryRecordingHook). After this returns, the turn is "done"
    # and the Agent's handle() can return its summary.
    await runtime.hooks.on_turn_complete(steps, ctx)

    return steps
