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
  4. Pre-act synthesis (multi-query fan-out)
                  → ModelCallStep + ToolStep (×N, no LLM call)  (v1.3)
  5. Act loop     → ModelCallStep + ToolStep (×N)
       After each tool dispatch, OPTIONAL auto-followup synthesis
       fires (rule-based, no LLM call) gated by
       ``policy.auto_followup_after`` / ``policy.auto_followup_tool``.
  6. Critique     → CritiqueStep + ReviseStep (v1, gated by policy)
  7. Reply        → ReplyStep

Each step is appended to ``steps`` and broadcast via the hook registry
both as it lands (``on_step``) and again at the end (``on_turn_complete``).

The returned ``list[Step]`` is the canonical record of the turn — the
Application (or hooks) can summarise it however they want.

Persistence is NOT performed by the executor; the standard pattern is
to register ``owela.hooks_builtin.MemoryRecordingHook`` in the
HookRegistry. See that module for the contract.

Synthesized phases (v1.3): the executor can dispatch tool calls
WITHOUT an LLM call when the policy declares a deterministic
sequence:

  - ``planner_query_tool`` + ``PlanStep.queries`` → turn-1 fan-out
  - ``auto_followup_after`` + ``auto_followup_tool`` → rule-based
    follow-up after any matching ToolStep

Synthesized turns do NOT count toward ``policy.max_steps`` (that
budget is for model-driven turns only). All synthesized steps carry
``attrs["synthesized_by_policy"] = policy.name`` and
``attrs["decision_source"] = "plan.queries" | "auto_followup"`` for
the audit trail.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from .hooks import TurnContext
from .model import ModelRequest
from .policy import AUTO, Policy
from .step import ModelCallStep, PlanStep, ReplyStep, RouterStep, Step, ToolStep
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

    # 4a. Pre-loop synthesis: if the planner emitted query variants AND
    # the policy says they should fan out, synthesise N parallel tool
    # calls before the model takes its first turn. The synthesised step
    # does NOT count toward ``policy.max_steps`` — that budget is for
    # model-driven turns.
    synthesised_initial = await _maybe_synthesize_planner_fanout(
        steps=steps,
        messages=messages,
        runtime=runtime,
        ctx=ctx,
        tool_ctx=tool_ctx,
        policy=policy,
    )
    if synthesised_initial:
        # The initial fan-out already produced search results in
        # context. Any auto-followup after those results also fires.
        await _maybe_synthesize_auto_followup(
            recent_tool_steps=_recent_synth_tool_steps(steps),
            steps=steps,
            messages=messages,
            runtime=runtime,
            ctx=ctx,
            tool_ctx=tool_ctx,
            policy=policy,
        )

    # If we already did the initial search via synthesis, the model's
    # first turn shouldn't re-force the same tool. Otherwise, honour
    # ``policy.first_tool`` as before.
    first_tool_override = AUTO if synthesised_initial else policy.first_tool

    # Model-driven turns. Only these increment toward ``max_steps``.
    model_turns_taken = 0
    while model_turns_taken < policy.max_steps:
        model_turns_taken += 1
        first_turn = (model_turns_taken == 1)
        tc = first_tool_override if first_turn else AUTO
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
            turn=model_turns_taken,
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

        # Append the model's assistant turn + (truncated) tool results
        # so the next iteration sees the loop history.
        _append_act_iteration_to_messages(
            messages, resp.content, resp.tool_calls, tool_steps, policy,
        )

        # Rule-based auto-follow-up: synthesise a follow-up call if any
        # ToolStep in this turn matches the policy's trigger. Synthesised
        # steps don't count toward max_steps.
        await _maybe_synthesize_auto_followup(
            recent_tool_steps=tool_steps,
            steps=steps,
            messages=messages,
            runtime=runtime,
            ctx=ctx,
            tool_ctx=tool_ctx,
            policy=policy,
        )

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


# ----------------------------------------------------------------------
# Synthesis helpers (v1.3)
# ----------------------------------------------------------------------


def _truncate_for_messages(content: str, tool_name: str, policy: Policy) -> str:
    """Cap a tool result for the model-visible message list.

    The FULL result text remains in ``ToolStep.attrs["result"]`` for
    reviewer and trace consumers; this only bounds what the model sees
    in its context window. Policy supplies per-tool caps via
    ``policy.tool_result_message_caps``; tools without a cap pass
    through unchanged.
    """
    cap = policy.tool_result_message_caps.get(tool_name, 0)
    if cap <= 0 or len(content) <= cap:
        return content
    return content[:cap] + f"\n[truncated at {cap} chars for context budget]"


def _append_act_iteration_to_messages(
    messages: list[dict[str, Any]],
    assistant_content: str,
    tool_calls: list[dict[str, Any]],
    tool_steps: list[ToolStep],
    policy: Policy,
) -> None:
    """Append one assistant turn (with its tool_calls) + each tool
    result message, applying per-tool truncation."""
    messages.append({
        "role": "assistant",
        "content": assistant_content,
        "tool_calls": list(tool_calls),
    })
    for ts in tool_steps:
        result_text = ts.attrs.get("result", "")
        truncated = _truncate_for_messages(result_text, ts.tool_name, policy)
        messages.append({
            "role": "tool",
            "tool_call_id": ts.tool_call_id,
            "content": truncated,
        })


async def _maybe_synthesize_planner_fanout(
    *,
    steps: list[Step],
    messages: list[dict[str, Any]],
    runtime: "Runtime",
    ctx: TurnContext,
    tool_ctx: ToolContext,
    policy: Policy,
) -> bool:
    """If the policy declares a planner_query_tool AND the latest
    PlanStep carries query variants, synthesise one tool call per
    variant and dispatch them in parallel — no LLM call.

    Returns True if synthesis happened, False otherwise.
    """
    if not policy.planner_query_tool:
        return False
    plan = _latest_plan_step(steps)
    if plan is None or not plan.queries:
        return False

    tool_calls: list[dict[str, Any]] = []
    for variant in plan.queries:
        # Args precedence: policy defaults → variant.extra → primary
        # query. variant.extra wins over policy defaults so a planner
        # can opt out of a default for a specific variant; the primary
        # query string is always last so it can't be accidentally
        # overwritten by either.
        args: dict[str, Any] = dict(policy.planner_query_default_args)
        if variant.extra:
            args.update(variant.extra)
        args[policy.planner_query_arg] = variant.query
        tool_calls.append({
            "id": f"call_synth_q_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": policy.planner_query_tool,
                "arguments": json.dumps(args),
            },
        })

    # Synthesised ModelCallStep — represents the "decision" to fan out,
    # made by the policy, not by the model. No LLM consumed.
    synth_call = ModelCallStep(
        started_at=time.monotonic(),
        turn=0,
        finish_reason="tool_calls",
        tool_calls=list(tool_calls),    # defensive copy
        synthesized=True,
    )
    synth_call.ended_at = synth_call.started_at
    synth_call.attrs["synthesized_by_policy"] = policy.name
    synth_call.attrs["decision_source"] = "plan.queries"
    steps.append(synth_call)
    await runtime.hooks.on_step(synth_call, ctx)

    # Dispatch in parallel. ``execute_parallel`` returns one ToolStep
    # per tool_call in matching order.
    tool_steps = await runtime.tools.execute_parallel(tool_calls, tool_ctx)
    for i, ts in enumerate(tool_steps):
        ts.attrs["synthesized_by_policy"] = policy.name
        ts.attrs["decision_source"] = "plan.queries"
        ts.attrs["query_variant_index"] = i
    steps.extend(tool_steps)
    for ts in tool_steps:
        await runtime.hooks.on_step(ts, ctx)

    # Append to messages so the model's first real turn sees the search
    # results in context.
    _append_act_iteration_to_messages(
        messages, "", tool_calls, tool_steps, policy,
    )
    return True


async def _maybe_synthesize_auto_followup(
    *,
    recent_tool_steps: list[ToolStep],
    steps: list[Step],
    messages: list[dict[str, Any]],
    runtime: "Runtime",
    ctx: TurnContext,
    tool_ctx: ToolContext,
    policy: Policy,
) -> None:
    """Rule-based follow-up synthesis: if any recent ToolStep matches
    the policy trigger AND carries non-empty
    ``attrs[auto_followup_attr]``, synthesise a single follow-up call
    with the consolidated values.

    Idempotent: if no recent step matches the trigger, this is a no-op.
    """
    if not policy.auto_followup_after or not policy.auto_followup_tool:
        return

    # Skip ToolSteps that came from the auto-followup synthesis itself
    # — feeding the follow-up machinery its own output would loop if a
    # misconfigured policy set ``auto_followup_after == auto_followup_tool``.
    # Fan-out outputs (``decision_source == "plan.queries"``) are kept —
    # those are exactly what we WANT to trigger the follow-up.
    matching: list[ToolStep] = [
        ts for ts in recent_tool_steps
        if ts.tool_name == policy.auto_followup_after
        and ts.attrs.get("decision_source") != "auto_followup"
    ]
    if not matching:
        return

    pool = _consolidate_attr_values(
        [ts.attrs.get(policy.auto_followup_attr, []) for ts in matching],
        max_count=policy.auto_followup_max_items,
        one_per_host=policy.auto_followup_one_per_host,
    )
    if not pool:
        return

    synth_call = {
        "id": f"call_synth_f_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": policy.auto_followup_tool,
            "arguments": json.dumps({policy.auto_followup_arg: pool}),
        },
    }
    synth_call_step = ModelCallStep(
        started_at=time.monotonic(),
        turn=0,
        finish_reason="tool_calls",
        tool_calls=[synth_call],
        synthesized=True,
    )
    synth_call_step.ended_at = synth_call_step.started_at
    synth_call_step.attrs["synthesized_by_policy"] = policy.name
    synth_call_step.attrs["decision_source"] = "auto_followup"
    steps.append(synth_call_step)
    await runtime.hooks.on_step(synth_call_step, ctx)

    tool_steps = await runtime.tools.execute_parallel([synth_call], tool_ctx)
    for ts in tool_steps:
        ts.attrs["synthesized_by_policy"] = policy.name
        ts.attrs["decision_source"] = "auto_followup"
    steps.extend(tool_steps)
    for ts in tool_steps:
        await runtime.hooks.on_step(ts, ctx)
    errored = [ts for ts in tool_steps if ts.error is not None]
    if errored:
        log.warning(
            "auto_followup dispatch produced %d errored step(s) for tool %r",
            len(errored), policy.auto_followup_tool,
        )

    _append_act_iteration_to_messages(
        messages, "", [synth_call], tool_steps, policy,
    )


def _consolidate_attr_values(
    pools: list[Any], *, max_count: int, one_per_host: bool = True,
) -> list[str]:
    """Merge multiple lists, dedupe by canonical form, optionally
    enforce one item per URL host, cap at ``max_count``.

    Generic: callers pass arbitrary string lists. When
    ``one_per_host=True`` (default), URL hosts are extracted via
    ``urllib.parse.urlparse`` and only the first occurrence per host
    is kept — strings that don't parse as URLs (empty netloc) bypass
    the host filter. When ``one_per_host=False``, only canonical-form
    dedup applies.
    """
    seen_canonical: set[str] = set()
    seen_hosts: set[str] = set()
    out: list[str] = []
    for pool in pools:
        if not isinstance(pool, (list, tuple)):
            continue
        for item in pool:
            if not isinstance(item, str) or not item:
                continue
            canon = item.strip().rstrip("/").lower()
            if canon in seen_canonical:
                continue
            seen_canonical.add(canon)
            if one_per_host:
                host = urlparse(item).netloc.lower()
                if host and host in seen_hosts:
                    continue
                if host:
                    seen_hosts.add(host)
            out.append(item)
            if len(out) >= max_count:
                return out
    return out


def _latest_plan_step(steps: list[Step]) -> PlanStep | None:
    for s in reversed(steps):
        if isinstance(s, PlanStep):
            return s
    return None


def _recent_synth_tool_steps(steps: list[Step]) -> list[ToolStep]:
    """Walk back over the step list collecting ToolSteps that were
    appended in the most recent synthesis batch (until a step of any
    other type is hit, which marks the prior synthesis boundary)."""
    out: list[ToolStep] = []
    for s in reversed(steps):
        if isinstance(s, ToolStep):
            out.append(s)
        else:
            break
    return list(reversed(out))
