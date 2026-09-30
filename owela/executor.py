"""The Owela executor — one function, one place where the loop lives.

``execute_turn`` runs a single user turn end-to-end. Behaviour is
driven entirely by the Policy returned from the Classifier; the
executor itself contains no conditional logic about any transport,
model or product. Anti-trap principle #1: no special cases in here.

Shape of one turn (each numbered phase corresponds to a Step):

  1. Router       → RouterStep (+ ErrorStep if the classifier raised)
  2. Degrade      → DegradeStep, if a ``requires_tools`` tool is
                    unavailable before the turn starts
  3. Plan         → PlanStep                 (gated by policy)
  4. Initial synthesis (no LLM call)
                  → ModelCallStep + ToolStep ×N
  5. Act loop     → ModelCallStep + ToolStep ×N
       After each tool dispatch: mid-turn degrade if a required tool
       errored, then OPTIONAL auto-followup synthesis.
  6. Critique     → CritiqueStep + ReviseStep (gated by policy)
  7. Reply        → ReplyStep

Guarantees:
  - A ReplyStep is always appended and ``on_turn_complete`` always
    fires, even when a phase raises (→ ErrorStep + ``fallback_reply``).
  - ``policy.deadline_s`` is soft: past it, the next model call gets no
    tools (it composes from what the turn already has) and critique is
    skipped.
  - ``policy.interstitial_after_s`` schedules the transport's "still
    working" message; it is cancelled as soon as the reply is ready.

Synthesised calls (no LLM round-trip) do NOT count toward
``policy.max_steps``. They carry ``attrs["synthesized_by_policy"]`` and
``attrs["decision_source"]`` ∈ {"plan.queries", "message",
"forced_tool_fallback", "auto_followup"} for the audit trail.

The returned ``list[Step]`` is the canonical record of the turn.
Persistence is NOT performed by the executor; register
``owela.hooks_builtin.MemoryRecordingHook``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from .errors import PolicyNotFound
from .hooks import TurnContext
from .model import ModelRequest
from .policy import AUTO, THINKING_OFF, Policy, forced_tool_name
from .router import ClassifierResult
from .step import (
    REPLY_DEADLINE, REPLY_ERROR, REPLY_MAX_STEPS, REPLY_OK, TOOL_URLS_ATTR,
    DegradeStep, ErrorStep, ModelCallStep, PlanStep, QueryVariant, ReplyStep,
    RouterStep, Step, ToolStep,
)
from .tools import ToolContext
from .transport import InboundMessage, ReplyContext, SendResult

if TYPE_CHECKING:
    from .runtime import Runtime

log = logging.getLogger("owela.executor")


async def execute_turn(runtime: "Runtime", msg: InboundMessage) -> list[Step]:
    """Run one turn. See module docstring for phase ordering."""
    steps: list[Step] = []
    clock = _TurnClock()
    tool_ctx = ToolContext(user_id=msg.user_id, runtime=runtime, msg=msg)

    # Surface "we got it" UX immediately — read receipt + typing indicator.
    try:
        await runtime.transport.acknowledge(msg)
    except Exception as exc:                          # noqa: BLE001 — UX is soft-fail
        log.warning("transport.acknowledge failed: %s", exc)

    # 1. Router. Classifier failures never abort the turn.
    rstep, router_error = await _route(runtime, msg)
    steps.append(rstep)
    policy = runtime.policies.lookup(rstep.verdict, rstep.depth)
    ctx = TurnContext(msg=msg, policy=policy, runtime=runtime)
    await runtime.hooks.on_step(rstep, ctx)
    if router_error is not None:
        steps.append(router_error)
        await runtime.hooks.on_step(router_error, ctx)
    clock.deadline_s = policy.deadline_s

    interstitial = _start_interstitial_timer(runtime, msg, policy, clock)
    draft = ""
    reason = REPLY_OK
    truncated = False
    try:
        # 2. Pre-turn degrade: a required capability is known to be down.
        policy = await _maybe_degrade(
            runtime, ctx, steps,
            runtime.tools.unavailable(policy.requires_tools), "breaker_open",
        )

        # 3. Planner (only if policy says so AND a planner is wired in)
        if policy.enable_planner and runtime.planner is not None:
            plan_step = await runtime.planner.plan(msg, policy, steps)
            steps.append(plan_step)
            await runtime.hooks.on_step(plan_step, ctx)

        messages = await runtime.memory.assemble_messages(msg, policy, steps)
        synth = _Synth(runtime, ctx, tool_ctx, steps, messages)

        # 4. Initial synthesis: plan query variants, or one call built from
        # the message. If it ran, the first model call must not re-force.
        variants, source = _initial_variants(policy, steps, msg)
        synthesised_initial = bool(variants)
        if synthesised_initial:
            policy = await synth.dispatch_and_follow_up(policy, variants, source)
        first_tool = AUTO if synthesised_initial else policy.first_tool

        # 5. Act loop. Only model-driven turns count toward max_steps.
        model_turns_taken = 0
        while model_turns_taken < policy.max_steps:
            model_turns_taken += 1
            compose_only = clock.over()
            tc = AUTO if (compose_only or model_turns_taken > 1) else first_tool
            req = _model_request(runtime, messages, policy, tc, compose_only)
            call_start = time.monotonic()
            resp = await runtime.model.complete(req)
            call_step = _model_call_step(req, resp, call_start, model_turns_taken)
            steps.append(call_step)
            await runtime.hooks.on_step(call_step, ctx)

            if call_step.forced_tool_honoured is False and call_step.forced_tool == policy.synth_tool:
                # The engine accepted a named tool_choice but the model did
                # not call the tool. Discard the prose and dispatch the call
                # deterministically from the message instead.
                policy = await synth.dispatch_and_follow_up(
                    policy, [QueryVariant(query=msg.text)], "forced_tool_fallback",
                )
                continue

            if not resp.tool_calls:
                draft = resp.content
                truncated = resp.finish_reason == "length"
                if compose_only:
                    reason = REPLY_DEADLINE
                break

            tool_steps = await runtime.tools.execute_parallel(resp.tool_calls, tool_ctx)
            steps.extend(tool_steps)
            for ts in tool_steps:
                await runtime.hooks.on_step(ts, ctx)
            _append_act_iteration_to_messages(
                messages, resp.content, resp.tool_calls, tool_steps, policy,
            )
            policy = await _degrade_on_tool_error(runtime, ctx, steps, tool_steps)
            await synth.auto_followup(policy, tool_steps)
        else:
            draft = policy.fallback_reply
            reason = REPLY_MAX_STEPS

        # 6. Critique + revise — only for a real draft, with time left, on a
        # turn that still has the evidence it was meant to have.
        degraded = any(isinstance(s, DegradeStep) for s in steps)
        if (policy.enable_critique and runtime.reviewer is not None
                and reason == REPLY_OK and not degraded and not clock.over()):
            crit = await runtime.reviewer.critique(msg, draft, steps, policy)
            steps.append(crit)
            await runtime.hooks.on_step(crit, ctx)
            if crit.verdict == "REVISE" and not clock.over():
                revised = await runtime.reviewer.revise(msg, draft, crit, steps, policy)
                steps.append(revised)
                await runtime.hooks.on_step(revised, ctx)
                draft = revised.attrs.get("revised_reply", draft)
    except Exception as exc:                           # noqa: BLE001 — the turn must still reply
        log.exception("turn failed for policy %s: %s", ctx.policy.name, exc)
        err = ErrorStep(phase="turn", exc_type=type(exc).__name__, message=str(exc)[:300])
        err.ended_at = time.monotonic()
        steps.append(err)
        await runtime.hooks.on_step(err, ctx)
        policy = ctx.policy
        draft = policy.fallback_reply
        reason = REPLY_ERROR
        truncated = False
    finally:
        if interstitial is not None:
            interstitial.cancel()

    # 7. Reply — the transport owns reply hygiene. Hooks observe only.
    reply_step = await _reply(runtime, msg, policy, steps, draft, reason, truncated, clock)
    steps.append(reply_step)
    await runtime.hooks.on_step(reply_step, ctx)

    # Final fan-out — billing, tracing, eval recording, memory persistence.
    await runtime.hooks.on_turn_complete(steps, ctx)
    return steps


# ----------------------------------------------------------------------
# Turn plumbing
# ----------------------------------------------------------------------


class _TurnClock:
    """Wall time since the turn started, against an optional soft deadline."""

    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.deadline_s: float | None = None

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def over(self) -> bool:
        return self.deadline_s is not None and self.elapsed() >= self.deadline_s


async def _route(runtime: "Runtime", msg: InboundMessage) -> tuple[RouterStep, ErrorStep | None]:
    """Classify and wrap into a RouterStep. A raising classifier yields the
    default verdict with ``fallback_reason`` set, plus an ErrorStep."""
    start = time.monotonic()
    error: ErrorStep | None = None
    try:
        result = await runtime.classifier.classify(msg)
    except Exception as exc:                          # noqa: BLE001 — routing is fail-safe
        log.warning("classifier raised: %s", exc)
        result = ClassifierResult(fallback_reason=f"exception:{type(exc).__name__}")
        error = ErrorStep(phase="router", exc_type=type(exc).__name__, message=str(exc)[:300])
        error.ended_at = time.monotonic()
    rstep = RouterStep(
        started_at=start,
        ended_at=time.monotonic(),
        verdict=result.verdict,
        depth=result.depth,
        fallback_reason=result.fallback_reason,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cached_tokens=result.cached_tokens,
    )
    rstep.attrs.update(result.attrs)
    return rstep, error


def _start_interstitial_timer(
    runtime: "Runtime", msg: InboundMessage, policy: Policy, clock: _TurnClock,
) -> asyncio.Task | None:
    """Schedule the transport's "still working" message. The delay is
    capped at the transport's typing window, measured from turn start."""
    if policy.interstitial_after_s is None:
        return None
    delay = min(policy.interstitial_after_s, runtime.transport.typing_window_s)

    async def _fire() -> None:
        await asyncio.sleep(max(0.0, delay - clock.elapsed()))
        try:
            await runtime.transport.send_interstitial(msg.user_id, policy)
        except Exception as exc:                      # noqa: BLE001 — UX is soft-fail
            log.warning("transport.send_interstitial failed: %s", exc)

    return asyncio.create_task(_fire())


async def _maybe_degrade(
    runtime: "Runtime",
    ctx: TurnContext,
    steps: list[Step],
    missing: tuple[str, ...],
    trigger: str,
) -> Policy:
    """Swap to ``policy.on_unavailable`` when required tools are missing.
    Records one DegradeStep per turn; rebinds ``ctx.policy``."""
    policy = ctx.policy
    if not missing or any(isinstance(s, DegradeStep) for s in steps):
        return policy
    target = policy
    if policy.on_unavailable:
        try:
            target = runtime.policies.by_name(policy.on_unavailable)
        except PolicyNotFound:
            log.warning("policy %s names unknown on_unavailable %r",
                        policy.name, policy.on_unavailable)
    step = DegradeStep(
        from_policy=policy.name, to_policy=target.name,
        missing_tools=tuple(missing), trigger=trigger,
    )
    step.ended_at = step.started_at
    steps.append(step)
    ctx.policy = target
    await runtime.hooks.on_step(step, ctx)
    return target


async def _degrade_on_tool_error(
    runtime: "Runtime", ctx: TurnContext, steps: list[Step], tool_steps: list[ToolStep],
) -> Policy:
    required = ctx.policy.requires_tools
    failed = tuple(dict.fromkeys(
        ts.tool_name for ts in tool_steps if ts.error is not None and ts.tool_name in required
    ))
    return await _maybe_degrade(runtime, ctx, steps, failed, "tool_error")


def _model_request(
    runtime: "Runtime",
    messages: list[dict[str, Any]],
    policy: Policy,
    tool_choice: Any,
    compose_only: bool,
) -> ModelRequest:
    thinking_on = policy.thinking != THINKING_OFF
    return ModelRequest(
        messages=messages,
        tools=[] if compose_only else runtime.tools.schemas(expose=policy.expose_tools),
        tool_choice=tool_choice,
        max_tokens=policy.max_reply_tokens,
        temperature=policy.temperature,
        thinking=policy.thinking,
        thinking_budget=policy.max_thinking_tokens if thinking_on else None,
        policy=policy,
    )


def _model_call_step(req: ModelRequest, resp: Any, started: float, turn: int) -> ModelCallStep:
    forced = forced_tool_name(req.tool_choice) if req.tools else None
    step = ModelCallStep(
        started_at=started,
        turn=turn,
        finish_reason=resp.finish_reason,
        thinking=req.thinking,
        thinking_budget=req.thinking_budget,
        max_tokens=req.max_tokens,
        tool_calls=list(resp.tool_calls),
        tokens_in=resp.tokens_in,
        tokens_out=resp.tokens_out,
        cached_tokens=resp.cached_tokens,
        forced_tool=forced,
        forced_tool_honoured=(
            None if forced is None
            else any((tc.get("function") or {}).get("name") == forced for tc in resp.tool_calls)
        ),
    )
    step.ended_at = time.monotonic()
    # Merge adapter-supplied audit attrs BEFORE content so a stray
    # "content" key in resp.attrs can't shadow the real reply text.
    if resp.attrs:
        step.attrs.update(resp.attrs)
    step.attrs["content"] = resp.content
    return step


async def _reply(
    runtime: "Runtime",
    msg: InboundMessage,
    policy: Policy,
    steps: list[Step],
    draft: str,
    reason: str,
    truncated: bool,
    clock: _TurnClock,
) -> ReplyStep:
    tool_steps = [s for s in steps if isinstance(s, ToolStep)]
    reply_ctx = ReplyContext(
        used_tools=tuple(dict.fromkeys(s.tool_name for s in tool_steps)),
        allowed_urls=_cited_urls(tool_steps),
        truncated=truncated,
        degraded=any(isinstance(s, DegradeStep) for s in steps),
        deadline_exceeded=clock.over(),
        reason=reason,
    )
    step = ReplyStep(
        started_at=time.monotonic(),
        reply_len=len(draft),
        reason=reason,
        truncated=truncated,
        degraded=reply_ctx.degraded,
        deadline_exceeded=reply_ctx.deadline_exceeded,
    )
    try:
        result = await runtime.transport.send(msg.user_id, draft, policy, reply_ctx)
    except Exception as exc:                           # noqa: BLE001 — transport errors logged, turn completes
        log.exception("transport.send failed: %s", exc)
        result = SendResult(sent=False, attrs={"send_error": type(exc).__name__})
    if isinstance(result, bool):                       # tolerate bool-returning transports
        result = SendResult(sent=result)
    step.sent = bool(result.sent)
    step.attrs.update(result.attrs)
    step.ended_at = time.monotonic()
    step.attrs["reply_text"] = draft                   # full draft for hook visibility
    return step


def _cited_urls(tool_steps: list[ToolStep]) -> tuple[str, ...]:
    urls: list[str] = []
    for ts in tool_steps:
        if ts.error is not None:
            continue
        value = ts.attrs.get(TOOL_URLS_ATTR)
        if isinstance(value, (list, tuple)):
            urls.extend(u for u in value if isinstance(u, str) and u)
    return tuple(dict.fromkeys(urls))


# ----------------------------------------------------------------------
# Synthesis (tool dispatch without an LLM call)
# ----------------------------------------------------------------------


def _initial_variants(
    policy: Policy, steps: list[Step], msg: InboundMessage,
) -> tuple[list[QueryVariant], str]:
    """Variants for the pre-loop synthesis: the latest plan's queries if
    any, else one call from the message text if the policy asks for it."""
    if not policy.synth_tool:
        return [], ""
    plan = _latest_plan_step(steps)
    if plan is not None and plan.queries:
        return list(plan.queries), "plan.queries"
    if policy.synth_first_call_from_message:
        return [QueryVariant(query=msg.text)], "message"
    return [], ""


class _Synth:
    """Dispatches policy-synthesised tool calls for one turn and keeps the
    step list, hooks and model-visible messages in sync."""

    def __init__(
        self,
        runtime: "Runtime",
        ctx: TurnContext,
        tool_ctx: ToolContext,
        steps: list[Step],
        messages: list[dict[str, Any]],
    ) -> None:
        self.runtime = runtime
        self.ctx = ctx
        self.tool_ctx = tool_ctx
        self.steps = steps
        self.messages = messages

    async def dispatch_and_follow_up(
        self, policy: Policy, variants: list[QueryVariant], source: str,
    ) -> Policy:
        """Synthesise one ``policy.synth_tool`` call per variant, then apply
        mid-turn degrade and auto-followup exactly like a model turn.
        Returns the (possibly degraded) policy."""
        calls = []
        for variant in variants:
            # Precedence: policy defaults → variant.extra → primary query,
            # so the query string can't be overwritten by either.
            args: dict[str, Any] = dict(policy.synth_default_args)
            if variant.extra:
                args.update(variant.extra)
            if policy.synth_arg:
                args[policy.synth_arg] = variant.query
            calls.append(_tool_call(policy.synth_tool or "", args, "q"))
        tool_steps = await self._dispatch(policy, calls, source)
        if source == "plan.queries":
            for i, ts in enumerate(tool_steps):
                ts.attrs["query_variant_index"] = i
        policy = await _degrade_on_tool_error(self.runtime, self.ctx, self.steps, tool_steps)
        await self.auto_followup(policy, tool_steps)
        return policy

    async def auto_followup(self, policy: Policy, recent: list[ToolStep]) -> None:
        """Rule-based follow-up: if any recent ToolStep matches the policy
        trigger AND carries ``attrs[auto_followup_attr]``, synthesise one
        call with the consolidated values. No-op otherwise."""
        if not policy.auto_followup_after or not policy.auto_followup_tool:
            return
        # Never feed the follow-up machinery its own output — a policy with
        # ``auto_followup_after == auto_followup_tool`` would loop.
        matching = [
            ts for ts in recent
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
        call = _tool_call(policy.auto_followup_tool, {policy.auto_followup_arg: pool}, "f")
        tool_steps = await self._dispatch(policy, [call], "auto_followup")
        errored = [ts for ts in tool_steps if ts.error is not None]
        if errored:
            log.warning(
                "auto_followup dispatch produced %d errored step(s) for tool %r",
                len(errored), policy.auto_followup_tool,
            )

    async def _dispatch(
        self, policy: Policy, calls: list[dict[str, Any]], source: str,
    ) -> list[ToolStep]:
        # Synthesised ModelCallStep — the "decision" made by the policy,
        # not by the model. No LLM consumed.
        decision = ModelCallStep(
            started_at=time.monotonic(),
            turn=0,
            finish_reason="tool_calls",
            tool_calls=list(calls),
            synthesized=True,
        )
        decision.ended_at = decision.started_at
        decision.attrs["synthesized_by_policy"] = policy.name
        decision.attrs["decision_source"] = source
        self.steps.append(decision)
        await self.runtime.hooks.on_step(decision, self.ctx)

        tool_steps = await self.runtime.tools.execute_parallel(calls, self.tool_ctx)
        for ts in tool_steps:
            ts.attrs["synthesized_by_policy"] = policy.name
            ts.attrs["decision_source"] = source
        self.steps.extend(tool_steps)
        for ts in tool_steps:
            await self.runtime.hooks.on_step(ts, self.ctx)
        _append_act_iteration_to_messages(self.messages, "", calls, tool_steps, policy)
        return tool_steps


def _tool_call(name: str, args: dict[str, Any], tag: str) -> dict[str, Any]:
    return {
        "id": f"call_synth_{tag}_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


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
