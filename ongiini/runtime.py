"""Composition root — builds the Owela ``Runtime`` for Ongiini.

This is the ONE place that knows every Ongiini-specific choice:

  - vLLM endpoint URL + model id
  - WhatsApp transport settings
  - Which long-term + short-term memory backends to wire
  - The depth-aware Gemma classifier prompt
  - The Ongiini tool catalogue (web_search, fetch_url, fetch_urls,
    delete_my_data, whats_in_my_memory, my_token_usage,
    lookup_ongiini_docs)
  - The PolicyTable mapping classifier verdicts → loop shape

If a future change needs to switch one of these — different model,
different transport, different memory store — this is the only file
to touch. Anti-trap principle #6: one Runtime object holds everything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from owela import Agent, BreakerConfig, CircuitBreaker, HookRegistry, Runtime, ToolRegistry

from . import pii, summary, usage
from .config import settings
from .hooks import (
    BillingHook, ContributeHallucinationGuardHook, HealthAlertHook,
    OngiiniMemoryRecordingHook, ReviseEvalCaptureHook, SourceIndexHook,
    TracingHook,
)
from .memory import (
    OngiiniMemoryProvider,
    long_term as mem,
    short_term as memory,
    source_index,
)
from .models import VLLMGemmaModel
from .planner import OngiiniPlanner
from .reviewer import OngiiniReviewer
from .routers import GemmaClassifier
from .routers.state_gated_classifier import StateGatedClassifier
from .policies import SECTIONS_SEARCH, build_policy_table
from .skills_loader import load_skills
from .system_prompt import build_system_prompt
from .tools import ALL_TOOLS
from .transports import WhatsAppTransport

log = logging.getLogger("ongiini.runtime")


@dataclass(frozen=True)
class SharedComponents:
    """Transport-agnostic Owela components, built ONCE at app startup
    and shared across both the WhatsApp runtime and any per-request
    chat runtime. Frozen so callers can't accidentally mutate the
    classifier/tools/policies mid-flight."""
    model: Any
    classifier: Any
    planner: Any
    reviewer: Any
    tools: Any
    policies: Any
    skills: Any
    trace_destination: Path


def build_shared_components(*, trace_path: Path | None = None) -> SharedComponents:
    """Build all transport-agnostic Owela components.

    The returned `SharedComponents` is used by `build_whatsapp_runtime`
    (single Runtime constructed at startup) and `build_chat_runtime`
    (per-request Runtime for the chat.ongiini.ai endpoint). Sharing is
    safe because every component is either stateless or holds its own
    concurrency protection (AsyncOpenAI clients are thread-safe per the
    SDK contract, ToolRegistry/PolicyTable/SkillRegistry are read-only
    after construction).
    """
    log.info("building Ongiini shared components…")

    # One model adapter for every call — act loop AND the auxiliary
    # classifier / planner / reviewer / summariser calls — so they all
    # share the same output sanitising and the same 90 s ceiling.
    model = VLLMGemmaModel(
        base_url=settings.vllm_base_url,
        model_id=settings.vllm_model,
        temperature=0.6,
        max_tokens=1500,
    )

    summary.set_model(model)
    skills = load_skills()

    # State-gated wrapper around GemmaClassifier. The inner classifier
    # emits its best opinion; the wrapper blocks CONTRIBUTE_* verdicts
    # that contradict fresh state (no pending → SAVE redirected to NONE
    # etc.). See ongiini/routers/state_gated_classifier.py.
    classifier = StateGatedClassifier(GemmaClassifier(model=model))

    # Planner runs only when policy.enable_planner fires (SEARCH_DEEP);
    # reviewer only when policy.enable_critique fires. Both soft-fail.
    planner = OngiiniPlanner(model=model)
    reviewer = OngiiniReviewer(model=model, system_prompt=build_system_prompt(SECTIONS_SEARCH))

    # The breaker stops calling a tool after 3 failures in 10 minutes and
    # re-probes after 15 — the Tavily 402 outage (Sep 2026) ran for
    # 3.5 weeks because nothing acted on ToolStep.error.
    tools = ToolRegistry(
        list(ALL_TOOLS),
        breaker=CircuitBreaker(BreakerConfig(failure_threshold=3, window_s=600, cooldown_s=900)),
    )
    policies = build_policy_table()

    trace_destination = trace_path or (settings.data_dir / "trace.jsonl")
    if settings.trace_critique_detail:
        log.warning(
            "ONGIINI_TRACE_CRITIQUE_DETAIL is set — trace.jsonl will include "
            "raw critique bodies and plan-text snippets. These contain LLM "
            "output that may reference user-question content. Do NOT export "
            "this file to external systems while the flag is on."
        )
    if settings.capture_revise_eval:
        log.warning(
            "ONGIINI_CAPTURE_REVISE_EVAL is set — every REVISE turn will "
            "write BOTH drafts (compose + revise) plus the user question "
            "to data/revise_eval/<msg_id>.json. This breaks the no-content- "
            "on-disk PII contract by design (the data is the question + "
            "reply by definition). Local-only, gitignored, never export. "
            "Disable this flag and rm -rf data/revise_eval/ when the "
            "evaluation window closes."
        )

    return SharedComponents(
        model=model,
        classifier=classifier,
        planner=planner,
        reviewer=reviewer,
        tools=tools,
        policies=policies,
        skills=skills,
        trace_destination=trace_destination,
    )


def build_whatsapp_runtime(
    shared: SharedComponents | None = None,
    *,
    trace_path: Path | None = None,
) -> Runtime:
    """Build the Runtime used by the WhatsApp webhook path.

    Uses the existing per-msisdn disk + mem0 memory provider and the
    full hook chain. ``shared`` is reused when provided (chat path also
    needs the model/classifier/etc); falls back to building components
    fresh when called standalone.
    """
    if shared is None:
        shared = build_shared_components(trace_path=trace_path)

    memory_provider = OngiiniMemoryProvider(
        prompt_builder=build_system_prompt,
        short_term=memory,
        long_term=mem,
        source_index_loader=source_index.load,
        source_index_formatter=source_index.format_for_injection,
        source_index_deleter=source_index.delete,
        skills=shared.skills,
    )

    # Hooks observe step events for billing, tracing, and persistence.
    # Order matters: BillingHook + TracingHook run before
    # MemoryRecordingHook so the trace line is written even if mem0 is
    # down. All are soft-fail at the registry level.
    hooks_list = [
        BillingHook(recorder=usage),
        TracingHook(
            trace_path=shared.trace_destination,
            include_critique_detail=settings.trace_critique_detail,
        ),
        HealthAlertHook(operator_msisdn=settings.operator_msisdn),
        OngiiniMemoryRecordingHook(sanitiser=pii.sanitize),
        SourceIndexHook(),
        # Recovers bot-hallucinated translation tasks. When the model
        # serves an English sentence without calling contribute_next,
        # this hook detects the pattern in the reply and retroactively
        # sets pending_save so the user's next reply (their translation)
        # lands normally.
        ContributeHallucinationGuardHook(),
    ]
    if settings.capture_revise_eval:
        hooks_list.append(ReviseEvalCaptureHook())
    hooks = HookRegistry(hooks_list)

    rt = Runtime(
        model=shared.model,
        transport=WhatsAppTransport(),
        memory=memory_provider,
        classifier=shared.classifier,
        tools=shared.tools,
        policies=shared.policies,
        hooks=hooks,
        planner=shared.planner,
        reviewer=shared.reviewer,
        skills=shared.skills,
    )
    log.info(
        "Ongiini WhatsApp runtime ready — tools=%d, policies=%d, hooks=%d, "
        "skills=%d",
        len(shared.tools.names()), len(shared.policies.all()),
        len(hooks.hooks), len(shared.skills.all()),
    )
    return rt


def build_chat_runtime(
    shared: SharedComponents,
    *,
    transport,         # WebChatTransport — per-request instance
    memory_provider,   # SessionMemoryProvider bound to the session id
) -> Runtime:
    """Build a per-request Runtime for the chat.ongiini.ai endpoint.

    ``shared`` is the singleton built once at startup. ``transport`` and
    ``memory_provider`` are created per HTTP request so each request has
    its own reply-capture slot and session-specific memory write target.

    The hook chain is intentionally trimmed vs. WhatsApp:
    - BillingHook stays (per-session token totals feed the cap check)
    - TracingHook stays (web chat turns trace like WA turns — grep for
      transport_name=web_chat in trace.jsonl)
    - OngiiniMemoryRecordingHook STAYS — despite the "no disk for
      sessions" promise, this hook is what actually invokes
      ``runtime.memory.record_turn(...)`` at end-of-turn. For the chat
      runtime that dispatches to ``SessionMemoryProvider.record_turn``
      which writes to the in-process SessionStore (no disk / no mem0).
      Without it, every turn looks like the first one to the bot —
      production bug 2026-06-02 (conversation context lost mid-thread).
    - SourceIndexHook DROPPED (writes per-user JSON to disk and the
      web-chat memory provider isn't wired to read it back — sessions
      get cited URLs via the rolling history alone)
    - ContributeHallucinationGuardHook DROPPED (no contribute flow when
      memory is per-session — the recovery would have nowhere to land)
    - ReviseEvalCaptureHook DROPPED (PII capture flag doesn't apply to
      anonymous web sessions; we don't persist content)
    """
    hooks_list = [
        BillingHook(recorder=usage),
        TracingHook(
            trace_path=shared.trace_destination,
            include_critique_detail=settings.trace_critique_detail,
        ),
        # PII sanitiser runs on user text + reply before they reach
        # the memory provider's record_turn. Mirrors the WhatsApp
        # contract: model sees raw text, persistence sees sanitised.
        OngiiniMemoryRecordingHook(sanitiser=pii.sanitize),
    ]
    hooks = HookRegistry(hooks_list)

    return Runtime(
        model=shared.model,
        transport=transport,
        memory=memory_provider,
        classifier=shared.classifier,
        tools=shared.tools,
        policies=shared.policies,
        hooks=hooks,
        planner=shared.planner,
        reviewer=shared.reviewer,
        skills=shared.skills,
    )


# Backwards-compat alias for existing callers (tests + scripts that
# import build_runtime). The WhatsApp path is what build_runtime ever
# built; keeping the name avoids touching unrelated callsites.
def build_runtime(*, trace_path: Path | None = None) -> Runtime:
    """Alias for build_whatsapp_runtime — the historical entry point."""
    return build_whatsapp_runtime(trace_path=trace_path)


def build_agent() -> Agent:
    """Convenience wrapper for the WhatsApp path: build_runtime() + Agent(rt)."""
    return Agent(build_runtime())
