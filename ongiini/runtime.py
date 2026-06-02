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

from owela import (
    Agent, AUTO, DEPTH_DEEP, DEPTH_SHALLOW, HookRegistry, Policy, PolicyTable,
    Runtime, ToolRegistry, VERDICT_ADMIN, VERDICT_DOCS, VERDICT_NONE,
    VERDICT_SEARCH, force_tool,
)

from . import pii, usage
from .config import settings
from .hooks import (
    BillingHook, ContributeHallucinationGuardHook, OngiiniMemoryRecordingHook,
    ReviseEvalCaptureHook, SourceIndexHook, TracingHook,
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
from .routers.gemma_classifier import (
    VERDICT_CONTRIB_DECLINE, VERDICT_CONTRIB_DIALECT, VERDICT_CONTRIB_INVITE,
    VERDICT_CONTRIB_NEXT, VERDICT_CONTRIB_SAVE, VERDICT_CONTRIB_SKIP,
    VERDICT_CONTRIB_STATS, VERDICT_OPT_OUT_BROADCAST,
)
from .skills_loader import load_skills
from .system_prompt import SYSTEM_PROMPT
from .tools import ALL_TOOLS
from .transports import WhatsAppTransport

log = logging.getLogger("ongiini.runtime")


def build_policy_table() -> PolicyTable:
    """The orchestration policy table — verdict + depth → Policy.

    v1.1 flips the planner / critique / interstitial flags ON for the
    policies that benefit. Gating rationale:

      - **Planner** only on SEARCH_DEEP. Casual / ADMIN / DOCS / SHALLOW
        turns don't need decomposition; running a planner there just
        adds latency for no quality gain.
      - **Critique** on every turn that fired a tool: DOCS + both SEARCH
        depths. This is the biggest single quality unlock — catches
        confabulation before it ships. ADMIN / NONE turns return short
        tool-bounded or trivial replies; critique is wasted there.
      - **Interstitial** only on SEARCH_DEEP. The planner + multiple
        tool calls + critique + maybe revise can push past the 25s
        WhatsApp typing window; the interstitial message tells the
        user "still working" so the wait feels intentional.
    """
    table = PolicyTable()

    # Default fallback — looked up when the router returns NONE or an
    # unknown verdict. tool_choice="auto" so Gemma can still call any
    # tool it wants on a casual turn (e.g. user pivots mid-chat). No
    # v1 phases — casual chat must stay snappy.
    table.set(
        VERDICT_NONE, DEPTH_SHALLOW,
        Policy(name="none", first_tool=AUTO, max_steps=6),
    )

    # ADMIN — actions on the user's own data. Tool_choice=AUTO so the
    # model can pick between delete_my_data / whats_in_my_memory /
    # my_token_usage based on the phrasing. The router lifts the
    # decision OUT of DOCS (which is the historical confusion case).
    # No critique — admin tool replies are templated, nothing to fact-check.
    table.set(
        VERDICT_ADMIN, DEPTH_SHALLOW,
        Policy(name="admin", first_tool=AUTO, max_steps=4),
    )

    # DOCS — meta questions about Ongiini itself. Force the docs tool
    # on turn 1 so Gemma always grounds in the canonical product.md.
    # Critique ON: catches paraphrase drift from the source doc.
    table.set(
        VERDICT_DOCS, DEPTH_SHALLOW,
        Policy(
            name="docs",
            first_tool=force_tool("lookup_ongiini_docs"),
            max_steps=4,
            enable_critique=_critique_on(),
            # DOCS turns load the full product.md verbatim — the model
            # just needs to paraphrase the relevant section, not
            # synthesise across multiple sources. Thinking-after-long-
            # results is overkill here and risks runaway reasoning that
            # leaks via the truncated-thinking path (production bug
            # 2026-05-23T18:33: model spent 1500 tokens reasoning about
            # a one-line capability question and shipped raw
            # chain-of-thought to a WhatsApp user).
            enable_thinking_after_long_results=False,
        ),
    )

    # SEARCH_SHALLOW — single tool call is enough. Force web_search on
    # turn 1; subsequent turns fall back to AUTO so the model can
    # optionally call fetch_url for one deeper read.
    # Critique ON: this is the path the hackathon-confabulation
    # incident travelled — exactly where critique adds value.
    #
    # v1.3 truncation caps: Tavily's upgraded /search (advanced +
    # include_raw_content + chunks_per_source=3 + max_results=10) can
    # return 100K+ chars per call. Without per-tool caps, the model
    # would spend ~30s digesting a single search result. Cap web_search
    # at 6K chars (slightly larger than DEEP's 4K because SHALLOW only
    # gets one call), fetch_url/fetch_urls at 12K.
    table.set(
        VERDICT_SEARCH, DEPTH_SHALLOW,
        Policy(
            name="search_shallow",
            first_tool=force_tool("web_search"),
            max_steps=6,
            enable_critique=_critique_on(),
            tool_result_message_caps={
                "web_search": 6000,
                "fetch_url": 12000,
                "fetch_urls": 12000,
            },
        ),
    )

    # SEARCH_DEEP — multi-source research. v1.3 turns this into a
    # fully-deterministic search-quality pipeline:
    #
    #   - Planner emits structured QueryVariants (JSON output)
    #   - Executor synthesises N parallel web_search calls from those
    #     queries (``planner_query_tool="web_search"``)
    #   - After every web_search ToolStep, executor consolidates URLs
    #     and synthesises a fetch_urls call automatically
    #     (``auto_followup_after`` / ``auto_followup_tool``)
    #   - Per-tool message truncation bounds the model's context window
    #
    # ``first_tool=force_tool("web_search")`` is still set as a safety
    # net: if the planner soft-fails (returns empty queries), the model
    # gets forced into web_search on turn 1 instead of trying to answer
    # from training data. With successful planner JSON, the executor
    # bypasses ``first_tool`` because the synthesised fan-out already
    # ran the searches.
    #
    # ALL THREE v1 phases on: plan decomposes the question, critique
    # checks the answer, interstitial tells the user we're working.
    table.set(
        VERDICT_SEARCH, DEPTH_DEEP,
        Policy(
            name="search_deep",
            first_tool=force_tool("web_search"),
            max_steps=8,
            enable_planner=_planner_on(),
            enable_critique=_critique_on(),
            enable_interstitial=_interstitial_on(),
            # v1.3 deterministic search pipeline:
            planner_query_tool="web_search",
            planner_query_arg="query",
            # v1.3.1: skip include_raw_content for SEARCH_DEEP — the
            # auto-followup fetch_urls supplies depth. Embedding raw
            # content into the search response too would double-fetch
            # the same pages (~10× tokens for no quality gain).
            planner_query_default_args={"include_raw_content": False},
            auto_followup_after="web_search",
            auto_followup_tool="fetch_urls",
            auto_followup_attr="urls",
            auto_followup_arg="urls",
            auto_followup_max_items=5,
            auto_followup_one_per_host=True,
            # Per-tool context caps. v1.3.1 lowered web_search 4000→
            # 2500 — with include_raw_content off and chunks_per_source
            # at 1, search snippets are much smaller now; the cap
            # follows. fetch_urls cap unchanged (it's still the depth
            # tool).
            tool_result_message_caps={
                "web_search": 2500,
                "fetch_urls": 12000,
                "fetch_url": 12000,
            },
        ),
    )

    # ---------- Contribution-loop policies (v2 classifier-driven) ----------
    #
    # Each CONTRIBUTE_* verdict forces a specific contribute_* tool
    # on turn 1. The tool reads pending state + ctx.msg.text and
    # writes to the contributions sqlite directly — model can't skip
    # the call (force_tool guarantees it) and can't fake args (tools
    # take no model-fillable parameters). After turn 1 the model gets
    # the tool result and composes the user-facing reply on turn 2.
    #
    # Critique is OFF for the contribute path — the tool's effect is
    # binary (saved/not-saved, served/not-served), the user-facing
    # reply is a templated thank-you / sentence presentation, and the
    # critique loop would add ~3-7s latency to a turn that's already
    # one model call + one tool execution. If we later see the model
    # composing weirdly off-result replies we can flip critique on.
    _CONTRIB_MAX_STEPS = 3   # turn 1: forced tool call. turn 2: compose reply. headroom: 1.

    table.set(
        VERDICT_CONTRIB_INVITE, DEPTH_SHALLOW,
        Policy(
            name="contribute_invite",
            first_tool=force_tool("contribute_invite_check"),
            max_steps=_CONTRIB_MAX_STEPS,
        ),
    )
    table.set(
        VERDICT_CONTRIB_DIALECT, DEPTH_SHALLOW,
        Policy(
            name="contribute_dialect",
            first_tool=force_tool("contribute_set_dialect"),
            max_steps=_CONTRIB_MAX_STEPS,
        ),
    )
    table.set(
        VERDICT_CONTRIB_NEXT, DEPTH_SHALLOW,
        Policy(
            name="contribute_next",
            first_tool=force_tool("contribute_next"),
            max_steps=_CONTRIB_MAX_STEPS,
        ),
    )
    table.set(
        VERDICT_CONTRIB_SAVE, DEPTH_SHALLOW,
        Policy(
            name="contribute_save",
            first_tool=force_tool("contribute_save"),
            max_steps=_CONTRIB_MAX_STEPS,
        ),
    )
    table.set(
        VERDICT_CONTRIB_SKIP, DEPTH_SHALLOW,
        Policy(
            name="contribute_skip",
            first_tool=force_tool("contribute_skip"),
            max_steps=_CONTRIB_MAX_STEPS,
        ),
    )
    table.set(
        VERDICT_CONTRIB_DECLINE, DEPTH_SHALLOW,
        Policy(
            name="contribute_decline",
            first_tool=force_tool("contribute_decline"),
            max_steps=_CONTRIB_MAX_STEPS,
        ),
    )
    table.set(
        VERDICT_CONTRIB_STATS, DEPTH_SHALLOW,
        Policy(
            name="contribute_stats",
            first_tool=force_tool("contribute_stats"),
            max_steps=_CONTRIB_MAX_STEPS,
        ),
    )

    # OPT_OUT_BROADCAST → force opt_out_broadcast tool, same shape as
    # the contribute policies: turn 1 records the opt-out, turn 2
    # composes a warm confirmation reply.
    table.set(
        VERDICT_OPT_OUT_BROADCAST, DEPTH_SHALLOW,
        Policy(
            name="opt_out_broadcast",
            first_tool=force_tool("opt_out_broadcast"),
            max_steps=_CONTRIB_MAX_STEPS,
        ),
    )

    return table


# v1 quality-phase kill switches. Each is gated by an env var in
# ``ongiini.config`` so we can disable a phase WITHOUT redeploying if
# we discover (e.g.) the REVISE rate is too high or the planner is
# misfiring on questions the classifier mis-tagged as DEEP. Default
# is ON for every flag.
def _planner_on() -> bool:
    return not settings.disable_planner


def _critique_on() -> bool:
    return not settings.disable_critique


def _interstitial_on() -> bool:
    return not settings.disable_interstitial


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

    model = VLLMGemmaModel(
        base_url=settings.vllm_base_url,
        model_id=settings.vllm_model,
        temperature=0.6,
        max_tokens=1500,
    )

    skills = load_skills()

    # State-gated wrapper around GemmaClassifier. The inner classifier
    # emits its best opinion; the wrapper blocks CONTRIBUTE_* verdicts
    # that contradict fresh state (no pending → SAVE redirected to NONE
    # etc.). See ongiini/routers/state_gated_classifier.py.
    classifier = StateGatedClassifier(
        GemmaClassifier(
            base_url=settings.vllm_base_url,
            model_id=settings.vllm_model,
        )
    )

    # v1.1 components — planner runs only when policy.enable_planner
    # fires (SEARCH_DEEP); reviewer runs only when policy.enable_critique
    # fires (DOCS + both SEARCH depths). Both are soft-fail by contract.
    planner = OngiiniPlanner(
        base_url=settings.vllm_base_url,
        model_id=settings.vllm_model,
    )
    reviewer = OngiiniReviewer(
        base_url=settings.vllm_base_url,
        model_id=settings.vllm_model,
    )

    tools = ToolRegistry(list(ALL_TOOLS))
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
        system_prompt=SYSTEM_PROMPT,
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

    The hook chain is intentionally trimmed:
    - BillingHook stays (we need per-session token totals for the cap)
    - TracingHook stays (web chat turns trace just like WA turns —
      grep for transport_name=web_chat to find session activity)
    - OngiiniMemoryRecordingHook DROPPED (no disk / no mem0 for
      sessions; the SessionMemoryProvider's record_turn appends to the
      in-process store directly)
    - SourceIndexHook DROPPED (writes per-user JSON to disk and the
      web-chat OngiiniMemoryProvider isn't wired to read it back —
      sessions get cited URLs via the rolling history alone)
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
