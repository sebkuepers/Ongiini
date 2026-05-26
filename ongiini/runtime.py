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
from pathlib import Path

from owela import (
    Agent, AUTO, DEPTH_DEEP, DEPTH_SHALLOW, HookRegistry, Policy, PolicyTable,
    Runtime, ToolRegistry, VERDICT_ADMIN, VERDICT_DOCS, VERDICT_NONE,
    VERDICT_SEARCH, force_tool,
)

from . import pii, usage
from .config import settings
from .hooks import (
    BillingHook, OngiiniMemoryRecordingHook, ReviseEvalCaptureHook,
    SourceIndexHook, TracingHook,
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


def build_runtime(*, trace_path: Path | None = None) -> Runtime:
    """Build the Ongiini Runtime — called once at startup.

    ``trace_path`` defaults to ``{data_dir}/trace.jsonl`` to match the
    pre-migration tracing destination. Tests can pass a tmp path.
    """
    log.info("building Ongiini runtime…")

    model = VLLMGemmaModel(
        base_url=settings.vllm_base_url,
        model_id=settings.vllm_model,
        temperature=0.6,
        max_tokens=1500,
    )

    transport = WhatsAppTransport()

    # Skills load from ongiini/skills/<name>/SKILL.md (Claude-compatible
    # layout). The MemoryProvider renders the manifest into the system
    # prompt; load_skill (in ALL_TOOLS) fetches on-demand skill content.
    skills = load_skills()

    memory_provider = OngiiniMemoryProvider(
        system_prompt=SYSTEM_PROMPT,
        short_term=memory,
        long_term=mem,
        source_index_loader=source_index.load,
        source_index_formatter=source_index.format_for_injection,
        source_index_deleter=source_index.delete,
        skills=skills,
    )

    classifier = GemmaClassifier(
        base_url=settings.vllm_base_url,
        model_id=settings.vllm_model,
    )

    # v1.1 components — planner runs only when policy.enable_planner
    # fires (SEARCH_DEEP); reviewer runs only when policy.enable_critique
    # fires (DOCS + both SEARCH depths). Both are soft-fail by contract
    # so a flaky reviewer never blocks a reply.
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

    # Hooks observe step events for billing, tracing, and persistence.
    # Order matters: BillingHook + TracingHook run before
    # MemoryRecordingHook so the trace line is written even if mem0 is
    # down. All three are soft-fail at the registry level.
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
    hooks_list = [
        BillingHook(recorder=usage),
        TracingHook(
            trace_path=trace_destination,
            include_critique_detail=settings.trace_critique_detail,
        ),
        OngiiniMemoryRecordingHook(sanitiser=pii.sanitize),
        SourceIndexHook(),
    ]
    if settings.capture_revise_eval:
        hooks_list.append(ReviseEvalCaptureHook())
    hooks = HookRegistry(hooks_list)

    rt = Runtime(
        model=model,
        transport=transport,
        memory=memory_provider,
        classifier=classifier,
        tools=tools,
        policies=policies,
        hooks=hooks,
        planner=planner,
        reviewer=reviewer,
        skills=skills,
    )
    log.info(
        "Ongiini runtime ready — tools=%d, policies=%d, hooks=%d, "
        "skills=%d, planner=on, reviewer=on",
        len(tools.names()), len(policies.all()), len(hooks.hooks),
        len(skills.all()),
    )
    return rt


def build_agent() -> Agent:
    """Convenience wrapper: build_runtime() + Agent(rt)."""
    return Agent(build_runtime())
