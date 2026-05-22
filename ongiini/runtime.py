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
from .hooks import BillingHook, OngiiniMemoryRecordingHook, TracingHook
from .memory import (
    OngiiniMemoryProvider,
    long_term as mem,
    short_term as memory,
)
from .models import VLLMGemmaModel
from .planner import OngiiniPlanner
from .reviewer import OngiiniReviewer
from .routers import GemmaClassifier
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
        ),
    )

    # SEARCH_SHALLOW — single tool call is enough. Force web_search on
    # turn 1; subsequent turns fall back to AUTO so the model can
    # optionally call fetch_url for one deeper read.
    # Critique ON: this is the path the hackathon-confabulation
    # incident travelled — exactly where critique adds value.
    table.set(
        VERDICT_SEARCH, DEPTH_SHALLOW,
        Policy(
            name="search_shallow",
            first_tool=force_tool("web_search"),
            max_steps=6,
            enable_critique=_critique_on(),
        ),
    )

    # SEARCH_DEEP — multi-source research. Force web_search on turn 1
    # for the same reason SEARCH_SHALLOW does: with first_tool=AUTO,
    # Gemma 4 26B was observed (May 2026) deciding to skip the search
    # entirely and reply from training data — even when the planner
    # explicitly identified things to look up. Forcing the tool on
    # turn 1 removes that option. Subsequent turns fall back to AUTO
    # so the model can chain fetch_urls or another web_search freely.
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

    memory_provider = OngiiniMemoryProvider(
        system_prompt=SYSTEM_PROMPT,
        short_term=memory,
        long_term=mem,
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
    hooks = HookRegistry([
        BillingHook(recorder=usage),
        TracingHook(trace_path=trace_destination),
        OngiiniMemoryRecordingHook(sanitiser=pii.sanitize),
    ])

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
    )
    log.info(
        "Ongiini runtime ready — tools=%d, policies=%d, hooks=%d, "
        "planner=on, reviewer=on",
        len(tools.names()), len(policies.all()), len(hooks.hooks),
    )
    return rt


def build_agent() -> Agent:
    """Convenience wrapper: build_runtime() + Agent(rt)."""
    return Agent(build_runtime())
