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

from . import mem, memory, pii, usage
from .config import settings
from .hooks_app import BillingHook, OngiiniMemoryRecordingHook, TracingHook
from .memory_provider import OngiiniMemoryProvider
from .models import VLLMGemmaModel
from .routers import GemmaClassifier
from .system_prompt import SYSTEM_PROMPT
from .tools_pkg import ALL_TOOLS
from .transports import WhatsAppTransport

log = logging.getLogger("ongiini.runtime")


def build_policy_table() -> PolicyTable:
    """The orchestration policy table — verdict + depth → Policy.

    v0 keeps all the v1 flags (enable_planner, enable_critique,
    enable_interstitial) at False everywhere. Step 12 of the migration
    plan flips SEARCH_DEEP into the more agentic shape; nothing else
    needs to change to support that.
    """
    table = PolicyTable()

    # Default fallback — looked up when the router returns NONE or an
    # unknown verdict. tool_choice="auto" so Gemma can still call any
    # tool it wants on a casual turn (e.g. user pivots mid-chat).
    table.set(
        VERDICT_NONE, DEPTH_SHALLOW,
        Policy(name="none", first_tool=AUTO, max_steps=6),
    )

    # ADMIN — actions on the user's own data. Tool_choice=AUTO so the
    # model can pick between delete_my_data / whats_in_my_memory /
    # my_token_usage based on the phrasing. The router lifts the
    # decision OUT of DOCS (which is the historical confusion case).
    table.set(
        VERDICT_ADMIN, DEPTH_SHALLOW,
        Policy(name="admin", first_tool=AUTO, max_steps=4),
    )

    # DOCS — meta questions about Ongiini itself. Force the docs tool
    # on turn 1 so Gemma always grounds in the canonical product.md.
    table.set(
        VERDICT_DOCS, DEPTH_SHALLOW,
        Policy(name="docs", first_tool=force_tool("lookup_ongiini_docs"), max_steps=4),
    )

    # SEARCH_SHALLOW — single tool call is enough. Force web_search on
    # turn 1; subsequent turns fall back to AUTO so the model can
    # optionally call fetch_url for one deeper read.
    table.set(
        VERDICT_SEARCH, DEPTH_SHALLOW,
        Policy(name="search_shallow", first_tool=force_tool("web_search"), max_steps=6),
    )

    # SEARCH_DEEP — multi-source research. Let the model decide which
    # tool to call first (web_search vs fetch_urls); allow more steps
    # to give it room to iterate. v0 keeps planner/critique OFF; v1
    # turns them on here.
    table.set(
        VERDICT_SEARCH, DEPTH_DEEP,
        Policy(
            name="search_deep",
            first_tool=AUTO,
            max_steps=8,
            # v1 — flip these on when planner / reviewer / interstitial
            # are wired in (step v1 of the plan):
            #   enable_planner=True,
            #   enable_critique=True,
            #   enable_interstitial=True,
        ),
    )

    return table


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
    )
    log.info(
        "Ongiini runtime ready — tools=%d, policies=%d, hooks=%d",
        len(tools.names()), len(policies.all()), len(hooks.hooks),
    )
    return rt


def build_agent() -> Agent:
    """Convenience wrapper: build_runtime() + Agent(rt)."""
    return Agent(build_runtime())
