"""Policy table — the orchestration spine.

A Policy is a frozen value object that describes the shape of one turn:
which tool (if any) is forced on the first model call, whether to plan
beforehand, whether to critique afterwards, how many tool/model loops
are allowed, and which model knobs to use.

The router classifies the incoming message into a (verdict, depth) pair,
and the PolicyTable maps that pair to a Policy. The executor then runs
the turn according to the Policy — no conditional behaviour lives in
the executor itself. Anti-trap principle #1: behaviour is policy-driven.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import PolicyNotFound


# Verdict / depth string constants. Plain strings (not Enum) so they
# serialise transparently into traces and JSON logs.
VERDICT_NONE = "NONE"
VERDICT_ADMIN = "ADMIN"
VERDICT_DOCS = "DOCS"
VERDICT_SEARCH = "SEARCH"
ALL_VERDICTS = (VERDICT_NONE, VERDICT_ADMIN, VERDICT_DOCS, VERDICT_SEARCH)

DEPTH_SHALLOW = "SHALLOW"
DEPTH_DEEP = "DEEP"
ALL_DEPTHS = (DEPTH_SHALLOW, DEPTH_DEEP)

# ToolChoice: matches the OpenAI tool_choice argument — either the string
# "auto" / "none" / "required", or a dict forcing a specific function.
ToolChoice = str | dict[str, Any]
AUTO: ToolChoice = "auto"


def force_tool(name: str) -> ToolChoice:
    """Build an OpenAI tool_choice dict that forces ``name`` on the first call."""
    return {"type": "function", "function": {"name": name}}


@dataclass(frozen=True)
class Policy:
    """One named row of the policy table.

    Fields with ``enable_*`` prefix are feature flags. v0 ships them all
    declarable but the v1-only ones (planner, critique, interstitial)
    must stay False until the corresponding Planner/Reviewer is wired
    into the Runtime. The executor enforces this — it skips a flagged
    phase if the corresponding runtime component is None.
    """
    name: str
    first_tool: ToolChoice = AUTO
    max_steps: int = 6

    # v1 feature flags. v0 should keep all of these False.
    enable_planner: bool = False
    enable_critique: bool = False
    enable_interstitial: bool = False

    # Gemma 4 reasoning knobs. None means "let the model adapter pick".
    reasoning_budget: int | None = 500
    enable_thinking_after_long_results: bool = True
    long_result_threshold_chars: int = 1000

    # Tool exposure. None = expose every registered tool; a tuple = expose
    # only those names. Useful for narrowing the surface on cheap turns
    # (e.g. casual chat exposing no tools at all).
    expose_tools: tuple[str, ...] | None = None

    # Planner-driven multi-query fan-out. When ``planner_query_tool``
    # is set AND a prior PlanStep carries non-empty ``queries``, the
    # executor synthesises one parallel call to this tool per
    # QueryVariant on turn 1 — bypassing the model's first call.
    #
    # The variant's ``query`` string becomes the kwarg named by
    # ``planner_query_arg``; ``variant.extra`` (a dict) is spread as
    # additional kwargs. The string is opaque to Owela — the
    # application registers a tool by this name. Per anti-trap #8 no
    # tool name is hardcoded in the executor.
    planner_query_tool: str | None = None
    planner_query_arg: str = "query"
    # Default kwargs merged into every synthesised planner-fan-out tool
    # call BEFORE ``QueryVariant.extra``. Use for policy-level tool
    # configuration that should apply to all variants (e.g. a search
    # policy that wants every search to skip full-page content because
    # an auto-followup will fetch it). Variant-level ``extra`` overrides.
    planner_query_default_args: dict[str, Any] = field(default_factory=dict)

    # Deterministic follow-up tool synthesis. When ``auto_followup_after``
    # is set, the executor watches each ToolStep: if its ``tool_name``
    # matches the trigger AND ``attrs[auto_followup_attr]`` is non-empty,
    # the executor synthesises a single call to ``auto_followup_tool``
    # with that attribute value passed as the ``auto_followup_arg``
    # kwarg. Rule-based (not one-shot): fires after every matching
    # ToolStep, including subsequent iterations on the same turn loop.
    #
    # ``auto_followup_after`` AND ``auto_followup_tool`` must be set
    # together; either None disables follow-up. ``auto_followup_attr``
    # and ``auto_followup_arg`` default to "urls" — the conventional
    # name for the typical search→fetch escalation — but apps may
    # override.
    auto_followup_after: str | None = None
    auto_followup_tool: str | None = None
    auto_followup_attr: str = "urls"
    auto_followup_arg: str = "urls"
    # Cap on items passed to ``auto_followup_tool``. The executor
    # consolidates ``attrs[auto_followup_attr]`` from every matching
    # ToolStep this turn (multi-query fan-out can produce many), then
    # dedupes and trims to this count. Default 5 mirrors typical search-
    # fetch pipelines (e.g. Tavily's /extract batch cap).
    auto_followup_max_items: int = 5
    # When True (default), the consolidated pool keeps only one item
    # per URL host — typical case is search results where multiple
    # pages from the same domain compete. Set False when you want
    # multiple items per host (e.g. fetching multiple PDFs from one
    # government site).
    auto_followup_one_per_host: bool = True

    # Per-tool char caps for results placed into the model's message
    # list. The FULL pre-truncation text is preserved in
    # ``ToolStep.attrs["result"]`` for reviewer + trace use; this only
    # bounds the model-visible context. Default empty dict = no
    # truncation (apps explicitly opt in by tool name).
    #
    # Frozen dataclass field caveat: dicts are technically mutable
    # internals. Treat as immutable by convention; do not mutate after
    # Policy construction.
    tool_result_message_caps: dict[str, int] = field(default_factory=dict)


class PolicyTable:
    """Maps (verdict, depth) tuples to Policies.

    A missing exact match falls back through three lookups in order:
       1. (verdict, depth) exact
       2. (verdict, SHALLOW) — default depth for that verdict
       3. ("NONE", SHALLOW) — global fallback
    If none of those resolves, ``PolicyNotFound`` is raised.
    """
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], Policy] = {}

    def set(self, verdict: str, depth: str, policy: Policy) -> "PolicyTable":
        self._entries[(verdict, depth)] = policy
        return self

    def lookup(self, verdict: str, depth: str = DEPTH_SHALLOW) -> Policy:
        for key in ((verdict, depth), (verdict, DEPTH_SHALLOW), (VERDICT_NONE, DEPTH_SHALLOW)):
            if key in self._entries:
                return self._entries[key]
        raise PolicyNotFound(f"no policy for verdict={verdict!r} depth={depth!r}")

    def all(self) -> dict[tuple[str, str], Policy]:
        return dict(self._entries)
