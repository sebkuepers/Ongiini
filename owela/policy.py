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


def forced_tool_name(choice: ToolChoice) -> str | None:
    """The tool name a named ``tool_choice`` forces, or None for
    "auto" / "none" / "required"."""
    if isinstance(choice, dict):
        return (choice.get("function") or {}).get("name") or None
    return None


THINKING_OFF = "off"
THINKING_LOW = "low"
THINKING_ON = "on"
ALL_THINKING = (THINKING_OFF, THINKING_LOW, THINKING_ON)


@dataclass(frozen=True)
class Policy:
    """One named row of the policy table.

    Fields with ``enable_*`` prefix are feature flags. A flagged phase
    only runs when the corresponding runtime component (Planner,
    Reviewer) is also wired in — missing component + flag-on means the
    phase is skipped.

    Product strings (``fallback_reply``, ``reply_notice``) are carried
    as opaque fields: the application writes them into its own policy
    rows, the framework never inspects their content.
    """
    name: str
    first_tool: ToolChoice = AUTO
    max_steps: int = 6

    # Optional phases.
    enable_planner: bool = False
    enable_critique: bool = False

    # Output budget and sampling. ``max_reply_tokens`` bounds the visible
    # reply; thinking gets its own budget on top (``max_thinking_tokens``)
    # so reasoning can never eat the answer. None = adapter default.
    max_reply_tokens: int | None = None
    temperature: float | None = None
    thinking: str = THINKING_OFF        # "off" | "low" | "on"
    max_thinking_tokens: int | None = None

    # Tool exposure. None = expose every registered tool; a tuple = expose
    # only those names. Useful for narrowing the surface on cheap turns
    # (e.g. casual chat exposing no tools at all).
    expose_tools: tuple[str, ...] | None = None

    # Named prompt sections the application's MemoryProvider should
    # include for this turn. Opaque to Owela — it only carries the names.
    prompt_sections: tuple[str, ...] = ()

    # Capability requirements. If any tool in ``requires_tools`` is
    # unavailable before the turn (circuit breaker open) or errors during
    # it, the executor swaps to the Policy named ``on_unavailable`` and
    # records a DegradeStep. ``on_unavailable=None`` keeps the policy but
    # still records the DegradeStep so the turn is visibly degraded.
    requires_tools: tuple[str, ...] = ()
    on_unavailable: str | None = None
    # A line the transport may append to the reply (e.g. an honesty note
    # on a degraded policy). The executor never reads it.
    reply_notice: str = ""

    # Turn guarantees. ``deadline_s`` is soft: once exceeded, no new
    # optional work starts (the next model call composes without tools;
    # critique and revise are skipped). ``interstitial_after_s`` sends
    # the transport's "still working" message if the turn is still
    # running after that many seconds (capped at the transport's typing
    # window). ``fallback_reply`` is sent when the loop exhausts
    # ``max_steps`` or a phase raises; "" lets the transport supply its
    # own empty-reply text.
    deadline_s: float | None = None
    interstitial_after_s: float | None = None
    fallback_reply: str = ""

    # Deterministic first-call synthesis. When ``synth_tool`` is set the
    # executor can dispatch calls to it WITHOUT a model call:
    #   - one call per ``PlanStep.queries`` variant, if a plan has any;
    #   - else one call built from the message text, if
    #     ``synth_first_call_from_message`` is True;
    #   - and, as a recovery, when ``first_tool`` forced ``synth_tool``
    #     but the model's first response did not call it.
    # The query string becomes the kwarg named ``synth_arg`` ("" = pass
    # no text argument, for tools that take none); ``synth_default_args``
    # are merged into every call BEFORE ``QueryVariant.extra``. Tool
    # names are opaque to Owela (anti-trap #8).
    synth_tool: str | None = None
    synth_arg: str = "query"
    synth_default_args: dict[str, Any] = field(default_factory=dict)
    synth_first_call_from_message: bool = False
    # Deterministic reply: when set, the synthesised tool returns JSON and
    # this field of it IS the reply — no model call, so nothing the tool
    # produced (a link, a code) can be paraphrased or mangled. Falls back
    # to a normal compose if the field is missing or empty.
    reply_from_tool: str = ""

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
    # dedupes and trims to this count.
    auto_followup_max_items: int = 5
    # When True (default), the consolidated pool keeps only one item
    # per URL host. Set False when you want multiple items per host.
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

    def __post_init__(self) -> None:
        if self.thinking not in ALL_THINKING:
            raise ValueError(
                f"Policy {self.name!r}: thinking must be one of {ALL_THINKING}, "
                f"got {self.thinking!r}"
            )


class PolicyTable:
    """Maps (verdict, depth) tuples to Policies.

    A missing exact match falls back through three lookups in order:
       1. (verdict, depth) exact
       2. (verdict, SHALLOW) — default depth for that verdict
       3. ("NONE", SHALLOW) — global fallback
    If none of those resolves, ``PolicyNotFound`` is raised.

    Policies that are only reached by name (a degraded fallback named in
    ``Policy.on_unavailable``) are registered with ``add`` and resolved
    with ``by_name``.
    """
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], Policy] = {}
        self._named: dict[str, Policy] = {}

    def set(self, verdict: str, depth: str, policy: Policy) -> "PolicyTable":
        self._entries[(verdict, depth)] = policy
        self._named[policy.name] = policy
        return self

    def add(self, policy: Policy) -> "PolicyTable":
        """Register a policy reachable only by name."""
        self._named[policy.name] = policy
        return self

    def lookup(self, verdict: str, depth: str = DEPTH_SHALLOW) -> Policy:
        for key in ((verdict, depth), (verdict, DEPTH_SHALLOW), (VERDICT_NONE, DEPTH_SHALLOW)):
            if key in self._entries:
                return self._entries[key]
        raise PolicyNotFound(f"no policy for verdict={verdict!r} depth={depth!r}")

    def by_name(self, name: str) -> Policy:
        try:
            return self._named[name]
        except KeyError:
            raise PolicyNotFound(f"no policy named {name!r}") from None

    def all(self) -> dict[tuple[str, str], Policy]:
        return dict(self._entries)
