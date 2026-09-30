"""Typed step model.

A turn is a sequence of typed steps. Every observable event the
executor performs lands in this list. Hooks subscribe to step events
for tracing, billing, eval recording, etc.

Adding a new behaviour to the executor (planner, critique, revise)
means adding a new step type here and a new branch in
``executor.execute_turn``. Anti-trap principle: steps are typed
dataclasses, never freeform dicts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    """Base step. Every step kind extends this with kind-specific fields.

    ``started_at`` / ``ended_at`` are ``time.monotonic()`` values, not
    wall-clock — they're for measuring latency, not for logging dates.
    ``attrs`` is a free-form bag for adapter-specific extras (e.g. the
    raw model response object stashed for tests).
    """
    kind: str = "step"
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cached_tokens: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)

    def latency_ms(self) -> int:
        if self.ended_at is None:
            return 0
        return int((self.ended_at - self.started_at) * 1000)


@dataclass
class RouterStep(Step):
    """Classifier output. Determines which Policy drives the rest of the turn."""
    kind: str = "router"
    verdict: str = "NONE"        # application-defined label, e.g. NONE / SEARCH
    depth: str = "SHALLOW"        # SHALLOW / DEEP
    # Set when the classifier could not decide and returned its fallback
    # verdict (timeout, exception, unparseable output). None means the
    # verdict is a real decision. Lets traces tell "classifier said NONE"
    # apart from "classifier failed".
    fallback_reason: str | None = None


@dataclass(frozen=True)
class QueryVariant:
    """One query emitted by a Planner for multi-query fan-out.

    The Planner is a state-setter — it can hand the executor a list of
    structured queries that get synthesised into parallel tool calls
    on turn 1, instead of relying on the model to pick a single
    query. The executor materialises each variant into a tool call:
    ``query`` becomes the primary kwarg (name configured by
    ``Policy.synth_arg``), and ``extra`` is spread as additional kwargs.

    ``extra`` is opaque to Owela — Owela just forwards it to the
    application's tool implementation. Engine-specific knobs (search
    topic, time-range biasing, language hints, source filters, etc.)
    live there. Per anti-trap principle #8, the framework declares
    no specific keys.

    ``frozen=True`` prevents attribute reassignment. ``extra`` is
    a dict (mutable internally) — by convention, planners construct
    fresh variants per turn and don't mutate them.
    """
    query: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanStep(Step):
    """Planner output.

    ``plan_text`` is the prose context the MemoryProvider may inject as
    a system message before the act loop (preserved for backwards-
    compat / context priming).

    ``queries`` is the structured fan-out signal. When non-empty AND
    ``policy.synth_tool`` is set, the executor synthesises one
    parallel tool call per variant on turn 1. Empty list = soft-fail
    or single-query case; executor falls back to letting the model
    pick the query.
    """
    kind: str = "plan"
    plan_text: str = ""
    queries: list[QueryVariant] = field(default_factory=list)


@dataclass
class ModelCallStep(Step):
    """One model round-trip OR a policy-synthesised call.

    ``tool_calls`` is the OpenAI-shape list. For real model calls,
    it's what the model returned. For synthesised calls (v1.3 — the
    executor fabricates a tool dispatch without an LLM round-trip),
    it's what the executor built. ``finish_reason`` is the standard
    OpenAI value (``stop`` / ``tool_calls`` / ``length`` / ...).

    ``turn`` is the 1-indexed model-driven turn this call belongs to.
    Synthesised calls share the turn number of the upcoming/preceding
    model turn — disambiguate via the ``synthesized`` flag, not the
    turn integer.

    ``synthesized`` distinguishes policy-driven synthesis from real
    model calls. Set True by the executor's synthesis helpers; left
    False for normal calls. Hooks filtering "real model turns" should
    check this flag rather than relying on ``turn`` values.

    ``forced_tool`` is the tool name the request tried to force (named
    ``tool_choice``), or None. ``forced_tool_honoured`` records whether
    the response actually called it — some inference engines accept a
    named tool_choice and silently ignore it.
    """
    kind: str = "model_call"
    turn: int = 0
    finish_reason: str = ""
    thinking: str = "off"                # "off" | "low" | "on"
    thinking_budget: int | None = None
    max_tokens: int | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    synthesized: bool = False
    forced_tool: str | None = None
    forced_tool_honoured: bool | None = None


# Conventional ToolStep.attrs key: a list of URLs the tool's result cites.
# The executor collects these into ReplyContext.allowed_urls, and the
# default auto-followup attr uses the same name.
TOOL_URLS_ATTR = "urls"


@dataclass
class ToolStep(Step):
    """One tool execution. ``breaker_state`` is the tool's circuit-breaker
    state when the call was attempted (None when no breaker is wired).
    Tools may attach ``attrs[TOOL_URLS_ATTR]`` (a list of URL strings)."""
    kind: str = "tool"
    tool_name: str = ""
    tool_call_id: str = ""
    args_len: int = 0
    result_len: int = 0
    error: str | None = None
    breaker_state: str | None = None


@dataclass
class DegradeStep(Step):
    """The executor swapped the turn's Policy because a capability it
    ``requires_tools`` is unavailable (breaker open) or failed mid-turn.
    ``to_policy`` equals ``from_policy`` when no fallback was declared."""
    kind: str = "degrade"
    from_policy: str = ""
    to_policy: str = ""
    missing_tools: tuple[str, ...] = ()
    trigger: str = ""                    # "breaker_open" | "tool_error"


@dataclass
class ErrorStep(Step):
    """A phase raised. The executor still sends the policy's fallback reply
    and fires ``on_turn_complete``, so every turn leaves a trace."""
    kind: str = "error"
    phase: str = ""                      # "router" | "turn"
    exc_type: str = ""
    message: str = ""


@dataclass
class CritiqueStep(Step):
    """Reserved for v1: critique LLM output verdict + reasons."""
    kind: str = "critique"
    verdict: str = "PASS"          # PASS / REVISE
    reasons: list[str] = field(default_factory=list)


@dataclass
class ReviseStep(Step):
    """Reserved for v1: revised draft generated in response to a critique."""
    kind: str = "revise"


REPLY_OK = "ok"
REPLY_MAX_STEPS = "max_steps"
REPLY_ERROR = "error"
REPLY_DEADLINE = "deadline"


@dataclass
class ReplyStep(Step):
    """The terminal step. ``sent`` is True iff the transport accepted the
    payload.

    ``reason`` says how the draft came about: ``ok`` (the model finished),
    ``max_steps`` (loop budget exhausted, fallback reply), ``error`` (a
    phase raised, fallback reply) or ``deadline`` (the turn ran out of
    time and composed without further tools). ``truncated`` means the
    final model call stopped on its token limit. Transport-side hygiene
    counts (URLs dropped, chars capped) arrive via ``SendResult.attrs``
    and are merged into ``attrs``."""
    kind: str = "reply"
    reply_len: int = 0
    sent: bool = False
    reason: str = REPLY_OK
    truncated: bool = False
    degraded: bool = False
    deadline_exceeded: bool = False
