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
    verdict: str = "NONE"        # NONE / ADMIN / DOCS / SEARCH
    depth: str = "SHALLOW"        # SHALLOW / DEEP — only meaningful for SEARCH


@dataclass
class PlanStep(Step):
    """Reserved for v1: the planner's structured plan text."""
    kind: str = "plan"
    plan_text: str = ""


@dataclass
class ModelCallStep(Step):
    """One vLLM round-trip (chat.completions.create).

    ``tool_calls`` is the OpenAI-shape list returned by the model. The
    executor uses it to decide whether to loop or terminate. ``finish_reason``
    is the standard OpenAI value (``stop`` / ``tool_calls`` / ``length`` / ...).
    """
    kind: str = "model_call"
    turn: int = 0
    finish_reason: str = ""
    enable_thinking: bool = False
    reasoning_budget: int | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolStep(Step):
    """One tool execution. ``result_len`` is used by the executor to decide
    whether the NEXT turn should enable reasoning (long results need
    deliberation to digest + cite)."""
    kind: str = "tool"
    tool_name: str = ""
    tool_call_id: str = ""
    args_len: int = 0
    result_len: int = 0
    error: str | None = None


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


@dataclass
class ReplyStep(Step):
    """The terminal step. ``sent`` is True iff the transport accepted the
    payload. ``dead_urls_stripped`` is a transport-internal count for
    visibility into how often citation hygiene fires."""
    kind: str = "reply"
    reply_len: int = 0
    sent: bool = False
    dead_urls_stripped: int = 0
