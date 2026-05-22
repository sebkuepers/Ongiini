"""Runtime — the composition root.

A Runtime bundles the concrete adapter instances that drive one turn:
model, transport, memory, classifier, tools, hooks, policies, and
optional v1 planner/reviewer. It's built once at application startup
and passed to ``Agent.handle()`` for every inbound message.

Anti-trap principle #6: one Runtime object holds everything. Tests
substitute components via dependency injection. No module-level
globals leaking state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .hooks import HookRegistry
from .memory import MemoryProvider
from .model import Model
from .policy import Policy, PolicyTable
from .router import Classifier
from .step import CritiqueStep, PlanStep, ReviseStep, Step
from .tools import ToolRegistry
from .transport import InboundMessage, Transport


@runtime_checkable
class Planner(Protocol):
    """v1 — produce a structured plan before the act loop. Not used in v0.

    The plan text lands in PlanStep.plan_text and is injected as a system
    message into the act loop's message list (handled by the
    MemoryProvider impl that knows where to put it)."""
    async def plan(self, msg: InboundMessage, policy: Policy, prior_steps: list[Step]) -> PlanStep:
        ...


@runtime_checkable
class Reviewer(Protocol):
    """v1 — critique the draft and optionally produce a revised version."""
    async def critique(
        self,
        msg: InboundMessage,
        draft: str,
        prior_steps: list[Step],
        policy: Policy,
    ) -> CritiqueStep:
        ...

    async def revise(
        self,
        msg: InboundMessage,
        draft: str,
        critique: CritiqueStep,
        prior_steps: list[Step],
        policy: Policy,
    ) -> ReviseStep:
        ...


@dataclass(frozen=True)
class Runtime:
    """One per application. Build at startup; do NOT mutate during requests.

    Frozen at the outer level so accidental re-assignment of ``runtime.model``
    or similar raises. Inner mutability still exists (``HookRegistry.hooks``
    is a list, the ``PolicyTable`` entries dict can be added to) — that's
    intentional, since registering hooks/policies happens during runtime
    construction, not at request time.
    """
    model: Model
    transport: Transport
    memory: MemoryProvider
    classifier: Classifier
    tools: ToolRegistry
    policies: PolicyTable
    hooks: HookRegistry

    # v1 components — kept as Optional so v0 runtimes can omit them entirely.
    # Executor checks both the policy flag AND the component presence before
    # invoking; missing component + flag-on = phase silently skipped.
    planner: Planner | None = None
    reviewer: Reviewer | None = None
