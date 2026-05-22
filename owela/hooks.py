"""Hook protocol + HookRegistry — cross-cutting concerns.

Hooks subscribe to step events. They observe — they do not transform.
Reply transformations (dead-URL strip, format normalisation, char cap)
live inside the Transport adapter, not in hooks, because they're
medium-specific.

Examples of well-shaped hooks:
  - billing — records per-step token cost into usage.log
  - tracing — writes one JSON line per turn for offline analysis
  - eval — records selected turns for later replay
  - pii-scrub — checks step.attrs for accidental PII before any export
  - memory-recording — see ``owela.hooks_builtin.MemoryRecordingHook``

Anti-trap principle #4: cross-cutting concerns are hooks subscribed to
step events. They do not appear inline in the executor.

**Concurrency contract:** a single ``HookRegistry`` instance is shared
across all concurrent requests handled by the same ``Runtime``. Hook
impls MUST therefore be safe under concurrent invocation — multiple
``on_step`` / ``on_turn_complete`` calls for different inbound messages
may overlap on the same event loop. In practice this means: do not
keep mutable state on the hook instance for cross-call coordination
(write to per-call locals or external storage); if you must keep state
(e.g. an in-memory counter), protect it with an ``asyncio.Lock``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .policy import Policy
from .step import Step
from .transport import InboundMessage

if TYPE_CHECKING:
    from .runtime import Runtime

log = logging.getLogger("owela.hooks")


@dataclass
class TurnContext:
    """Read-only context passed to hooks alongside step events.

    Hooks should treat the runtime as a service locator (read components
    out of it) and never mutate it. Mutation during a turn is undefined."""
    msg: InboundMessage
    policy: Policy
    runtime: "Runtime"


@runtime_checkable
class Hook(Protocol):
    """Observe step events. Both methods are optional in impls — define
    only the one(s) you need; ``HookRegistry`` swallows AttributeError.

    See module docstring for the concurrency contract: impls must be
    safe under concurrent invocation."""

    async def on_step(self, step: Step, ctx: TurnContext) -> None: ...

    async def on_turn_complete(self, steps: list[Step], ctx: TurnContext) -> None: ...


class HookRegistry:
    """Holds the application's registered hooks and fires events to them.

    Hook failures are logged but NEVER re-raised — a broken billing
    hook must not break a user reply. The executor's contract: hook
    invocation is best-effort.
    """
    def __init__(self, hooks: list[Hook] | None = None) -> None:
        self.hooks: list[Hook] = list(hooks or [])

    def add(self, hook: Hook) -> "HookRegistry":
        self.hooks.append(hook)
        return self

    async def on_step(self, step: Step, ctx: TurnContext) -> None:
        for h in self.hooks:
            fn = getattr(h, "on_step", None)
            if fn is None:
                continue
            try:
                await fn(step, ctx)
            except Exception as exc:                 # noqa: BLE001 — hooks are soft-fail
                log.warning("hook %s.on_step raised: %s", type(h).__name__, exc)

    async def on_turn_complete(self, steps: list[Step], ctx: TurnContext) -> None:
        for h in self.hooks:
            fn = getattr(h, "on_turn_complete", None)
            if fn is None:
                continue
            try:
                await fn(steps, ctx)
            except Exception as exc:                 # noqa: BLE001 — hooks are soft-fail
                log.warning("hook %s.on_turn_complete raised: %s", type(h).__name__, exc)
