"""Owela ``Hook`` that raises the alarm when a capability goes down.

The Tavily outage of Sep 2026 (every web search returned 402 from
09-04 to 09-30) went unnoticed because nothing looked at ``ToolStep.error``.
The executor now degrades turns when a required tool is down; this hook
makes sure a human hears about it:

  - a tool's circuit breaker trips (``ToolStep.attrs["breaker_tripped"]``)
    → ``log.error`` and, if an operator number is configured, a WhatsApp
    message to the operator. At most one message per tool per
    ``alert_interval_s`` so a flapping provider can't spam.
  - a turn degrades (``DegradeStep``) → ``log.warning`` (volume is
    visible in the trace; ``scripts/trace_query.py health`` reports it).

Observe-only and soft-fail: sending the alert never blocks or breaks the
user's turn (it runs as a background task).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from owela import DegradeStep, Step, ToolStep, TurnContext

log = logging.getLogger("ongiini.hooks.health_alert")

Sender = Callable[[str, str], Awaitable[object]]


async def _default_sender(to: str, text: str) -> object:
    # Imported lazily so the hook can be unit-tested without WhatsApp config.
    # Template first (delivered outside the 24h window), text as fallback.
    from ..ops_alert import send_ops_alert
    return await send_ops_alert(to, text)


class HealthAlertHook:
    def __init__(
        self,
        *,
        operator_msisdn: str = "",
        sender: Sender | None = None,
        alert_interval_s: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.operator_msisdn = operator_msisdn
        self._send = sender or _default_sender
        self.alert_interval_s = alert_interval_s
        self._clock = clock
        self._last_alert: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    async def on_step(self, step: Step, ctx: TurnContext) -> None:
        if isinstance(step, ToolStep) and step.attrs.get("breaker_tripped"):
            log.error(
                "capability down: tool %s tripped its circuit breaker (last error: %s)",
                step.tool_name, (step.error or "")[:200],
            )
            await self._maybe_alert(step.tool_name, step.error or "")
        elif isinstance(step, DegradeStep):
            log.warning(
                "turn degraded: %s → %s (missing %s, trigger %s)",
                step.from_policy, step.to_policy,
                ",".join(step.missing_tools), step.trigger,
            )

    async def _maybe_alert(self, tool: str, error: str) -> None:
        if not self.operator_msisdn:
            return
        async with self._lock:
            now = self._clock()
            last = self._last_alert.get(tool)
            if last is not None and now - last < self.alert_interval_s:
                return
            self._last_alert[tool] = now
        text = (
            f"Ongiini AI alert: tool '{tool}' failed repeatedly and was switched "
            f"off for 15 minutes. Turns that need it are answering in degraded "
            f"mode. Last error: {error[:160]}"
        )
        task = asyncio.create_task(self._deliver(text))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _deliver(self, text: str) -> None:
        try:
            await self._send(self.operator_msisdn, text)
        except Exception as exc:                          # noqa: BLE001 — alerting is soft-fail
            log.warning("health alert delivery failed: %s", exc)
