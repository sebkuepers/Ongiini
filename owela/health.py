"""Capability health — a per-tool circuit breaker.

A tool that fails repeatedly (an API out of credit, DNS down, a provider
outage) should stop being called and stop shaping turns as if it worked.
The breaker tracks consecutive failures per tool name and opens after
``failure_threshold`` of them inside ``window_s``. While open, the
ToolRegistry short-circuits calls to that tool, and the executor can
degrade any Policy that ``requires_tools`` it. After ``cooldown_s`` the
breaker goes half-open: exactly one real call is admitted as a probe;
success closes the breaker, failure reopens it for another cooldown.

There are no synthetic health probes. The half-open real call is the
probe, so a recovered provider is picked up by the next turn that needs
it.

State lives on one event loop (the Runtime is a process singleton), so
no lock is needed: every method runs to completion without awaiting.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half_open"


@dataclass(frozen=True)
class BreakerConfig:
    failure_threshold: int = 3      # failures within window_s that trip the breaker
    window_s: float = 600.0
    cooldown_s: float = 900.0       # open → half_open after this long


class CircuitBreaker:
    """Per-tool failure tracker. See module docstring for the state machine."""

    def __init__(
        self,
        config: BreakerConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or BreakerConfig()
        self._clock = clock
        self._failures: dict[str, deque[float]] = {}
        self._opened_at: dict[str, float] = {}
        self._probe_in_flight: set[str] = set()

    def state(self, tool: str) -> str:
        opened = self._opened_at.get(tool)
        if opened is None:
            return STATE_CLOSED
        if self._clock() - opened >= self.config.cooldown_s:
            return STATE_HALF_OPEN
        return STATE_OPEN

    def allow(self, tool: str) -> bool:
        """True if a call to ``tool`` may run now. In half-open state only
        the first caller gets True until its outcome is recorded."""
        st = self.state(tool)
        if st == STATE_CLOSED:
            return True
        if st == STATE_HALF_OPEN and tool not in self._probe_in_flight:
            self._probe_in_flight.add(tool)
            return True
        return False

    def record(self, tool: str, ok: bool) -> bool:
        """Record one call outcome. Returns True iff this call tripped the
        breaker open (a transition, so callers can alert exactly once)."""
        was_open = tool in self._opened_at
        self._probe_in_flight.discard(tool)
        if ok:
            self._failures.pop(tool, None)
            self._opened_at.pop(tool, None)
            return False
        now = self._clock()
        if was_open:
            # Failed half-open probe: restart the cooldown, not a new trip.
            self._opened_at[tool] = now
            return False
        window = self._failures.setdefault(tool, deque())
        window.append(now)
        while window and now - window[0] > self.config.window_s:
            window.popleft()
        if len(window) >= self.config.failure_threshold:
            self._opened_at[tool] = now
            window.clear()
            return True
        return False

    def unavailable(self, names: tuple[str, ...]) -> tuple[str, ...]:
        """The subset of ``names`` whose breaker is fully open. Half-open
        tools count as available — the turn may be the recovery probe."""
        return tuple(n for n in names if self.state(n) == STATE_OPEN)
