"""CircuitBreaker + ToolRegistry integration."""

from __future__ import annotations

import pytest

from owela.errors import ToolError
from owela.health import (
    STATE_CLOSED, STATE_HALF_OPEN, STATE_OPEN, BreakerConfig, CircuitBreaker,
)
from owela.tools import ToolContext, ToolRegistry, tool


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _breaker(clock: _Clock) -> CircuitBreaker:
    return CircuitBreaker(BreakerConfig(failure_threshold=3, window_s=60, cooldown_s=300), clock=clock)


def test_breaker_trips_after_threshold():
    clock = _Clock()
    b = _breaker(clock)
    assert b.record("search", ok=False) is False
    assert b.record("search", ok=False) is False
    assert b.state("search") == STATE_CLOSED
    assert b.record("search", ok=False) is True        # the tripping call
    assert b.state("search") == STATE_OPEN
    assert b.allow("search") is False
    assert b.unavailable(("search", "docs")) == ("search",)


def test_breaker_failures_outside_window_do_not_accumulate():
    clock = _Clock()
    b = _breaker(clock)
    b.record("search", ok=False)
    b.record("search", ok=False)
    clock.t += 120                                      # beyond window_s=60
    assert b.record("search", ok=False) is False
    assert b.state("search") == STATE_CLOSED


def test_success_resets_failure_count():
    clock = _Clock()
    b = _breaker(clock)
    b.record("search", ok=False)
    b.record("search", ok=False)
    b.record("search", ok=True)
    assert b.record("search", ok=False) is False
    assert b.state("search") == STATE_CLOSED


def test_breaker_half_open_admits_one_probe_then_closes_or_reopens():
    clock = _Clock()
    b = _breaker(clock)
    for _ in range(3):
        b.record("search", ok=False)
    clock.t += 301
    assert b.state("search") == STATE_HALF_OPEN
    assert b.unavailable(("search",)) == ()             # half-open counts as available
    assert b.allow("search") is True                    # the probe
    assert b.allow("search") is False                   # only one probe in flight
    # Failed probe: reopen for another cooldown, not reported as a new trip.
    assert b.record("search", ok=False) is False
    assert b.state("search") == STATE_OPEN
    clock.t += 301
    assert b.allow("search") is True
    b.record("search", ok=True)
    assert b.state("search") == STATE_CLOSED
    assert b.allow("search") is True


@pytest.mark.asyncio
async def test_open_breaker_short_circuits_execute():
    calls = 0

    @tool(name="flaky_health_test")
    async def flaky() -> str:
        """Always fails."""
        nonlocal calls
        calls += 1
        raise ToolError("provider returned 402")

    clock = _Clock()
    reg = ToolRegistry([flaky], breaker=_breaker(clock))
    ctx = ToolContext(user_id="u", runtime=None, msg=None)   # type: ignore[arg-type]
    call = {"id": "1", "type": "function",
            "function": {"name": "flaky_health_test", "arguments": "{}"}}

    steps = [await reg.execute(call, ctx) for _ in range(3)]
    assert calls == 3
    assert [s.attrs["breaker_tripped"] for s in steps] == [False, False, True]
    assert steps[0].breaker_state == STATE_CLOSED

    blocked = await reg.execute(call, ctx)
    assert calls == 3                                   # not called again
    assert blocked.error == "circuit_open"
    assert blocked.breaker_state == STATE_OPEN
    assert "temporarily unavailable" in blocked.attrs["result"]
    assert reg.unavailable(("flaky_health_test",)) == ("flaky_health_test",)


def test_unavailable_reports_unregistered_tools():
    reg = ToolRegistry([])
    assert reg.unavailable(("missing",)) == ("missing",)
    assert reg.unavailable(()) == ()
