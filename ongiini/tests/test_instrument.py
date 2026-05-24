"""Tests for ongiini/instrument.py — resource snapshot logger."""

from __future__ import annotations

import asyncio
import logging

import pytest

from ongiini.instrument import snapshot, snapshot_loop


def test_snapshot_returns_expected_keys():
    s = snapshot()
    assert set(s.keys()) == {"threads", "fds", "asyncio_tasks", "rss_mb"}
    assert isinstance(s["threads"], int)
    assert s["threads"] >= 1


def test_snapshot_returns_int_values():
    s = snapshot()
    for k, v in s.items():
        assert isinstance(v, int), f"{k} should be int, got {type(v).__name__}"


@pytest.mark.asyncio
async def test_snapshot_loop_emits_log_lines(caplog: pytest.LogCaptureFixture):
    """The loop emits at least one snapshot line before being cancelled."""
    with caplog.at_level(logging.INFO, logger="ongiini.instrument"):
        task = asyncio.create_task(snapshot_loop(interval_s=0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    msgs = [r.message for r in caplog.records if r.name == "ongiini.instrument"]
    snapshot_msgs = [m for m in msgs if "resource_snapshot" in m]
    assert len(snapshot_msgs) >= 1, f"expected at least one snapshot log, got {msgs}"
    assert "threads=" in snapshot_msgs[0]
    assert "fds=" in snapshot_msgs[0]
    assert "asyncio_tasks=" in snapshot_msgs[0]
    assert "rss_mb=" in snapshot_msgs[0]


@pytest.mark.asyncio
async def test_snapshot_loop_emits_shutdown_snapshot(caplog: pytest.LogCaptureFixture):
    """On cancellation, the loop emits one final snapshot tagged
    'resource_snapshot.shutdown' so the post-mortem log captures
    end-state."""
    with caplog.at_level(logging.INFO, logger="ongiini.instrument"):
        task = asyncio.create_task(snapshot_loop(interval_s=10.0))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    msgs = [r.message for r in caplog.records if r.name == "ongiini.instrument"]
    assert any("resource_snapshot.shutdown" in m for m in msgs), msgs
