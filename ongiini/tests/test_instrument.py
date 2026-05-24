"""Tests for ongiini/instrument.py — resource snapshot logger + Prometheus
exporter."""

from __future__ import annotations

import asyncio
import logging

import pytest
from prometheus_client import REGISTRY

from ongiini.instrument import (
    _G_ASYNCIO_TASKS,
    _G_FDS,
    _G_PYTHON_THREADS,
    _G_RSS_MB,
    _G_THREADS,
    _G_THREADS_BY_GROUP,
    _classify_thread_name,
    _format_groups,
    _python_thread_groups,
    _update_gauges,
    snapshot,
    snapshot_loop,
)


def test_snapshot_returns_expected_keys():
    s = snapshot()
    assert set(s.keys()) == {
        "threads", "python_threads", "python_thread_groups",
        "fds", "asyncio_tasks", "rss_mb",
    }
    assert isinstance(s["threads"], int)
    assert s["threads"] >= 1
    assert isinstance(s["python_thread_groups"], dict)


def test_snapshot_returns_int_values():
    s = snapshot()
    for k, v in s.items():
        if k == "python_thread_groups":
            assert isinstance(v, dict)
            continue
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


def test_update_gauges_pushes_snapshot_into_prometheus():
    """_update_gauges() writes the snapshot dict into the module-level
    Prometheus gauges so they show up at /metrics."""
    _update_gauges({
        "threads": 42, "python_threads": 8, "fds": 17,
        "asyncio_tasks": 3, "rss_mb": 512,
        "python_thread_groups": {"MainThread": 1, "ThreadPoolExecutor-0": 7},
    })
    assert REGISTRY.get_sample_value("ongiini_webhook_threads") == 42
    assert REGISTRY.get_sample_value("ongiini_webhook_python_threads") == 8
    assert REGISTRY.get_sample_value("ongiini_webhook_open_fds") == 17
    assert REGISTRY.get_sample_value("ongiini_webhook_asyncio_tasks") == 3
    assert REGISTRY.get_sample_value("ongiini_webhook_rss_mb") == 512
    assert REGISTRY.get_sample_value(
        "ongiini_webhook_python_threads_by_group", {"group": "MainThread"}
    ) == 1
    assert REGISTRY.get_sample_value(
        "ongiini_webhook_python_threads_by_group",
        {"group": "ThreadPoolExecutor-0"},
    ) == 7


def test_update_gauges_skips_negative_sentinels():
    """-1 sentinel means /proc read failed; the gauge should keep its
    last good value rather than report -1."""
    _update_gauges({
        "threads": 100, "python_threads": 5, "fds": 50,
        "asyncio_tasks": 5, "rss_mb": 200, "python_thread_groups": {},
    })
    _update_gauges({
        "threads": -1, "python_threads": -1, "fds": -1,
        "asyncio_tasks": -1, "rss_mb": -1, "python_thread_groups": {},
    })
    # Gauges should retain the good values, not be clobbered with -1
    assert REGISTRY.get_sample_value("ongiini_webhook_threads") == 100
    assert REGISTRY.get_sample_value("ongiini_webhook_open_fds") == 50
    assert REGISTRY.get_sample_value("ongiini_webhook_asyncio_tasks") == 5
    assert REGISTRY.get_sample_value("ongiini_webhook_rss_mb") == 200


@pytest.mark.asyncio
async def test_snapshot_loop_updates_gauges():
    """The loop updates the Prometheus gauges in addition to logging.
    Only asserts on the thread gauge — it has a portable fallback
    (threading.active_count) so the assertion holds even on platforms
    without /proc (CI on macOS, the dev box). RSS / FDs come from /proc
    only and are exercised end-to-end on Linux by the container itself."""
    _G_THREADS.set(0)
    task = asyncio.create_task(snapshot_loop(interval_s=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert REGISTRY.get_sample_value("ongiini_webhook_threads") > 0


# ── Per-thread-group classification (added for the leak diagnostic) ─


class TestThreadNameClassification:
    """Pinned mappings so the group labels stay stable across deploys
    (Grafana dashboards filter by these strings)."""

    @pytest.mark.parametrize("name,group", [
        ("MainThread", "MainThread"),
        ("ThreadPoolExecutor-0_0", "ThreadPoolExecutor-0"),
        ("ThreadPoolExecutor-0_7", "ThreadPoolExecutor-0"),
        ("ThreadPoolExecutor-3_2", "ThreadPoolExecutor-3"),
        ("asyncio_0", "asyncio"),
        ("asyncio 0", "asyncio"),
        ("Thread-1", "anonymous_Thread-N"),
        ("Thread-42 (target)", "anonymous_Thread-N"),
        ("resource-snapshot", "resource-snapshot"),
        ("stats-analyses", "stats-analyses"),
        ("Thread-3 (WSGIServer)", "prometheus_http"),
        ("totally-unknown-name", "other"),
    ])
    def test_thread_name_classified_to_expected_group(self, name, group):
        assert _classify_thread_name(name) == group


def test_python_thread_groups_returns_at_least_mainthread():
    """In any process there's always a MainThread — sanity check that
    enumeration + classification round-trips correctly."""
    groups = _python_thread_groups()
    assert groups.get("MainThread", 0) >= 1


def test_format_groups_is_sorted_by_count_desc():
    """The log line shows noisiest groups first so a leak post-mortem
    can eyeball the dominant pool without searching."""
    out = _format_groups({"a": 1, "b": 5, "c": 3})
    assert out == "b=5,c=3,a=1"


def test_format_groups_handles_empty():
    assert _format_groups({}) == ""
