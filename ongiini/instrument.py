"""Lightweight runtime resource instrumentation.

Diagnostic background task for the 2026-05-23 process-leak incident.
The webhook accumulated 11,154 OS threads over 10 hours before wedging.
We had no historical thread/fd/memory data to identify the cause.

This module:

- Emits one structured log line per minute with thread count, file
  descriptor count, asyncio task count, and resident memory. Visible in
  ``docker logs ongiini-webhook | grep resource_snapshot``.
- Exposes the same numbers as Prometheus gauges on a dedicated HTTP
  server (default :9101) so they become long-lived time series in
  Grafana, not just ephemeral log lines. The server is started from
  ``start_metrics_server()`` — main.py calls it once during lifespan.
- Cheap (~5 ms per snapshot) — runs from a single asyncio task.
- Crash-safe — any failure to read /proc is logged but never raises.

To investigate a leak, correlate timestamps with usage.log to see
whether thread/FD growth coincides with specific message types (voice
notes, images, mem0 adds) or grows linearly regardless of traffic.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading

from prometheus_client import Gauge, start_http_server

log = logging.getLogger("ongiini.instrument")


# Gauges are module-level singletons. Prometheus' default registry
# auto-collects them — no separate registry plumbing needed.
_G_THREADS = Gauge(
    "ongiini_webhook_threads",
    "Number of OS-level threads in the webhook process (the 2026-05-23 leak signal)",
)
_G_FDS = Gauge(
    "ongiini_webhook_open_fds",
    "Open file descriptors in the webhook process",
)
_G_ASYNCIO_TASKS = Gauge(
    "ongiini_webhook_asyncio_tasks",
    "Number of pending asyncio tasks in the event loop",
)
_G_RSS_MB = Gauge(
    "ongiini_webhook_rss_mb",
    "Resident set size of the webhook process in MiB",
)

# Tracks whether start_metrics_server() has already run so a duplicate
# call (e.g. from a test that imports lifespan twice) doesn't crash
# with "address already in use".
_metrics_server_started = False


def _thread_count() -> int:
    """Number of OS-level threads in this process. Uses /proc on Linux
    (counts kernel-visible threads, which is what wedged us). Falls
    back to Python-visible threads on platforms without /proc."""
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return threading.active_count()


def _fd_count() -> int:
    """Open file-descriptor count (sockets, files, pipes). FD exhaustion
    is a separate failure mode worth watching alongside thread leaks."""
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return -1


def _rss_mb() -> int:
    """Resident set size in MiB. Read from /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def snapshot() -> dict[str, int]:
    """One-shot resource snapshot. Returned as a dict so callers (e.g.
    future /metrics endpoint) can serialise it however they want."""
    try:
        asyncio_tasks = len(asyncio.all_tasks())
    except RuntimeError:
        asyncio_tasks = -1
    return {
        "threads": _thread_count(),
        "fds": _fd_count(),
        "asyncio_tasks": asyncio_tasks,
        "rss_mb": _rss_mb(),
    }


def _update_gauges(s: dict[str, int]) -> None:
    """Push the latest snapshot into Prometheus gauges. -1 sentinels
    (set by helpers when /proc read fails) skip the gauge so we keep
    the last good value instead of clobbering it with a fake -1."""
    if s["threads"] >= 0:
        _G_THREADS.set(s["threads"])
    if s["fds"] >= 0:
        _G_FDS.set(s["fds"])
    if s["asyncio_tasks"] >= 0:
        _G_ASYNCIO_TASKS.set(s["asyncio_tasks"])
    if s["rss_mb"] >= 0:
        _G_RSS_MB.set(s["rss_mb"])


def start_metrics_server(port: int = 9101) -> None:
    """Start the Prometheus exporter HTTP server on its own thread.

    Bound to 0.0.0.0 inside the container; the docker-compose host port
    mapping must restrict the host-side bind to 127.0.0.1 so /metrics
    never leaks via the public Cloudflare-fronted webhook port.

    Idempotent — the second call is a no-op rather than crashing with
    "address already in use" (which would matter for tests that import
    the lifespan twice).
    """
    global _metrics_server_started
    if _metrics_server_started:
        return
    start_http_server(port)
    _metrics_server_started = True
    log.info("metrics server listening on :%d/metrics", port)


async def snapshot_loop(interval_s: int = 60) -> None:
    """Background task — emit one resource-snapshot log line per
    ``interval_s`` seconds until cancelled and push the same numbers
    into Prometheus gauges. Safe to start from a FastAPI lifespan
    handler.

    Log format chosen to be easy to grep + easy to parse with awk or
    a small Python script when investigating a leak::

        resource_snapshot threads=543 fds=124 asyncio_tasks=12 rss_mb=1411
    """
    while True:
        try:
            s = snapshot()
            _update_gauges(s)
            log.info(
                "resource_snapshot threads=%d fds=%d asyncio_tasks=%d rss_mb=%d",
                s["threads"], s["fds"], s["asyncio_tasks"], s["rss_mb"],
            )
        except Exception as exc:  # noqa: BLE001 — instrumentation never breaks the app
            log.warning("resource_snapshot failed: %s", exc)
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            # Clean shutdown — emit a final snapshot for the post-mortem log.
            try:
                s = snapshot()
                _update_gauges(s)
                log.info(
                    "resource_snapshot.shutdown threads=%d fds=%d "
                    "asyncio_tasks=%d rss_mb=%d",
                    s["threads"], s["fds"], s["asyncio_tasks"], s["rss_mb"],
                )
            except Exception:
                pass
            raise
