"""Lightweight runtime resource instrumentation.

Diagnostic background task for the 2026-05-23 process-leak incident.
The webhook accumulated 11,154 OS threads over 10 hours before wedging.
We had no historical thread/fd/memory data to identify the cause.

This module:

- Emits one structured log line per minute with thread count, file
  descriptor count, asyncio task count, and resident memory. Visible in
  ``docker logs ongiini-webhook | grep resource_snapshot``.
- Cheap (~5 ms per snapshot) — runs from a single asyncio task.
- Crash-safe — any failure to read /proc is logged but never raises.
- Format chosen so a future Prometheus exporter can parse the same line
  if we wire metrics scraping later.

To investigate a leak later, correlate timestamps with usage.log to see
whether thread/FD growth coincides with specific message types (voice
notes, images, mem0 adds) or grows linearly regardless of traffic.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading

log = logging.getLogger("ongiini.instrument")


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


async def snapshot_loop(interval_s: int = 60) -> None:
    """Background task — emit one resource-snapshot log line per
    ``interval_s`` seconds until cancelled. Safe to start from a
    FastAPI lifespan handler.

    Log format chosen to be easy to grep + easy to parse with awk or
    a small Python script when investigating a leak::

        resource_snapshot threads=543 fds=124 asyncio_tasks=12 rss_mb=1411
    """
    while True:
        try:
            s = snapshot()
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
                log.info(
                    "resource_snapshot.shutdown threads=%d fds=%d "
                    "asyncio_tasks=%d rss_mb=%d",
                    s["threads"], s["fds"], s["asyncio_tasks"], s["rss_mb"],
                )
            except Exception:
                pass
            raise
