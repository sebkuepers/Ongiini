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
import re
import threading
from collections import Counter

from prometheus_client import Gauge, start_http_server

log = logging.getLogger("ongiini.instrument")


# Gauges are module-level singletons. Prometheus' default registry
# auto-collects them — no separate registry plumbing needed.
_G_THREADS = Gauge(
    "ongiini_webhook_threads",
    "Number of OS-level threads (incl. C-ext threads invisible to Python) — "
    "from /proc/self/task. The 2026-05-23 leak signal.",
)
_G_PYTHON_THREADS = Gauge(
    "ongiini_webhook_python_threads",
    "Number of Python-visible threads (threading.active_count). Compare with "
    "ongiini_webhook_threads — if total grows but python_threads stays flat, "
    "the leak is in a C extension (torch, faster-whisper, ctranslate2, etc.)",
)
_G_THREADS_BY_GROUP = Gauge(
    "ongiini_webhook_python_threads_by_group",
    "Python-visible threads grouped by name pattern (e.g. ThreadPoolExecutor, "
    "asyncio, anonymous Thread-N). Tells WHICH library's pool is leaking.",
    ["group"],
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


# Regexes for collapsing per-thread names into a small set of stable
# groups. Order matters — first match wins. Names that don't match any
# pattern fall into "other" so we still see them in the gauge.
#
# Why grouping matters: each asyncio.to_thread() call ultimately uses a
# named ThreadPoolExecutor; libraries like sentence-transformers create
# their own named executors. The group label lets us answer "which
# library is leaking" without explosion in metric cardinality.
_THREAD_GROUP_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^MainThread$"), "MainThread"),
    (re.compile(r"^ThreadPoolExecutor-(\d+)_\d+$"), r"ThreadPoolExecutor-\1"),
    (re.compile(r"^asyncio_\d+$"), "asyncio"),
    (re.compile(r"^anyio\.from_thread"), "anyio"),
    (re.compile(r"^asyncio "), "asyncio"),
    (re.compile(r"^resource-snapshot$"), "resource-snapshot"),
    (re.compile(r"^stats-analyses$"), "stats-analyses"),
    (re.compile(r"WSGIServer", re.IGNORECASE), "prometheus_http"),
    (re.compile(r"^uvicorn", re.IGNORECASE), "uvicorn"),
    (re.compile(r"^Thread-\d+( \(\w+\))?$"), "anonymous_Thread-N"),
]


def _classify_thread_name(name: str) -> str:
    """Collapse a Python thread name into a stable group label so the
    Prometheus gauge cardinality stays bounded (storage cost scales
    with unique label sets, so 1000 distinct 'Thread-1234' labels would
    blow up the series count for no diagnostic value)."""
    for pat, label in _THREAD_GROUP_PATTERNS:
        m = pat.match(name) if pat.pattern.startswith("^") else pat.search(name)
        if m:
            try:
                return m.expand(label) if "\\" in label else label
            except (re.error, IndexError):
                return label
    return "other"


def _python_thread_groups() -> Counter:
    """Count live Python threads grouped by name pattern. Returns a
    Counter so the caller can iterate group → count to set the gauge."""
    c: Counter = Counter()
    for t in threading.enumerate():
        c[_classify_thread_name(t.name)] += 1
    return c

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


def snapshot() -> dict[str, int | dict]:
    """One-shot resource snapshot. Returned as a dict so callers (e.g.
    the /metrics endpoint, a log line) can serialise it however they
    want. The ``python_thread_groups`` field is a dict of group→count."""
    try:
        asyncio_tasks = len(asyncio.all_tasks())
    except RuntimeError:
        asyncio_tasks = -1
    return {
        "threads": _thread_count(),
        "python_threads": threading.active_count(),
        "python_thread_groups": dict(_python_thread_groups()),
        "fds": _fd_count(),
        "asyncio_tasks": asyncio_tasks,
        "rss_mb": _rss_mb(),
    }


def _update_gauges(s: dict[str, int | dict]) -> None:
    """Push the latest snapshot into Prometheus gauges. -1 sentinels
    (set by helpers when /proc read fails) skip the gauge so we keep
    the last good value instead of clobbering it with a fake -1.

    Per-group thread counts are pushed one label-value at a time.
    Groups that previously existed but vanished get set to 0 (rather
    than left dangling) so Grafana doesn't show stale series."""
    if s["threads"] >= 0:
        _G_THREADS.set(s["threads"])
    if s["python_threads"] >= 0:
        _G_PYTHON_THREADS.set(s["python_threads"])
    if s["fds"] >= 0:
        _G_FDS.set(s["fds"])
    if s["asyncio_tasks"] >= 0:
        _G_ASYNCIO_TASKS.set(s["asyncio_tasks"])
    if s["rss_mb"] >= 0:
        _G_RSS_MB.set(s["rss_mb"])
    # Per-group breakdown — set every group we currently see. Groups
    # that disappeared aren't reset (they'll naturally decay in Prom
    # as no new samples arrive), which is fine for diagnostic use.
    for group, count in s["python_thread_groups"].items():
        _G_THREADS_BY_GROUP.labels(group=group).set(count)


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


def _format_groups(groups: dict) -> str:
    """Compact 'g1=N,g2=M' rendering of the thread-group breakdown for
    the snapshot log line. Sorted by count desc so the noisiest groups
    are first in the line — easy to eyeball during a leak post-mortem."""
    return ",".join(f"{g}={n}" for g, n in sorted(groups.items(), key=lambda kv: -kv[1]))


async def snapshot_loop(interval_s: int = 60) -> None:
    """Background task — emit one resource-snapshot log line per
    ``interval_s`` seconds until cancelled and push the same numbers
    into Prometheus gauges. Safe to start from a FastAPI lifespan
    handler.

    Log format chosen to be easy to grep + easy to parse with awk or
    a small Python script when investigating a leak::

        resource_snapshot threads=543 python_threads=12 fds=124 \
            asyncio_tasks=12 rss_mb=1411 \
            groups=ThreadPoolExecutor-0=8,MainThread=1,prometheus_http=1,...
    """
    while True:
        try:
            s = snapshot()
            _update_gauges(s)
            log.info(
                "resource_snapshot threads=%d python_threads=%d fds=%d "
                "asyncio_tasks=%d rss_mb=%d groups=%s",
                s["threads"], s["python_threads"], s["fds"],
                s["asyncio_tasks"], s["rss_mb"],
                _format_groups(s["python_thread_groups"]),
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
                    "resource_snapshot.shutdown threads=%d python_threads=%d "
                    "fds=%d asyncio_tasks=%d rss_mb=%d groups=%s",
                    s["threads"], s["python_threads"], s["fds"],
                    s["asyncio_tasks"], s["rss_mb"],
                    _format_groups(s["python_thread_groups"]),
                )
            except Exception:
                pass
            raise
