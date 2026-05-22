"""In-process TTL cache for the assembled /stats.json payload.

Aggregation is fast (a few tens of thousands of log lines), but every
page-view shouldn't trigger a full filesystem walk. We compute once,
hold the result for settings.stats_cache_ttl_seconds, then recompute
on the next request after expiry.

Single-slot cache — there is only one payload to cache (the global
aggregates). Concurrency-safe via an asyncio.Lock around the
recompute path so two parallel requests after expiry don't both
trigger the (somewhat expensive) full recompute; one waits.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from ..config import settings


class TTLCache:
    def __init__(self) -> None:
        self._payload: dict[str, Any] | None = None
        self._stored_at: float = 0.0
        self._lock = asyncio.Lock()

    def fresh(self) -> bool:
        if self._payload is None:
            return False
        return (time.monotonic() - self._stored_at) < settings.stats_cache_ttl_seconds

    async def get_or_compute(
        self, compute: Callable[[], Awaitable[dict[str, Any]]]
    ) -> dict[str, Any]:
        if self.fresh():
            assert self._payload is not None  # for type checkers
            return self._payload
        async with self._lock:
            # Re-check inside the lock — another waiter may have just
            # refreshed.
            if self.fresh():
                assert self._payload is not None
                return self._payload
            payload = await compute()
            self._payload = payload
            self._stored_at = time.monotonic()
            return payload


cache = TTLCache()
