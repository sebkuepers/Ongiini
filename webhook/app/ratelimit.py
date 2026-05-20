"""Per-phone-number in-memory rate limiter.

Tracks the timestamps of recent messages per sender and rejects bursts.
In-memory only — limits reset on container restart. That's fine for the
pilot scale (one webhook container). If we scale out to multiple workers
this needs to move to Redis or a shared file.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from .config import settings

_buckets: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def check(msisdn: str) -> tuple[bool, str]:
    """Return (allowed, reason). When False, `reason` is a user-facing message."""
    now = time.time()
    with _lock:
        bucket = _buckets[msisdn]

        # Purge entries older than 24h.
        while bucket and bucket[0] < now - 86400:
            bucket.popleft()

        last_5min = sum(1 for t in bucket if t > now - 300)
        last_day = len(bucket)

        if last_5min >= settings.rate_limit_per_5min:
            return (
                False,
                "You're sending messages a bit fast. Take a short break and try again "
                "in a few minutes.",
            )
        if last_day >= settings.rate_limit_per_day:
            return (
                False,
                "You've reached today's message limit. The counter resets every 24 hours "
                "— please try again tomorrow.",
            )

        bucket.append(now)
        return True, ""
