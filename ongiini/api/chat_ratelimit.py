"""Per-IP sliding-window rate-limit for the chat.ongiini.ai endpoint.

Mirrors the shape of ``ongiini.ratelimit`` (the WhatsApp per-msisdn
limiter) but keys on source IP instead of phone number. In-memory only;
resets on container restart. Cloudflare WAF/Bot Management sits in
front of this and handles the heavy DDoS cases — this layer is the
backend's own ceiling for the slow-and-steady abusers.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from ..config import settings

_buckets: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def check(ip: str) -> tuple[bool, str]:
    """Return (allowed, reason). When False, ``reason`` is a user-facing
    message to surface in the 429 response body.

    Single sliding-window per IP: the configured per-hour cap from
    ``settings.chat_ip_rate_limit_per_hour``. Empty IP or "unknown"
    (when the request has no source identifier) is always allowed —
    refusing those would lock out genuine traffic during transient
    middleware misconfig.
    """
    if not ip or ip == "unknown":
        return True, ""

    now = time.time()
    with _lock:
        bucket = _buckets[ip]
        # Purge entries older than one hour.
        cutoff = now - 3600
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= settings.chat_ip_rate_limit_per_hour:
            return (
                False,
                "You're going fast — give it a few minutes and try again.",
            )

        bucket.append(now)
        return True, ""


def reset_for_tests() -> None:
    """Clear the rate-limit state. ONLY for tests."""
    with _lock:
        _buckets.clear()
