"""Owela ``Transport`` implementation for WhatsApp Cloud API.

This is the only place in the codebase that knows about Meta's
``graph.facebook.com`` endpoint and the 25s typing-window constraint.
The Owela executor only sees ``transport.acknowledge(msg)``,
``transport.send_interstitial(user_id, policy)``, and
``transport.send(user_id, body, policy)``.

Transport-internal reply hygiene (anti-confabulation):
  1. **Dead-URL HEAD check** — strip lines containing 404/410 URLs
     before sending. Saves the user from clicking citation links that
     go nowhere.
  2. **HTML-fragment URL filter** — Tavily sometimes returns URLs
     with embedded HTML tag fragments; treat those as broken too.
  3. **Char cap** — WhatsApp's per-message limit is 4096; the truncation
     happens at the API layer anyway, but cap deliberately so we don't
     produce orphaned half-citations.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx

from owela import InboundMessage, Policy

# We delegate the actual Meta HTTP calls to the existing low-level helpers
# in ``webhook.app.whatsapp`` so the retry policy, signature checks, and
# media handling all keep working unchanged. The Transport is purely a
# wrapper that imposes the Owela protocol shape.
from ..whatsapp import mark_as_read as _mark_as_read
from ..whatsapp import send_text as _send_text

log = logging.getLogger("ongiini.transports.whatsapp")


# Match a full https?:// URL up to the first whitespace or closing
# bracket. Trailing punctuation is trimmed at use sites.
_URL_RE = re.compile(r"https?://[^\s)>\"]+")


class WhatsAppTransport:
    """The Ongiini transport. Constructed once at runtime build."""

    # Owela protocol metadata.
    name = "whatsapp"
    typing_window_s = 25.0          # Meta-side hard limit, not configurable.
    max_message_chars = 4096        # WhatsApp's per-text limit.
    format = "plain_text"           # No markdown — WhatsApp shows raw chars.

    def __init__(
        self,
        *,
        interstitial_text: str = "Looking into this — give me about 20-30 seconds.",
        dead_url_check_timeout_s: float = 2.0,
    ) -> None:
        self.interstitial_text = interstitial_text
        self.dead_url_check_timeout_s = dead_url_check_timeout_s

    async def acknowledge(self, msg: InboundMessage) -> None:
        """Read receipt + typing indicator. Soft-fail (logged inside
        the underlying helper) — UX is not allowed to break a reply."""
        if not msg.msg_id:
            return
        await _mark_as_read(msg.msg_id, with_typing=True)

    async def send_interstitial(self, user_id: str, policy: Policy) -> None:
        """v1 — sends a "still working" message during long turns. The
        text is configurable at construction time so other transports
        with different latency budgets (Signal, Telegram) can override.
        """
        await _send_text(user_id, self.interstitial_text)

    async def send(
        self,
        user_id: str,
        body: str,
        policy: Policy,
        *,
        used_search: bool = False,
    ) -> bool:
        """Post-process and deliver the final reply.

        Steps, in order:
          1. trim whitespace
          2. if used_search: dead-URL strip (HEAD-check every URL in parallel)
          3. cap at ``max_message_chars``
          4. send via the underlying WhatsApp helper

        Dead-URL stripping is gated on ``used_search`` to match the
        original behaviour in ``ongiini/llm.py::respond`` — non-search
        turns should not pay the 2s HEAD-check latency or risk stripping
        legitimately-quoted URLs the model wrote from its own context.

        Returns True on successful send. Transport-side failures bubble
        up as RuntimeError; the executor catches and records ReplyStep.sent=False.
        """
        cleaned = (body or "").strip()
        if not cleaned:
            cleaned = "Sorry, I couldn't come up with a reply."

        if used_search:
            cleaned = await self._strip_dead_urls(cleaned)
            # If the strip removed every line (all URLs were dead), the
            # cleaned body might be empty or just whitespace. Fall back
            # to a graceful explanation rather than sending an empty
            # WhatsApp message body (Meta 400s on those).
            if not cleaned.strip():
                cleaned = (
                    "I had sources for this but they didn't come back as "
                    "live links right now. Want me to search again with "
                    "different terms?"
                )
        if len(cleaned) > self.max_message_chars:
            cleaned = cleaned[: self.max_message_chars]

        await _send_text(user_id, cleaned)
        return True

    # ----------- internal hygiene -----------

    async def _strip_dead_urls(self, reply: str) -> str:
        """HEAD-check every URL in the reply; remove lines containing a
        definitely-dead URL (404 / 410) or a malformed URL (embedded
        HTML fragment). Soft-fail: timeouts and other errors keep the
        URL.

        Empty replies / no URLs short-circuit. URLs are checked in
        parallel via ``asyncio.gather`` to bound extra latency to one
        round-trip.
        """
        if not reply:
            return reply

        raw_urls = _URL_RE.findall(reply)
        # Trim trailing punctuation that snuck into the match.
        urls = [u.rstrip(".,;:!?)") for u in raw_urls]

        # URLs with embedded HTML tag fragments come from Tavily snippets
        # that didn't strip an <i> or </a>. HEAD-check will not catch
        # these (httpx percent-encodes the brackets and the server
        # typically returns 200 + a redirect to the homepage).
        malformed = {u for u in urls if "<" in u or ">" in u}
        for u in malformed:
            log.info("stripping malformed URL (embedded HTML): %s", u)

        urls = [u for u in urls if u not in malformed]
        urls = list(dict.fromkeys(urls))   # dedupe, preserve order

        cleaned = reply

        if malformed:
            cleaned = self._drop_lines_containing(cleaned, malformed)

        if urls:
            dead = await self._head_check_for_dead(urls)
            if dead:
                for u in dead:
                    log.info("stripping dead URL (404/410): %s", u)
                cleaned = self._drop_lines_containing(cleaned, dead)

        return cleaned

    async def _head_check_for_dead(self, urls: list[str]) -> set[str]:
        timeout = self.dead_url_check_timeout_s

        async def _check(url: str) -> tuple[str, bool]:
            try:
                async with httpx.AsyncClient(
                    timeout=timeout, follow_redirects=True,
                ) as client:
                    r = await client.head(url)
                    if r.status_code == 405:
                        # Some servers reject HEAD — fall back to a range
                        # GET that we don't read.
                        r = await client.get(url, headers={"Range": "bytes=0-0"})
                    alive = r.status_code not in (404, 410)
                    return url, alive
            except Exception:                  # noqa: BLE001 — soft-fail keeps URL
                return url, True

        results = await asyncio.gather(*[_check(u) for u in urls])
        return {u for u, alive in results if not alive}

    @staticmethod
    def _drop_lines_containing(text: str, bad: set[str] | list[str]) -> str:
        bad_set = set(bad)
        out_lines: list[str] = []
        for line in text.split("\n"):
            if any(b in line for b in bad_set):
                continue
            out_lines.append(line)
        cleaned = "\n".join(out_lines)
        # Collapse any 3+ blank-line runs introduced by line removal.
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
