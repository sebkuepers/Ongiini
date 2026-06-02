"""WebChatTransport — the Owela Transport used by chat.ongiini.ai.

Unlike WhatsAppTransport (which POSTs the reply to Meta's API), this
transport "sends" by writing the body into an in-process slot that the
HTTP handler is awaiting. The handler creates a transport per request,
calls ``Agent.handle(msg)``, then reads ``.captured_reply`` to return
to the browser.

Owela's executor doesn't know any of this — it just calls ``send()``
like any other transport. The capture pattern lives entirely in the
adapter.

Markdown is preserved (the browser renders it). Dead-URL stripping is
shared with WhatsAppTransport — both transports talk to the same
search-backed model and shouldn't differ in how they handle stale
citations.
"""
from __future__ import annotations

import asyncio
import logging
import re

import httpx

from owela import InboundMessage, Policy

log = logging.getLogger("ongiini.transports.web_chat")


# Same URL regex WhatsAppTransport uses. The constant lives in
# whatsapp_transport.py too — kept local here so the two transports
# stay independent (avoids accidental coupling later).
_URL_RE = re.compile(r"https?://[^\s)>\"]+")


class WebChatTransport:
    """Owela Transport for the anonymous web chat endpoint.

    One instance per HTTP request. Constructed by the chat handler,
    passed to ``build_chat_runtime``, awaited via ``await_reply()``
    after ``Agent.handle()`` returns.
    """

    # Owela protocol metadata. ``typing_window_s`` here is the budget
    # the executor uses to schedule interstitials — HTTP itself has no
    # native typing UX, but interstitial messages are no-ops anyway for
    # web chat (see ``send_interstitial`` below).
    name = "web_chat"
    typing_window_s = 60.0
    max_message_chars = 10_000
    format = "markdown"

    def __init__(
        self,
        *,
        dead_url_check_timeout_s: float = 2.0,
        reply_timeout_s: float = 90.0,
    ) -> None:
        self.dead_url_check_timeout_s = dead_url_check_timeout_s
        self.reply_timeout_s = reply_timeout_s
        # The captured reply slot. ``send()`` sets ``_reply`` and signals
        # ``_event``; ``await_reply()`` blocks on the event then returns
        # the slot. None until set; empty string is a legal value if the
        # executor decided there was nothing to say.
        self._reply: str | None = None
        self._event = asyncio.Event()
        self._send_ok = False
        # Set by fail() when the surrounding handler's Agent.handle()
        # raises before reaching transport.send(). Re-raised by
        # await_reply() so the HTTP handler can return a clean error
        # immediately instead of hanging until the reply_timeout_s
        # budget (90s) expires.
        self._error: BaseException | None = None

    async def acknowledge(self, msg: InboundMessage) -> None:
        """No-op for web chat. HTTP request/response is synchronous, the
        browser shows its own typing indicator while waiting on the
        response."""
        return None

    async def send_interstitial(self, user_id: str, policy: Policy) -> None:
        """No-op for web chat. ``policy.enable_interstitial`` only fires
        on SEARCH_DEEP today — when we eventually stream replies we
        might surface progress events here, but for the
        wait-for-complete MVP there's nothing to deliver."""
        return None

    async def send(
        self,
        user_id: str,
        body: str,
        policy: Policy,
        *,
        used_search: bool = False,
    ) -> bool:
        """Capture the reply into the per-request slot.

        Same post-processing pipeline as WhatsAppTransport with two
        differences:
          - Markdown is NOT flattened to WhatsApp syntax. The browser
            renders `**bold**`, `[text](url)`, etc. natively. We still
            strip raw HTML tag fragments via the URL-malformed check.
          - The char cap is much higher (10_000 vs 4096) — there's no
            external API limit constraining browser delivery.

        Dead-URL stripping is shared logic with WhatsAppTransport; only
        runs when ``used_search`` is set so plain conversational turns
        don't eat the HEAD-check latency budget.

        Note: markdown tables (``| col | col |``) are NOT converted
        here. The frontend renderer (see ``website/chat-app/index.html``)
        is responsible for rendering tables; this transport stays in
        the post-processing role and trusts the client.
        """
        # Owela's contract is one send() per turn. A second call would
        # silently overwrite the first reply — log it and ignore so the
        # captured body matches what the executor first decided was the
        # final answer. Defensive: production wouldn't trigger this
        # without a real executor bug.
        if self._send_ok:
            log.warning(
                "web_chat send() called twice; ignoring second body "
                "(len=%d)", len(body or ""),
            )
            return True

        cleaned = (body or "").strip()
        if not cleaned:
            cleaned = "Sorry, I couldn't come up with a reply."

        if used_search:
            cleaned = await self._strip_dead_urls(cleaned)
            if not cleaned.strip():
                cleaned = (
                    "I had sources for this but they didn't come back as "
                    "live links right now. Want me to search again with "
                    "different terms?"
                )
        if len(cleaned) > self.max_message_chars:
            cleaned = cleaned[: self.max_message_chars]

        self._reply = cleaned
        self._send_ok = True
        self._event.set()
        return True

    async def await_reply(self) -> str:
        """Block until ``send()`` fires, return the captured body.

        If ``fail()`` was called the stored exception is re-raised so
        the HTTP handler can return an error response immediately
        instead of hanging until ``reply_timeout_s`` expires. Raises
        ``asyncio.TimeoutError`` if neither send() nor fail() fires
        within the budget.
        """
        await asyncio.wait_for(self._event.wait(), timeout=self.reply_timeout_s)
        if self._error is not None:
            raise self._error
        return self._reply or ""

    def fail(self, exc: BaseException) -> None:
        """Inject an exception so a pending ``await_reply()`` unblocks
        immediately with the exception instead of timing out.

        The HTTP handler wraps ``Agent.handle(msg)`` in try/except and
        calls ``fail()`` when handle raises before reaching the
        transport's send() — without this, every executor-side failure
        (classifier crash, hook raise, model timeout) would burn the
        full reply_timeout_s budget per request, turning the chat
        endpoint into a DoS amplifier under any backend hiccup.

        Soft-fail / idempotent: if the slot is already set (send() ran
        successfully OR a previous fail() already fired) we log and
        return without overwriting.
        """
        if self._send_ok or self._error is not None:
            log.warning(
                "web_chat fail() called after slot already set; ignoring "
                "(%s: %s)", type(exc).__name__, exc,
            )
            return
        self._error = exc
        self._event.set()

    @property
    def reply_received(self) -> bool:
        """True once ``send()`` has fired. Lets the handler decide
        whether to attempt graceful degradation."""
        return self._send_ok

    # ─── internal hygiene (shared logic with WhatsAppTransport) ──────

    async def _strip_dead_urls(self, reply: str) -> str:
        """HEAD-check every URL in the reply; drop lines containing
        404/410 URLs or malformed (embedded HTML tag) URLs. Soft-fail:
        timeouts and other errors keep the URL in place."""
        if not reply:
            return reply
        raw_urls = _URL_RE.findall(reply)
        urls = [u.rstrip(".,;:!?)") for u in raw_urls]
        malformed = {u for u in urls if "<" in u or ">" in u}
        urls = [u for u in urls if u not in malformed]
        urls = list(dict.fromkeys(urls))

        cleaned = reply
        if malformed:
            cleaned = self._drop_lines_containing(cleaned, malformed)
        if urls:
            dead = await self._head_check_for_dead(urls)
            if dead:
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
                        r = await client.get(url, headers={"Range": "bytes=0-0"})
                    return url, r.status_code not in (404, 410)
            except Exception:                      # noqa: BLE001 — soft-fail
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
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
