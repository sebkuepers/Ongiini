"""WebChatTransport — the Owela Transport used by chat.ongiini.ai.

Unlike WhatsAppTransport (which POSTs the reply to Meta's API), this
transport "sends" by writing the body into an in-process slot that the
HTTP handler is awaiting. The handler creates a transport per request,
calls ``Agent.handle(msg)``, then reads ``.captured_reply`` to return
to the browser.

Owela's executor doesn't know any of this — it just calls ``send()``
like any other transport. The capture pattern lives entirely in the
adapter.

Markdown is preserved (the browser renders it). URL hygiene and
boundary trimming are shared with WhatsAppTransport via
``reply_hygiene`` — both transports talk to the same search-backed
model and shouldn't differ in how they handle citations.
"""
from __future__ import annotations

import asyncio
import logging

from owela import InboundMessage, Policy, ReplyContext, SendResult

from . import reply_hygiene

log = logging.getLogger("ongiini.transports.web_chat")


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
        reply_timeout_s: float = 90.0,
        empty_reply_text: str = "Sorry, I couldn't come up with a reply.",
        continue_offer_text: str = "Want the rest? Reply **more**.",
    ) -> None:
        self.reply_timeout_s = reply_timeout_s
        self.empty_reply_text = empty_reply_text
        self.continue_offer_text = continue_offer_text
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
        """No-op for web chat. When we eventually stream replies we might
        surface progress events here, but for the wait-for-complete MVP
        there's nothing to deliver."""
        return None

    async def send(
        self,
        user_id: str,
        body: str,
        policy: Policy,
        ctx: ReplyContext,
    ) -> SendResult:
        """Capture the reply into the per-request slot.

        Same post-processing pipeline as WhatsAppTransport with two
        differences:
          - Markdown is NOT flattened to WhatsApp syntax. The browser
            renders `**bold**`, `[text](url)`, etc. natively. We still
            strip raw HTML tag fragments via the URL-malformed check.
          - The char cap is much higher (10_000 vs 4096) — there's no
            external API limit constraining browser delivery.

        URL hygiene is shared with WhatsAppTransport and only runs when
        a citation tool fired this turn.

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
            return SendResult(sent=True)

        attrs: dict[str, object] = {}
        cleaned = (body or "").strip()
        if cleaned and ctx.truncated:
            cleaned = reply_hygiene.trim_to_boundary(cleaned)
            cleaned = f"{cleaned}\n\n{self.continue_offer_text}"
            attrs["trimmed_at_boundary"] = True
        if not cleaned:
            cleaned = self.empty_reply_text
            attrs["empty_fallback"] = True

        if reply_hygiene.cites_tools(ctx.used_tools):
            cleaned, dropped = reply_hygiene.drop_unlisted_urls(cleaned, ctx.allowed_urls)
            attrs["urls_dropped"] = dropped
            if not cleaned:
                cleaned = self.empty_reply_text
                attrs["empty_fallback"] = True

        notice = f"\n\n_{policy.reply_notice}_" if policy.reply_notice else ""
        limit = self.max_message_chars - len(notice)
        if len(cleaned) > limit:
            cleaned = reply_hygiene.cap_at_boundary(cleaned, limit)
            attrs["chars_capped"] = True
        cleaned += notice

        self._reply = cleaned
        self._send_ok = True
        self._event.set()
        return SendResult(sent=True, attrs=attrs)

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
