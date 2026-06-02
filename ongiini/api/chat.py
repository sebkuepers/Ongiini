"""HTTP endpoints for the chat.ongiini.ai anonymous web chat.

Two routes live here:

  * ``POST /v1/chat`` — one chat turn. Request carries the browser's
    session_id (UUID v4 in localStorage), the user's text, and
    optionally an image as base64. Response carries the model's reply.

  * ``POST /v1/chat/clear`` — wipe the named session. Anonymous users
    can't "delete my data" via WhatsApp ADMIN; this is the equivalent.

Per-request flow:

  1. Validate session_id format (UUID v4). Reject otherwise.
  2. IP rate-limit check (sliding window per source IP).
  3. Per-session token-cap check (read SessionState.tokens_used).
  4. (If image) base64-decode + resize for Gemma 4.
  5. Read session history out of the SessionStore.
  6. Construct an InboundMessage.
  7. Build a per-request Runtime via build_chat_runtime() with a fresh
     WebChatTransport + the bound SessionMemoryProvider.
  8. ``await Agent.handle(msg)`` with a timeout.
  9. ``await transport.await_reply()`` to capture the reply.
  10. Return reply + token usage in JSON.

The session store is created at app startup (in api/main.py lifespan)
and passed via ``shared_components`` style — this module exposes the
``build_router(store, shared, sanitiser)`` factory so the wiring stays
in the application layer.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import re
import time
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from owela import Agent, InboundMessage

from . import chat_ratelimit
from ..config import settings
from ..memory import SessionMemoryProvider, SessionStore
from ..runtime import SharedComponents, build_chat_runtime
from ..system_prompt import SYSTEM_PROMPT
from ..transports import WebChatTransport

log = logging.getLogger("ongiini.api.chat")


# UUID v4 format: 8-4-4-4-12 hex characters. We accept the
# canonical-lowercase form the browser's crypto.randomUUID() produces;
# this regex deliberately rejects anything else (no uppercase, no
# braces, no hyphens-removed forms) so the server has a single
# session-id format to deal with.
_UUID_V4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

# Largest base64-encoded image payload we'll accept. Plain JPEGs from a
# modern phone camera are 1-3 MB; the base64 form inflates to ~33%
# more. 12 MB encoded covers the realistic upper bound; anything
# larger is rejected before we burn memory on a decode.
_IMAGE_MAX_B64_BYTES = 12 * 1024 * 1024

# Per-turn wall-clock limit for the whole pipeline (classifier +
# planner + act loop + critique + transport.send). Matches the chat
# transport's reply timeout so the two budgets stay aligned.
_TURN_TIMEOUT_S = 90.0


class ChatRequest(BaseModel):
    """Single chat turn input."""
    session_id: str = Field(..., min_length=36, max_length=36)
    text: str = Field(..., max_length=4000)
    image_b64: str | None = Field(default=None, max_length=_IMAGE_MAX_B64_BYTES)


class ChatResponse(BaseModel):
    """Single chat turn output."""
    reply: str
    tokens_used: int            # tokens spent on THIS turn
    session_tokens_used: int    # running total for the session
    session_token_cap: int      # the limit, surfaced for client display


class ClearRequest(BaseModel):
    session_id: str = Field(..., min_length=36, max_length=36)


def _validate_session_id(session_id: str) -> None:
    if not _UUID_V4_RE.match(session_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id must be a UUID v4 (lowercase, hyphenated).",
        )


def _client_ip(request: Request) -> str:
    """Pull the source IP from Cloudflare's CF-Connecting-IP header
    (set on every request that traverses the Cloudflare proxy), with
    fallback to X-Forwarded-For and the direct peer if those don't
    exist. Used for the rate-limit key only — never logged with the
    request body."""
    cf = request.headers.get("cf-connecting-ip", "").strip()
    if cf:
        return cf
    xff = request.headers.get("x-forwarded-for", "").strip()
    if xff:
        # XFF is a comma-separated list; the leftmost is the original
        # client. Trim whitespace.
        return xff.split(",")[0].strip()
    if request.client:
        return request.client.host or "unknown"
    return "unknown"


def build_router(
    store: SessionStore,
    shared: SharedComponents,
    *,
    pii_sanitiser=None,
    resize_image=None,
) -> APIRouter:
    """Build the chat router with the dependencies it needs.

    ``store`` is the singleton SessionStore from app startup.
    ``shared`` is the singleton SharedComponents (model, classifier,
    tools, policies, hooks helpers). ``pii_sanitiser`` is the function
    that scrubs emails/IDs/cards before in-memory storage. ``resize_image``
    is the Gemma-4 image resizer from api/main.py (we pass it as a
    callable to avoid the api/chat.py → api/main.py circular import).
    """
    # Router has NO prefix because it gets mounted under a sub-app
    # at "/v1" — see api/main.py::lifespan. The sub-app pattern is
    # used so chat's CORSMiddleware only applies to /v1/chat* and
    # doesn't leak POST/OPTIONS onto the parent /whatsapp + /status
    # routes.
    router = APIRouter(tags=["chat"])

    @router.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, request: Request) -> Any:
        if not settings.chat_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Chat is temporarily disabled. Please try again later.",
            )

        _validate_session_id(req.session_id)

        # 1. IP rate-limit. Cloudflare WAF/Bot Management is the first
        #    line; this is the in-app ceiling for slow steady abusers.
        ip = _client_ip(request)
        ok, reason = chat_ratelimit.check(ip)
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=reason,
            )

        # 2. Per-session token cap. Read the running total without
        #    bumping last_used_at — we don't want to count the
        #    rejection toward keep-alive. Only the get_or_create that
        #    runs later (inside the actual processing path) should
        #    refresh the session.
        existing = store.peek(req.session_id)
        if existing is not None and existing.tokens_used >= settings.chat_session_token_cap:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    "This conversation has reached its token limit. "
                    "Refresh the page to start a fresh one."
                ),
            )

        # 3. Image handling (optional).
        has_image = False
        content_parts: list[dict[str, Any]] = []
        if req.image_b64:
            if resize_image is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Image processing is unavailable on this server.",
                )
            try:
                image_bytes = base64.b64decode(req.image_b64, validate=True)
            except (binascii.Error, ValueError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="image_b64 is not valid base64.",
                )
            try:
                image_bytes = resize_image(image_bytes)
            except Exception:                          # noqa: BLE001
                log.exception("chat: resize_image failed")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Couldn't process that image — try a different one.",
                )
            data_url = (
                "data:image/jpeg;base64,"
                + base64.standard_b64encode(image_bytes).decode("ascii")
            )
            has_image = True

        # 4. Construct the text + (optional) image content.
        safe_caption = (
            pii_sanitiser(req.text) if pii_sanitiser and req.text else (req.text or "")
        )
        if has_image:
            text_part = safe_caption or (
                "I just sent you a photo. Have a look and tell me what you see — "
                "if there's something specific worth pointing out, mention it."
            )
            content_parts = [
                {"type": "text", "text": text_part},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        else:
            if not safe_caption:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="text or image_b64 is required.",
                )
            content_parts = [{"type": "text", "text": safe_caption}]

        # 5. Read the existing session history (or create the session
        #    on first request).
        state = store.get_or_create(req.session_id)
        history_snapshot = list(state.history)

        # 6. Build per-request transport + memory + runtime.
        transport = WebChatTransport()
        memory_provider = SessionMemoryProvider(
            system_prompt=SYSTEM_PROMPT,
            store=store,
            skills=shared.skills,
            pii_sanitiser=pii_sanitiser,
        )

        rt = build_chat_runtime(
            shared,
            transport=transport,
            memory_provider=memory_provider,
        )

        msg = InboundMessage(
            user_id=req.session_id,
            msg_id=uuid4().hex,
            text=safe_caption or "",
            content_parts=content_parts,
            has_image=has_image,
            history=history_snapshot,
            storage_text="",
            raw_payload=None,
        )

        # 7. Run the turn. The watcher task forwards any executor-side
        #    failure to the transport via fail(), so await_reply()
        #    raises immediately instead of burning the full
        #    reply_timeout_s budget. After agent.handle() completes
        #    successfully without producing a reply (e.g., a tool-only
        #    turn or a policy that short-circuits), the watcher trips
        #    fail() with a sentinel so await_reply also resolves
        #    instead of hanging.
        agent = Agent(rt)
        turn_start = time.monotonic()
        handle_task = asyncio.create_task(
            asyncio.wait_for(agent.handle(msg), timeout=_TURN_TIMEOUT_S),
        )

        async def _watch_handle():
            try:
                await handle_task
            except asyncio.CancelledError:
                # Watcher cancellation is initiated by the request
                # handler's finally block (timeout / client disconnect).
                # Don't log it as an error, don't push into the
                # transport (await_reply has already returned or is
                # being torn down).
                raise
            except Exception as exc:                   # noqa: BLE001
                log.exception(
                    "chat: agent.handle failed for session %s",
                    req.session_id[:8],
                )
                transport.fail(exc)
                return
            # agent.handle completed without raising. If the transport
            # didn't receive a body (executor decided not to reply),
            # unblock await_reply with a sentinel so the handler can
            # return a clean 500 instead of timing out at 90s.
            if not transport.reply_received:
                transport.fail(RuntimeError("agent produced no reply"))

        # Keep a reference so the watcher gets cancelled by the
        # finally block below (client disconnect, timeout, etc.) —
        # without this it would leak on every request.
        watcher = asyncio.create_task(_watch_handle())

        try:
            try:
                reply = await transport.await_reply()
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="That took too long — try again in a moment.",
                )
            except HTTPException:
                raise
            except Exception as exc:                   # noqa: BLE001
                log.warning(
                    "chat: turn failed for session %s: %s",
                    req.session_id[:8], type(exc).__name__,
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Something went wrong on our side. Try again.",
                )
        finally:
            # Tie the watcher + handle tasks to the request lifecycle.
            # Cancellation is safe — both have shielded blocks.
            for t in (handle_task, watcher):
                if not t.done():
                    t.cancel()

        # 8. Token accounting.
        #
        # MVP placeholder: estimate tokens from char counts at ~3
        # chars/token. This UNDER-counts system prompt + tool-call
        # context, so the session cap (50k tokens) will be loose:
        # sessions will run ~3-5× longer than the cap suggests before
        # the rejection fires.
        #
        # TODO before scaling — wire `usage.summary_for(session_id)`
        # snapshots around agent.handle() to record the real delta the
        # BillingHook saw. The recorder is already in the chat
        # runtime's hook chain; the only missing piece is reading it
        # back here. Tracked as a follow-up; for the launch the loose
        # cap is acceptable because the IP rate limit gates abuse and
        # the LRU on max_sessions bounds total memory regardless.
        tokens_used = max(1, (len(safe_caption) + len(reply)) // 3)
        session_total = store.touch_tokens(req.session_id, tokens_used)

        elapsed_ms = int((time.monotonic() - turn_start) * 1000)
        log.info(
            "chat: session=%s tokens=%d total=%d elapsed=%dms",
            req.session_id[:8], tokens_used, session_total, elapsed_ms,
        )

        return ChatResponse(
            reply=reply,
            tokens_used=tokens_used,
            session_tokens_used=session_total,
            session_token_cap=settings.chat_session_token_cap,
        )

    @router.post("/chat/clear")
    async def clear(req: ClearRequest) -> JSONResponse:
        _validate_session_id(req.session_id)
        removed = store.delete(req.session_id)
        return JSONResponse({"ok": True, "removed": removed})

    return router
