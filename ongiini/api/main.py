import asyncio
import base64
import io
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image

from owela import InboundMessage

from .. import audio, contributions, instrument, pii, ratelimit
from ..broadcast import opt_outs as broadcast_opt_outs
from ..config import settings
from ..filters import InvalidMsisdn, is_allowed, normalize
from ..memory import SessionStore, long_term as mem, short_term as memory
from ..runtime import build_shared_components, build_whatsapp_runtime
from ..stats import analyses as stats_analyses
from ..stats.api import router as stats_router
from ..summary import maybe_summarize
from ..delivery_log import record_status as record_delivery_status
from ..whatsapp import (
    download_media,
    extract_messages,
    extract_statuses,
    mark_as_read,
    send_text,
    verify_signature,
)
from .chat import build_router as build_chat_router
from .learn import build_router as build_learn_router
from ..learning import db as learning_db

# Single per-process Owela agent. Built lazily in lifespan so the
# embedded mem0/qdrant + whisper warmup logs land before this prints
# "runtime ready".
agent = None    # set during lifespan startup
chat_session_store = None    # SessionStore — set during lifespan startup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("ongiini")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    # Eagerly initialise the mem0 client + embedding model on startup
    # so the first real message doesn't pay the ~10s cold-load cost.
    # mem.warmup() is no-op-on-failure; if it fails, search/add calls
    # will retry the lazy init on demand.
    log.info("warming mem0 (loading embedding model)…")
    await asyncio.to_thread(mem.warmup)
    log.info("mem0 ready")
    # Whisper init downloads ~1GB on first container start and takes a
    # few seconds even when cached. Doing it now keeps the first real
    # voice note from paying the cold cost on top of transcription.
    log.info("warming faster-whisper…")
    await asyncio.to_thread(audio.warmup)
    log.info("faster-whisper ready")
    # Create the contributions sqlite if it doesn't exist yet. Idempotent
    # — no-op when tables are already in place. Soft-warn (don't crash
    # startup) if CONTRIBUTIONS_HASH_SALT is missing; the tool itself
    # will surface a clean error when first invoked.
    try:
        contributions.warmup()
        if not settings.contributions_hash_salt:
            log.warning(
                "CONTRIBUTIONS_HASH_SALT is not set — contribute_translation "
                "tool will refuse to write until it's configured."
            )
    except Exception:
        log.exception("contributions warmup failed; tool will be unavailable")
    # Broadcast opt-outs sqlite. Same shape as contributions — soft-warn
    # on warmup failure, don't crash startup. Shares the contributions
    # hash salt so opting out is deterministic across restarts.
    try:
        broadcast_opt_outs.warmup()
    except Exception:
        log.exception(
            "broadcast opt-outs warmup failed; STOP keyword + broadcast "
            "exclusion will be unavailable"
        )
    # Build all transport-agnostic Owela components ONCE here, then
    # compose into:
    #   - one WhatsApp Runtime (singleton, used for every webhook turn)
    #   - the chat router (which builds per-request Runtimes via
    #     build_chat_runtime with this same `shared` reference, so we
    #     don't pay the vLLM-client / classifier / planner construction
    #     cost twice)
    from owela import Agent
    shared = build_shared_components()
    agent = Agent(build_whatsapp_runtime(shared))

    # Singleton SessionStore for the chat endpoint. Bounded by the
    # max_sessions config (LRU eviction) so memory stays predictable
    # under load. The store is process-local — no Redis, no disk —
    # matching the "browser session only" promise.
    global chat_session_store
    chat_session_store = SessionStore(
        max_sessions=settings.chat_max_sessions,
        ttl_minutes=settings.chat_session_ttl_min,
    )

    # Mount the chat endpoint. CORS is handled by the parent app's
    # CORSMiddleware (declared at module scope above) — it's the first
    # middleware in the stack and sees every OPTIONS preflight, so
    # adding a second per-route CORS layer wouldn't change behaviour
    # for preflights and would only add an unneeded inner middleware
    # hop on the actual POST.
    # ── learn.ongiini.ai surface (must register BEFORE chat mount) ────
    # Starlette matches routes in registration order. The chat sub-app
    # mounts at "/v1" and so catches /v1/* greedily — registering the
    # learn router on the parent app first ensures /v1/learn/* routes
    # are matched before the chat catch-all sub-app.
    if settings.learn_enabled:
        if not settings.learn_token_secret:
            # Magic-link issuance (Phase 2) requires the secret, but the
            # cold-visit flow doesn't — operate degraded rather than
            # refusing to start. Log loudly so the deployer notices.
            log.warning(
                "ONGIINI_LEARN_TOKEN_SECRET is not set — magic-link "
                "tokens will fail to verify. Cold-visit learn.ongiini.ai "
                "still works."
            )
        try:
            learning_db.warmup()
            # Pass the per-pair skill renderer directly — it composes
            # the skill text from the language-agnostic core template +
            # the target-language anchor file on every turn so each
            # learner sees the right prompts for their (source, target)
            # pair.
            from ..learning.skill_renderer import render_skill_for_pair
            learn_router = build_learn_router(
                model=shared.model,
                skill_renderer=render_skill_for_pair,
            )
            app.include_router(learn_router, prefix="/v1/learn")
            log.info("learn endpoint enabled at /v1/learn")
        except Exception as exc:                            # noqa: BLE001
            # Don't bring the webhook down if the learn surface can't
            # warm up — same soft-fail discipline contributions uses.
            log.warning("learn endpoint warmup failed: %s", exc)

    if settings.chat_enabled:
        chat_sub = FastAPI(title="Ongiini Chat", openapi_url=None)
        # build_chat_router returns an APIRouter with no prefix; the
        # sub-app is mounted at "/v1" so chat routes resolve as
        # /v1/chat and /v1/chat/clear at the parent app's path layer.
        chat_sub.include_router(
            build_chat_router(
                store=chat_session_store,
                shared=shared,
                pii_sanitiser=pii.sanitize,
                resize_image=_resize_for_gemma4,
            )
        )
        app.mount("/v1", chat_sub)
        log.info("chat endpoint enabled at /v1/chat")
    else:
        log.info("chat endpoint DISABLED via ONGIINI_CHAT_ENABLED=false")
    # Kick off the LLM-driven qualitative-analysis loop (topics, roles).
    # Runs in the background; never blocks message handling. Pauses
    # between passes; one-shot failures are caught inside.
    stats_task = asyncio.create_task(
        stats_analyses.run_forever(), name="stats-analyses"
    )
    log.info("qualitative analysis background loop scheduled")
    # Diagnostic resource-snapshot logger — per-minute thread/fd/asyncio
    # task counts + RSS memory. Added 2026-05-24 after the webhook
    # accumulated 11,154 OS threads over 10 hours and wedged. We had
    # no historical resource data; this fills that gap so the next
    # leak (if any) leaves a trail.
    snapshot_task = asyncio.create_task(
        instrument.snapshot_loop(interval_s=60), name="resource-snapshot"
    )
    log.info("resource-snapshot loop scheduled (interval=60s)")
    # Prometheus exporter on a dedicated port. The container exposes
    # 9101 only to 127.0.0.1 on the host (via docker-compose), so the
    # /metrics endpoint never leaks through the public Cloudflare-fronted
    # webhook port.
    instrument.start_metrics_server(port=9101)
    try:
        yield
    finally:
        for t in (stats_task, snapshot_task):
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass


app = FastAPI(title="Ongiini Webhook", lifespan=lifespan)

# CORS for the public surface. Starlette's CORSMiddleware sits on the
# parent app and is the FIRST gatekeeper for every request, including
# OPTIONS preflights destined for sub-app routes — so the parent's
# allow_origins + allow_methods must cover EVERY origin × method
# combination the public surface needs:
#
#  - `https://ongiini.ai` / `https://www.ongiini.ai` need GET (for the
#    website's /status poll + Pages-Functions /api/stats proxy).
#  - `https://chat.ongiini.ai` (and the localhost dev variants from
#    settings.chat_allowed_origins) need POST + OPTIONS for /v1/chat
#    and /v1/chat/clear.
#
# Widening allow_methods to GET+POST+OPTIONS doesn't expose the
# webhook or admin endpoints because the WhatsApp webhook is gated by
# its own signature-verify (cross-origin POSTs without a valid Meta
# signature get 403), and /status / /api/stats have no POST routes —
# a stray POST would return 405. The webhook isn't even in any CORS
# origin so the browser couldn't reach it from a third-party page
# even if it tried.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ongiini.ai",
        "https://www.ongiini.ai",
    ] + settings.chat_allowed_origins + settings.learn_allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)

# Transparency-reporting endpoint (/stats.json). Same FastAPI app, same
# port; routed through Cloudflare Pages Function on the page side so the
# browser sees same-origin /api/stats.
app.include_router(stats_router)

NON_NAMIBIA_REPLY = (
    "Hi! Ongiini is currently only available for users in Namibia (+264 numbers). "
    "We're working on expanding — stay tuned! 🇳🇦"
)

# Sent when an image message couldn't be downloaded from Meta (expired URL,
# auth failure, transient 5xx). Kept short on purpose — the user just
# resends. We deliberately do NOT include the Meta error in the reply.
IMAGE_FETCH_FAILED_REPLY = (
    "I couldn't load the image you sent — could you try again? "
    "If it keeps failing, a text description of what you're looking at also works."
)

# Largest image payload we'll accept from Meta. WhatsApp Cloud API's own
# server-side cap is 5 MB; anything larger is either a relayed file we
# don't want to process or a misbehaving client. The Bearer-authed
# download URL doesn't expose Content-Length pre-flight, so this is a
# post-download guard.
_IMAGE_MAX_BYTES = 8 * 1024 * 1024

# Largest audio payload we'll accept. WhatsApp voice notes are usually
# 50-300 KB for typical durations; files can be larger. 16 MB covers
# realistic uploads and refuses anything looking like a transcription
# DoS. The transcribe() function additionally caps by audio DURATION
# (90s) — that's the real ceiling on compute, this is just an early
# size-based reject.
_AUDIO_MAX_BYTES = 16 * 1024 * 1024


AUDIO_FETCH_FAILED_REPLY = (
    "I couldn't load the voice note you sent — could you try again? "
    "Typing the message out also works if it keeps failing."
)
AUDIO_TRANSCRIBE_FAILED_REPLY = (
    "I tried to listen to your voice note but couldn't make it out. "
    "Could you try again, or type your message instead?"
)
AUDIO_TOO_LONG_REPLY = (
    "Your voice note is a bit long for me to listen to in one go — "
    "could you split it up or type the question instead?"
)

# Gemma 4's vision tower requires BOTH image dimensions to be exact
# multiples of 48 (patch 16 × pool kernel 3). Anything off-grid crashes
# `_avg_pool_by_positions` with a CUDA "operation not permitted" inside
# the pooler. We resize every inbound image to clean 48-aligned bounds
# BEFORE sending to vLLM. Min 336×192 keeps quality usable; max 896×896
# matches the model's effective input budget at max_soft_tokens=1120.
_GEMMA4_PATCH_GRID = 48
_GEMMA4_MIN_W = 336
_GEMMA4_MIN_H = 192
_GEMMA4_MAX_DIM = 896

# Decompression-bomb cap on inbound images. PIL allocates buffers
# based on the declared dimensions in the file header — a malicious
# upload can claim gigapixel dims and OOM the process during .convert().
# 50 megapixels comfortably covers any legitimate phone-camera shot
# (modern phones top out around 48 MP); anything bigger gets refused
# with a DecompressionBombError that _resize_for_gemma4 catches.
_IMAGE_MAX_PIXELS = 50_000_000
Image.MAX_IMAGE_PIXELS = _IMAGE_MAX_PIXELS


# Words / phrases in EN + AF that signal the user wants to do an
# ADMIN action against their own data — deletion, memory inspection, or
# token-balance check. These can't be served via the image path because
# we strip `tools=` from image-bearing calls (vLLM #41452 workaround),
# so the model can't fire delete_my_data / whats_in_my_memory /
# my_token_usage even if the caption clearly asks for it. If the caption
# matches, we IGNORE the image and route the caption text through the
# normal text handler so the right tool fires.
# Caption + image admin-intent detection used to live here as an
# ``_ADMIN_INTENT_RE`` regex pre-router. Removed 2026-05-29 — Ongiini
# is an LLM app, and routing an image+caption to a different handler
# based on regex was a brittle bypass of the model's own judgement.
# Vision-aware Gemma sees both the image and the caption; when the
# caption is clearly an admin command ("delete my data") the model
# fires the appropriate tool regardless of the image. The extra
# image-tokens spent on the "admin-with-attached-image" edge case
# (rare) are worth not maintaining a regex that drifts out of sync
# with the actual admin-tool surface.


def _resize_for_gemma4(image_bytes: bytes) -> bytes:
    """Snap an image's W and H to multiples of 48, clamped to Gemma 4's
    supported input range, and re-encode as JPEG.

    Returns the original bytes unchanged if PIL can't open them — the
    caller will pass them downstream where vLLM may or may not cope.
    Never raises.

    Decompression-bomb guard: a malicious upload can declare absurd
    dimensions in the header (gigapixels) and crash PIL with an
    OutOfMemory or runaway CPU before the resize ever runs. Set a
    hard cap on pixel count via Image.MAX_IMAGE_PIXELS so PIL refuses
    to load anything past it. The cap is generous enough for any
    legitimate phone-camera shot (24 MP ~= 24 million pixels) but
    well under the gigabyte-allocation threshold.
    """
    # MAX_IMAGE_PIXELS is set at module level (see _IMAGE_MAX_PIXELS) so
    # the guard applies on every PIL.Image.open invocation in the
    # process — including the WhatsApp media path and the chat upload
    # path.
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")   # Gemma 4 rejects palette / 1-bit PNGs
    except Image.DecompressionBombError:
        log.warning("inbound image exceeded MAX_IMAGE_PIXELS; rejecting")
        return image_bytes
    except Exception:
        log.warning("PIL couldn't open inbound image; passing through raw")
        return image_bytes

    w, h = img.size

    # 1) clamp to max box, preserving aspect ratio
    if max(w, h) > _GEMMA4_MAX_DIM:
        scale = _GEMMA4_MAX_DIM / max(w, h)
        w, h = int(w * scale), int(h * scale)

    # 2) snap down to multiples of 48
    w = max(_GEMMA4_MIN_W, (w // _GEMMA4_PATCH_GRID) * _GEMMA4_PATCH_GRID)
    h = max(_GEMMA4_MIN_H, (h // _GEMMA4_PATCH_GRID) * _GEMMA4_PATCH_GRID)

    img = img.resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/status")
async def status() -> JSONResponse:
    """Public status probe consumed by the landing page.

    Three states:
      online    — webhook + vLLM both reachable
      degraded  — webhook reachable, vLLM is not (model down or warming)
      (network error / 5xx from this endpoint itself means 'offline' to the
      caller — they decide that, we never return it)
    """
    try:
        async with httpx.AsyncClient(timeout=2.5) as c:
            r = await c.get(f"{settings.vllm_base_url.rstrip('/')}/models")
        if r.status_code == 200:
            return JSONResponse({"status": "online"})
    except Exception:
        pass
    return JSONResponse({"status": "degraded"}, status_code=200)


@app.get("/whatsapp")
async def verify(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge", "")
    if mode == "subscribe" and token and token == settings.whatsapp_verify_token:
        return PlainTextResponse(challenge)
    return Response(status_code=403)


async def _run_handler(coro, sender: str, kind: str) -> None:
    """Wrap an inbound-message handler with exception logging so it can
    be scheduled as a fire-and-forget background task. The webhook POST
    returns 200 OK to Meta synchronously (within milliseconds); the
    actual processing — router, search, Gemma, send — runs in the
    background and never delays Meta's webhook acknowledgement.
    """
    try:
        await coro
    except Exception:
        log.exception("Failed to handle %s message from %s", kind, sender)


# Module-level strong-reference set for in-flight tasks. asyncio's docs
# explicitly warn that tasks held only by weakrefs can be garbage-collected
# mid-flight on a quiet event loop. Keeping a reference here defends against
# that and silences the "Task was destroyed but it is pending" RuntimeWarning.
# discard() runs on task completion via add_done_callback.
_in_flight: set[asyncio.Task] = set()


def _spawn(coro, sender: str, kind: str) -> asyncio.Task:
    """Schedule a coroutine as a fire-and-forget task, wrapped with
    exception logging and held in `_in_flight` so the loop's GC can't
    drop it before completion."""
    task = asyncio.create_task(_run_handler(coro, sender, kind))
    _in_flight.add(task)
    task.add_done_callback(_in_flight.discard)
    return task


# ── Inbound message-id dedup ─────────────────────────────────────────
# Fire-and-forget acks the webhook in milliseconds, BUT if Meta retries
# a payload (because our 200 was delayed by network, or because the user's
# WhatsApp client itself fired the same wamid twice), the same message.id
# can legitimately arrive at our webhook twice within seconds. Without
# dedup, each arrival schedules its own _run_handler task → user gets
# duplicate replies. The per-user lock inside handle_message serialises
# them but does NOT deduplicate them.
#
# Bounded TTL set: 10-minute TTL (well past Meta's typical 1-3 retry
# window), capped at 1024 entries to bound memory under attack scenarios.
_SEEN_MSG_IDS: dict[str, float] = {}
_SEEN_TTL_S = 600.0
_SEEN_MAX_ENTRIES = 1024


def _is_duplicate_msg_id(msg_id: str) -> bool:
    """Return True if this message_id was processed within the last
    _SEEN_TTL_S seconds. Always records the id (refreshes its timestamp)
    so the most recently seen messages are also the most recently
    refreshed — which keeps the eviction policy LRU-shaped.
    """
    if not msg_id:
        return False
    now = time.monotonic()
    # Opportunistic eviction when the set grows past cap.
    if len(_SEEN_MSG_IDS) > _SEEN_MAX_ENTRIES:
        cutoff = now - _SEEN_TTL_S
        for k in list(_SEEN_MSG_IDS.keys()):
            if _SEEN_MSG_IDS[k] < cutoff:
                del _SEEN_MSG_IDS[k]
    prev = _SEEN_MSG_IDS.get(msg_id)
    _SEEN_MSG_IDS[msg_id] = now
    return prev is not None and (now - prev) < _SEEN_TTL_S


@app.post("/whatsapp")
async def receive(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None),
):
    raw_body = await request.body()

    # Reject forged webhook posts. Meta signs every body with the App Secret;
    # an attacker without the secret can't produce a valid signature.
    if not verify_signature(raw_body, x_hub_signature_256):
        log.warning("rejected webhook POST with invalid/missing signature")
        return Response(status_code=403)

    try:
        payload = await request.json()
    except Exception:
        log.warning("rejected webhook POST with non-JSON body")
        return Response(status_code=400)

    # Top-level safety net: if anything in the dispatch / scheduling
    # block below raises (bug, typo, library mismatch), we MUST still
    # return 200 to Meta so they don't retry the same broken payload
    # over and over. The original message is logged on our side; better
    # to drop one user message than to lock Meta in a retry loop against
    # our broken webhook.
    try:
        # Delivery status events ride the same `messages` field as inbound
        # messages (sent/delivered/read/failed). Capture before we early-
        # return on dedup, so failed-delivery diagnostics survive even
        # when Meta retries with the same status block.
        try:
            for st in extract_statuses(payload):
                record_delivery_status(st)
        except Exception:
            log.exception("extract_statuses failed; continuing with message dispatch")

        messages = extract_messages(payload)

        # Drop duplicates BEFORE doing anything. Meta may legitimately
        # re-deliver the same message.id (network blip, our 200 was slow,
        # client retransmitted) — without this filter, every retry would
        # generate a second reply.
        unique_messages = []
        for m in messages:
            mid = m.get("id", "")
            if _is_duplicate_msg_id(mid):
                log.info("dropping duplicate message_id %s from %s", mid, m.get("from", "?"))
                continue
            unique_messages.append(m)
        messages = unique_messages

        # Fire read receipt + typing indicator immediately for each
        # inbound message. Wrapped via _spawn so failures are logged
        # and tasks are held in _in_flight to survive GC.
        for m in messages:
            mid = m.get("id", "")
            if mid:
                _spawn(mark_as_read(mid), m.get("from", ""), "mark_as_read")

        # Schedule each handler as a fire-and-forget task and return
        # 200 to Meta IMMEDIATELY. Meta retries any webhook POST it
        # doesn't get a 200 on within ~5 seconds — and our full pipeline
        # (router classifier + Tavily search + 2 Gemma vLLM hops with
        # reasoning + WhatsApp send) easily takes 8-15s on a search-
        # grounded reply. Without this pattern, the user gets duplicate
        # replies because Meta keeps retrying while we're still working.
        # The per-user lock inside handle_message serialises concurrent
        # work for the same msisdn, so even if Meta DOES deliver a
        # retry before the original task finishes, they queue instead
        # of trampling state.
        for m in messages:
            sender = m["from"]
            kind = m.get("type", "text")

            if kind == "text":
                text = (m.get("text") or "").strip()
                if not text:
                    continue
                # Defensive: WhatsApp's own text limit is 4096 chars.
                # Anything larger is either a bug or an abuse attempt
                # — drop it without spending tokens.
                if len(text) > settings.message_max_chars:
                    log.warning(
                        "dropping oversize message from %s (%d chars)",
                        sender, len(text),
                    )
                    continue
                # `referral` is attached by extract_messages when the
                # inbound carries a click-to-WhatsApp ad referral block.
                # Threaded through to handle_message so the welcome
                # experiment can detect FB-ad arrivals.
                _spawn(
                    handle_message(sender, text, referral=m.get("referral")),
                    sender, "text",
                )

            elif kind == "image":
                media_id = m.get("media_id") or ""
                if not media_id:
                    continue
                caption = (m.get("caption") or "").strip()
                # WhatsApp's image caption limit is 1024 chars; same
                # behaviour as oversize text for consistency.
                if len(caption) > settings.message_max_chars:
                    log.warning(
                        "dropping image with oversize caption from %s (%d chars)",
                        sender, len(caption),
                    )
                    continue
                _spawn(
                    handle_image_message(
                        sender=sender,
                        media_id=media_id,
                        mime_type=m.get("mime_type") or "image/jpeg",
                        caption=caption,
                    ),
                    sender,
                    "image",
                )

            elif kind == "audio":
                media_id = m.get("media_id") or ""
                if not media_id:
                    continue
                _spawn(
                    handle_audio_message(
                        sender=sender,
                        media_id=media_id,
                        mime_type=m.get("mime_type") or "audio/ogg",
                    ),
                    sender,
                    "audio",
                )
    except Exception:
        # Always return 200 below — see the safety-net comment above.
        log.exception("webhook dispatch failed — returning 200 to Meta anyway")

    # WhatsApp expects a fast 200 OK acknowledgement.
    return {"status": "ok"}



async def handle_image_message(
    sender: str, media_id: str, mime_type: str, caption: str
) -> None:
    """Inbound WhatsApp image — same overall flow as text messages
    (normalise → allow check → rate limit → load history → agent.handle →
    Owela hooks persist + bill), with an extra step to pull the bytes
    from Meta and an OpenAI-style multipart user content payload.

    The image itself is NOT persisted. The OngiiniMemoryRecordingHook
    routes image-bearing turns through ``record_image_turn`` which
    stores only the placeholder + caption + reply (never the bytes).
    """
    try:
        msisdn = normalize(sender)
    except InvalidMsisdn as exc:
        log.warning("rejected image message with invalid sender field: %s", exc)
        return

    if not is_allowed(msisdn):
        log.info("blocked non-Namibian sender %s (image)", msisdn)
        await send_text(sender, NON_NAMIBIA_REPLY)
        return

    allowed, reason = ratelimit.check(msisdn)
    if not allowed:
        log.info("rate-limited %s: %s (image)", msisdn, reason)
        await send_text(sender, reason)
        return

    media = await download_media(media_id)
    if media is None:
        await send_text(sender, IMAGE_FETCH_FAILED_REPLY)
        return

    image_bytes, actual_mime = media
    if len(image_bytes) > _IMAGE_MAX_BYTES:
        log.warning(
            "dropping oversize image from %s (%d bytes > %d)",
            sender, len(image_bytes), _IMAGE_MAX_BYTES,
        )
        await send_text(sender, IMAGE_FETCH_FAILED_REPLY)
        return

    # Preprocess to Gemma 4's 48-multiple grid BEFORE sending to vLLM.
    # Off-grid dimensions crash the vision pooler with cudaErrorNotPermitted
    # (vLLM/transformers Gemma 4 #45482). Output is always re-encoded as
    # JPEG so the data URL mime is predictable from here on.
    image_bytes = _resize_for_gemma4(image_bytes)
    data_url = (
        f"data:image/jpeg;base64,"
        f"{base64.standard_b64encode(image_bytes).decode('ascii')}"
    )

    # OpenAI-style multipart content. When the caller didn't provide a
    # caption we steer the model with a tiny default — Gemma 4 with no
    # text prompt and only an image will sometimes just describe the
    # image without next-step or context awareness.
    #
    # PII scrub: the caption text is treated like any other user message
    # — emails / IDs / IBANs / card numbers go through the same redaction
    # the text path uses, BEFORE it reaches the LLM, mem0, or the on-disk
    # placeholder. This closes a gap where the raw caption could leak
    # PII into long-term memory.
    safe_caption = pii.sanitize(caption) if caption else ""
    user_text_part = safe_caption or (
        "I just sent you a photo. Have a look and tell me what you see — "
        "if there's something specific worth pointing out, mention it."
    )
    user_content = [
        {"type": "text", "text": user_text_part},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    async with memory.lock_for(msisdn):
        history = memory.load(msisdn)
        history = await maybe_summarize(history, msisdn=msisdn)

        # msg_id="" because main.py already fired mark_as_read on webhook
        # receipt — the executor's transport.acknowledge would no-op.
        msg = InboundMessage(
            user_id=msisdn,
            msg_id="",
            text=safe_caption,
            content_parts=user_content,
            has_image=True,
            history=history,
        )
        await agent.handle(msg)
        # Persistence, billing, tracing all handled by the Owela hooks
        # fired inside the executor's on_turn_complete.


async def handle_audio_message(
    sender: str, media_id: str, mime_type: str
) -> None:
    """Inbound WhatsApp voice note / audio file.

    Pipeline: download bytes from Meta → faster-whisper CPU transcribe →
    route the transcript through `handle_message` so it shares the text
    path's memory writes, tool dispatch, EN/AF language redirect, and
    PII scrub. Audio bytes are NEVER stored — only the transcript flows
    further and only the transcript hits mem0 / disk.
    """
    try:
        msisdn = normalize(sender)
    except InvalidMsisdn as exc:
        log.warning("rejected audio message with invalid sender field: %s", exc)
        return

    if not is_allowed(msisdn):
        log.info("blocked non-Namibian sender %s (audio)", msisdn)
        await send_text(sender, NON_NAMIBIA_REPLY)
        return

    allowed, reason = ratelimit.check(msisdn)
    if not allowed:
        log.info("rate-limited %s: %s (audio)", msisdn, reason)
        await send_text(sender, reason)
        return

    media = await download_media(media_id)
    if media is None:
        await send_text(sender, AUDIO_FETCH_FAILED_REPLY)
        return

    audio_bytes, _actual_mime = media
    if len(audio_bytes) > _AUDIO_MAX_BYTES:
        log.warning(
            "dropping oversize audio from %s (%d bytes > %d)",
            sender, len(audio_bytes), _AUDIO_MAX_BYTES,
        )
        await send_text(sender, AUDIO_TOO_LONG_REPLY)
        return

    # Whisper is CPU-bound — run in a worker thread so we don't pin the
    # asyncio event loop for the 2-5s typical transcription window.
    transcript, lang, duration_s = await asyncio.to_thread(
        audio.transcribe, audio_bytes
    )

    if duration_s > 0 and not transcript:
        # transcribe() returns "" with a non-zero duration when the
        # audio was too long for our cap — distinguish that from the
        # generic transcription failure so the user gets the right hint.
        if duration_s > 90.0:
            await send_text(sender, AUDIO_TOO_LONG_REPLY)
        else:
            await send_text(sender, AUDIO_TRANSCRIBE_FAILED_REPLY)
        return

    if not transcript:
        await send_text(sender, AUDIO_TRANSCRIBE_FAILED_REPLY)
        return

    # Defensive: reject transcripts that somehow exceed the text-path
    # length cap. WhatsApp's own voice-note limit plus our 90s duration
    # cap makes this unlikely, but a chatty Whisper output on background
    # noise could be lengthy.
    if len(transcript) > settings.message_max_chars:
        log.warning(
            "dropping oversize transcript from %s (%d chars)",
            msisdn, len(transcript),
        )
        await send_text(sender, AUDIO_TRANSCRIBE_FAILED_REPLY)
        return

    log.info(
        "voice note from %s — %.1fs, lang=%s, %d chars",
        msisdn, duration_s, lang or "?", len(transcript),
    )

    # Route through the text handler so the transcript benefits from the
    # full pipeline: per-user lock, memory load/save, mem0 fact extraction,
    # tools dispatch, EN/AF-only redirect (the language rule in
    # SYSTEM_PROMPT fires on the transcript text itself — no separate
    # gating needed here). Reply is text-only for v1.
    #
    # `memory_prefix="[voice note]"` marks the stored turn — the LLM
    # still receives the raw transcript for THIS call, but future turns
    # (and the transparency aggregator's voice-note counter) see the
    # marker. Same pattern as `[image attached]` for images.
    await handle_message(sender, transcript, memory_prefix="[voice note]")


async def handle_message(
    sender: str,
    text: str,
    *,
    memory_prefix: str = "",
    referral: dict | None = None,
) -> None:
    """Inbound text-or-transcript handler.

    Loads history under a per-user lock, summarises if it crossed the
    rolling threshold, hands an InboundMessage to the Owela agent.
    Persistence (short-term + mem0), billing, and tracing all happen
    inside ``agent.handle`` via Owela hooks — main.py's only job is
    transport-layer concerns (signature verify, msisdn normalise, allow
    check, rate limit, lock acquisition).
    """
    try:
        msisdn = normalize(sender)
    except InvalidMsisdn as exc:
        # The sender field didn't survive validation — could be a forged
        # webhook before signature verification, a Meta-side oddity, or a
        # genuine bug. Either way: drop the message without touching disk.
        log.warning("rejected message with invalid sender field: %s", exc)
        return

    if not is_allowed(msisdn):
        log.info("blocked non-Namibian sender %s", msisdn)
        await send_text(sender, NON_NAMIBIA_REPLY)
        return

    allowed, reason = ratelimit.check(msisdn)
    if not allowed:
        log.info("rate-limited %s: %s", msisdn, reason)
        await send_text(sender, reason)
        return

    # Serialise the load → handle → persist block per-user so rapid-fire
    # messages from the same number can't race and clobber each other's
    # memory file. Different users run concurrently.
    async with memory.lock_for(msisdn):
        history = memory.load(msisdn)
        history = await maybe_summarize(history, msisdn=msisdn)

        # ``storage_text`` carries the persistence-side label (e.g.
        # "[voice note] <transcript>" for audio turns). The model sees
        # the raw ``text``; the hook sees ``storage_text`` if set.
        storage_text = (
            f"{memory_prefix} {text}".strip() if memory_prefix else ""
        )

        # raw_payload carries the FB click-to-WhatsApp referral block
        # when present — read by welcome_experiment downstream to decide
        # whether to assign a welcome variant. None for organic arrivals.
        msg = InboundMessage(
            user_id=msisdn,
            msg_id="",   # main.py already mark_as_read'd on receipt
            text=text,
            content_parts=[{"type": "text", "text": text}],
            history=history,
            storage_text=storage_text,
            raw_payload={"referral": referral} if referral else None,
        )
        await agent.handle(msg)
        # Persistence + billing + tracing handled by Owela hooks.
