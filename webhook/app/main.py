import asyncio
import base64
import io
import logging
import re
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image

from . import audio, mem, memory, pii, ratelimit, usage
from .config import settings
from .filters import InvalidMsisdn, is_allowed, normalize
from .llm import maybe_summarize, respond
from .stats import analyses as stats_analyses
from .stats.api import router as stats_router
from .whatsapp import download_media, extract_messages, mark_as_read, send_text, verify_signature

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("ongiini")

@asynccontextmanager
async def lifespan(app: FastAPI):
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
    # Kick off the LLM-driven qualitative-analysis loop (topics, roles).
    # Runs in the background; never blocks message handling. Pauses
    # between passes; one-shot failures are caught inside.
    stats_task = asyncio.create_task(
        stats_analyses.run_forever(), name="stats-analyses"
    )
    log.info("qualitative analysis background loop scheduled")
    try:
        yield
    finally:
        stats_task.cancel()
        try:
            await stats_task
        except (asyncio.CancelledError, Exception):
            pass


app = FastAPI(title="Ongiini Webhook", lifespan=lifespan)

# Allow the public website (which may live on Cloudflare Pages) to poll
# /status from the browser. GET only, no credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ongiini.ai",
        "https://www.ongiini.ai",
    ],
    allow_methods=["GET"],
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


# Words / phrases in EN + AF that signal the user wants to do an
# ADMIN action against their own data — deletion, memory inspection, or
# token-balance check. These can't be served via the image path because
# we strip `tools=` from image-bearing calls (vLLM #41452 workaround),
# so the model can't fire delete_my_data / whats_in_my_memory /
# my_token_usage even if the caption clearly asks for it. If the caption
# matches, we IGNORE the image and route the caption text through the
# normal text handler so the right tool fires.
_ADMIN_INTENT_RE = re.compile(
    r"\b("
    r"delete\s+(?:my|all)\s+data|forget\s+(?:everything|all|me|my)|wipe\s+(?:my|all)|"
    r"vergeet\s+(?:alles|my)|wis\s+(?:my|alle)|verwyder\s+(?:my|alle)|"
    r"what\s+do\s+you\s+remember|what\s+have\s+you\s+stored|show\s+me\s+(?:my\s+)?data|"
    r"wat\s+onthou\s+jy|wat\s+het\s+(?:julle|jy)\s+gestoor|"
    r"how\s+many\s+tokens|hoeveel\s+tokens|my\s+(?:token|monthly)\s+(?:usage|limit|balance)"
    r")\b",
    re.IGNORECASE,
)


def _caption_is_admin_intent(caption: str) -> bool:
    """True iff the caption clearly asks for an admin operation that
    requires tools (delete_my_data / whats_in_my_memory / my_token_usage)."""
    return bool(caption and _ADMIN_INTENT_RE.search(caption))


def _resize_for_gemma4(image_bytes: bytes) -> bytes:
    """Snap an image's W and H to multiples of 48, clamped to Gemma 4's
    supported input range, and re-encode as JPEG.

    Returns the original bytes unchanged if PIL can't open them — the
    caller will pass them downstream where vLLM may or may not cope.
    Never raises.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")   # Gemma 4 rejects palette / 1-bit PNGs
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
                _spawn(handle_message(sender, text), sender, "text")

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
                # Admin-intent captions (delete / recall / usage) route
                # through the text handler so the proper tool fires.
                if _caption_is_admin_intent(caption):
                    log.info(
                        "image caption matched admin intent — handling as text from %s",
                        sender,
                    )
                    _spawn(handle_message(sender, caption), sender, "admin-caption")
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
    (normalise → allow check → rate limit → load history → respond →
    send → save → record usage), with an extra step to pull the bytes
    from Meta and an OpenAI-style multipart user content payload.

    The image itself is NOT persisted. Short-term memory stores a
    compact "[image] <caption>" placeholder so the model on the next
    text turn knows an image was shared, while mem0 (vision-enabled)
    extracts a durable typed fact like '[SITUATION] Shared photo of
    yellowing maize leaves' that persists across all future sessions.
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
    # placeholder. This closes a gap noted in code review where the raw
    # caption could leak PII into long-term memory.
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
        result = await respond(history, user_content, msisdn)
        await send_text(sender, result.reply)

        if not result.deleted_data:
            # Short-term memory: compact textual placeholder, NOT the
            # base64 image bytes. The model on future turns sees that
            # an image was shared (and the PII-scrubbed caption if any)
            # — enough context to continue the conversation. Durable
            # image-aware facts live in mem0 from the add_image_turn
            # call below.
            placeholder = "[image attached]"
            if safe_caption:
                placeholder += f" {safe_caption}"
            history.append(
                pii.sanitize_message({"role": "user", "content": placeholder})
            )
            history.append(
                pii.sanitize_message({"role": "assistant", "content": result.reply})
            )
            history = await maybe_summarize(history, msisdn=msisdn)
            memory.save(msisdn, history)

            # Long-term: feed mem0 a synthesised text-only version of
            # the image turn. mem0's extraction LLM never sees the raw
            # base64 bytes — its prompt is calibrated for "[image
            # attached] <caption>" + the assistant's textual description
            # of what it saw, which is far more reliable than asking
            # mem0's vision path to do another visual pass. The caption
            # is passed PII-sanitised (closes the gap that mem0 could
            # otherwise persist raw user data as a typed fact).
            await asyncio.to_thread(
                mem.add_image_turn, msisdn, safe_caption, pii.sanitize(result.reply)
            )

        usage.record(msisdn, result.tokens_in, result.tokens_out, result.used_search)


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


async def handle_message(sender: str, text: str, *, memory_prefix: str = "") -> None:
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

    # Serialize the load → respond → save block per-user so rapid-fire
    # messages from the same number can't race and clobber each other's
    # memory file. Different users run concurrently.
    async with memory.lock_for(msisdn):
        history = memory.load(msisdn)
        result = await respond(history, text, msisdn)

        await send_text(sender, result.reply)

        # When the model fires the deletion tool, leave no trace of this turn
        # either — the file is already wiped by the tool handler, and we
        # deliberately skip the history.append/save below so the deletion
        # request itself isn't re-persisted.
        if not result.deleted_data:
            # PII sanitisation happens at WRITE time: the LLM call above
            # already saw the un-redacted user text (so it could answer the
            # actual question). What lands on disk for future-turn replay
            # is the scrubbed version.
            stored_text = f"{memory_prefix} {text}".strip() if memory_prefix else text
            history.append(pii.sanitize_message({"role": "user", "content": stored_text}))
            history.append(pii.sanitize_message({"role": "assistant", "content": result.reply}))
            history = await maybe_summarize(history, msisdn=msisdn)
            memory.save(msisdn, history)

            # Long-term semantic memory: feed the just-completed turn to
            # mem0 so it can extract or update durable facts about this
            # user. Done AFTER send_text so it never blocks the live
            # reply. We pass the PII-sanitised text — mem0 should not see
            # raw emails or ID numbers any more than the disk does.
            #
            # Awaited inside the per-user lock so the next message from
            # this same user starts with fresh memory; different users
            # never block each other (their locks are independent).
            await asyncio.to_thread(
                mem.add_turn, msisdn, pii.sanitize(text), pii.sanitize(result.reply)
            )

        usage.record(msisdn, result.tokens_in, result.tokens_out, result.used_search)
