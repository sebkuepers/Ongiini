import asyncio
import base64
import io
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image

from . import mem, memory, pii, ratelimit, usage
from .config import settings
from .filters import InvalidMsisdn, is_allowed, normalize
from .llm import maybe_summarize, respond
from .whatsapp import download_media, extract_messages, send_text, verify_signature

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
    yield


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

    messages = extract_messages(payload)

    for m in messages:
        sender = m["from"]
        kind = m.get("type", "text")

        if kind == "text":
            text = (m.get("text") or "").strip()
            if not text:
                continue
            # Defensive: WhatsApp's own text limit is 4096 chars. Anything
            # larger is either a bug or an abuse attempt — drop it without
            # spending tokens.
            if len(text) > settings.message_max_chars:
                log.warning(
                    "dropping oversize message from %s (%d chars)", sender, len(text)
                )
                continue
            try:
                await handle_message(sender, text)
            except Exception:
                log.exception("Failed to handle message from %s", sender)

        elif kind == "image":
            media_id = m.get("media_id") or ""
            if not media_id:
                continue
            try:
                await handle_image_message(
                    sender=sender,
                    media_id=media_id,
                    mime_type=m.get("mime_type") or "image/jpeg",
                    caption=(m.get("caption") or "").strip(),
                )
            except Exception:
                log.exception("Failed to handle image message from %s", sender)

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
    user_text_part = caption or (
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
            # an image was shared (and the caption if any) — enough
            # context to continue the conversation. Durable image-aware
            # facts live in mem0 from the add_turn call below.
            placeholder = "[image attached]"
            if caption:
                placeholder += f" {caption}"
            history.append(
                pii.sanitize_message({"role": "user", "content": placeholder})
            )
            history.append(
                pii.sanitize_message({"role": "assistant", "content": result.reply})
            )
            history = await maybe_summarize(history)
            memory.save(msisdn, history)

            # Long-term: full multipart content to mem0 so the
            # vision-enabled extractor can describe the image and store
            # typed facts about what was shown.
            await asyncio.to_thread(
                mem.add_turn, msisdn, user_content, pii.sanitize(result.reply)
            )

        usage.record(msisdn, result.tokens_in, result.tokens_out, result.used_search)


async def handle_message(sender: str, text: str) -> None:
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
            history.append(pii.sanitize_message({"role": "user", "content": text}))
            history.append(pii.sanitize_message({"role": "assistant", "content": result.reply}))
            history = await maybe_summarize(history)
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
