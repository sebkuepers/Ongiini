import logging

import httpx
from fastapi import FastAPI, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from . import memory, ratelimit, usage
from .config import settings
from .filters import InvalidMsisdn, is_allowed, normalize
from .llm import respond
from .whatsapp import extract_messages, send_text, verify_signature

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("ongiini")

app = FastAPI(title="Ongiini Webhook")

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
        text = (m["text"] or "").strip()
        if not text:
            continue

        # Defensive: WhatsApp's own text limit is 4096 chars. Anything larger
        # is either a bug or an abuse attempt — drop it without spending tokens.
        if len(text) > settings.message_max_chars:
            log.warning(
                "dropping oversize message from %s (%d chars)", sender, len(text)
            )
            continue

        try:
            await handle_message(sender, text)
        except Exception:
            log.exception("Failed to handle message from %s", sender)

    # WhatsApp expects a fast 200 OK acknowledgement.
    return {"status": "ok"}


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
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": result.reply})
            memory.save(msisdn, history)

        usage.record(msisdn, result.tokens_in, result.tokens_out, result.used_search)
