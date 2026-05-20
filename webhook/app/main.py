import logging

from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import PlainTextResponse

from . import memory, ratelimit, usage
from .config import settings
from .filters import is_allowed, normalize
from .llm import respond
from .whatsapp import extract_messages, send_text, verify_signature

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("ongiini")

app = FastAPI(title="Ongiini Webhook")

NON_NAMIBIA_REPLY = (
    "Hi! Ongiini is currently only available for users in Namibia (+264 numbers). "
    "We're working on expanding — stay tuned! 🇳🇦"
)


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


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
    msisdn = normalize(sender)

    if not is_allowed(msisdn):
        log.info("blocked non-Namibian sender %s", msisdn)
        await send_text(sender, NON_NAMIBIA_REPLY)
        return

    allowed, reason = ratelimit.check(msisdn)
    if not allowed:
        log.info("rate-limited %s: %s", msisdn, reason)
        await send_text(sender, reason)
        return

    history = memory.load(msisdn)
    result = await respond(history, text, msisdn)

    await send_text(sender, result.reply)

    # When the model fires the deletion tool, leave no trace of this turn either —
    # the file is already wiped by the tool handler, and we deliberately skip the
    # history.append/save below so the deletion request itself isn't re-persisted.
    if not result.deleted_data:
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": result.reply})
        memory.save(msisdn, history)

    usage.record(msisdn, result.tokens_in, result.tokens_out, result.used_search)
