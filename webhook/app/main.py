import logging

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse

from . import memory, usage
from .config import settings
from .filters import is_allowed, normalize
from .llm import respond
from .whatsapp import extract_messages, send_text

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


@app.get("/webhooks/whatsapp")
async def verify(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge", "")
    if mode == "subscribe" and token and token == settings.whatsapp_verify_token:
        return PlainTextResponse(challenge)
    return Response(status_code=403)


@app.post("/webhooks/whatsapp")
async def receive(request: Request):
    payload = await request.json()
    messages = extract_messages(payload)

    for m in messages:
        sender = m["from"]
        text = (m["text"] or "").strip()
        if not text:
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

    history = memory.load(msisdn)
    result = await respond(history, text)

    await send_text(sender, result.reply)

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": result.reply})
    memory.save(msisdn, history)

    usage.record(msisdn, result.tokens_in, result.tokens_out, result.used_search)
