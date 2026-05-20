import hashlib
import hmac
import logging

import httpx

from .config import settings

log = logging.getLogger("ongiini.whatsapp")

GRAPH_URL = "https://graph.facebook.com/v21.0"


def verify_signature(body: bytes, signature_header: str | None) -> bool:
    """Verify Meta's X-Hub-Signature-256 against the App Secret.

    When WHATSAPP_APP_SECRET is unset we accept the request and log a
    warning — useful for local dev. In production the env var MUST be set.
    """
    if not settings.whatsapp_app_secret:
        log.warning("WHATSAPP_APP_SECRET not set — skipping webhook signature check")
        return True

    if not signature_header or not signature_header.startswith("sha256="):
        return False

    expected = signature_header[len("sha256="):]
    digest = hmac.new(
        settings.whatsapp_app_secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, digest)


async def send_text(to: str, body: str) -> None:
    if not settings.whatsapp_token or not settings.whatsapp_phone_id:
        log.warning("WhatsApp not configured; would send to %s: %s", to, body)
        return

    url = f"{GRAPH_URL}/{settings.whatsapp_phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": body[:4096]},
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            log.error("WhatsApp send failed (%s): %s", r.status_code, r.text)
        r.raise_for_status()


def extract_messages(payload: dict) -> list[dict]:
    """Yield simplified message dicts from a WhatsApp webhook payload.

    Each item: {"from": msisdn, "text": str, "id": message_id}
    """
    out: list[dict] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for msg in value.get("messages", []) or []:
                if msg.get("type") != "text":
                    continue
                text = (msg.get("text") or {}).get("body", "")
                out.append(
                    {
                        "from": msg.get("from", ""),
                        "text": text,
                        "id": msg.get("id", ""),
                    }
                )
    return out
