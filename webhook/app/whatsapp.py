import asyncio
import hashlib
import hmac
import logging

import httpx

from .config import settings

log = logging.getLogger("ongiini.whatsapp")

GRAPH_URL = "https://graph.facebook.com/v21.0"

# Retry policy for send_text. Three attempts total — first immediate, then
# 0.5s, then 2.5s after that. Covers most transient Meta 5xx / network
# blips without holding the webhook open for >3s on the failure path.
_SEND_RETRY_DELAYS_S: tuple[float, ...] = (0.0, 0.5, 2.0)


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
        # Deliberately do NOT log the body — we never log message content.
        # The eval harness in webhook/tests/eval.py is the right tool for
        # inspecting actual replies during development.
        log.warning(
            "WhatsApp not configured — would send to %s (%d chars)", to, len(body)
        )
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

    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=15.0) as client:
        for attempt, delay in enumerate(_SEND_RETRY_DELAYS_S, start=1):
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                r = await client.post(url, headers=headers, json=payload)
            except httpx.RequestError as exc:
                # Transport-level failure (DNS, connection reset, timeout).
                # Transient by nature — retry.
                last_error = exc
                log.warning(
                    "WhatsApp send transport error (attempt %d/%d): %s",
                    attempt, len(_SEND_RETRY_DELAYS_S), exc,
                )
                continue

            # 4xx is a permanent error — bad token, blocked recipient,
            # malformed body. Retrying won't help; surface immediately.
            if 400 <= r.status_code < 500:
                log.error("WhatsApp send 4xx (%s): %s", r.status_code, r.text)
                r.raise_for_status()

            if r.status_code >= 500:
                last_error = httpx.HTTPStatusError(
                    f"server error {r.status_code}", request=r.request, response=r
                )
                log.warning(
                    "WhatsApp send 5xx (attempt %d/%d, %s): %s",
                    attempt, len(_SEND_RETRY_DELAYS_S), r.status_code, r.text,
                )
                continue

            # 2xx — success.
            return

    log.error(
        "WhatsApp send failed after %d attempts: %s",
        len(_SEND_RETRY_DELAYS_S), last_error,
    )
    if last_error is not None:
        raise last_error


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
