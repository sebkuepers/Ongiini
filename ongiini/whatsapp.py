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
        # The eval harness in ongiini/tests/eval.py is the right tool for
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


async def send_template(
    to: str,
    template_name: str,
    language_code: str,
    body_params: list[str],
    button_url_param: str | None = None,
) -> dict:
    """Send a pre-approved WhatsApp template message via the Cloud API.

    Distinct from send_text: outbound-only path used for proactive
    broadcasts. The recipient may have no active 24h session — only
    pre-approved MARKETING / UTILITY / AUTHENTICATION templates are
    permitted in that case.

    Args:
        to: Recipient msisdn (with or without leading +).
        template_name: Template name as registered in Meta Business Manager.
        language_code: e.g. 'en', 'af'. Must match an approved variant
            of `template_name`.
        body_params: One string per {{N}} placeholder in the template
            body, in order.
        button_url_param: If the template's button is a URL button with
            a dynamic suffix, pass the suffix value here. None means no
            button params (static URL button, or no button).

    Returns the Meta API response dict (contains `messages[0].id` for
    delivery tracking). Raises on permanent failure after retries.
    """
    if not settings.whatsapp_token or not settings.whatsapp_phone_id:
        log.warning(
            "WhatsApp not configured — would send template %s to %s",
            template_name, to,
        )
        return {"skipped": True}

    components: list[dict] = []
    if body_params:
        components.append({
            "type": "body",
            "parameters": [{"type": "text", "text": p} for p in body_params],
        })
    if button_url_param is not None:
        # Dynamic URL suffix button. Meta expects sub_type='url',
        # index='0' (only one button in our template), parameters=[text].
        components.append({
            "type": "button",
            "sub_type": "url",
            "index": "0",
            "parameters": [{"type": "text", "text": button_url_param}],
        })

    url = f"{GRAPH_URL}/{settings.whatsapp_phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": language_code},
            "components": components,
        },
    }

    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=15.0) as client:
        for attempt, delay in enumerate(_SEND_RETRY_DELAYS_S, start=1):
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                r = await client.post(url, headers=headers, json=payload)
            except httpx.RequestError as exc:
                last_error = exc
                log.warning(
                    "WhatsApp template send transport error (attempt %d/%d): %s",
                    attempt, len(_SEND_RETRY_DELAYS_S), exc,
                )
                continue
            if 400 <= r.status_code < 500:
                # Permanent errors: bad token, recipient never opted in,
                # template not approved, etc. Surface immediately so the
                # broadcaster can log + skip rather than waste retries.
                log.error(
                    "WhatsApp template send 4xx (%s): %s",
                    r.status_code, r.text,
                )
                r.raise_for_status()
            if r.status_code >= 500:
                last_error = httpx.HTTPStatusError(
                    f"server error {r.status_code}", request=r.request, response=r
                )
                log.warning(
                    "WhatsApp template send 5xx (attempt %d/%d, %s): %s",
                    attempt, len(_SEND_RETRY_DELAYS_S), r.status_code, r.text,
                )
                continue
            return r.json()

    log.error(
        "WhatsApp template send failed after %d attempts: %s",
        len(_SEND_RETRY_DELAYS_S), last_error,
    )
    if last_error is not None:
        raise last_error
    return {}


async def mark_as_read(message_id: str, with_typing: bool = True) -> None:
    """Tell Meta the user's message has been read, optionally with a
    typing indicator to show the bot is composing a reply.

    Read receipt flips the user's checkmarks from grey ✓✓ (delivered) to
    blue ✓✓ (read). The typing indicator adds an animated "..." or
    "typing" status under the bot's name in the user's chat — perfect
    for the 5-15s gap between read receipt and the actual reply landing
    (after router classification, web_search, Gemma 4 reasoning, etc.).
    WhatsApp auto-dismisses the typing indicator the moment we send the
    reply (or after ~25s, whichever comes first), so we don't need to
    actively "stop typing".

    Fired immediately upon webhook receipt — before any processing —
    so the UX feedback ("read + composing") appears within a second of
    the user sending. Soft-fail: a billing-blocked WABA or transient
    Meta 5xx shouldn't break the reply path.
    """
    if not settings.whatsapp_token or not settings.whatsapp_phone_id or not message_id:
        return
    url = f"{GRAPH_URL}/{settings.whatsapp_phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
    }
    if with_typing:
        # Meta's API: typing indicator goes in the same /messages call as
        # the read receipt, requires status=read alongside it.
        payload["typing_indicator"] = {"type": "text"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.post(url, headers=headers, json=payload)
            if r.status_code >= 300:
                log.warning(
                    "mark_as_read non-2xx for %s: %s %s",
                    message_id, r.status_code, r.text[:200],
                )
    except Exception as exc:
        log.warning("mark_as_read failed for %s: %s", message_id, exc)


def extract_messages(payload: dict) -> list[dict]:
    """Yield simplified message dicts from a WhatsApp webhook payload.

    Each item is one of:
      {"from": msisdn, "type": "text",  "text": str,                       "id": ...}
      {"from": msisdn, "type": "image", "media_id": str, "mime_type": str,
                                        "caption": str,                    "id": ...}

    Non-text, non-image types are dropped silently — we'll add audio /
    document handling separately when the model is ready for them.
    """
    out: list[dict] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for msg in value.get("messages", []) or []:
                msg_type = msg.get("type")
                if msg_type == "text":
                    text = (msg.get("text") or {}).get("body", "")
                    out.append(
                        {
                            "from": msg.get("from", ""),
                            "type": "text",
                            "text": text,
                            "id": msg.get("id", ""),
                        }
                    )
                elif msg_type == "image":
                    image = msg.get("image") or {}
                    out.append(
                        {
                            "from": msg.get("from", ""),
                            "type": "image",
                            "media_id": image.get("id", ""),
                            "mime_type": image.get("mime_type", "image/jpeg"),
                            "caption": image.get("caption", "") or "",
                            "id": msg.get("id", ""),
                        }
                    )
                elif msg_type in ("audio", "voice"):
                    # WhatsApp sends both `audio` (file upload) and `voice`
                    # (in-app voice note) — same payload shape with a
                    # `voice: true` flag on the latter. faster-whisper
                    # handles both via ffmpeg.
                    audio = msg.get("audio") or {}
                    out.append(
                        {
                            "from": msg.get("from", ""),
                            "type": "audio",
                            "media_id": audio.get("id", ""),
                            "mime_type": audio.get("mime_type", "audio/ogg"),
                            "voice": bool(audio.get("voice", False)),
                            "id": msg.get("id", ""),
                        }
                    )
    return out


async def download_media(media_id: str) -> tuple[bytes, str] | None:
    """Two-step Meta media download: resolve the short-lived URL, then fetch.

    Returns (bytes, mime_type) on success, or None if anything goes wrong.
    Never raises — caller decides how to respond when media is unavailable
    (typically: send a "couldn't load your image" reply to the user).

    Meta's media API requires the WHATSAPP_TOKEN even on the download URL
    (the URL points at lookaside.fbsbx.com and rejects unauthenticated
    requests), so we re-use the same auth header for both calls.
    """
    if not settings.whatsapp_token:
        log.warning("WhatsApp token not configured — can't download media %s", media_id)
        return None

    headers = {"Authorization": f"Bearer {settings.whatsapp_token}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: resolve the short-lived signed URL for this media id.
        try:
            r = await client.get(f"{GRAPH_URL}/{media_id}", headers=headers)
        except httpx.RequestError as exc:
            log.warning("Meta media metadata fetch failed for %s: %s", media_id, exc)
            return None
        if r.status_code >= 400:
            log.warning(
                "Meta media metadata %s for %s: %s", r.status_code, media_id, r.text
            )
            return None

        try:
            data = r.json()
        except Exception:
            log.warning("Meta media metadata for %s was not JSON", media_id)
            return None

        url = data.get("url")
        mime = data.get("mime_type") or "application/octet-stream"
        if not url:
            log.warning("Meta media metadata for %s missing url field", media_id)
            return None

        # Step 2: pull the bytes. Same Bearer auth required.
        try:
            r = await client.get(url, headers=headers)
        except httpx.RequestError as exc:
            log.warning("Meta media bytes fetch failed for %s: %s", media_id, exc)
            return None
        if r.status_code >= 400:
            log.warning(
                "Meta media bytes %s for %s: %s", r.status_code, media_id, r.text[:200]
            )
            return None

    return r.content, mime
