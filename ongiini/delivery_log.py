"""Delivery-status observability for outbound WhatsApp messages.

Meta posts a status callback (sent/delivered/read/failed) for every
outbound message we send — broadcast templates, free-form replies,
mark_as_read. We currently silently drop these. This module captures
them so we can:

  - Confirm marketing templates actually reach the recipient
  - Surface failures (code, title, details) for diagnostics
  - Audit which conversations were billed as marketing vs utility vs
    free in-window service

Log format: one JSON line per status event, appended to
``/data/delivery_status.log``. The recipient msisdn is hashed (first
12 chars of SHA-256, no salt) — same pattern as broadcast.log. We
do NOT persist raw msisdns per the PII contract.

Errors here MUST never propagate: a write failure should not break
the webhook return path. Caller wraps in try/except as defense in
depth; the helper also catches internally.
"""
import hashlib
import json
import logging
from pathlib import Path

from .config import settings

log = logging.getLogger("ongiini.delivery_log")

LOG_PATH = settings.data_dir / "delivery_status.log"


def _hash_recipient(msisdn: str) -> str:
    if not msisdn:
        return ""
    return hashlib.sha256(msisdn.encode("utf-8")).hexdigest()[:12]


def record_status(status: dict) -> None:
    """Append one status event as a JSON line. Soft-fail on any error."""
    try:
        record = {
            "ts": status.get("ts", ""),
            "msg_id": status.get("msg_id", ""),
            "recipient_hash": _hash_recipient(status.get("recipient", "")),
            "status": status.get("status", ""),
            "conversation_origin": status.get("conversation_origin", ""),
            "pricing_category": status.get("pricing_category", ""),
            "errors": status.get("errors", []),
        }
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception:
        log.exception("delivery_log write failed for msg_id=%s", status.get("msg_id", "?"))
