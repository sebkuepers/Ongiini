"""HMAC-signed magic-link tokens for the learning surface.

When normal chat (WhatsApp or chat.ongiini.ai) detects a learning
intent, the assistant offers a URL like::

    https://learn.ongiini.ai/start?t=<token>

The token carries: a `learner_id` (UUID v4 for chat sessions, or the
salted hash of an msisdn for WhatsApp), an optional `goal_text`, and
an expiry. We sign it with HMAC-SHA256 over the payload using the
shared `learn_token_secret` so the browser frontend can prove the
visitor actually came from a legitimately-issued chat reply.

Concretely the token looks like::

    base64url(json_payload) + "." + base64url(hmac_sha256(json_payload))

JSON instead of `key=value` for forward-compat — adding a new field
(say, a referrer code) doesn't break older verifier code.

Tokens are single-issue but NOT single-use: the same magic link can
be tapped twice within its expiry window (network retry, accidental
double tap, sharing with a tutor) and still resolve to the same
learner. The DB layer is what enforces "this learner already started,
resume them" via `learner_id` PRIMARY KEY.

If `settings.learn_token_secret` is empty, ``sign()`` raises — the
deployer has to set ONGIINI_LEARN_TOKEN_SECRET in env. ``verify()``
returns ``None`` for any failure (bad sig, expired, malformed,
secret-missing) — never raises — so callers can treat invalid tokens
as "not authenticated; fall back to cold-start" without try/except.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from ..config import settings

log = logging.getLogger("ongiini.learning.tokens")


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    # Re-pad up to a multiple of 4 — urlsafe_b64encode strips padding.
    padding = (-len(s)) % 4
    return base64.urlsafe_b64decode(s + ("=" * padding))


def _secret_bytes() -> bytes:
    secret = settings.learn_token_secret or ""
    if not secret:
        raise RuntimeError(
            "ONGIINI_LEARN_TOKEN_SECRET is not set — refusing to sign "
            "a magic-link token without it. Set the env var or disable "
            "the learning surface via ONGIINI_LEARN_ENABLED=false."
        )
    return secret.encode("utf-8")


def sign(
    *,
    learner_id: str,
    goal_text: str | None = None,
    expiry_hours: int | None = None,
) -> str:
    """Return a signed magic-link token string.

    ``learner_id`` is opaque to this module — usually a UUID v4 for
    cold/anon sessions or ``"wa:<hashed_msisdn>"`` for WhatsApp-bound
    learners. ``goal_text`` is the free-text learning objective if
    we have one. ``expiry_hours`` defaults to
    ``settings.learn_token_expiry_hours``.

    Raises ``RuntimeError`` if the deployer hasn't set the secret.
    """
    if not learner_id:
        raise ValueError("learner_id is required to sign a token")

    expiry_h = expiry_hours if expiry_hours is not None else settings.learn_token_expiry_hours
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=expiry_h)).isoformat(timespec="seconds")
    payload: dict[str, Any] = {
        "lid": learner_id,
        "exp": expires_at,
    }
    if goal_text:
        # Cap so an over-long goal can't bloat the URL beyond what
        # WhatsApp will linkify cleanly (~2 KB practical limit).
        payload["g"] = goal_text[:200]

    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    payload_b = payload_json.encode("utf-8")
    sig = hmac.new(_secret_bytes(), payload_b, hashlib.sha256).digest()
    return f"{_b64url_encode(payload_b)}.{_b64url_encode(sig)}"


def verify(token: str) -> dict[str, Any] | None:
    """Return the verified payload dict or ``None`` if the token is
    invalid / expired / tampered with / malformed.

    Never raises. Soft-fail by design — callers should treat a None
    return as 'not authenticated; show the cold-start UI'.
    """
    if not token or "." not in token:
        return None

    try:
        payload_part, sig_part = token.split(".", 1)
        payload_b = _b64url_decode(payload_part)
        provided_sig = _b64url_decode(sig_part)
    except (ValueError, base64.binascii.Error):
        return None

    try:
        secret = _secret_bytes()
    except RuntimeError:
        # Secret not set — refuse to verify anything. Soft-fail.
        log.warning("verify: ONGIINI_LEARN_TOKEN_SECRET missing; rejecting token")
        return None

    expected_sig = hmac.new(secret, payload_b, hashlib.sha256).digest()
    if not hmac.compare_digest(provided_sig, expected_sig):
        return None

    try:
        payload = json.loads(payload_b.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    # Expiry check — accept anything <= the embedded `exp` timestamp.
    # ``fromisoformat`` returns a naive datetime when the input lacks a
    # tz suffix; comparing a naive value to a tz-aware ``now()`` raises
    # TypeError. That would propagate out of verify() and violate the
    # "never raises, returns None" contract. Normalise instead.
    exp_str = payload.get("exp")
    if not isinstance(exp_str, str):
        return None
    try:
        exp_dt = datetime.fromisoformat(exp_str)
    except ValueError:
        return None
    if exp_dt.tzinfo is None:
        exp_dt = exp_dt.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > exp_dt:
        return None

    # Minimum required field.
    if not isinstance(payload.get("lid"), str) or not payload["lid"]:
        return None

    return payload
