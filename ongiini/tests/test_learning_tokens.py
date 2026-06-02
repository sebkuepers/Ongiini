"""HMAC magic-link tokens — round-trip, tamper, expiry, naive-tz, etc.

Every `verify()` failure mode is exercised. The cardinal rule the
module docstring states — ``verify`` returns None for any failure,
never raises — is the load-bearing invariant; the tests prove it.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone

import pytest

from ongiini.learning import tokens


@pytest.fixture
def secret(monkeypatch):
    """Set the HMAC secret to a stable value so signing works in tests
    without requiring the deployer's env var."""
    from ongiini import config
    monkeypatch.setattr(config.settings, "learn_token_secret", "test-secret-value")
    return "test-secret-value"


# ---------- round-trip ----------

def test_sign_then_verify_returns_payload(secret):
    tok = tokens.sign(learner_id="learner-abc", goal_text="job interview")
    payload = tokens.verify(tok)
    assert payload is not None
    assert payload["lid"] == "learner-abc"
    assert payload["g"] == "job interview"
    assert "exp" in payload


def test_sign_without_goal_omits_goal_field(secret):
    tok = tokens.sign(learner_id="learner-x")
    payload = tokens.verify(tok)
    assert payload is not None
    assert "g" not in payload


def test_sign_truncates_overlong_goal(secret):
    long_goal = "x" * 5000
    tok = tokens.sign(learner_id="lx", goal_text=long_goal)
    payload = tokens.verify(tok)
    assert payload is not None
    assert len(payload["g"]) == 200    # cap from sign()


# ---------- secret missing → sign raises, verify soft-fails ----------

def test_sign_without_secret_raises(monkeypatch):
    from ongiini import config
    monkeypatch.setattr(config.settings, "learn_token_secret", "")
    with pytest.raises(RuntimeError, match="LEARN_TOKEN_SECRET"):
        tokens.sign(learner_id="lx")


def test_verify_without_secret_returns_none_does_not_raise(monkeypatch, secret):
    # First sign with a known secret.
    tok = tokens.sign(learner_id="lx")
    # Then strip the secret. Verifier must soft-fail, never raise.
    from ongiini import config
    monkeypatch.setattr(config.settings, "learn_token_secret", "")
    assert tokens.verify(tok) is None


# ---------- malformed inputs ----------

@pytest.mark.parametrize("bad", [
    "",
    "no-dot-anywhere",
    ".",
    "a.b",                  # not valid base64
    "!!!.!!!",
    "abc.def.ghi",          # too many dots after the split — handled by maxsplit
])
def test_verify_garbage_returns_none(secret, bad):
    assert tokens.verify(bad) is None


def test_verify_payload_not_a_dict_returns_none(secret):
    # Hand-craft a token whose decoded JSON is a list, not a dict.
    payload_b = b'["lid", "x"]'
    sig = tokens._b64url_encode(
        tokens.hmac.new(
            tokens._secret_bytes(), payload_b, tokens.hashlib.sha256
        ).digest()
    )
    tok = f"{tokens._b64url_encode(payload_b)}.{sig}"
    assert tokens.verify(tok) is None


def test_verify_missing_lid_returns_none(secret):
    payload_b = json.dumps({
        "exp": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(timespec="seconds"),
    }).encode()
    sig = tokens._b64url_encode(
        tokens.hmac.new(
            tokens._secret_bytes(), payload_b, tokens.hashlib.sha256
        ).digest()
    )
    tok = f"{tokens._b64url_encode(payload_b)}.{sig}"
    assert tokens.verify(tok) is None


# ---------- tamper detection ----------

def test_verify_rejects_tampered_payload(secret):
    tok = tokens.sign(learner_id="learner-real")
    # Decode payload, swap the lid, keep the original signature.
    payload_part, sig_part = tok.split(".", 1)
    decoded = json.loads(tokens._b64url_decode(payload_part).decode())
    decoded["lid"] = "learner-evil"
    tampered_payload = tokens._b64url_encode(
        json.dumps(decoded, separators=(",", ":"), sort_keys=True).encode()
    )
    tampered_tok = f"{tampered_payload}.{sig_part}"
    assert tokens.verify(tampered_tok) is None


def test_verify_rejects_tampered_signature(secret):
    tok = tokens.sign(learner_id="learner-real")
    payload_part, sig_part = tok.split(".", 1)
    # Flip one byte of the sig.
    sig_bytes = bytearray(tokens._b64url_decode(sig_part))
    sig_bytes[0] ^= 0xff
    bad_sig = tokens._b64url_encode(bytes(sig_bytes))
    assert tokens.verify(f"{payload_part}.{bad_sig}") is None


# ---------- expiry ----------

def test_verify_rejects_expired_token(secret):
    # Sign with -1h expiry → already expired.
    tok = tokens.sign(learner_id="learner-old", expiry_hours=-1)
    assert tokens.verify(tok) is None


def test_verify_accepts_just_in_time_token(secret):
    # 1 second of headroom.
    tok = tokens.sign(learner_id="lx", expiry_hours=1)
    payload = tokens.verify(tok)
    assert payload is not None


def test_verify_handles_naive_exp_timestamp(secret):
    """The reviewer flagged that an `exp` like '2099-01-01T00:00:00'
    (no tz suffix) used to crash with TypeError when compared to
    tz-aware now(). The fix normalises naive datetimes to UTC; verify
    that it still works and doesn't raise."""
    # Craft a token with a naive future expiry.
    payload = {"lid": "lx", "exp": "2099-01-01T00:00:00"}
    payload_b = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    sig = tokens._b64url_encode(
        tokens.hmac.new(
            tokens._secret_bytes(), payload_b, tokens.hashlib.sha256
        ).digest()
    )
    tok = f"{tokens._b64url_encode(payload_b)}.{sig}"
    # Should validate, not raise.
    out = tokens.verify(tok)
    assert out is not None
    assert out["lid"] == "lx"


def test_verify_handles_naive_past_exp_timestamp(secret):
    """Same normalisation, but past — should reject, still not raise."""
    payload = {"lid": "lx", "exp": "2001-01-01T00:00:00"}
    payload_b = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    sig = tokens._b64url_encode(
        tokens.hmac.new(
            tokens._secret_bytes(), payload_b, tokens.hashlib.sha256
        ).digest()
    )
    tok = f"{tokens._b64url_encode(payload_b)}.{sig}"
    assert tokens.verify(tok) is None


def test_verify_handles_bad_exp_string(secret):
    payload = {"lid": "lx", "exp": "not-a-date"}
    payload_b = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    sig = tokens._b64url_encode(
        tokens.hmac.new(
            tokens._secret_bytes(), payload_b, tokens.hashlib.sha256
        ).digest()
    )
    tok = f"{tokens._b64url_encode(payload_b)}.{sig}"
    assert tokens.verify(tok) is None


# ---------- single-issue but not single-use ----------

def test_same_token_verifies_twice(secret):
    """A magic link can be tapped twice (network retry, share with
    a tutor). Verifier must be idempotent — same payload back."""
    tok = tokens.sign(learner_id="lx", goal_text="job interview")
    a = tokens.verify(tok)
    b = tokens.verify(tok)
    assert a == b
    assert a is not None


# ---------- sign requires learner_id ----------

def test_sign_empty_learner_id_raises(secret):
    with pytest.raises(ValueError, match="learner_id"):
        tokens.sign(learner_id="")
