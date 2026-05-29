"""Tests for the welcome A/B/C experiment for FB-ad arrivers."""

from __future__ import annotations

import json

import pytest

from ongiini import welcome_experiment as we


# ---------- is_fb_ad_arrival ----------

def test_is_fb_ad_arrival_true_when_referral_present():
    payload = {"referral": {"source_type": "ad", "source_id": "X"}}
    assert we.is_fb_ad_arrival(payload) is True


def test_is_fb_ad_arrival_false_for_organic():
    assert we.is_fb_ad_arrival(None) is False
    assert we.is_fb_ad_arrival({}) is False
    assert we.is_fb_ad_arrival({"referral": None}) is False
    assert we.is_fb_ad_arrival({"referral": {}}) is False  # empty dict is falsy


# ---------- assign_variant ----------

def test_assign_variant_is_deterministic():
    """Same msisdn always → same variant. This is THE property the
    sticky-routing depends on; if it breaks, the experiment leaks."""
    msisdn = "264811234567"
    first = we.assign_variant(msisdn)
    for _ in range(50):
        assert we.assign_variant(msisdn) == first


def test_assign_variant_returns_only_a_b_c():
    for m in (f"26481000{i:04d}" for i in range(200)):
        assert we.assign_variant(m) in ("A", "B", "C")


def test_assign_variant_roughly_balanced():
    """200 distinct msisdns should split ~33/33/33. Tolerance ±10
    percentage points (i.e. each bucket between 23% and 43%) — generous
    enough to never flake but tight enough to catch a routing bug."""
    from collections import Counter
    counts = Counter(we.assign_variant(f"26481000{i:04d}") for i in range(200))
    assert set(counts.keys()) == {"A", "B", "C"}
    for v in ("A", "B", "C"):
        share = counts[v] / 200
        assert 0.23 <= share <= 0.43, f"variant {v}: {share*100:.1f}% — routing not balanced"


# ---------- variant_directive ----------

@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_variant_directive_includes_body_for_each_variant(variant):
    directive = we.variant_directive(variant, "en")
    assert "WELCOME OVERRIDE" in directive
    assert f"variant {variant}" in directive
    # Sanity: the actual body string is present in the directive
    body = we._BODY_EN[variant]
    assert body in directive


def test_variant_directive_supports_afrikaans():
    directive_af = we.variant_directive("A", "af")
    directive_en = we.variant_directive("A", "en")
    assert "Hoe kan ek help?" in directive_af
    assert "How can I help?" in directive_en
    assert directive_af != directive_en


def test_variant_directive_unknown_variant_falls_back_to_a():
    """Defensive: if a future bug passes a bad variant id, we still
    emit a valid directive rather than crashing the webhook reply."""
    directive = we.variant_directive("X", "en")
    assert we._BODY_EN["A"] in directive


# ---------- log_assignment ----------

@pytest.fixture
def tmp_log(tmp_path, monkeypatch):
    p = tmp_path / "welcome_experiment.log"
    monkeypatch.setattr(we, "LOG_PATH", p)
    return p


def test_log_assignment_appends_one_jsonl_line_with_hashed_msisdn(tmp_log):
    msisdn = "264811234567"
    we.log_assignment(msisdn, "B", {"source_id": "ad123", "headline": "Try Ongiini AI"}, "en")
    lines = tmp_log.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["variant"] == "B"
    assert rec["language"] == "en"
    assert rec["referral"]["source_id"] == "ad123"
    assert rec["referral"]["headline"] == "Try Ongiini AI"
    # PII contract — raw msisdn must NOT appear on disk
    assert msisdn not in tmp_log.read_text()
    import hashlib
    expected = hashlib.sha256(msisdn.encode()).hexdigest()[:12]
    assert rec["msisdn_hash"] == expected


def test_log_assignment_handles_no_referral_gracefully(tmp_log):
    """The experiment is scoped to FB-ad arrivers, so referral should
    always be present in production — but if a caller invokes
    log_assignment with referral=None, we still write a valid record
    rather than crashing."""
    we.log_assignment("264811234567", "A", None, "en")
    rec = json.loads(tmp_log.read_text().splitlines()[0])
    assert rec["referral"] is None


def test_log_assignment_soft_fails_on_disk_error(tmp_path, monkeypatch):
    """Webhook reply path MUST NOT break when the log dir is broken.
    Use a non-existent parent to force open() to fail."""
    bogus = tmp_path / "does_not_exist" / "welcome_experiment.log"
    monkeypatch.setattr(we, "LOG_PATH", bogus)
    # No raise — the function eats the exception
    we.log_assignment("264811234567", "C", {"source_id": "X"}, "en")


# ---------- referral extraction in whatsapp.py ----------

def test_extract_messages_attaches_referral_block():
    """Cross-module sanity: a payload with a referral block should
    surface that block on the simplified message dict so downstream
    code (welcome_experiment) can read it."""
    from ongiini.whatsapp import extract_messages
    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": "264811234567",
                        "id": "wamid.X",
                        "type": "text",
                        "text": {"body": "Hi"},
                        "referral": {
                            "source_url": "https://fb.me/X",
                            "source_id": "ad123",
                            "source_type": "ad",
                            "headline": "Try Ongiini AI",
                            "ctwa_clid": "abc",
                        },
                    }],
                },
            }],
        }],
    }
    out = extract_messages(payload)
    assert len(out) == 1
    assert out[0]["referral"]["source_id"] == "ad123"


def test_extract_messages_referral_is_none_for_organic():
    from ongiini.whatsapp import extract_messages
    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": "264811234567", "id": "wamid.X",
                        "type": "text", "text": {"body": "Hi"},
                    }],
                },
            }],
        }],
    }
    out = extract_messages(payload)
    assert out[0]["referral"] is None
