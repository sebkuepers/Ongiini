"""Tests for delivery-status webhook parsing + logging."""

from __future__ import annotations

import hashlib
import json

import pytest

from ongiini import delivery_log
from ongiini.whatsapp import extract_statuses


# ---------- extract_statuses ----------

def _wrap(value: dict) -> dict:
    """Wrap a status `value` dict in the same envelope Meta posts."""
    return {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "WABA_ID",
            "changes": [{"field": "messages", "value": value}],
        }],
    }


def test_extract_statuses_sent_delivered_read():
    """The three happy-path statuses for a normal in-window message."""
    payload = _wrap({
        "messaging_product": "whatsapp",
        "metadata": {"display_phone_number": "+1...", "phone_number_id": "P"},
        "statuses": [
            {
                "id": "wamid.SENT",
                "status": "sent",
                "timestamp": "1748428880",
                "recipient_id": "264811234567",
                "conversation": {"id": "C1", "origin": {"type": "service"}},
                "pricing": {"billable": False, "pricing_model": "CBP", "category": "service"},
            },
            {
                "id": "wamid.DELIVERED",
                "status": "delivered",
                "timestamp": "1748428881",
                "recipient_id": "264811234567",
                "conversation": {"id": "C1", "origin": {"type": "service"}},
                "pricing": {"billable": False, "pricing_model": "CBP", "category": "service"},
            },
            {
                "id": "wamid.READ",
                "status": "read",
                "timestamp": "1748428895",
                "recipient_id": "264811234567",
            },
        ],
    })
    out = extract_statuses(payload)
    assert [s["status"] for s in out] == ["sent", "delivered", "read"]
    assert out[0]["msg_id"] == "wamid.SENT"
    assert out[0]["recipient"] == "264811234567"
    assert out[0]["ts"] == "1748428880"
    assert out[0]["conversation_origin"] == "service"
    assert out[0]["pricing_category"] == "service"
    assert out[0]["errors"] == []


def test_extract_statuses_marketing_template():
    """A marketing template send should surface origin=marketing — that's the
    signal that lets us distinguish billable broadcasts from free replies."""
    payload = _wrap({
        "statuses": [{
            "id": "wamid.MKT",
            "status": "sent",
            "timestamp": "1748428900",
            "recipient_id": "264811234567",
            "conversation": {"id": "C2", "origin": {"type": "marketing"}},
            "pricing": {"billable": True, "pricing_model": "CBP", "category": "marketing"},
        }],
    })
    out = extract_statuses(payload)
    assert out[0]["conversation_origin"] == "marketing"
    assert out[0]["pricing_category"] == "marketing"


def test_extract_statuses_failed_with_errors():
    """`failed` status carries an errors[] array with code+title+details —
    this is where we'd diagnose 'undeliverable to recipient' or 'recipient
    not opted in for marketing'."""
    payload = _wrap({
        "statuses": [{
            "id": "wamid.FAIL",
            "status": "failed",
            "timestamp": "1748428910",
            "recipient_id": "491701234567",
            "errors": [{
                "code": 131026,
                "title": "Message undeliverable",
                "message": "Message undeliverable",
                "error_data": {"details": "Receiver is incapable of receiving this message."},
            }],
        }],
    })
    out = extract_statuses(payload)
    assert out[0]["status"] == "failed"
    assert out[0]["errors"][0]["code"] == 131026
    # When no conversation block, origin + pricing default to ""
    assert out[0]["conversation_origin"] == ""
    assert out[0]["pricing_category"] == ""


def test_extract_statuses_inbound_only_payload_returns_empty():
    """A pure inbound-message payload (no statuses[]) yields []."""
    payload = _wrap({
        "messages": [{"id": "wamid.IN", "from": "264811234567", "type": "text",
                      "text": {"body": "hi"}}],
    })
    assert extract_statuses(payload) == []


def test_extract_statuses_empty_payload():
    assert extract_statuses({}) == []
    assert extract_statuses({"entry": []}) == []
    assert extract_statuses(_wrap({})) == []
    assert extract_statuses(_wrap({"statuses": []})) == []


def test_extract_statuses_malformed_blocks_are_tolerated():
    """Defensive: a half-formed status (missing fields) yields a row with
    empty defaults rather than raising. Meta has shipped payload schema
    drift in the past; better to log incomplete data than crash the webhook."""
    payload = _wrap({"statuses": [{"id": "wamid.X"}]})
    out = extract_statuses(payload)
    assert out[0]["msg_id"] == "wamid.X"
    assert out[0]["status"] == ""
    assert out[0]["recipient"] == ""
    assert out[0]["errors"] == []


# ---------- record_status / delivery_log ----------

@pytest.fixture
def tmp_log(tmp_path, monkeypatch):
    """Redirect the module-level LOG_PATH at the place it's used."""
    p = tmp_path / "delivery_status.log"
    monkeypatch.setattr(delivery_log, "LOG_PATH", p)
    return p


def test_record_status_writes_one_jsonl_line(tmp_log):
    delivery_log.record_status({
        "msg_id": "wamid.A",
        "status": "delivered",
        "recipient": "264811234567",
        "ts": "1748428881",
        "errors": [],
        "conversation_origin": "service",
        "pricing_category": "service",
    })
    lines = tmp_log.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["msg_id"] == "wamid.A"
    assert rec["status"] == "delivered"
    assert rec["ts"] == "1748428881"
    assert rec["conversation_origin"] == "service"
    assert rec["pricing_category"] == "service"


def test_record_status_hashes_recipient_msisdn(tmp_log):
    """PII contract: no raw msisdns on disk. Use the same hashing as
    broadcast.log — first 12 chars of sha256(msisdn)."""
    msisdn = "264811234567"
    delivery_log.record_status({"msg_id": "x", "status": "sent",
                                "recipient": msisdn})
    rec = json.loads(tmp_log.read_text().splitlines()[0])
    assert msisdn not in tmp_log.read_text()
    expected = hashlib.sha256(msisdn.encode()).hexdigest()[:12]
    assert rec["recipient_hash"] == expected
    assert "recipient" not in rec   # we ONLY persist the hash


def test_record_status_appends_not_overwrites(tmp_log):
    delivery_log.record_status({"msg_id": "1", "status": "sent"})
    delivery_log.record_status({"msg_id": "2", "status": "delivered"})
    delivery_log.record_status({"msg_id": "3", "status": "read"})
    lines = tmp_log.read_text().splitlines()
    assert [json.loads(l)["msg_id"] for l in lines] == ["1", "2", "3"]


def test_record_status_soft_fails_on_write_error(tmp_path, monkeypatch):
    """If the disk write blows up, the webhook MUST keep going — Meta gets
    a 200, no retry loop, no user impact from broken observability."""
    # Point LOG_PATH at a path whose parent does not exist → open() raises
    # FileNotFoundError, which the helper must catch.
    bogus = tmp_path / "nonexistent_dir" / "delivery_status.log"
    monkeypatch.setattr(delivery_log, "LOG_PATH", bogus)
    delivery_log.record_status({"msg_id": "x", "status": "failed",
                                "recipient": "491701234567"})
    # No exception means soft-fail caught it.


def test_record_status_empty_recipient_hash(tmp_log):
    """Some Meta payloads omit recipient_id (rare, mostly malformed).
    Should write the record with empty hash, not crash."""
    delivery_log.record_status({"msg_id": "y", "status": "sent", "recipient": ""})
    rec = json.loads(tmp_log.read_text().splitlines()[0])
    assert rec["recipient_hash"] == ""
