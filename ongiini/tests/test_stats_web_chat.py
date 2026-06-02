"""Tests for the web-chat aggregation block in stats.aggregator.

Specifically the transport-classification helpers (UUID-v4 fallback for
historical traces vs. explicit `transport` field on new traces) and
``_compute_web_chat()`` which produces the ``web_chat`` block in
/stats.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ongiini.stats import aggregator


UTC = timezone.utc

WA_MSISDN = "264812345678"
CHAT_UUID_A = "11111111-1111-4111-8111-111111111111"
CHAT_UUID_B = "22222222-2222-4222-8222-222222222222"


# ---------- transport classification ----------


def test_is_web_chat_msisdn_uuid_v4_matches():
    assert aggregator._is_web_chat_msisdn(CHAT_UUID_A) is True


def test_is_web_chat_msisdn_namibian_phone_does_not_match():
    assert aggregator._is_web_chat_msisdn(WA_MSISDN) is False


def test_is_web_chat_msisdn_rejects_uppercase_hex():
    # Browser crypto.randomUUID() always emits lowercase; we reject
    # uppercase hex so a manually-typed UUID can't pose as a session.
    # (The all-digit fixture would upper-case to the same string, so
    # use one with hex letters here.)
    uuid_with_letters = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    assert aggregator._is_web_chat_msisdn(uuid_with_letters) is True
    assert aggregator._is_web_chat_msisdn(uuid_with_letters.upper()) is False


def test_resolve_transport_prefers_explicit_field():
    record = {"msisdn": WA_MSISDN, "transport": "web_chat"}
    assert aggregator._resolve_transport(record) == "web_chat"


def test_resolve_transport_falls_back_to_msisdn_shape():
    # No transport field → UUID v4 msisdn classifies as web_chat
    record = {"msisdn": CHAT_UUID_A}
    assert aggregator._resolve_transport(record) == "web_chat"


def test_resolve_transport_defaults_to_whatsapp():
    # No transport field + non-UUID msisdn → whatsapp
    record = {"msisdn": WA_MSISDN}
    assert aggregator._resolve_transport(record) == "whatsapp"


def test_resolve_transport_ignores_unknown_explicit_value():
    # An unrecognised string in the transport field falls through to
    # the msisdn-shape heuristic — keeps the classifier strict.
    record = {"msisdn": WA_MSISDN, "transport": "facebook"}
    assert aggregator._resolve_transport(record) == "whatsapp"


# ---------- _compute_web_chat ----------


def _write_trace(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "trace.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def _trace_row(
    ts: datetime,
    msisdn: str,
    *,
    tokens_in: int = 100,
    tokens_out: int = 80,
    latency_ms: int = 1500,
    has_image: bool = False,
    tool_called: bool = False,
    truncated: bool = False,
    transport: str | None = None,
) -> dict:
    row = {
        "ts": ts.isoformat(timespec="seconds"),
        "msisdn": msisdn,
        "total_tokens_in": tokens_in,
        "total_tokens_out": tokens_out,
        "total_latency_ms": latency_ms,
        "has_image": has_image,
        "calls": [{"tool_calls": [{"name": "web_search"}]}] if tool_called else [],
        "truncated": truncated,
    }
    if transport is not None:
        row["transport"] = transport
    return row


def test_returns_none_when_no_web_chat_rows(tmp_path, monkeypatch):
    """When the trace contains only WhatsApp rows, the web_chat block
    is None so the /statistics page hides the section."""
    rows = [
        _trace_row(datetime(2026, 6, 1, 10, 0, tzinfo=UTC), WA_MSISDN),
        _trace_row(datetime(2026, 6, 1, 11, 0, tzinfo=UTC), WA_MSISDN),
    ]
    p = _write_trace(tmp_path, rows)
    monkeypatch.setattr(aggregator, "_trace_path", lambda: p)
    assert aggregator._compute_web_chat() is None


def test_aggregates_only_web_chat_rows(tmp_path, monkeypatch):
    """A mixed trace with WhatsApp + web-chat rows: the web_chat block
    counts only the web-chat half, with no cross-contamination."""
    rows = [
        # WhatsApp — must be ignored entirely
        _trace_row(datetime(2026, 6, 1, 10, 0, tzinfo=UTC), WA_MSISDN, tokens_out=999),
        _trace_row(datetime(2026, 6, 1, 11, 0, tzinfo=UTC), WA_MSISDN, tokens_out=888),

        # Web-chat — historical record (no transport field, UUID msisdn)
        _trace_row(datetime(2026, 6, 1, 12, 0, tzinfo=UTC), CHAT_UUID_A,
                   tokens_in=100, tokens_out=80),

        # Web-chat — new record (explicit transport field + same session)
        _trace_row(datetime(2026, 6, 1, 12, 5, tzinfo=UTC), CHAT_UUID_A,
                   tokens_in=120, tokens_out=90, transport="web_chat",
                   has_image=True),

        # Web-chat — second session
        _trace_row(datetime(2026, 6, 1, 13, 0, tzinfo=UTC), CHAT_UUID_B,
                   tokens_in=150, tokens_out=110, tool_called=True),
    ]
    p = _write_trace(tmp_path, rows)
    monkeypatch.setattr(aggregator, "_trace_path", lambda: p)

    block = aggregator._compute_web_chat()
    assert block is not None

    totals = block["totals"]
    assert totals["sessions"] == 2          # A and B; WhatsApp rows excluded
    assert totals["messages"] == 3          # 3 web-chat rows
    assert totals["tokens_in_total"] == 100 + 120 + 150
    assert totals["tokens_out_total"] == 80 + 90 + 110
    assert totals["images"] == 1
    assert totals["tool_call_turns"] == 1


def test_per_day_timeseries(tmp_path, monkeypatch):
    """Sessions / messages / tokens are bucketed per UTC date."""
    rows = [
        _trace_row(datetime(2026, 6, 1, 12, 0, tzinfo=UTC), CHAT_UUID_A,
                   tokens_in=100, tokens_out=80),
        _trace_row(datetime(2026, 6, 1, 12, 5, tzinfo=UTC), CHAT_UUID_A,
                   tokens_in=100, tokens_out=80),
        _trace_row(datetime(2026, 6, 2, 9, 0, tzinfo=UTC), CHAT_UUID_B,
                   tokens_in=200, tokens_out=150),
    ]
    p = _write_trace(tmp_path, rows)
    monkeypatch.setattr(aggregator, "_trace_path", lambda: p)

    block = aggregator._compute_web_chat()
    ts = block["timeseries"]

    assert ts["messages_per_day"] == [
        ["2026-06-01", 2],
        ["2026-06-02", 1],
    ]
    assert ts["sessions_per_day"] == [
        ["2026-06-01", 1],   # only session A on day 1
        ["2026-06-02", 1],   # only session B on day 2
    ]
    assert ts["tokens_per_day"] == [
        ["2026-06-01", 200, 160],
        ["2026-06-02", 200, 150],
    ]


def test_performance_block(tmp_path, monkeypatch):
    """median / p95 latency + tool-call & truncation rates match
    the WhatsApp `perf` block's shape."""
    rows = [
        _trace_row(datetime(2026, 6, 1, 10, 0, tzinfo=UTC), CHAT_UUID_A,
                   latency_ms=1000),
        _trace_row(datetime(2026, 6, 1, 10, 1, tzinfo=UTC), CHAT_UUID_A,
                   latency_ms=1500, tool_called=True),
        _trace_row(datetime(2026, 6, 1, 10, 2, tzinfo=UTC), CHAT_UUID_A,
                   latency_ms=2000, truncated=True),
        _trace_row(datetime(2026, 6, 1, 10, 3, tzinfo=UTC), CHAT_UUID_A,
                   latency_ms=2500, tool_called=True),
    ]
    p = _write_trace(tmp_path, rows)
    monkeypatch.setattr(aggregator, "_trace_path", lambda: p)

    perf = aggregator._compute_web_chat()["performance"]
    assert perf["median_latency_ms"] == 1750     # (1500 + 2000) / 2
    # p95 index = max(0, int(4*0.95)-1) = 2, so latencies[2] = 2000.
    # Matches the existing WhatsApp `perf` block's percentile rule.
    assert perf["p95_latency_ms"] == 2000
    assert perf["tool_call_rate"] == 0.5         # 2/4
    assert perf["truncation_rate"] == 0.25       # 1/4


def test_wow_deltas(tmp_path, monkeypatch):
    """Current-vs-prior 7-day deltas, anchored on the latest observed
    timestamp (not wall-clock)."""
    # Anchor will be 2026-06-15 10:00. Current window: (06-08, 06-15].
    # Prior window: (06-01, 06-08].
    rows = [
        # Prior window — 2 messages on 06-05, 1 session
        _trace_row(datetime(2026, 6, 5, 10, 0, tzinfo=UTC), CHAT_UUID_A),
        _trace_row(datetime(2026, 6, 5, 10, 5, tzinfo=UTC), CHAT_UUID_A),

        # Current window — 4 messages across 2 sessions
        _trace_row(datetime(2026, 6, 10, 10, 0, tzinfo=UTC), CHAT_UUID_A),
        _trace_row(datetime(2026, 6, 12, 10, 0, tzinfo=UTC), CHAT_UUID_B),
        _trace_row(datetime(2026, 6, 14, 10, 0, tzinfo=UTC), CHAT_UUID_B),
        _trace_row(datetime(2026, 6, 15, 10, 0, tzinfo=UTC), CHAT_UUID_A),
    ]
    p = _write_trace(tmp_path, rows)
    monkeypatch.setattr(aggregator, "_trace_path", lambda: p)

    deltas = aggregator._compute_web_chat()["totals_deltas"]
    assert deltas["messages"]["current"] == 4
    assert deltas["messages"]["prior"] == 2
    assert deltas["messages"]["pct_change"] == 100.0
    assert deltas["sessions"]["current"] == 2
    assert deltas["sessions"]["prior"] == 1
    assert deltas["sessions"]["pct_change"] == 100.0


def test_pct_change_omitted_when_prior_is_zero(tmp_path, monkeypatch):
    """When the prior 7-day window has zero observations we drop the
    pct_change field entirely (no inf, no misleading large number)."""
    rows = [
        # Only current window has anything
        _trace_row(datetime(2026, 6, 15, 10, 0, tzinfo=UTC), CHAT_UUID_A),
    ]
    p = _write_trace(tmp_path, rows)
    monkeypatch.setattr(aggregator, "_trace_path", lambda: p)

    deltas = aggregator._compute_web_chat()["totals_deltas"]
    assert deltas["messages"]["current"] == 1
    assert deltas["messages"]["prior"] == 0
    assert "pct_change" not in deltas["messages"]
