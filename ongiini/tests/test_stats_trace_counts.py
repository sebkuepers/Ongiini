"""2026-10-01 stats fixes: executed-tool counts, failed replies, wall time.

The old counters read model ``tool_calls`` (missing policy-synthesised
fetch_urls) and a ``deleted_data`` trace field that never existed, so the
public page showed 0 URL fetches and 0 deletions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from ongiini.stats import aggregator

UTC = timezone.utc
WA = "264812345678"
CHAT = "11111111-1111-4111-8111-111111111111"


def _row(msisdn, **extra):
    row = {
        "ts": datetime(2026, 10, 1, 10, 0, tzinfo=UTC).isoformat(timespec="seconds"),
        "msisdn": msisdn, "policy": "none", "total_tokens_in": 10, "total_tokens_out": 10,
        "total_latency_ms": 5000, "has_image": False, "calls": [], "truncated": False,
    }
    row.update(extra)
    return row


def _write(tmp_path, rows):
    p = tmp_path / "trace.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_turn_latency_prefers_wall_ms():
    assert aggregator._turn_latency_ms({"wall_ms": 30000, "total_latency_ms": 9000}) == 30000
    assert aggregator._turn_latency_ms({"total_latency_ms": 9000}) == 9000
    assert aggregator._turn_latency_ms({}) == 0


def test_web_chat_failed_reply_rate_counts_only_rows_with_a_reason(tmp_path, monkeypatch):
    rows = [
        _row(CHAT, transport="web_chat", reply_reason="ok"),
        _row(CHAT, transport="web_chat", reply_reason="error"),
        _row(CHAT, transport="web_chat"),               # pre-v2 row: not counted
    ]
    p = _write(tmp_path, rows)
    monkeypatch.setattr(aggregator, "_trace_path", lambda: p)
    perf = aggregator._compute_web_chat()["performance"]
    assert perf["failed_reply_rate"] == 0.5


def test_whatsapp_counts_executed_fetches_and_deletions(tmp_path, monkeypatch):
    rows = [
        _row(WA, wall_ms=41000, reply_reason="ok", calls=[
            {"tool_calls": [{"name": "web_search"}],
             "tool_results": [{"name": "web_search", "error": None}]},
            {"synthesized": True, "tool_calls": [{"name": "fetch_urls"}],
             "tool_results": [{"name": "fetch_urls", "error": None}]},
        ]),
        _row(WA, reply_reason="max_steps", calls=[
            {"tool_calls": [{"name": "delete_my_data"}],
             "tool_results": [{"name": "delete_my_data", "error": None}]},
        ]),
    ]
    trace = _write(tmp_path, rows)
    usage = tmp_path / "usage.log"
    usage.write_text("")
    monkeypatch.setattr(aggregator, "_trace_path", lambda: trace)
    monkeypatch.setattr(aggregator, "_usage_path", lambda: usage)
    monkeypatch.setattr(aggregator, "_memory_glob", lambda: iter(()))
    monkeypatch.setattr(aggregator, "_objections_path", lambda: tmp_path / "none.txt")
    out = aggregator._compute_sync()
    assert out["totals"]["url_fetches"] == 1
    assert out["totals"]["deletions_invoked"] == 1
    assert out["performance"]["failed_reply_rate"] == 0.5
    assert out["performance"]["p95_latency_ms"] in (5000, 41000)
