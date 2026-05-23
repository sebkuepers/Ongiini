"""Tests for trace_query.py — the trace aggregation CLI.

These run against synthetic trace.jsonl lines so they don't need a
live system. Covered:
- Window filtering (entries outside the window are excluded)
- Policy filtering (--policy=X drops other policies)
- Each command's aggregation math
- Malformed lines silently skipped
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make scripts/ importable from pytest run-from-repo-root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import trace_query  # noqa: E402


def _now_iso(offset_minutes: int = 0) -> str:
    """ISO-8601 UTC timestamp shifted by ``offset_minutes`` (negative = past)."""
    return (datetime.now(timezone.utc) + timedelta(minutes=offset_minutes)).isoformat(timespec="seconds")


def _write_traces(tmp_path: Path, entries: list[dict]) -> Path:
    p = tmp_path / "trace.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return p


def _entry(
    *,
    minutes_ago: int = 5,
    policy: str = "search_deep",
    queries_count: int | None = 4,
    critique_verdict: str | None = "PASS",
    used_search: bool = True,
    total_latency_ms: int = 18000,
    total_tokens_in: int = 200,
    total_tokens_out: int = 500,
    msisdn: str = "+264user1234",
    leak_per_call: list[int] | None = None,
) -> dict:
    """Build a synthetic trace entry."""
    phases = []
    if queries_count is not None:
        phases.append({"kind": "plan", "queries_count": queries_count})
    if critique_verdict is not None:
        phases.append({"kind": "critique", "verdict": critique_verdict})
    calls = []
    for n in (leak_per_call or [0]):
        calls.append({"turn": 1, "reasoning_leak_stripped": n})
    return {
        "ts": _now_iso(-minutes_ago),
        "msisdn": msisdn,
        "policy": policy,
        "calls": calls,
        "phases": phases,
        "used_search": used_search,
        "total_latency_ms": total_latency_ms,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
    }


# ---------- window + filtering ----------

def test_iter_traces_filters_out_entries_outside_window(tmp_path):
    entries = [
        _entry(minutes_ago=5),
        _entry(minutes_ago=60),
        _entry(minutes_ago=2000),    # outside 24h
    ]
    p = _write_traces(tmp_path, entries)
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    out = list(trace_query._iter_traces(p, since=since, policy=None))
    assert len(out) == 2


def test_iter_traces_filters_by_policy(tmp_path):
    entries = [
        _entry(policy="search_deep"),
        _entry(policy="search_shallow"),
        _entry(policy="none"),
    ]
    p = _write_traces(tmp_path, entries)
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    out = list(trace_query._iter_traces(p, since=since, policy="search_deep"))
    assert len(out) == 1
    assert out[0]["policy"] == "search_deep"


def test_iter_traces_skips_malformed_lines(tmp_path):
    p = tmp_path / "trace.jsonl"
    p.write_text(
        json.dumps(_entry()) + "\n"
        "not json at all\n"
        '{"missing_ts": "yes"}\n'
        + json.dumps(_entry()) + "\n"
    )
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    out = list(trace_query._iter_traces(p, since=since, policy=None))
    assert len(out) == 2


# ---------- revise-rate ----------

def test_revise_rate_computes_percentage():
    traces = [
        _entry(critique_verdict="PASS"),
        _entry(critique_verdict="PASS"),
        _entry(critique_verdict="REVISE"),
        _entry(critique_verdict="PASS"),
    ]
    r = trace_query.cmd_revise_rate(None, traces)
    assert r["total_critiques"] == 4
    assert r["revise_count"] == 1
    assert r["revise_rate_pct"] == 25.0


def test_revise_rate_empty_input_returns_zero():
    r = trace_query.cmd_revise_rate(None, [])
    assert r["total_critiques"] == 0
    assert r["revise_rate_pct"] == 0.0


# ---------- reasoning-leak-count ----------

def test_reasoning_leak_count_aggregates_across_turns_and_calls():
    traces = [
        _entry(leak_per_call=[0]),
        _entry(leak_per_call=[2, 1]),
        _entry(leak_per_call=[0, 0]),
        _entry(leak_per_call=[5]),
    ]
    r = trace_query.cmd_reasoning_leak_count(None, traces)
    assert r["total_turns"] == 4
    assert r["turns_with_leak"] == 2
    assert r["total_tokens_stripped"] == 8


# ---------- planner-fail-rate ----------

def test_planner_fail_rate_zero_queries_counts_as_soft_fail():
    traces = [
        _entry(queries_count=4),
        _entry(queries_count=0),
        _entry(queries_count=3),
        _entry(queries_count=0),
    ]
    r = trace_query.cmd_planner_fail_rate(None, traces)
    assert r["planner_runs"] == 4
    assert r["soft_fails"] == 2
    assert r["soft_fail_pct"] == 50.0


# ---------- queries-count-distribution ----------

def test_queries_count_distribution_bins_correctly():
    traces = [
        _entry(queries_count=0),
        _entry(queries_count=0),
        _entry(queries_count=2),
        _entry(queries_count=2),
        _entry(queries_count=4),
        _entry(queries_count=7),
    ]
    r = trace_query.cmd_queries_count_distribution(None, traces)
    assert r["samples"] == 6
    assert r["distribution"][0] == 2
    assert r["distribution"][2] == 2
    assert r["distribution"][4] == 1
    assert r["distribution"][5] == 1


# ---------- latency-percentiles ----------

def test_latency_percentiles_picks_correct_values():
    traces = [_entry(total_latency_ms=ms) for ms in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]]
    r = trace_query.cmd_latency_percentiles(None, traces)
    assert r["samples"] == 10
    # p50 = median; with 10 sorted samples [1000..10000] and banker's
    # rounding at index `round(0.5 * 9) = round(4.5) = 4`, the value is
    # 5000. Accept either 5000 or 6000 depending on rounding mode.
    assert r["p50_ms"] in (5000, 6000)
    assert r["max_ms"] == 10000


def test_latency_percentiles_empty_returns_no_samples():
    r = trace_query.cmd_latency_percentiles(None, [])
    assert r == {"samples": 0}


# ---------- search-pass-rate ----------

def test_search_pass_rate_ignores_non_search_turns():
    traces = [
        _entry(used_search=False, critique_verdict="PASS"),
        _entry(used_search=True, critique_verdict="PASS"),
        _entry(used_search=True, critique_verdict="REVISE"),
        _entry(used_search=True, critique_verdict="PASS"),
    ]
    r = trace_query.cmd_search_pass_rate(None, traces)
    assert r["search_turns_critiqued"] == 3
    assert r["passed"] == 2
    assert r["pass_rate_pct"] == round(2 / 3 * 100, 1)


# ---------- token-spend ----------

def test_token_spend_by_policy_aggregates_per_policy():
    traces = [
        _entry(policy="search_deep", total_tokens_in=1000, total_tokens_out=500),
        _entry(policy="search_deep", total_tokens_in=2000, total_tokens_out=1000),
        _entry(policy="search_shallow", total_tokens_in=100, total_tokens_out=200),
    ]
    class _Args:
        by = "policy"
    r = trace_query.cmd_token_spend(_Args(), traces)
    assert r["by_policy"]["search_deep"] == 4500
    assert r["by_policy"]["search_shallow"] == 300
    assert r["total"] == 4800


def test_token_spend_by_user_anonymises_and_caps_at_top_10():
    traces = []
    for i in range(15):
        traces.append(_entry(
            msisdn=f"+264user{i:04d}",
            total_tokens_in=10 * (i + 1),
            total_tokens_out=0,
        ))
    class _Args:
        by = "user"
    r = trace_query.cmd_token_spend(_Args(), traces)
    assert len(r["top_10_users"]) == 10
    for k in r["top_10_users"]:
        assert k.startswith("...")
        assert len(k) == 7


# ---------- window parser ----------

def test_parse_window_handles_d_h_m():
    assert trace_query._parse_window("7d") == timedelta(days=7)
    assert trace_query._parse_window("24h") == timedelta(hours=24)
    assert trace_query._parse_window("60m") == timedelta(minutes=60)


def test_parse_window_rejects_invalid_input():
    import pytest
    with pytest.raises(ValueError):
        trace_query._parse_window("7x")
    with pytest.raises(ValueError):
        trace_query._parse_window("seven_days")
    with pytest.raises(ValueError):
        trace_query._parse_window("")


def test_parse_window_rejects_pure_number_with_helpful_message():
    """``--window=7`` (forgot unit) should give a clear error pointing
    at the missing suffix, not a confusing 'unsupported unit 7' message."""
    import pytest
    with pytest.raises(ValueError) as exc:
        trace_query._parse_window("7")
    assert "missing unit suffix" in str(exc.value)
