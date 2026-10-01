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


# ---------- critique-timeout-rate ----------

def test_critique_timeout_rate_counts_timeouts():
    traces = [
        {"phases": [{"kind": "critique", "verdict": "PASS", "error": None}]},
        {"phases": [{"kind": "critique", "verdict": "PASS", "error": "timeout"}]},
        {"phases": [{"kind": "critique", "verdict": "REVISE", "error": None}]},
        {"phases": [{"kind": "critique", "verdict": "PASS", "error": "timeout"}]},
    ]
    r = trace_query.cmd_critique_timeout_rate(None, traces)
    assert r["critique_runs"] == 4
    assert r["timeouts"] == 2
    assert r["timeout_rate_pct"] == 50.0


def test_critique_timeout_rate_ignores_turns_without_critique():
    traces = [
        {"phases": [{"kind": "plan", "queries_count": 3}]},          # no critique
        {"phases": [{"kind": "critique", "verdict": "PASS", "error": None}]},
    ]
    r = trace_query.cmd_critique_timeout_rate(None, traces)
    assert r["critique_runs"] == 1
    assert r["timeouts"] == 0


def test_critique_timeout_rate_empty():
    r = trace_query.cmd_critique_timeout_rate(None, [])
    assert r["critique_runs"] == 0
    assert r["timeout_rate_pct"] == 0.0


# ---------- Owela v2 health commands ----------

def _v2_entry(*, policy="search_shallow", tool_error=None, degraded=False,
              fallback_reason=None, forced_honoured=None, deadline=False,
              wall_ms=12000, reply_len=500, minutes_ago=5) -> dict:
    call = {"turn": 1, "tool_results": [{"name": "web_search", "error": tool_error}]}
    if forced_honoured is not None:
        call["forced_tool"] = "web_search"
        call["forced_tool_honoured"] = forced_honoured
    return {
        "ts": _now_iso(-minutes_ago),
        "policy": policy,
        "router": {"verdict": "SEARCH", "fallback_reason": fallback_reason},
        "calls": [call],
        "phases": [],
        "degraded": degraded,
        "degrade": {"trigger": "tool_error"} if degraded else None,
        "deadline_exceeded": deadline,
        "wall_ms": wall_ms,
        "reply_len": reply_len,
        "reply_reason": "ok",
    }


def _args(**kw):
    import argparse
    return argparse.Namespace(by="policy", **kw)


def test_tool_errors_reports_rate_and_prefixes():
    traces = [_v2_entry(tool_error="Web search is temporarily unavailable (provider returned HTTP 402)")] * 3
    traces += [_v2_entry()]
    out = trace_query.cmd_tool_errors(_args(), traces)
    ws = out["by_tool"]["web_search"]
    assert (ws["calls"], ws["errors"], ws["error_rate_pct"]) == (4, 3, 75.0)
    assert list(ws["top_errors"].values()) == [3]


def test_health_flags_a_dead_search_provider():
    """The Sep 2026 shape: every search errors → breach, exit 1."""
    traces = [_v2_entry(tool_error="HTTP 402", degraded=True) for _ in range(25)]
    out = trace_query.cmd_health(_args(), traces)
    assert out["_exit"] == 1
    assert any("tool web_search" in b for b in out["breaches"])
    assert any("degraded_rate_pct" in b for b in out["breaches"])


def test_health_is_quiet_when_healthy():
    traces = [_v2_entry() for _ in range(30)]
    out = trace_query.cmd_health(_args(), traces)
    assert out["breaches"] == []
    assert out["_exit"] == 0


def test_health_ignores_low_volume():
    traces = [_v2_entry(tool_error="boom", degraded=True) for _ in range(5)]
    out = trace_query.cmd_health(_args(), traces)
    assert out["breaches"] == []


def test_router_fallback_and_forced_tool_miss_rates():
    traces = [_v2_entry(fallback_reason="timeout", forced_honoured=False),
              _v2_entry(forced_honoured=True), _v2_entry(), _v2_entry()]
    rf = trace_query.cmd_router_fallback_rate(_args(), traces)
    assert rf["router_fallback_rate_pct"] == 25.0
    assert rf["by_reason"] == {"timeout": 1}
    ft = trace_query.cmd_forced_tool_miss_rate(_args(), traces)
    assert (ft["forced_calls"], ft["not_honoured"]) == (2, 1)


def test_wall_latency_uses_wall_ms_and_falls_back_to_model_time():
    traces = [_v2_entry(wall_ms=30000), _v2_entry(wall_ms=10000)]
    legacy = _v2_entry()
    legacy.pop("wall_ms")
    legacy["total_latency_ms"] = 5000
    out = trace_query.cmd_wall_latency(_args(), traces + [legacy])
    assert out["over_25s_pct"] == 33.3
    assert out["by_policy"]["search_shallow"]["n"] == 3


def test_health_cli_exit_code(tmp_path):
    p = _write_traces(tmp_path, [_v2_entry(tool_error="HTTP 402") for _ in range(25)])
    assert trace_query.main(["health", "--path", str(p), "--window", "24h"]) == 1
    p2 = _write_traces(tmp_path, [_v2_entry() for _ in range(25)])
    assert trace_query.main(["health", "--path", str(p2), "--window", "24h"]) == 0


def test_health_explains_breaches_in_plain_language():
    slow = [_v2_entry(policy="search_deep", wall_ms=40000) for _ in range(10)]
    fast = [_v2_entry(policy="none", wall_ms=3000) for _ in range(15)]
    out = trace_query.cmd_health(_args(), slow + fast)
    assert len(out["explanations"]) == len(out["breaches"]) == 1
    text = out["explanations"][0]
    assert "länger als 25 Sekunden" in text
    assert "Recherche über mehrere Quellen 10/10" in text
    assert "kurzer Chat" not in text            # only slow policies are listed


def test_health_explains_a_dead_tool():
    out = trace_query.cmd_health(_args(), [_v2_entry(tool_error="HTTP 402") for _ in range(25)])
    assert any("Web-Suche' fällt aus" in e and "HTTP 402" in e for e in out["explanations"])
