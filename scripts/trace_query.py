"""Trace aggregation CLI for Ongiini.

Reads ``trace.jsonl`` (typically ``/data/trace.jsonl`` in the webhook
container, or ``~/dev/Ongiini/data/trace.jsonl`` on Spark) and produces
aggregate quality / cost / latency metrics over recent activity.

Why this exists: v1.4 inverted the "ship more features" loop. Before
that we'd add an architecture change and HOPE it moved the needle.
This script lets operators ask the trace directly — what's the REVISE
rate after the v1.3.1 reviewer rebalance? How often does the planner
soft-fail on follow-up questions? Are Gemma 4 reasoning leaks still
recurring after the v1.3.2 hotfix?

Usage examples::

    python scripts/trace_query.py revise-rate --window=7d --policy=search_deep
    python scripts/trace_query.py reasoning-leak-count --window=24h
    python scripts/trace_query.py latency-percentiles --policy=search_deep
    python scripts/trace_query.py planner-fail-rate --window=7d
    python scripts/trace_query.py queries-count-distribution --window=7d
    python scripts/trace_query.py token-spend --by=policy --window=30d
    python scripts/trace_query.py token-spend --by=user --window=30d
    python scripts/trace_query.py tool-errors --window=24h
    python scripts/trace_query.py wall-latency --window=7d
    python scripts/trace_query.py health --window=24h   # exit 1 on breach

``health`` is the "is anything broken right now" check: tool error
rates, degraded turns, classifier fallbacks, forced tools the engine
ignored, deadline overruns, reply length and wall latency, each with a
threshold. It exits non-zero when a threshold is breached, so a cron
job can alert. (The Tavily 402 outage of Aug–Sep 2026 would have
tripped ``tool-errors`` on its first day.)

Output is JSON to stdout (single object) — pipe to ``jq`` for
operator workflows.

Override the trace path with ``--path`` or ``ONGIINI_TRACE_PATH=...``
in the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


# Default trace location matches the Spark deployment (data/ is bind-
# mounted into the webhook container, so the host path on Spark is
# ``~/dev/Ongiini/data/trace.jsonl``).
_DEFAULT_TRACE_RELPATH = "data/trace.jsonl"


def _default_trace_path() -> Path:
    env = os.environ.get("ONGIINI_TRACE_PATH")
    if env:
        return Path(env)
    # Walk up from cwd looking for a Ongiini repo root with data/.
    here = Path.cwd().resolve()
    for parent in (here, *here.parents):
        candidate = parent / _DEFAULT_TRACE_RELPATH
        if candidate.exists():
            return candidate
    # Fallback to ~/dev/Ongiini/data/trace.jsonl
    return Path.home() / "dev" / "Ongiini" / _DEFAULT_TRACE_RELPATH


def _parse_window(window: str) -> timedelta:
    """Parse ``7d`` / ``24h`` / ``60m`` into a ``timedelta``."""
    if not window:
        raise ValueError("empty window")
    unit_char = window[-1].lower()
    if not unit_char.isalpha():
        raise ValueError(
            f"invalid window {window!r}: missing unit suffix (use d/h/m, e.g. 7d)"
        )
    try:
        n = int(window[:-1])
    except ValueError as exc:
        raise ValueError(f"invalid window {window!r}; use e.g. 7d, 24h, 60m") from exc
    if unit_char == "d":
        return timedelta(days=n)
    if unit_char == "h":
        return timedelta(hours=n)
    if unit_char == "m":
        return timedelta(minutes=n)
    raise ValueError(f"unsupported window unit {unit_char!r}; use d/h/m")


def _iter_traces(
    path: Path, *, since: datetime, policy: str | None,
) -> Iterator[dict[str, Any]]:
    """Yield trace entries from ``path`` whose ``ts`` is at/after
    ``since``, optionally filtered to a single ``policy`` name.

    Malformed lines / missing-ts entries are silently skipped — a
    corrupt trace shouldn't break the aggregation.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts_str = d.get("ts")
            if not isinstance(ts_str, str):
                continue
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < since:
                continue
            if policy and d.get("policy") != policy:
                continue
            yield d


# ----------------------------- commands -------------------------------


def cmd_revise_rate(args, traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Critique outcome distribution. A REVISE means the critique flagged
    a problem AND the reviewer was given a chance to fix it. Lower is
    better — but 0% might indicate the reviewer is rubber-stamping."""
    total = 0
    revise = 0
    for t in traces:
        for p in t.get("phases", []):
            if p.get("kind") == "critique":
                total += 1
                if p.get("verdict") == "REVISE":
                    revise += 1
    pct = (revise / total * 100) if total else 0.0
    return {
        "total_critiques": total,
        "revise_count": revise,
        "revise_rate_pct": round(pct, 1),
    }


def cmd_reasoning_leak_count(args, traces: list[dict[str, Any]]) -> dict[str, Any]:
    """v1.4 audit: how many turns had Gemma 4 channel tokens stripped
    by the v1.3.2 scrubber? Non-zero means vLLM's reasoning-parser
    still occasionally lets channel tokens through."""
    leak_count = 0
    leak_turns = 0
    total_turns = 0
    for t in traces:
        total_turns += 1
        leak_in_turn = 0
        for c in t.get("calls", []):
            leak_in_turn += int(c.get("reasoning_leak_stripped", 0) or 0)
        if leak_in_turn > 0:
            leak_turns += 1
            leak_count += leak_in_turn
    return {
        "total_turns": total_turns,
        "turns_with_leak": leak_turns,
        "total_tokens_stripped": leak_count,
    }


def cmd_critique_timeout_rate(args, traces: list[dict[str, Any]]) -> dict[str, Any]:
    """How often does the critique time out (and therefore revise
    doesn't fire)? A high rate means the critique timeout budget is
    too tight — drafts ship without the quality-control pass.

    Counts only phases marked ``kind=critique`` — does NOT include
    turns that didn't run critique at all (e.g. NONE-policy casual
    chat) since their critique was never even attempted.

    TODO: this function and ``cmd_planner_fail_rate`` are structurally
    identical — both walk phases counting an error/predicate. If a
    third near-identical phase-counter lands, extract a generic
    ``_phase_error_rate(traces, phase_kind, predicate)`` helper.
    """
    runs = 0
    timeouts = 0
    for t in traces:
        for p in t.get("phases", []):
            if p.get("kind") == "critique":
                runs += 1
                if p.get("error") == "timeout":
                    timeouts += 1
    pct = (timeouts / runs * 100) if runs else 0.0
    return {
        "critique_runs": runs,
        "timeouts": timeouts,
        "timeout_rate_pct": round(pct, 1),
    }


def cmd_planner_fail_rate(args, traces: list[dict[str, Any]]) -> dict[str, Any]:
    """How often does the planner emit zero queries when invoked?
    ``queries_count == 0`` means the planner soft-failed (malformed
    JSON, refused to decompose, etc.). Auto-followup masks this at
    runtime but it's the most direct signal of planner-prompt
    calibration drift."""
    runs = 0
    soft_fails = 0
    for t in traces:
        for p in t.get("phases", []):
            if p.get("kind") == "plan":
                runs += 1
                if int(p.get("queries_count", 0) or 0) == 0:
                    soft_fails += 1
    pct = (soft_fails / runs * 100) if runs else 0.0
    return {
        "planner_runs": runs,
        "soft_fails": soft_fails,
        "soft_fail_pct": round(pct, 1),
    }


def cmd_queries_count_distribution(args, traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Histogram of planner ``queries_count``. Most SEARCH_DEEP turns
    should produce 2-5 queries; outliers tell us about failure modes."""
    counts: list[int] = []
    for t in traces:
        for p in t.get("phases", []):
            if p.get("kind") == "plan":
                counts.append(int(p.get("queries_count", 0) or 0))
    if not counts:
        return {"samples": 0}
    bins: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for c in counts:
        bins[c if c in bins else 5] += 1
    return {
        "samples": len(counts),
        "distribution": bins,
        "median": statistics.median(counts),
        "mean": round(statistics.mean(counts), 2),
    }


def cmd_latency_percentiles(args, traces: list[dict[str, Any]]) -> dict[str, Any]:
    """End-to-end per-turn latency. The transport already gates on the
    25s WhatsApp typing-indicator window; p95 above 25s means users
    are seeing the typing indicator disappear before the reply arrives.
    """
    lats = [int(t["total_latency_ms"]) for t in traces if isinstance(t.get("total_latency_ms"), (int, float))]
    if not lats:
        return {"samples": 0}
    lats_sorted = sorted(lats)

    def pct(p: float) -> int:
        idx = max(0, int(round((p / 100.0) * (len(lats_sorted) - 1))))
        return lats_sorted[idx]

    return {
        "samples": len(lats_sorted),
        "p50_ms": pct(50),
        "p90_ms": pct(90),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "max_ms": lats_sorted[-1],
    }


def cmd_search_pass_rate(args, traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Of turns where any search tool fired (``used_search=True``),
    what fraction had the critique return PASS? PASS implies dim 3
    ("citation present") and dim 2 ("grounded") both held. A coarse
    proxy for "did the model produce a citation-correct, grounded
    reply on a search turn?"."""
    total = 0
    passed = 0
    for t in traces:
        if not t.get("used_search"):
            continue
        critique = next((p for p in t.get("phases", []) if p.get("kind") == "critique"), None)
        if not critique:
            continue
        total += 1
        if critique.get("verdict") == "PASS":
            passed += 1
    pct = (passed / total * 100) if total else 0.0
    return {
        "search_turns_critiqued": total,
        "passed": passed,
        "pass_rate_pct": round(pct, 1),
    }


def cmd_token_spend(args, traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Total token consumption (raw — what we pay vLLM, not what the
    user is billed). Group by policy or user. Use ``--by=user`` to
    find heavy testers; ``--by=policy`` to see which depths dominate
    the cost."""
    by_policy: dict[str, int] = {}
    by_user: dict[str, int] = {}
    for t in traces:
        policy = t.get("policy", "?")
        msisdn = t.get("msisdn", "?")
        tokens = int(t.get("total_tokens_in", 0) or 0) + int(t.get("total_tokens_out", 0) or 0)
        by_policy[policy] = by_policy.get(policy, 0) + tokens
        by_user[msisdn] = by_user.get(msisdn, 0) + tokens
    if args.by == "user":
        # Anonymise: only show last-4 digits of the msisdn.
        top = sorted(by_user.items(), key=lambda kv: -kv[1])[:10]
        return {
            "top_10_users": {f"...{m[-4:]}": tok for m, tok in top},
            "total": sum(by_user.values()),
        }
    return {
        "by_policy": dict(sorted(by_policy.items(), key=lambda kv: -kv[1])),
        "total": sum(by_policy.values()),
    }


# ---------- health commands (Owela v2 trace fields) ----------

# Thresholds for ``health``. A breach is reported and makes the process
# exit 1. Rates only count once there is enough volume to mean anything.
_HEALTH_MIN_SAMPLES = 20
_HEALTH_THRESHOLDS = {
    "tool_error_rate_pct": 50.0,        # per tool
    "degraded_rate_pct": 20.0,
    "router_fallback_rate_pct": 10.0,
    "forced_tool_miss_rate_pct": 10.0,
    "deadline_exceeded_rate_pct": 15.0,
    "over_25s_pct": 25.0,
}


def _pct(part: int, total: int) -> float:
    return round(part / total * 100, 1) if total else 0.0


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))]


def _tool_results(t: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for c in t.get("calls") or []:
        for r in c.get("tool_results") or []:
            yield r


def cmd_tool_errors(args, traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-tool call count, error count and rate, plus the most common
    error prefixes (exception text only — never tool output)."""
    by_tool: dict[str, dict[str, Any]] = {}
    for t in traces:
        for r in _tool_results(t):
            name = r.get("name") or "?"
            row = by_tool.setdefault(name, {"calls": 0, "errors": 0, "prefixes": {}})
            row["calls"] += 1
            err = r.get("error")
            if err:
                row["errors"] += 1
                prefix = str(err)[:60]
                row["prefixes"][prefix] = row["prefixes"].get(prefix, 0) + 1
    out = {}
    for name, row in sorted(by_tool.items(), key=lambda kv: -kv[1]["calls"]):
        top = sorted(row["prefixes"].items(), key=lambda kv: -kv[1])[:3]
        out[name] = {
            "calls": row["calls"],
            "errors": row["errors"],
            "error_rate_pct": _pct(row["errors"], row["calls"]),
            "top_errors": dict(top),
        }
    return {"by_tool": out}


def cmd_degraded_rate(args, traces: list[dict[str, Any]]) -> dict[str, Any]:
    degraded = [t for t in traces if t.get("degraded")]
    triggers: dict[str, int] = {}
    for t in degraded:
        trig = (t.get("degrade") or {}).get("trigger", "?")
        triggers[trig] = triggers.get(trig, 0) + 1
    return {
        "turns": len(traces),
        "degraded": len(degraded),
        "degraded_rate_pct": _pct(len(degraded), len(traces)),
        "by_trigger": triggers,
    }


def cmd_router_fallback_rate(args, traces: list[dict[str, Any]]) -> dict[str, Any]:
    routed = [t for t in traces if t.get("router")]
    reasons: dict[str, int] = {}
    for t in routed:
        reason = t["router"].get("fallback_reason")
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
    fallbacks = sum(reasons.values())
    return {
        "routed": len(routed),
        "fallbacks": fallbacks,
        "router_fallback_rate_pct": _pct(fallbacks, len(routed)),
        "by_reason": reasons,
    }


def cmd_forced_tool_miss_rate(args, traces: list[dict[str, Any]]) -> dict[str, Any]:
    forced = misses = 0
    for t in traces:
        for c in t.get("calls") or []:
            if c.get("forced_tool"):
                forced += 1
                if c.get("forced_tool_honoured") is False:
                    misses += 1
    return {
        "forced_calls": forced,
        "not_honoured": misses,
        "forced_tool_miss_rate_pct": _pct(misses, forced),
    }


def cmd_deadline_exceeded_rate(args, traces: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy: dict[str, list[int]] = {}
    for t in traces:
        row = by_policy.setdefault(t.get("policy") or "?", [0, 0])
        row[0] += 1
        if t.get("deadline_exceeded"):
            row[1] += 1
    total = sum(r[0] for r in by_policy.values())
    exceeded = sum(r[1] for r in by_policy.values())
    return {
        "deadline_exceeded_rate_pct": _pct(exceeded, total),
        "by_policy": {p: {"turns": n, "exceeded": e, "rate_pct": _pct(e, n)}
                      for p, (n, e) in sorted(by_policy.items())},
    }


def cmd_reply_length(args, traces: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy: dict[str, list[int]] = {}
    reasons: dict[str, int] = {}
    for t in traces:
        by_policy.setdefault(t.get("policy") or "?", []).append(int(t.get("reply_len") or 0))
        reason = t.get("reply_reason")
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "by_policy": {
            p: {"n": len(v), "p50": _percentile(v, 0.5), "p95": _percentile(v, 0.95)}
            for p, v in sorted(by_policy.items())
        },
        "reply_reasons": reasons,
        "truncated": sum(1 for t in traces if t.get("truncated")),
    }


def cmd_wall_latency(args, traces: list[dict[str, Any]]) -> dict[str, Any]:
    """End-to-end turn time (router start → reply sent), tools included.
    Falls back to model-only ``total_latency_ms`` for traces written
    before ``wall_ms`` existed."""
    by_policy: dict[str, list[float]] = {}
    for t in traces:
        ms = t.get("wall_ms")
        if ms is None:
            ms = t.get("total_latency_ms")
        if ms is None:
            continue
        by_policy.setdefault(t.get("policy") or "?", []).append(ms / 1000)
    all_s = [x for v in by_policy.values() for x in v]
    return {
        "p50_s": _percentile(all_s, 0.5),
        "p90_s": _percentile(all_s, 0.9),
        "p95_s": _percentile(all_s, 0.95),
        "over_25s_pct": _pct(sum(1 for x in all_s if x > 25), len(all_s)),
        "by_policy": {
            p: {"n": len(v), "p50_s": _percentile(v, 0.5), "p95_s": _percentile(v, 0.95),
                "over_25s_pct": _pct(sum(1 for x in v if x > 25), len(v))}
            for p, v in sorted(by_policy.items())
        },
    }


def cmd_health(args, traces: list[dict[str, Any]]) -> dict[str, Any]:
    """All health signals with thresholds. ``breaches`` lists every
    signal over its threshold; the CLI exits 1 when it is non-empty."""
    report = {
        "tool_errors": cmd_tool_errors(args, traces)["by_tool"],
        "degraded": cmd_degraded_rate(args, traces),
        "router_fallback": cmd_router_fallback_rate(args, traces),
        "forced_tool_miss": cmd_forced_tool_miss_rate(args, traces),
        "deadline_exceeded": cmd_deadline_exceeded_rate(args, traces),
        "wall_latency": cmd_wall_latency(args, traces),
        "reply_length": cmd_reply_length(args, traces),
    }
    th = _HEALTH_THRESHOLDS
    breaches: list[str] = []
    for tool, row in report["tool_errors"].items():
        if row["calls"] >= _HEALTH_MIN_SAMPLES and row["error_rate_pct"] > th["tool_error_rate_pct"]:
            breaches.append(f"tool {tool}: {row['error_rate_pct']}% errors over {row['calls']} calls")
    checks = [
        ("degraded_rate_pct", report["degraded"]["degraded_rate_pct"], len(traces)),
        ("router_fallback_rate_pct", report["router_fallback"]["router_fallback_rate_pct"],
         report["router_fallback"]["routed"]),
        ("forced_tool_miss_rate_pct", report["forced_tool_miss"]["forced_tool_miss_rate_pct"],
         report["forced_tool_miss"]["forced_calls"]),
        ("deadline_exceeded_rate_pct", report["deadline_exceeded"]["deadline_exceeded_rate_pct"],
         len(traces)),
        ("over_25s_pct", report["wall_latency"]["over_25s_pct"], len(traces)),
    ]
    for key, value, n in checks:
        if n >= _HEALTH_MIN_SAMPLES and value > th[key]:
            breaches.append(f"{key}: {value}% > {th[key]}% (n={n})")
    report["breaches"] = breaches
    report["_exit"] = 1 if breaches else 0
    return report


COMMANDS = {
    "revise-rate": cmd_revise_rate,
    "critique-timeout-rate": cmd_critique_timeout_rate,
    "reasoning-leak-count": cmd_reasoning_leak_count,
    "planner-fail-rate": cmd_planner_fail_rate,
    "queries-count-distribution": cmd_queries_count_distribution,
    "latency-percentiles": cmd_latency_percentiles,
    "search-pass-rate": cmd_search_pass_rate,
    "token-spend": cmd_token_spend,
    "tool-errors": cmd_tool_errors,
    "degraded-rate": cmd_degraded_rate,
    "router-fallback-rate": cmd_router_fallback_rate,
    "forced-tool-miss-rate": cmd_forced_tool_miss_rate,
    "deadline-exceeded-rate": cmd_deadline_exceeded_rate,
    "reply-length": cmd_reply_length,
    "wall-latency": cmd_wall_latency,
    "health": cmd_health,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate metrics from Ongiini trace.jsonl",
    )
    parser.add_argument("command", choices=list(COMMANDS.keys()))
    parser.add_argument(
        "--window", default="24h",
        help="time window: e.g. 7d, 24h, 60m (default: 24h)",
    )
    parser.add_argument(
        "--policy", default=None,
        help="filter to one policy name (e.g. search_deep)",
    )
    parser.add_argument(
        "--path", default=None,
        help="trace.jsonl path (overrides ONGIINI_TRACE_PATH env)",
    )
    parser.add_argument(
        "--by", default="policy", choices=["policy", "user"],
        help="grouping for token-spend (default: policy); ignored for other commands",
    )
    args = parser.parse_args(argv)

    path = Path(args.path) if args.path else _default_trace_path()
    delta = _parse_window(args.window)
    since = datetime.now(timezone.utc) - delta

    traces = list(_iter_traces(path, since=since, policy=args.policy))
    result = COMMANDS[args.command](args, traces)
    exit_code = int(result.pop("_exit", 0))
    result["window"] = args.window
    if args.policy:
        result["policy"] = args.policy
    result["sample_size"] = len(traces)
    result["trace_path"] = str(path)

    print(json.dumps(result, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
