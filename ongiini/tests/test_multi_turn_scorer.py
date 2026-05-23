"""Unit tests for the pure-logic scoring functions in
``multi_turn_eval.py``. The live runner that exercises Agent.handle()
isn't unit-testable (needs vLLM + Tavily), but the scorers ARE."""

from __future__ import annotations

from ongiini.tests.multi_turn_eval import (
    aggregate,
    score_citation_present,
    score_latency_under_25s,
    score_no_reasoning_leak,
    score_planner_queries,
)


# ---------- planner_queries ----------

def test_score_planner_queries_passes_when_count_ge_2():
    trace = {"phases": [{"kind": "plan", "queries_count": 3}]}
    assert score_planner_queries(trace) is True


def test_score_planner_queries_fails_when_count_lt_2():
    trace = {"phases": [{"kind": "plan", "queries_count": 1}]}
    assert score_planner_queries(trace) is False
    trace_zero = {"phases": [{"kind": "plan", "queries_count": 0}]}
    assert score_planner_queries(trace_zero) is False


def test_score_planner_queries_fails_when_no_plan_phase():
    trace = {"phases": [{"kind": "critique", "verdict": "PASS"}]}
    assert score_planner_queries(trace) is False


# ---------- citation_present ----------

def test_score_citation_present_accepts_em_dash_source_line():
    reply = (
        "Here's the comparison...\n\n"
        "— source: https://www.bon.com.na/Rates/Exchange-Rates.aspx\n"
    )
    assert score_citation_present(reply) is True


def test_score_citation_present_accepts_hyphen_source_line():
    reply = "Some text\n- source: https://example.com/path/to/article"
    assert score_citation_present(reply) is True


def test_score_citation_present_rejects_homepage_only_url():
    reply = "Some text\n— source: https://www.namibian.com.na"
    assert score_citation_present(reply) is False


def test_score_citation_present_rejects_url_without_source_marker():
    reply = "Read more at https://example.com/article today."
    assert score_citation_present(reply) is False


def test_score_citation_present_empty_reply():
    assert score_citation_present("") is False


# ---------- latency_under_25s ----------

def test_score_latency_under_25s_passes_at_20s():
    assert score_latency_under_25s({"total_latency_ms": 20_000}) is True


def test_score_latency_under_25s_fails_at_25s():
    assert score_latency_under_25s({"total_latency_ms": 25_000}) is False


def test_score_latency_under_25s_missing_field_defaults_to_zero_passes():
    assert score_latency_under_25s({}) is True


# ---------- no_reasoning_leak ----------

def test_score_no_reasoning_leak_passes_clean_reply():
    reply = "Namibia has 4 commercial banks. — source: https://x.com/y"
    assert score_no_reasoning_leak(reply) is True


def test_score_no_reasoning_leak_fails_on_channel_token():
    reply = "thought<|channel|>some reasoning\n\nActual answer here."
    assert score_no_reasoning_leak(reply) is False


def test_score_no_reasoning_leak_fails_on_thought_preamble():
    reply = "thought there are 4 banks in Namibia."
    assert score_no_reasoning_leak(reply) is False


def test_score_no_reasoning_leak_thoughtfully_known_limitation():
    """A reply legitimately starting with 'Thoughtfully' is currently
    flagged as a leak (heuristic is greedy on the 'thought' prefix).
    Pinning this so a future improvement is intentional."""
    reply = "Thoughtfully designed budgeting starts with understanding fixed costs."
    assert score_no_reasoning_leak(reply) is False


def test_score_no_reasoning_leak_fails_on_wait_preamble():
    reply = "Wait, the user asked about banks in English\n\nNamibia has..."
    assert score_no_reasoning_leak(reply) is False


# ---------- aggregate ----------

def test_aggregate_computes_per_dimension_pass_rates():
    results = [
        {"id": "a", "scores": {"planner_queries": True, "citation_present": True,
                               "latency_under_25s": True, "no_reasoning_leak": True}},
        {"id": "b", "scores": {"planner_queries": False, "citation_present": True,
                               "latency_under_25s": True, "no_reasoning_leak": True}},
        {"id": "c", "scores": {"planner_queries": True, "citation_present": False,
                               "latency_under_25s": True, "no_reasoning_leak": False}},
        {"id": "d", "scores": {"planner_queries": True, "citation_present": True,
                               "latency_under_25s": True, "no_reasoning_leak": True}},
    ]
    out = aggregate(results)
    assert out["samples"] == 4
    assert out["planner_queries_pass_rate"] == 75.0
    assert out["citation_present_pass_rate"] == 75.0
    assert out["latency_under_25s_pass_rate"] == 100.0
    assert out["no_reasoning_leak_pass_rate"] == 75.0
    assert out["all_dimensions_pass_rate"] == 50.0


def test_aggregate_empty_returns_zero_samples():
    out = aggregate([])
    assert out == {"samples": 0}
