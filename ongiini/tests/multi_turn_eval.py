"""Multi-turn research eval — runs against a LIVE backend.

Each scenario seeds ``InboundMessage.history`` with 2-3 prior turns
then asks a context-dependent follow-up. Scores on 4 dimensions to
catch the failure modes we've observed in production:

  1. **planner_queries**     — `queries_count >= 2` (planner resolved
     pronouns and decomposed the follow-up; soft-fail to 0 = fail).
  2. **citation_present**    — reply contains a `— source:` line with
     a deep URL (path, not just publication homepage).
  3. **latency_under_25s**   — total turn within WhatsApp's typing
     indicator window.
  4. **no_reasoning_leak**   — reply doesn't contain `<|` or a
     leading "thought" preamble (v1.3.2 hotfix audit).

Run via::

    docker exec -i ongiini-webhook python -m ongiini.tests.multi_turn_eval > /tmp/multi-turn.json

Output is JSON with per-scenario scores + an aggregate. The eval is
intentionally permissive — it's looking for systemic failure
patterns, not pixel-perfect answers.

NOT a pytest target — it requires a live agent (vLLM, Tavily, etc.).
The scoring helpers below are pure-logic and ARE unit-tested in
``test_multi_turn_scorer.py``.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Reasoning-leak markers — see v1.3.2 hotfix in ongiini/models/vllm_gemma.py.
_LEAK_TOKEN_RE = re.compile(r"<\|?[a-zA-Z0-9_\-]+\|?>")
_LEAK_PREAMBLE_HEADS = ("thought", "wait", "(wait")

# Deep-URL citation: a "— source:" or "- source:" line with a URL
# that has a path beyond the domain (so a homepage doesn't count).
_DEEP_URL_RE = re.compile(
    r"(?:[—-]\s*source:?\s*|^source:?\s*)(https?://[^\s/]+/\S+)",
    re.MULTILINE,
)


# ----- pure-logic scoring (unit-tested in test_multi_turn_scorer.py) -----


def score_planner_queries(trace_entry: dict[str, Any]) -> bool:
    """Did the planner emit >= 2 structured queries for this turn?"""
    for phase in trace_entry.get("phases", []):
        if phase.get("kind") == "plan":
            return int(phase.get("queries_count", 0) or 0) >= 2
    return False


def score_citation_present(reply_text: str) -> bool:
    """Reply contains at least one `— source: <deep URL>` line."""
    return bool(_DEEP_URL_RE.search(reply_text))


def score_latency_under_25s(trace_entry: dict[str, Any]) -> bool:
    return int(trace_entry.get("total_latency_ms", 0) or 0) < 25_000


def score_no_reasoning_leak(reply_text: str) -> bool:
    """Reply is clean of Gemma 4 reasoning detritus.

    Two checks:
      - No ``<|...|>`` / ``<|...>`` special tokens.
      - Doesn't START with the "thought" / "wait" / "(wait" reasoning
        preamble that the v1.3.2 scrubber sometimes misses on long
        single-line leaks.

    TODO: the "thought" head check is greedy on legitimate words like
    "Thoughtfully" (see ``test_score_no_reasoning_leak_thoughtfully_known_limitation``).
    Future fix: tighten the check to require a non-alphanumeric character
    after "thought" (i.e. boundary detection).
    """
    if _LEAK_TOKEN_RE.search(reply_text):
        return False
    head = reply_text.lstrip()[:20].lower()
    return not any(head.startswith(p) for p in _LEAK_PREAMBLE_HEADS)


# Aggregate scoring across a list of scenario results.

def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute pass rates per dimension across all scenarios."""
    if not results:
        return {"samples": 0}
    dims = ("planner_queries", "citation_present", "latency_under_25s", "no_reasoning_leak")
    out: dict[str, Any] = {"samples": len(results)}
    for d in dims:
        passed = sum(1 for r in results if r["scores"].get(d) is True)
        out[f"{d}_pass_rate"] = round(passed / len(results) * 100, 1)
    # Composite: scenario passes iff ALL four dimensions pass.
    all_pass = sum(1 for r in results if all(r["scores"].get(d) for d in dims))
    out["all_dimensions_pass_rate"] = round(all_pass / len(results) * 100, 1)
    return out


# ------------------------- scenarios + runner -------------------------


# Each scenario is a list of (role, content) tuples followed by the
# final user question. The runner replays history + sends the final
# message via Agent.handle(), then reads the latest trace entry.

SCENARIOS = [
    {
        "id": "pronoun_compare_them",
        "history": [
            ("user", "How many data centers exist in Namibia?"),
            ("assistant",
             "There are about 6 main data centers in Namibia, including "
             "Paratus Armada, IT Guru, MTN Windhoek, UNAM Datacenter, "
             "SALT DC and Telecom Namibia."),
        ],
        "question": "Compare them and tell me which is best for a small business",
    },
    {
        "id": "numerical_drill_down",
        "history": [
            ("user", "What are home loan rates at 3 Namibian banks?"),
            ("assistant",
             "Bank Windhoek offers home loans from around prime + 1.5%, "
             "FNB Namibia from prime + 1.25%, and Nedbank from prime + 1.0%-2.0%."),
        ],
        "question": "Which would be cheapest for a young family with one income?",
    },
    {
        "id": "source_request_after_shallow",
        "history": [
            ("user", "What's the BoN exchange rate today?"),
            ("assistant",
             "The BoN benchmark is currently about N$18.40 to the US dollar; "
             "the NAD is pegged 1:1 to the South African Rand."),
        ],
        "question": "What's the source for that — can you give me the actual URL?",
    },
    {
        "id": "topic_switch_within_thread",
        "history": [
            ("user", "Tell me about Namibian banks"),
            ("assistant",
             "Namibia has four main commercial banks: Bank Windhoek, FNB Namibia, "
             "Nedbank Namibia and Standard Bank Namibia, plus several development banks."),
        ],
        "question": "Now tell me about Namibian medical aid schemes",
    },
    {
        "id": "recency_followup",
        "history": [
            ("user", "Is there a medicine shortage in Namibia?"),
            ("assistant",
             "Yes, Namibia has been experiencing pharmaceutical supply chain "
             "stress, with several reports from health ministry officials."),
        ],
        "question": "Has anything changed in the past week?",
    },
]


async def _run_scenario(scenario: dict, agent, trace_path: Path) -> dict[str, Any]:
    """Run one scenario end-to-end and capture its scores.

    Imports happen inside so this module is importable without a
    live agent (the test_multi_turn_scorer.py file imports the
    scoring helpers without spinning up Runtime).
    """
    from owela import InboundMessage

    user_id = f"+264v14eval_{scenario['id']}"
    history = [{"role": role, "content": content} for role, content in scenario["history"]]
    msg = InboundMessage(
        user_id=user_id, msg_id="",
        text=scenario["question"],
        content_parts=[{"type": "text", "text": scenario["question"]}],
        history=history,
    )

    started = time.monotonic()
    reply_text = ""
    try:
        result = await agent.handle(msg)
        # Agent.handle() returns a HandleResult with reply_text
        # populated from ReplyStep.attrs — that's the post-revise
        # final text the user would see on WhatsApp.
        reply_text = result.reply_text or ""
        error: str | None = None
    except Exception as exc:                       # noqa: BLE001
        error = str(exc)
    elapsed_s = time.monotonic() - started

    # Find the trace entry for the structural fields (latency, planner
    # queries_count). Reply text comes from HandleResult above.
    trace_entry, _ = _find_latest_trace(trace_path, user_id)

    scores: dict[str, bool] = {}
    if trace_entry is not None:
        scores["planner_queries"] = score_planner_queries(trace_entry)
        scores["latency_under_25s"] = score_latency_under_25s(trace_entry)
    else:
        scores["planner_queries"] = False
        scores["latency_under_25s"] = False
    scores["citation_present"] = score_citation_present(reply_text or "")
    scores["no_reasoning_leak"] = score_no_reasoning_leak(reply_text or "")

    return {
        "id": scenario["id"],
        "elapsed_s": round(elapsed_s, 1),
        "scores": scores,
        "trace_found": trace_entry is not None,
        "reply_len": len(reply_text or ""),
        "error": error,
    }


def _find_latest_trace(trace_path: Path, user_id: str) -> tuple[dict | None, str]:
    """Read trace.jsonl tail, return the most recent entry for
    ``user_id``. Reply text is sourced from ``HandleResult.reply_text``
    by the caller (NOT from trace.jsonl, which deliberately doesn't
    persist message content for privacy)."""
    if not trace_path.exists():
        return None, ""
    latest: dict | None = None
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("msisdn") == user_id:
                latest = d
    return latest, ""


async def main() -> int:
    from ongiini.runtime import build_agent
    from ongiini.config import settings

    trace_path = settings.data_dir / "trace.jsonl"
    agent = build_agent()
    results: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        print(f"[eval] {scenario['id']}...", file=sys.stderr, flush=True)
        result = await _run_scenario(scenario, agent, trace_path)
        results.append(result)

    report = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenarios": results,
        "aggregate": aggregate(results),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
