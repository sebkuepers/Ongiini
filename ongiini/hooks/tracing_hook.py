"""Owela ``Hook`` that writes one JSON line per turn to ``trace.jsonl``.

Re-implements the current ``ongiini/tracing.py`` ``MessageTrace``
behaviour on top of the typed Step model. Each turn's steps are
serialised to a single line. Content of replies / user messages /
tool args is NEVER included — only structural metadata (lengths,
names, token counts, latencies, finish reasons).

Privacy contract (unchanged from the old code):
  - no user message text
  - no assistant reply text
  - no tool args verbatim
  - no tool result content
Only: lengths, names, counts, durations, status flags.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from owela import (
    CritiqueStep, ModelCallStep, PlanStep, ReplyStep, ReviseStep, RouterStep,
    Step, ToolStep, TurnContext,
)

log = logging.getLogger("ongiini.hooks.tracing")


class TracingHook:
    """Writes one JSON line per turn to the given file. Constructed with
    the destination path so tests can point at a tempfile."""

    def __init__(self, trace_path: Path) -> None:
        self.trace_path = Path(trace_path)

    async def on_turn_complete(self, steps: list[Step], ctx: TurnContext) -> None:
        try:
            line = json.dumps(self._build(steps, ctx), ensure_ascii=False)
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:                        # noqa: BLE001 — soft-fail
            log.warning("tracing: write failed: %s", exc)

    def _build(self, steps: list[Step], ctx: TurnContext) -> dict[str, Any]:
        router = next((s for s in steps if isinstance(s, RouterStep)), None)
        reply = next((s for s in reversed(steps) if isinstance(s, ReplyStep)), None)

        calls = []
        # phases: the v1 pre-/post-loop steps (planner, critique, revise).
        # Kept separate from `calls` so the trace shape stays clean — calls
        # are the act-loop iterations; phases are the agentic-quality
        # pre/post-processing.
        phases = []
        total_latency_ms = 0
        total_tokens_in = 0
        total_tokens_out = 0
        used_search = False

        for s in steps:
            if isinstance(s, ModelCallStep):
                calls.append({
                    "turn": s.turn,
                    "tokens_in": s.tokens_in,
                    "tokens_out": s.tokens_out,
                    "cached_tokens": s.cached_tokens,
                    "enable_thinking": s.enable_thinking,
                    "reasoning_budget": s.reasoning_budget,
                    "finish_reason": s.finish_reason,
                    "latency_ms": s.latency_ms(),
                    "tool_calls": [
                        {
                            "name": tc.get("function", {}).get("name", ""),
                            "args_len": len(tc.get("function", {}).get("arguments", "")),
                        }
                        for tc in s.tool_calls
                    ],
                })
                total_latency_ms += s.latency_ms()
                total_tokens_in += s.tokens_in
                total_tokens_out += s.tokens_out
            elif isinstance(s, PlanStep):
                phases.append({
                    "kind": "plan",
                    "tokens_in": s.tokens_in,
                    "tokens_out": s.tokens_out,
                    "cached_tokens": s.cached_tokens,
                    "plan_len": len(s.plan_text),
                    "latency_ms": s.latency_ms(),
                    "error": s.attrs.get("error"),
                })
                total_latency_ms += s.latency_ms()
                total_tokens_in += s.tokens_in
                total_tokens_out += s.tokens_out
            elif isinstance(s, CritiqueStep):
                phases.append({
                    "kind": "critique",
                    "tokens_in": s.tokens_in,
                    "tokens_out": s.tokens_out,
                    "cached_tokens": s.cached_tokens,
                    "verdict": s.verdict,
                    "reasons_count": len(s.reasons),
                    "latency_ms": s.latency_ms(),
                    "error": s.attrs.get("error"),
                })
                total_latency_ms += s.latency_ms()
                total_tokens_in += s.tokens_in
                total_tokens_out += s.tokens_out
            elif isinstance(s, ReviseStep):
                phases.append({
                    "kind": "revise",
                    "tokens_in": s.tokens_in,
                    "tokens_out": s.tokens_out,
                    "cached_tokens": s.cached_tokens,
                    "revised_len": len(s.attrs.get("revised_reply", "")),
                    "latency_ms": s.latency_ms(),
                    "error": s.attrs.get("error"),
                })
                total_latency_ms += s.latency_ms()
                total_tokens_in += s.tokens_in
                total_tokens_out += s.tokens_out
            elif isinstance(s, ToolStep):
                # Locate the parent call we attribute this tool to — the
                # most recent ModelCallStep before this one. If we can't
                # find one, attach a separate entry.
                if s.tool_name in ("web_search", "fetch_url", "fetch_urls"):
                    used_search = True
                if calls:
                    calls[-1].setdefault("tool_results", []).append({
                        "name": s.tool_name,
                        "args_len": s.args_len,
                        "result_len": s.result_len,
                        "error": s.error,
                        "latency_ms": s.latency_ms(),
                    })

        return {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "msisdn": ctx.msg.user_id,
            "msg_id": ctx.msg.msg_id,
            "user_msg_len": len(ctx.msg.text or ""),
            "history_len": len(ctx.msg.history),
            "policy": ctx.policy.name,
            "router": {
                "verdict": router.verdict if router else None,
                "depth": router.depth if router else None,
                "tokens_in": router.tokens_in if router else 0,
                "tokens_out": router.tokens_out if router else 0,
                "latency_ms": router.latency_ms() if router else 0,
            } if router else None,
            "calls": calls,
            "phases": phases,
            "reply_len": reply.reply_len if reply else 0,
            "sent": bool(reply and reply.sent),
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_latency_ms": total_latency_ms,
            "used_search": used_search,
            "truncated": reply is None,    # no ReplyStep = loop fell through
        }
