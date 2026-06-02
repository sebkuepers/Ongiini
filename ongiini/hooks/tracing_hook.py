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
    the destination path so tests can point at a tempfile.

    When ``include_critique_detail=True``, also embeds the raw critique
    text + a plan_text snippet in the per-phase trace entries — useful
    for debugging WHY the critique flipped REVISE or WHAT the planner
    actually said. Default OFF so production traces stay structural-
    only (no content). The flag is plumbed from
    ``settings.trace_critique_detail`` in the composition root.
    """

    def __init__(self, trace_path: Path, *, include_critique_detail: bool = False) -> None:
        self.trace_path = Path(trace_path)
        self.include_critique_detail = include_critique_detail

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
                    # v1.3 audit: distinguish real model calls from
                    # policy-synthesised dispatches (multi-query fan-out
                    # or auto-followup). EU AI Act provenance — make it
                    # clear which tool calls the model chose vs which
                    # were forced by deterministic policy.
                    "synthesized": s.synthesized,
                    "decision_source": s.attrs.get("decision_source"),
                    # v1.4 audit: count of Gemma 4 channel tokens the
                    # model adapter scrubbed from this call's content.
                    # 0 means the reasoning parser worked; >0 means the
                    # scrubber fired (operators can monitor recurrence
                    # via `trace_query.py reasoning-leak-count`).
                    "reasoning_leak_stripped": s.attrs.get("reasoning_leak_stripped", 0),
                })
                total_latency_ms += s.latency_ms()
                total_tokens_in += s.tokens_in
                total_tokens_out += s.tokens_out
            elif isinstance(s, PlanStep):
                plan_entry = {
                    "kind": "plan",
                    "tokens_in": s.tokens_in,
                    "tokens_out": s.tokens_out,
                    "cached_tokens": s.cached_tokens,
                    "plan_len": len(s.plan_text),
                    # v1.3: planner emits structured query variants
                    # that the executor synthesises as parallel tool
                    # calls. Surface the count for trace consumers.
                    "queries_count": len(s.queries),
                    "latency_ms": s.latency_ms(),
                    "error": s.attrs.get("error"),
                }
                if self.include_critique_detail:
                    # First 400 chars of the actual plan body — useful
                    # to verify what Gemma is suggesting and whether the
                    # plan is shaping the act-loop's tool choices.
                    plan_entry["plan_text"] = s.plan_text[:400]
                phases.append(plan_entry)
                total_latency_ms += s.latency_ms()
                total_tokens_in += s.tokens_in
                total_tokens_out += s.tokens_out
            elif isinstance(s, CritiqueStep):
                critique_entry = {
                    "kind": "critique",
                    "tokens_in": s.tokens_in,
                    "tokens_out": s.tokens_out,
                    "cached_tokens": s.cached_tokens,
                    "verdict": s.verdict,
                    "reasons_count": len(s.reasons),
                    "latency_ms": s.latency_ms(),
                    "error": s.attrs.get("error"),
                }
                if self.include_critique_detail:
                    # Extracted reasons + raw critique body. The raw
                    # body is the ground truth — reasons are what our
                    # parser extracted from it. Comparing the two tells
                    # us if the parser is missing things.
                    critique_entry["reasons"] = list(s.reasons)
                    critique_entry["raw_critique"] = s.attrs.get("raw_critique", "")
                phases.append(critique_entry)
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
                    # v1.3: tool result entry always carries the audit
                    # keys with None defaults — stable trace schema for
                    # downstream JSON parsers. None means "model chose
                    # this tool", a non-None value means "policy
                    # synthesised it" (EU AI Act provenance).
                    calls[-1].setdefault("tool_results", []).append({
                        "name": s.tool_name,
                        "args_len": s.args_len,
                        "result_len": s.result_len,
                        "error": s.error,
                        "latency_ms": s.latency_ms(),
                        "synthesized_by_policy": s.attrs.get("synthesized_by_policy"),
                        "decision_source": s.attrs.get("decision_source"),
                        "query_variant_index": s.attrs.get("query_variant_index"),
                    })

        return {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "msisdn": ctx.msg.user_id,
            "msg_id": ctx.msg.msg_id,
            # Channel the turn came in on — "whatsapp" or "web_chat".
            # Surfaced on the /statistics page (web-chat block) and used
            # by the stats aggregator to bucket trace rows. The aggregator
            # also has a UUID-v4 msisdn fallback for older records that
            # predate this field.
            "transport": getattr(ctx.runtime.transport, "name", None),
            # True when the user attached a photo this turn. Powers the
            # "Images" KPI tile on /statistics; cheaper than scanning
            # per-user memory files like the WhatsApp path used to.
            "has_image": bool(getattr(ctx.msg, "has_image", False)),
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
