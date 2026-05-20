"""Structured per-message tracing.

One JSON line per handled WhatsApp message lands in /data/trace.jsonl,
capturing what the model decided turn-by-turn. Distinct from usage.log
(which is a summary line per message) and from message memory files
(which store assistant <-> user turns for follow-up context).

Trace data deliberately never includes:
- the user's message text
- the assistant's reply text
- tool arguments verbatim
- tool result content

Only structural signals: lengths, names, token counts, latencies,
finish reasons. Enough to diagnose model behaviour without leaking
content.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from .config import settings

TRACE_PATH = settings.data_dir / "trace.jsonl"


@dataclass
class CallTrace:
    turn: int
    tokens_in: int
    tokens_out: int
    finish_reason: str | None
    latency_ms: int
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class MessageTrace:
    ts: str
    msisdn: str
    user_msg_len: int
    history_len: int
    calls: list[CallTrace] = field(default_factory=list)
    reply_len: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_latency_ms: int = 0
    used_search: bool = False
    deleted_data: bool = False
    truncated: bool = False  # True if we hit the loop cap

    def add_call(
        self,
        turn: int,
        tokens_in: int,
        tokens_out: int,
        finish_reason: str | None,
        started_at: float,
    ) -> CallTrace:
        ct = CallTrace(
            turn=turn,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            finish_reason=finish_reason,
            latency_ms=int((time.monotonic() - started_at) * 1000),
        )
        self.calls.append(ct)
        return ct

    def write(self) -> None:
        self.total_tokens_in = sum(c.tokens_in for c in self.calls)
        self.total_tokens_out = sum(c.tokens_out for c in self.calls)
        self.total_latency_ms = sum(c.latency_ms for c in self.calls)
        line = json.dumps(asdict(self), ensure_ascii=False)
        with TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
