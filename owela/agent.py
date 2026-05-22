"""Agent — the public entry point.

Most applications need exactly two things: a Runtime built at startup
and a handle() method to call per inbound. Agent wraps that. The
executor lives one layer below and is used by tests directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .executor import execute_turn
from .runtime import Runtime
from .step import ReplyStep, Step
from .transport import InboundMessage


@dataclass
class HandleResult:
    """Lightweight summary the application can act on. The full ``steps``
    list is also returned for callers that need richer observability."""
    sent: bool
    reply_text: str
    steps: list[Step]


class Agent:
    """Top-level orchestrator. Construct once per Runtime; call ``handle``
    per inbound message.

    Thread-safety: the Agent itself is stateless. Concurrency control
    (per-user locks, duplicate detection) lives in the application's
    transport receiver, NOT here.
    """
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    async def handle(self, msg: InboundMessage) -> HandleResult:
        steps = await execute_turn(self.runtime, msg)
        # Pull the terminal ReplyStep for the summary; fall back gracefully
        # if the executor returned early without one (shouldn't happen, but
        # better than a crash on a malformed step list).
        reply = next(
            (s for s in reversed(steps) if isinstance(s, ReplyStep)),
            None,
        )
        return HandleResult(
            sent=bool(reply and reply.sent),
            reply_text=(reply.attrs.get("reply_text", "") if reply else ""),
            steps=steps,
        )
