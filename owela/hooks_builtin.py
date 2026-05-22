"""Built-in Hook implementations shipped with Owela.

These are opt-in: an application that wants the behaviour adds the
hook to its ``HookRegistry``. Anti-trap principle #4: cross-cutting
concerns are hooks, including the ones the framework itself offers.

If you find yourself adding new behaviour here, ask: is this truly
framework-generic, or is it application-specific? Application-specific
hooks belong in ``ongiini/hooks/``, not here.
"""

from __future__ import annotations

import logging

from .hooks import Hook, TurnContext
from .step import ReplyStep, Step

log = logging.getLogger("owela.hooks_builtin")


class MemoryRecordingHook(Hook):
    """Calls ``runtime.memory.record_turn`` after the reply is sent.

    Standard pattern for Owela applications: add this to ``HookRegistry``
    at runtime construction time. If you don't want persistence (e.g. a
    stateless CLI demo), simply omit it.

    Soft-fail: persistence errors are logged and swallowed — a broken
    memory write must not prevent us reporting a successful reply.
    """

    async def on_turn_complete(self, steps: list[Step], ctx: TurnContext) -> None:
        reply_step = next(
            (s for s in reversed(steps) if isinstance(s, ReplyStep)), None
        )
        if reply_step is None or not reply_step.sent:
            return
        reply_text = reply_step.attrs.get("reply_text", "")
        try:
            await ctx.runtime.memory.record_turn(
                ctx.msg.user_id, ctx.msg.text, reply_text,
            )
        except Exception as exc:                       # noqa: BLE001 — soft-fail
            log.warning("MemoryRecordingHook.record_turn failed: %s", exc)
