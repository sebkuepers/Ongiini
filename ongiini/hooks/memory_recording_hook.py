"""Ongiini's MemoryRecordingHook.

Subclassing the built-in pattern with Ongiini-specific persistence
rules:

  1. **Skip persistence when delete_my_data fired.** When the user
     asked to wipe their record, we MUST NOT re-persist this turn —
     the deletion request itself would be the new oldest memory.

  2. **PII sanitisation at write time.** The LLM sees the raw user
     text (so it can answer the actual question), but what lands on
     disk and in mem0 is sanitised. Closes the gap that mem0 could
     otherwise persist raw emails / ID numbers / IBANs as typed facts.

  3. **Image vs text routing.** Image-bearing inbound messages route
     through ``record_image_turn`` (synthesises a text placeholder for
     mem0); text messages go through ``record_turn``.

  4. **Storage text override.** ``InboundMessage.storage_text`` (if
     set) replaces the raw text for the persisted form, e.g.
     "[voice note] <transcript>" for audio turns.

This hook lives in the application package, not in ``owela/``,
because PII rules and storage labels are product-specific. Other
Owela applications would write their own.
"""

from __future__ import annotations

import logging
from typing import Callable

from owela import ReplyStep, Step, ToolStep, TurnContext

from ..memory import OngiiniMemoryProvider

log = logging.getLogger("ongiini.hooks.memory_recording")


# Type alias for the PII sanitiser callable. ``pii.sanitize(text) -> text``
# is the contract; the application injects the actual module function.
Sanitiser = Callable[[str], str]


class OngiiniMemoryRecordingHook:
    """One persistence write per successful turn, with Ongiini's policy."""

    def __init__(self, sanitiser: Sanitiser) -> None:
        self._sanitise = sanitiser

    async def on_turn_complete(self, steps: list[Step], ctx: TurnContext) -> None:
        # Locate the reply step. If there isn't one, nothing was sent —
        # nothing to persist.
        reply_step = next(
            (s for s in reversed(steps) if isinstance(s, ReplyStep)), None
        )
        if reply_step is None or not reply_step.sent:
            return

        # Skip persistence when the user asked to delete their data.
        # The deletion is already done by the tool; we don't want to
        # re-create memory by storing this turn.
        if self._delete_fired(steps):
            log.info("memory hook: skipping persist (delete_my_data fired)")
            return

        reply_text = reply_step.attrs.get("reply_text", "")
        sanitised_reply = self._sanitise(reply_text)

        # Image branch — uses record_image_turn (special mem0 path).
        if ctx.msg.has_image:
            sanitised_caption = self._sanitise(ctx.msg.text)
            await _safe(
                ctx.runtime.memory.record_image_turn(
                    ctx.msg.user_id, sanitised_caption, sanitised_reply,
                )
            )
            return

        # Text branch — storage_text override or raw text.
        raw_user_text = ctx.msg.storage_text or ctx.msg.text
        sanitised_user = self._sanitise(raw_user_text)
        await _safe(
            ctx.runtime.memory.record_turn(
                ctx.msg.user_id, sanitised_user, sanitised_reply,
            )
        )

    @staticmethod
    def _delete_fired(steps: list[Step]) -> bool:
        for s in steps:
            if isinstance(s, ToolStep) and s.tool_name == "delete_my_data":
                # Tool fired AND succeeded (error=None).
                if s.error is None:
                    return True
        return False


async def _safe(coro) -> None:
    """Run a coro, swallow + log any exception. Persistence must never
    break the success path."""
    try:
        await coro
    except Exception as exc:                            # noqa: BLE001 — soft-fail
        log.warning("memory recording hook write failed: %s", exc)
