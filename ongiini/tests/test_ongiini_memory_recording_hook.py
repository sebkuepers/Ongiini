"""Tests for OngiiniMemoryRecordingHook — the persistence hook that
handles PII sanitisation + deleted_data skip + image vs text routing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from owela import (
    InboundMessage, ModelCallStep, Policy, ReplyStep, ToolStep, TurnContext,
)
from ongiini.hooks.memory_recording_hook import OngiiniMemoryRecordingHook


def _sanitiser(prefix: str = "S:") -> MagicMock:
    """A fake sanitiser that prepends ``prefix`` so tests can confirm it ran."""
    s = MagicMock(side_effect=lambda t: f"{prefix}{t}" if t else "")
    return s


def _ctx(
    *, text: str = "user said hi", has_image: bool = False, storage_text: str = "",
) -> TurnContext:
    msg = InboundMessage(
        user_id="+264u",
        msg_id="m",
        text=text,
        content_parts=[{"type": "text", "text": text}],
        has_image=has_image,
        storage_text=storage_text,
    )
    runtime = MagicMock()
    runtime.memory.record_turn = AsyncMock()
    runtime.memory.record_image_turn = AsyncMock()
    return TurnContext(msg=msg, policy=Policy(name="test"), runtime=runtime)


def _reply_step(text: str = "bot replied", sent: bool = True) -> ReplyStep:
    rs = ReplyStep(reply_len=len(text), sent=sent)
    rs.attrs["reply_text"] = text
    return rs


# ---------- Happy paths ----------

@pytest.mark.asyncio
async def test_text_turn_records_sanitised_text_and_reply():
    hook = OngiiniMemoryRecordingHook(sanitiser=_sanitiser("S:"))
    ctx = _ctx(text="my email is foo@bar.com")
    await hook.on_turn_complete(
        [ModelCallStep(turn=1), _reply_step("bot reply")], ctx,
    )
    ctx.runtime.memory.record_turn.assert_awaited_once_with(
        "+264u", "S:my email is foo@bar.com", "S:bot reply",
    )
    ctx.runtime.memory.record_image_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_image_turn_records_via_record_image_turn():
    hook = OngiiniMemoryRecordingHook(sanitiser=_sanitiser("S:"))
    ctx = _ctx(text="look at my maize", has_image=True)
    await hook.on_turn_complete(
        [ModelCallStep(turn=1), _reply_step("I see leaves")], ctx,
    )
    ctx.runtime.memory.record_image_turn.assert_awaited_once_with(
        "+264u", "S:look at my maize", "S:I see leaves",
    )
    ctx.runtime.memory.record_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_storage_text_override_replaces_raw_text():
    """Voice notes set storage_text='[voice note] <transcript>' so the
    persisted form carries the marker — model still saw raw transcript."""
    hook = OngiiniMemoryRecordingHook(sanitiser=_sanitiser("S:"))
    ctx = _ctx(text="raw transcript here", storage_text="[voice note] raw transcript here")
    await hook.on_turn_complete(
        [ModelCallStep(turn=1), _reply_step("ok")], ctx,
    )
    ctx.runtime.memory.record_turn.assert_awaited_once_with(
        "+264u", "S:[voice note] raw transcript here", "S:ok",
    )


# ---------- Skip conditions ----------

@pytest.mark.asyncio
async def test_skips_when_delete_my_data_fired():
    """Privacy-critical: when the user asked to delete their data,
    the deletion request itself must NOT be re-persisted."""
    hook = OngiiniMemoryRecordingHook(sanitiser=_sanitiser())
    ctx = _ctx(text="delete my data")
    steps = [
        ModelCallStep(turn=1),
        ToolStep(tool_name="delete_my_data", result_len=10, error=None),
        ModelCallStep(turn=2),
        _reply_step("Done."),
    ]
    await hook.on_turn_complete(steps, ctx)
    ctx.runtime.memory.record_turn.assert_not_awaited()
    ctx.runtime.memory.record_image_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_does_not_skip_when_delete_tool_errored():
    """If delete_my_data failed, the turn should still persist —
    otherwise an error here would silently swallow the message."""
    hook = OngiiniMemoryRecordingHook(sanitiser=_sanitiser())
    ctx = _ctx(text="delete my data")
    steps = [
        ToolStep(tool_name="delete_my_data", error="boom"),
        _reply_step("Couldn't delete"),
    ]
    await hook.on_turn_complete(steps, ctx)
    ctx.runtime.memory.record_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_skips_when_no_reply_step():
    """Loop fell through max_steps with no ReplyStep — nothing was sent,
    nothing to persist."""
    hook = OngiiniMemoryRecordingHook(sanitiser=_sanitiser())
    ctx = _ctx()
    await hook.on_turn_complete([ModelCallStep(turn=1)], ctx)
    ctx.runtime.memory.record_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_when_reply_not_sent():
    """Transport said sent=False — don't persist a message that didn't
    arrive."""
    hook = OngiiniMemoryRecordingHook(sanitiser=_sanitiser())
    ctx = _ctx()
    rs = _reply_step()
    rs.sent = False
    await hook.on_turn_complete([ModelCallStep(turn=1), rs], ctx)
    ctx.runtime.memory.record_turn.assert_not_awaited()


# ---------- Failure modes ----------

@pytest.mark.asyncio
async def test_record_failure_does_not_propagate():
    """A broken persistence layer must NEVER break the turn — the user
    already got their reply, all we'd lose is one history entry."""
    hook = OngiiniMemoryRecordingHook(sanitiser=_sanitiser())
    ctx = _ctx()
    ctx.runtime.memory.record_turn = AsyncMock(side_effect=RuntimeError("disk gone"))
    # Should not raise.
    await hook.on_turn_complete([ModelCallStep(turn=1), _reply_step()], ctx)
