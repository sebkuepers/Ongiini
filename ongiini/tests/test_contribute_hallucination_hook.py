"""Tests for ContributeHallucinationGuardHook.

The hook is the recovery layer for bot-hallucinated translation tasks
— when the model serves an English sentence without calling
contribute_next, the hook detects the pattern in the reply and sets
pending_save retroactively so the next user reply lands as a normal save.

The hook's job is purely state-mutating; we verify by inspecting the
contributions DB after the hook fires."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from owela import InboundMessage, Policy, ReplyStep, ToolStep
from ongiini import contributions
from ongiini.hooks.contribute_hallucination_hook import (
    ContributeHallucinationGuardHook,
)


MSISDN = "264811234567"


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "contributions.sqlite"
    monkeypatch.setattr(contributions, "_db_path", lambda: db)
    monkeypatch.setattr(contributions.settings, "contributions_hash_salt", "test-salt")
    contributions.warmup()
    yield


@dataclass
class _Ctx:
    """Bare-minimum stand-in for TurnContext — the hook only reads
    ctx.msg.user_id."""
    msg: InboundMessage
    policy: Any = None
    runtime: Any = None


def _msg(msisdn: str = MSISDN) -> InboundMessage:
    return InboundMessage(
        user_id=msisdn, msg_id="t",
        text="yes", content_parts=[{"type": "text", "text": "yes"}],
    )


def _reply(text: str, sent: bool = True) -> ReplyStep:
    s = ReplyStep(reply_len=len(text), sent=sent)
    s.attrs["reply_text"] = text
    return s


# ── Hallucination detection: pattern matched, no tool fired ──────


@pytest.mark.asyncio
async def test_hook_detects_oshindonga_task_and_sets_pending():
    hook = ContributeHallucinationGuardHook()
    reply = _reply(
        'Tangi! Here\'s the next one — how would you say this in '
        'Oshindonga? "I am very grateful for your support."'
    )
    await hook.on_turn_complete([reply], _Ctx(msg=_msg()))

    h = contributions.hash_msisdn(MSISDN)
    pending = contributions.get_pending_save(h)
    assert pending is not None
    assert pending["dialect"] == "Oshindonga"
    # Confirm the task row was created with the right category
    import sqlite3
    conn = sqlite3.connect(contributions._db_path())
    row = conn.execute(
        "SELECT category, source_en FROM tasks WHERE id = ?",
        (pending["task_id"],),
    ).fetchone()
    conn.close()
    assert row[0] == "hallucinated_recovery"
    assert "grateful" in row[1]


@pytest.mark.asyncio
async def test_hook_detects_oshikwanyama_task_and_sets_pending():
    hook = ContributeHallucinationGuardHook()
    reply = _reply(
        'how would you say this in Oshikwanyama: "The weather is nice today"'
    )
    await hook.on_turn_complete([reply], _Ctx(msg=_msg()))

    h = contributions.hash_msisdn(MSISDN)
    pending = contributions.get_pending_save(h)
    assert pending is not None
    assert pending["dialect"] == "Oshikwanyama"


@pytest.mark.asyncio
async def test_hook_detects_case_insensitive():
    hook = ContributeHallucinationGuardHook()
    reply = _reply(
        'HOW WOULD YOU SAY THIS IN OSHINDONGA: "The weather is nice today"'
    )
    await hook.on_turn_complete([reply], _Ctx(msg=_msg()))
    h = contributions.hash_msisdn(MSISDN)
    assert contributions.get_pending_save(h) is not None


# ── Hook is NO-OP when legitimate tool fired ──────────────────────


@pytest.mark.asyncio
async def test_hook_skips_when_contribute_next_was_called():
    """If contribute_next fired this turn it set pending correctly —
    the bot's translation question is legitimate, NOT a hallucination."""
    hook = ContributeHallucinationGuardHook()
    reply = _reply(
        'how would you say this in Oshindonga: "I am grateful"'
    )
    tool = ToolStep(tool_name="contribute_next", error=None)
    await hook.on_turn_complete([tool, reply], _Ctx(msg=_msg()))

    h = contributions.hash_msisdn(MSISDN)
    # No pending should have been set by the hook (contribute_next would
    # set it via the tool path in production; here we're not running the
    # tool, just confirming the hook stays out of the way).
    pending = contributions.get_pending_save(h)
    assert pending is None  # only contribute_next's call would set it


@pytest.mark.asyncio
async def test_hook_skips_when_contribute_set_dialect_was_called():
    hook = ContributeHallucinationGuardHook()
    reply = _reply('how would you say this in Oshindonga: "test sentence x"')
    tool = ToolStep(tool_name="contribute_set_dialect", error=None)
    await hook.on_turn_complete([tool, reply], _Ctx(msg=_msg()))
    h = contributions.hash_msisdn(MSISDN)
    assert contributions.get_pending_save(h) is None


@pytest.mark.asyncio
async def test_hook_does_recover_when_tool_errored():
    """A tool call that errored didn't set state. The hook should
    still recover — error means the user's expectation wasn't met by
    the legitimate path."""
    hook = ContributeHallucinationGuardHook()
    reply = _reply('how would you say this in Oshindonga: "test sentence"')
    tool = ToolStep(tool_name="contribute_next", error="something failed")
    await hook.on_turn_complete([tool, reply], _Ctx(msg=_msg()))
    h = contributions.hash_msisdn(MSISDN)
    assert contributions.get_pending_save(h) is not None


# ── Hook is NO-OP when pattern not present ────────────────────────


@pytest.mark.asyncio
async def test_hook_skips_when_reply_has_no_task_pattern():
    hook = ContributeHallucinationGuardHook()
    reply = _reply("Tangi! Sure, here is the weather for Windhoek today...")
    await hook.on_turn_complete([reply], _Ctx(msg=_msg()))
    h = contributions.hash_msisdn(MSISDN)
    assert contributions.get_pending_save(h) is None


@pytest.mark.asyncio
async def test_hook_skips_when_reply_mentions_dialect_but_no_quoted_sentence():
    hook = ContributeHallucinationGuardHook()
    reply = _reply("Do you speak Oshindonga or Oshikwanyama?")
    await hook.on_turn_complete([reply], _Ctx(msg=_msg()))
    h = contributions.hash_msisdn(MSISDN)
    assert contributions.get_pending_save(h) is None


@pytest.mark.asyncio
async def test_hook_skips_when_no_reply_step():
    hook = ContributeHallucinationGuardHook()
    await hook.on_turn_complete([], _Ctx(msg=_msg()))
    h = contributions.hash_msisdn(MSISDN)
    assert contributions.get_pending_save(h) is None


@pytest.mark.asyncio
async def test_hook_skips_when_reply_not_sent():
    hook = ContributeHallucinationGuardHook()
    reply = _reply(
        'how would you say this in Oshindonga: "test sentence"',
        sent=False,
    )
    await hook.on_turn_complete([reply], _Ctx(msg=_msg()))
    h = contributions.hash_msisdn(MSISDN)
    assert contributions.get_pending_save(h) is None


# ── Resilience: hook never raises ─────────────────────────────────


@pytest.mark.asyncio
async def test_hook_soft_fails_on_internal_error(monkeypatch):
    """Per Hook contract: never raise. Internal errors swallowed + logged."""
    def _boom(*a, **k):
        raise RuntimeError("disk fire")
    monkeypatch.setattr(
        contributions, "create_hallucinated_recovery_task", _boom
    )
    hook = ContributeHallucinationGuardHook()
    reply = _reply('how would you say this in Oshindonga: "test sentence"')
    # Should not raise:
    await hook.on_turn_complete([reply], _Ctx(msg=_msg()))
