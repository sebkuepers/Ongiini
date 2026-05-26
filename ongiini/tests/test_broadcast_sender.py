"""Tests for ongiini.broadcast.sender — the proactive-message path.

Critical invariants tested here:
  - Memory MUST be written BEFORE the Meta API call. Otherwise a
    user replying to a broadcast hits a context-blind bot.
  - The synthetic memory turn is role='assistant' (so the agent's
    next-turn assembly sees it as "what I said last").
  - Failure modes: 4xx (permanent) and 5xx-after-retries surface
    as ok=False but the memory write still happened (orphan turn
    is the lesser evil vs the opposite).
  - Dry-run mode writes nothing and sends nothing.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest


os.environ.setdefault("CONTRIBUTIONS_HASH_SALT", "test-salt")


@pytest.fixture
def temp_data_dir(tmp_path: Path, monkeypatch):
    from ongiini.config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


def _read_memory(msisdn: str) -> list[dict]:
    from ongiini.memory import short_term as memory
    return memory.load(msisdn)


# ── render_template_body ───────────────────────────────────────────


def test_render_template_body_matches_template():
    """The local-render must mirror what WhatsApp actually shows.
    If the deployed template changes, this test is the canary."""
    from ongiini.broadcast.sender import render_template_body
    out = render_template_body("Voice notes are live — try sending one.")
    assert out == (
        "Update from Ongiini AI:\n\n"
        "Voice notes are live — try sending one."
    )


def test_render_includes_brand_with_ai_suffix():
    """Brand rule: it's 'Ongiini AI', never just 'Ongiini'."""
    from ongiini.broadcast.sender import render_template_body
    out = render_template_body("test")
    assert "Ongiini AI" in out


# ── broadcast_to: happy path ───────────────────────────────────────


@pytest.mark.asyncio
async def test_broadcast_writes_memory_before_meta_call(temp_data_dir: Path):
    """The CRITICAL invariant Sebastian flagged. Order:
      1. memory.append_synthetic_assistant_turn
      2. whatsapp.send_template
    Reversal would mean a user reply lands context-blind if step 2
    fires first and step 1 fails.
    """
    call_order: list[str] = []

    from ongiini.memory import short_term as st_mod
    real_append = st_mod.append_synthetic_assistant_turn

    def spy_append(msisdn, text):
        call_order.append("memory")
        real_append(msisdn, text)

    async def spy_send(**kwargs):
        call_order.append("meta")
        return {"messages": [{"id": "wamid.XYZ"}]}

    with patch.object(st_mod, "append_synthetic_assistant_turn", side_effect=spy_append):
        with patch("ongiini.broadcast.sender.send_template", side_effect=spy_send):
            from ongiini.broadcast.sender import broadcast_to
            result = await broadcast_to(
                "+264811000001", "Hello world", url_suffix=""
            )

    assert call_order == ["memory", "meta"], (
        f"memory write must happen BEFORE Meta send, got {call_order}"
    )
    assert result.ok is True
    assert result.memory_written is True
    assert result.meta_message_id == "wamid.XYZ"


@pytest.mark.asyncio
async def test_broadcast_appends_assistant_role_to_memory(temp_data_dir: Path):
    """The next-turn assembly reads {role, content} dicts. The
    broadcast must land as role='assistant', otherwise the agent
    sees it as something else (or nothing) and replies cold."""
    from ongiini.broadcast.sender import broadcast_to

    with patch("ongiini.broadcast.sender.send_template",
               AsyncMock(return_value={"messages": [{"id": "wamid.X"}]})):
        await broadcast_to("+264811000002", "Voice notes are live", url_suffix="")

    mem = _read_memory("+264811000002")
    assert len(mem) == 1
    assert mem[0]["role"] == "assistant"
    assert "Voice notes are live" in mem[0]["content"]
    assert "Ongiini AI" in mem[0]["content"]


@pytest.mark.asyncio
async def test_broadcast_passes_correct_template_params(temp_data_dir: Path):
    """body_text → {{1}}, url_suffix → {{2}} (button URL param)."""
    from ongiini.broadcast.sender import broadcast_to
    from ongiini.config import settings

    captured: dict = {}
    async def fake(**kw):
        captured.update(kw)
        return {"messages": [{"id": "x"}]}

    with patch("ongiini.broadcast.sender.send_template", side_effect=fake):
        await broadcast_to(
            "+264811000003",
            "We added voice notes",
            url_suffix="contribute/",
        )

    assert captured["to"] == "+264811000003"
    assert captured["template_name"] == settings.whatsapp_template_announcement_name
    assert captured["body_params"] == ["We added voice notes"]
    assert captured["button_url_param"] == "contribute/"


# ── broadcast_to: dry-run ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_dry_run_writes_no_memory_and_no_send(temp_data_dir: Path):
    from ongiini.broadcast.sender import broadcast_to
    fake_send = AsyncMock()
    with patch("ongiini.broadcast.sender.send_template", fake_send):
        result = await broadcast_to(
            "+264811000004", "test", url_suffix="", dry_run=True
        )
    assert result.ok is False
    assert result.skipped_reason == "dry_run"
    assert result.memory_written is False
    fake_send.assert_not_called()
    assert _read_memory("+264811000004") == []


# ── broadcast_to: failure modes ────────────────────────────────────


@pytest.mark.asyncio
async def test_broadcast_4xx_returns_error_but_keeps_memory(temp_data_dir: Path):
    """A 4xx (e.g. recipient blocked us) is permanent and surfaces as
    error. We still keep the memory write — it's the lesser-evil
    failure mode: user never got the message, so the AI's 'I told
    them X' assumption isn't reached by any inbound reply."""
    from ongiini.broadcast.sender import broadcast_to

    fake_resp = httpx.Response(status_code=400, text='{"error": "blocked"}')
    fake_req = httpx.Request("POST", "https://example.com")
    fake_resp._request = fake_req

    async def fake_send(**kw):
        raise httpx.HTTPStatusError("blocked", request=fake_req, response=fake_resp)

    with patch("ongiini.broadcast.sender.send_template", side_effect=fake_send):
        result = await broadcast_to(
            "+264811000005", "test announcement", url_suffix=""
        )

    assert result.ok is False
    assert result.error is not None and "http_400" in result.error
    assert result.memory_written is True   # memory was written first
    assert _read_memory("+264811000005")   # turn is in storage


@pytest.mark.asyncio
async def test_broadcast_unexpected_exception_returns_error(temp_data_dir: Path):
    from ongiini.broadcast.sender import broadcast_to

    async def boom(**kw):
        raise RuntimeError("kaboom")

    with patch("ongiini.broadcast.sender.send_template", side_effect=boom):
        result = await broadcast_to("+264811000006", "test", url_suffix="")

    assert result.ok is False
    assert "RuntimeError" in result.error
    assert "kaboom" in result.error
