"""Tests for the opt_out_broadcast tool."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest


os.environ.setdefault("CONTRIBUTIONS_HASH_SALT", "test-salt")


@pytest.fixture
def temp_data_dir(tmp_path: Path, monkeypatch):
    from ongiini.config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


def _ctx(user_id: str = "+264800000000"):
    from owela import ToolContext
    from owela.transport import InboundMessage
    msg = InboundMessage(user_id=user_id, msg_id="m", text="STOP", content_parts=[])
    return ToolContext(user_id=user_id, runtime=MagicMock(), msg=msg)


@pytest.mark.asyncio
async def test_opt_out_tool_records_and_returns_ok(temp_data_dir: Path):
    from ongiini.broadcast import opt_outs
    from ongiini.tools.opt_out import opt_out_broadcast
    opt_outs.warmup()

    raw = await opt_out_broadcast(_ctx("+264811000099"))
    result = json.loads(raw)
    assert result["status"] == "ok"
    assert result["newly_added"] is True
    assert "message_to_compose" in result
    assert opt_outs.is_opted_out("+264811000099") is True


@pytest.mark.asyncio
async def test_opt_out_tool_idempotent(temp_data_dir: Path):
    from ongiini.broadcast import opt_outs
    from ongiini.tools.opt_out import opt_out_broadcast
    opt_outs.warmup()
    # First call → newly_added True; second call → newly_added False
    first = json.loads(await opt_out_broadcast(_ctx("+264811000088")))
    second = json.loads(await opt_out_broadcast(_ctx("+264811000088")))
    assert first["newly_added"] is True
    assert second["newly_added"] is False
    assert second["status"] == "ok"
    assert opt_outs.count() == 1


@pytest.mark.asyncio
async def test_opt_out_tool_soft_fails_on_missing_salt(temp_data_dir: Path, monkeypatch):
    """If the deployment is misconfigured (no salt), the tool returns
    an error JSON rather than raising — so the agent can still send an
    apology instead of crashing the turn."""
    from ongiini.config import settings
    from ongiini.broadcast import opt_outs
    from ongiini.tools.opt_out import opt_out_broadcast

    opt_outs.warmup()
    monkeypatch.setattr(settings, "contributions_hash_salt", "")
    raw = await opt_out_broadcast(_ctx("+264811000077"))
    result = json.loads(raw)
    assert result["status"] == "error"
    assert result["reason"] == "config_missing"
