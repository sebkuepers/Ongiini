"""Tests for the rate_link tool (classifier verdict RATE_INVITE)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from ongiini import contributions, ratings
from ongiini.config import settings
from ongiini.tools.rate import rate_link


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(contributions.settings, "contributions_hash_salt", "test-salt")


def _ctx(user_id="264811234567"):
    from owela import ToolContext
    from owela.transport import InboundMessage
    msg = InboundMessage(user_id=user_id, msg_id="m", text="I'd like to help check translations",
                         content_parts=[])
    return ToolContext(user_id=user_id, runtime=MagicMock(), msg=msg)


@pytest.mark.asyncio
async def test_new_then_returning_rater_never_stores_the_number():
    first = json.loads(await rate_link(_ctx()))
    assert first["status"] == "ok" and not first["returning"] and "#t=" in first["url"]
    second = json.loads(await rate_link(_ctx()))
    assert second["returning"] and second["url"] != first["url"]
    con = ratings.connect()
    rows = [dict(r) for r in con.execute("SELECT * FROM raters")]
    assert len(rows) == 1 and "264811234567" not in json.dumps(rows)
    assert rows[0]["contributor_hash"] == contributions.hash_msisdn("264811234567")


@pytest.mark.asyncio
async def test_web_chat_session_gets_no_link():
    assert json.loads(await rate_link(_ctx("3f2c9a1e-uuid")))["status"] == "whatsapp_only"


@pytest.mark.asyncio
async def test_blocked_number():
    con = ratings.connect()
    ratings.block_contributor(con, contributions.hash_msisdn("264811234567"))
    assert json.loads(await rate_link(_ctx()))["status"] == "blocked"


@pytest.mark.asyncio
async def test_soft_fails_without_salt(monkeypatch):
    monkeypatch.setattr(contributions.settings, "contributions_hash_salt", "")
    assert json.loads(await rate_link(_ctx()))["status"] == "error"
