"""Tests for ``ongiini.tools.contribute.contribute_translation``."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from owela import ToolContext
from ongiini import contributions
from ongiini.tools.contribute import contribute_translation


@dataclass
class _FakeRuntime:
    """The tool only needs ctx.user_id — runtime isn't touched. Kept
    as a stub so ToolContext stays well-typed."""
    pass


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "contributions.sqlite"
    monkeypatch.setattr(contributions, "_db_path", lambda: db)
    monkeypatch.setattr(contributions.settings, "contributions_hash_salt", "test-salt")
    contributions.warmup()
    yield


def _ctx(msisdn: str = "264811234567") -> ToolContext:
    return ToolContext(
        user_id=msisdn,
        runtime=_FakeRuntime(),  # type: ignore[arg-type]
        msg=None,                # type: ignore[arg-type]
    )


def _seed(n: int = 3) -> None:
    contributions.seed_tasks([
        {"source_en": f"Sentence {i}", "category": "conversational", "seed_id": i}
        for i in range(1, n + 1)
    ])


# ── action validation ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_action_returns_error():
    out = json.loads(await contribute_translation(_ctx(), action="explode"))
    assert "error" in out
    assert "valid_actions" in out


@pytest.mark.asyncio
async def test_missing_hash_salt_returns_soft_error(monkeypatch):
    monkeypatch.setattr(contributions.settings, "contributions_hash_salt", "")
    out = json.loads(await contribute_translation(_ctx(), action="whoami"))
    assert "error" in out
    assert "temporarily" in out["error"]


# ── whoami ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_whoami_returns_new_for_unknown_contributor():
    out = json.loads(await contribute_translation(_ctx(), action="whoami"))
    assert out["status"] == "new"
    assert out["recently_declined"] is False
    assert out["total_contributions"] == 0


@pytest.mark.asyncio
async def test_whoami_returns_known_after_set_dialect():
    await contribute_translation(_ctx(), action="set_dialect", target_dialect="Oshindonga")
    out = json.loads(await contribute_translation(_ctx(), action="whoami"))
    assert out["status"] == "known:Oshindonga"
    assert out["recently_declined"] is False


@pytest.mark.asyncio
async def test_whoami_isolates_users():
    """Two different msisdns map to two different contributor states."""
    await contribute_translation(_ctx("264811111111"), action="set_dialect", target_dialect="Oshindonga")
    out = json.loads(await contribute_translation(_ctx("264822222222"), action="whoami"))
    assert out["status"] == "new"


@pytest.mark.asyncio
async def test_whoami_reflects_recent_decline():
    await contribute_translation(_ctx(), action="decline")
    out = json.loads(await contribute_translation(_ctx(), action="whoami"))
    assert out["recently_declined"] is True


# ── decline ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_decline_records_cooldown():
    out = json.loads(await contribute_translation(_ctx(), action="decline"))
    assert out["ok"] is True
    assert out["cooldown_days"] == 7


# ── set_dialect ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_dialect_oshindonga():
    out = json.loads(await contribute_translation(
        _ctx(), action="set_dialect", target_dialect="Oshindonga",
    ))
    assert out == {"ok": True, "preferred_dialect": "Oshindonga"}


@pytest.mark.asyncio
async def test_set_dialect_oshikwanyama():
    out = json.loads(await contribute_translation(
        _ctx(), action="set_dialect", target_dialect="Oshikwanyama",
    ))
    assert out == {"ok": True, "preferred_dialect": "Oshikwanyama"}


@pytest.mark.asyncio
async def test_set_dialect_rejects_invalid():
    out = json.loads(await contribute_translation(
        _ctx(), action="set_dialect", target_dialect="Klingon",
    ))
    assert "error" in out
    assert "invalid dialect" in out["error"]


@pytest.mark.asyncio
async def test_set_dialect_requires_target_dialect_param():
    out = json.loads(await contribute_translation(_ctx(), action="set_dialect"))
    assert "error" in out
    assert "target_dialect" in out["error"]


# ── next ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_next_returns_task_when_pool_has_rows():
    _seed(2)
    out = json.loads(await contribute_translation(_ctx(), action="next"))
    assert out["task"] is not None
    assert "id" in out["task"]
    assert "source_en" in out["task"]


@pytest.mark.asyncio
async def test_next_returns_null_when_pool_empty():
    out = json.loads(await contribute_translation(_ctx(), action="next"))
    assert out["task"] is None
    assert "message" in out


# ── save ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_writes_contribution():
    _seed(1)
    next_out = json.loads(await contribute_translation(_ctx(), action="next"))
    task_id = next_out["task"]["id"]
    save_out = json.loads(await contribute_translation(
        _ctx(),
        action="save",
        target_dialect="Oshindonga",
        task_id=task_id,
        translation="ondi ya nawa",
    ))
    assert save_out["ok"] is True
    assert save_out["total_for_contributor"] == 1
    assert "contribution_id" in save_out


@pytest.mark.asyncio
async def test_save_pii_sanitises_translation():
    _seed(1)
    next_out = json.loads(await contribute_translation(_ctx(), action="next"))
    task_id = next_out["task"]["id"]
    await contribute_translation(
        _ctx(),
        action="save",
        target_dialect="Oshindonga",
        task_id=task_id,
        translation="contact me at user@example.com",
    )
    # Verify the stored row is sanitised
    with contributions._conn() as c:
        row = c.execute(
            "SELECT target_translation FROM contributions ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert "user@example.com" not in row["target_translation"]


@pytest.mark.asyncio
async def test_save_rejects_invalid_dialect():
    _seed(1)
    next_out = json.loads(await contribute_translation(_ctx(), action="next"))
    out = json.loads(await contribute_translation(
        _ctx(),
        action="save",
        target_dialect="Spanish",
        task_id=next_out["task"]["id"],
        translation="hola",
    ))
    assert "error" in out


@pytest.mark.asyncio
async def test_save_rejects_missing_task_id():
    out = json.loads(await contribute_translation(
        _ctx(), action="save", target_dialect="Oshindonga", translation="x",
    ))
    assert "error" in out
    assert "task_id" in out["error"]


@pytest.mark.asyncio
async def test_save_rejects_missing_translation():
    _seed(1)
    next_out = json.loads(await contribute_translation(_ctx(), action="next"))
    out = json.loads(await contribute_translation(
        _ctx(),
        action="save",
        target_dialect="Oshindonga",
        task_id=next_out["task"]["id"],
        translation="   ",
    ))
    assert "error" in out
    assert "translation" in out["error"]


@pytest.mark.asyncio
async def test_save_rejects_nonexistent_task_id():
    out = json.loads(await contribute_translation(
        _ctx(),
        action="save",
        target_dialect="Oshindonga",
        task_id=99999,
        translation="something",
    ))
    assert "error" in out
    assert "does not exist" in out["error"]


# ── stats ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stats_returns_summary_shape():
    _seed(2)
    next_out = json.loads(await contribute_translation(_ctx(), action="next"))
    await contribute_translation(
        _ctx(),
        action="save",
        target_dialect="Oshindonga",
        task_id=next_out["task"]["id"],
        translation="test",
    )
    out = json.loads(await contribute_translation(_ctx(), action="stats"))
    assert out["total_contributions"] == 1
    assert out["by_dialect"]["Oshindonga"] == 1
    assert out["total_contributors"] == 1
    assert out["total_tasks"] == 2


# ── registry ──────────────────────────────────────────────────────


def test_contribute_translation_is_registered_with_owela():
    spec = contribute_translation.__owela_tool__  # type: ignore[attr-defined]
    assert spec.name == "contribute_translation"
    assert spec.needs_context is True
    assert "action" in spec.parameters["properties"]
