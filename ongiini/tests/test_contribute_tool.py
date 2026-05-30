"""Tests for the per-action contribute_* tools.

Replaces the v1 single-tool tests. The v2 design splits each action
into its own tool (no model-fillable args), force-called by the
classifier-driven policy table. Each tool reads its inputs from
ctx.msg.text + contributor state in sqlite."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from owela import InboundMessage, ToolContext
from ongiini import contributions
from ongiini.tools.contribute import (
    contribute_decline,
    contribute_invite_check,
    contribute_next,
    contribute_save,
    contribute_set_dialect,
    contribute_skip,
    contribute_stats,
)


@dataclass
class _FakeRuntime:
    pass


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "contributions.sqlite"
    monkeypatch.setattr(contributions, "_db_path", lambda: db)
    monkeypatch.setattr(contributions.settings, "contributions_hash_salt", "test-salt")
    contributions.warmup()
    yield


def _ctx(text: str = "", msisdn: str = "264811234567") -> ToolContext:
    msg = InboundMessage(
        user_id=msisdn,
        msg_id="m",
        text=text,
        content_parts=[{"type": "text", "text": text}],
    )
    return ToolContext(
        user_id=msisdn,
        runtime=_FakeRuntime(),  # type: ignore[arg-type]
        msg=msg,
    )


def _seed(n: int = 3) -> None:
    contributions.seed_tasks([
        {"source_en": f"Sentence {i}", "category": "conversational", "seed_id": i}
        for i in range(1, n + 1)
    ])


# ── contribute_invite_check ───────────────────────────────────────


@pytest.mark.asyncio
async def test_invite_check_returns_new_status_for_unknown_user():
    out = json.loads(await contribute_invite_check(_ctx()))
    assert out["status"] == "new"
    assert out["recently_declined"] is False
    assert out["total_contributions"] == 0


@pytest.mark.asyncio
async def test_invite_check_returns_recently_declined_true_after_decline():
    h = contributions.hash_msisdn("264811234567")
    contributions.record_decline(h)
    out = json.loads(await contribute_invite_check(_ctx()))
    assert out["recently_declined"] is True


@pytest.mark.asyncio
async def test_invite_check_returns_known_dialect_after_set():
    h = contributions.hash_msisdn("264811234567")
    contributions.set_dialect(h, "Oshindonga")
    out = json.loads(await contribute_invite_check(_ctx()))
    assert out["status"] == "known:Oshindonga"


# ── contribute_set_dialect ────────────────────────────────────────


@pytest.mark.parametrize("dialect_arg,expected_stored", [
    ("Oshindonga", "Oshindonga"),
    ("Oshikwanyama", "Oshikwanyama"),
    ("Either", "Oshindonga"),     # primary-target default
])
@pytest.mark.asyncio
async def test_set_dialect_stores_what_the_model_passes(dialect_arg, expected_stored):
    """The model — not the tool — interprets the user's free text. The
    tool's job is just to validate the model's choice and store it."""
    _seed(1)
    out = json.loads(await contribute_set_dialect(_ctx(text="anything"), dialect=dialect_arg))
    assert out["ok"] is True
    assert out["dialect"] == expected_stored
    h = contributions.hash_msisdn("264811234567")
    assert contributions.whoami(h) == f"known:{expected_stored}"


@pytest.mark.asyncio
async def test_set_dialect_chains_into_first_task():
    """After dialect-saving, the same call also fetches the first task
    so the user sees a real corpus sentence in a single turn."""
    _seed(1)
    out = json.loads(await contribute_set_dialect(_ctx(), dialect="Oshindonga"))
    assert out["ok"] is True
    assert out["task"] is not None
    h = contributions.hash_msisdn("264811234567")
    assert contributions.get_pending_save(h)["task_id"] == out["task"]["id"]


@pytest.mark.asyncio
async def test_set_dialect_with_empty_pool_returns_no_task():
    out = json.loads(await contribute_set_dialect(_ctx(), dialect="Oshindonga"))
    assert out["ok"] is True
    assert out["task"] is None
    assert "message" in out


@pytest.mark.asyncio
async def test_set_dialect_unclear_returns_error():
    """When the model genuinely can't tell what dialect the user
    picked, it passes 'Unclear' and the tool asks the user to
    clarify rather than storing a guess."""
    out = json.loads(await contribute_set_dialect(_ctx(), dialect="Unclear"))
    assert "error" in out


@pytest.mark.asyncio
async def test_set_dialect_invalid_value_returns_error():
    """Defensive: if the model passes something off-spec, treat as
    invalid rather than crashing or storing garbage."""
    out = json.loads(await contribute_set_dialect(_ctx(), dialect="Spanish"))
    assert "error" in out


# ── contribute_next ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_next_returns_task_and_sets_pending_when_dialect_known():
    _seed(1)
    h = contributions.hash_msisdn("264811234567")
    contributions.set_dialect(h, "Oshindonga")
    out = json.loads(await contribute_next(_ctx()))
    assert out["task"] is not None
    assert out["dialect"] == "Oshindonga"
    pending = contributions.get_pending_save(h)
    assert pending["task_id"] == out["task"]["id"]


@pytest.mark.asyncio
async def test_next_errors_when_dialect_not_known():
    _seed(1)
    out = json.loads(await contribute_next(_ctx()))
    assert "error" in out  # no dialect → can't serve


@pytest.mark.asyncio
async def test_next_returns_null_task_when_pool_empty():
    h = contributions.hash_msisdn("264811234567")
    contributions.set_dialect(h, "Oshindonga")
    out = json.loads(await contribute_next(_ctx()))
    assert out["task"] is None
    assert "message" in out


# ── contribute_save ───────────────────────────────────────────────


def _setup_pending() -> int:
    h = contributions.hash_msisdn("264811234567")
    contributions.set_dialect(h, "Oshindonga")
    task = contributions.next_task(h)
    contributions.set_pending_save(h, task["id"], "Oshindonga")
    return task["id"]


@pytest.mark.asyncio
async def test_save_writes_contribution_using_ctx_msg_text():
    _seed(1)
    task_id = _setup_pending()
    out = json.loads(await contribute_save(_ctx(text="ondi ya nawa")))
    assert out["ok"] is True
    # contribution_id is deliberately NOT in the response — the model
    # used to confuse it with total_for_contributor in the spoken reply.
    assert "contribution_id" not in out
    assert out["total_for_contributor"] == 1
    assert out["dialect"] == "Oshindonga"
    h = contributions.hash_msisdn("264811234567")
    assert contributions.get_pending_save(h) is None  # cleared
    # Save also sets awaiting_followup so the classifier routes the
    # user's next yes/no into CONTRIBUTE_NEXT or CONTRIBUTE_DECLINE.
    assert contributions.is_awaiting_followup(h) is True


@pytest.mark.asyncio
async def test_save_falls_back_to_orphan_when_no_pending_state():
    """Translation work is never silently dropped. When the classifier
    fires SAVE but no task was served, the text is saved as an orphan
    instead of returning an error."""
    out = json.loads(await contribute_save(_ctx(text="ondi ya nawa")))
    assert out.get("ok") is True
    assert out.get("orphan") is True
    assert out.get("dialect") == "unknown"
    # Confirm the text actually landed in the DB
    with contributions._conn() as c:
        row = c.execute(
            "SELECT target_translation, target_dialect FROM contributions "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row["target_translation"] == "ondi ya nawa"
    assert row["target_dialect"] == "unknown"


@pytest.mark.asyncio
async def test_save_still_errors_on_trivially_short_text_when_no_pending():
    """Don't pollute the dataset with classifier false positives on
    "yes" / "ok" / "thanks" — only save substantive text as orphans."""
    out = json.loads(await contribute_save(_ctx(text="yes")))
    assert "error" in out
    assert "pending" in out["error"]


@pytest.mark.asyncio
async def test_orphan_save_uses_known_dialect_when_available():
    """If the contributor has declared a dialect, the orphan save uses
    it instead of 'unknown' so curation is easier."""
    h = contributions.hash_msisdn("264811234567")
    contributions.set_dialect(h, "Oshindonga")
    out = json.loads(await contribute_save(_ctx(text="ondi ya nawa")))
    assert out.get("orphan") is True
    assert out.get("dialect") == "Oshindonga"


@pytest.mark.asyncio
async def test_save_pii_sanitises_translation_text():
    _seed(1)
    _setup_pending()
    await contribute_save(_ctx(text="email me at user@example.com please"))
    with contributions._conn() as c:
        row = c.execute(
            "SELECT target_translation FROM contributions ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert "user@example.com" not in row["target_translation"]


@pytest.mark.asyncio
async def test_save_rejects_empty_message():
    _seed(1)
    _setup_pending()
    out = json.loads(await contribute_save(_ctx(text="   ")))
    assert "error" in out


# ── contribute_skip ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_skip_drops_pending_and_serves_new_task():
    _seed(3)
    original_task = _setup_pending()
    out = json.loads(await contribute_skip(_ctx(text="skip")))
    assert out["task"] is not None
    assert out["task"]["id"] != original_task
    h = contributions.hash_msisdn("264811234567")
    new_pending = contributions.get_pending_save(h)
    assert new_pending["task_id"] == out["task"]["id"]
    # No contribution row was written
    assert contributions.total_contributions() == 0


@pytest.mark.asyncio
async def test_skip_returns_no_more_tasks_message_when_pool_exhausted():
    _seed(1)
    _setup_pending()
    # That seeded task is the only one; skipping it leaves nothing
    out = json.loads(await contribute_skip(_ctx(text="skip")))
    assert out["task"] is None


# ── contribute_decline ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_decline_records_cooldown_and_clears_pending():
    _seed(1)
    _setup_pending()
    h = contributions.hash_msisdn("264811234567")
    # Also ensure awaiting_followup gets cleared (decline can happen
    # from either the pending state or the awaiting_followup state)
    contributions.set_awaiting_followup(h)
    out = json.loads(await contribute_decline(_ctx(text="no thanks")))
    assert out["ok"] is True
    assert out["cooldown_days"] == contributions.DECLINE_COOLDOWN_DAYS
    assert contributions.get_pending_save(h) is None
    assert contributions.is_awaiting_followup(h) is False
    assert contributions.recently_declined(h) is True


# ── contribute_stats ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stats_returns_summary_dict():
    _seed(2)
    h = contributions.hash_msisdn("264811234567")
    contributions.set_dialect(h, "Oshindonga")
    contributions.next_task(h)  # serves task 1
    contributions.save_contribution(h, 1, "Oshindonga", "x")
    out = json.loads(await contribute_stats(_ctx()))
    assert out["total_contributions"] == 1
    assert out["by_dialect"]["Oshindonga"] == 1
    assert out["total_tasks"] == 2


# ── tool registration ────────────────────────────────────────────


def test_zero_param_contribute_tools_have_empty_schemas():
    """force_tool targets by name; for the no-arg tools the model
    never picks args — they should expose zero parameters."""
    for fn in (
        contribute_invite_check, contribute_next,
        contribute_save, contribute_skip, contribute_decline, contribute_stats,
    ):
        spec = fn.__owela_tool__  # type: ignore[attr-defined]
        assert spec.parameters["properties"] == {}, (
            f"{spec.name} should expose zero parameters"
        )
        assert spec.needs_context is True


def test_set_dialect_exposes_dialect_param():
    """contribute_set_dialect is the one contribute tool the model
    passes an arg to — `dialect`, the canonical name picked from the
    user's message. The model — not regex — interprets the message."""
    spec = contribute_set_dialect.__owela_tool__  # type: ignore[attr-defined]
    assert "dialect" in spec.parameters["properties"]
    assert spec.needs_context is True
