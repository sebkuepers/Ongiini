"""Unit tests for ongiini/contributions.py — the community-contribution
sqlite layer."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ongiini import contributions


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Every test gets its own sqlite file + a known hash salt.
    Patches the module's _db_path() to point at tmp_path and sets a
    fixed salt so hash_msisdn() works deterministically."""
    db = tmp_path / "contributions.sqlite"
    monkeypatch.setattr(contributions, "_db_path", lambda: db)
    monkeypatch.setattr(contributions.settings, "contributions_hash_salt", "test-salt")
    contributions.warmup()
    yield


def _h(msisdn: str = "264811234567") -> str:
    return contributions.hash_msisdn(msisdn)


# ── warmup / schema ────────────────────────────────────────────────


def test_warmup_creates_all_three_tables():
    with contributions._conn() as c:
        tables = {r["name"] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert {"tasks", "contributions", "contributors"} <= tables


def test_warmup_is_idempotent():
    contributions.warmup()
    contributions.warmup()
    contributions.warmup()
    with contributions._conn() as c:
        rows = c.execute(
            "SELECT COUNT(*) AS n FROM sqlite_master WHERE type='table' AND name='tasks'"
        ).fetchone()
    assert rows["n"] == 1


# ── hash_msisdn ────────────────────────────────────────────────────


def test_hash_msisdn_is_deterministic_with_same_salt():
    a = contributions.hash_msisdn("264811234567")
    b = contributions.hash_msisdn("264811234567")
    assert a == b
    assert len(a) == 64  # SHA-256 hex


def test_hash_msisdn_differs_between_users():
    assert contributions.hash_msisdn("264811234567") != contributions.hash_msisdn("264819876543")


def test_hash_msisdn_raises_without_salt(monkeypatch):
    monkeypatch.setattr(contributions.settings, "contributions_hash_salt", "")
    with pytest.raises(RuntimeError, match="CONTRIBUTIONS_HASH_SALT"):
        contributions.hash_msisdn("264811234567")


# ── seed_tasks / task_count / next_task ────────────────────────────


def _seed(n: int = 5) -> None:
    contributions.seed_tasks([
        {"source_en": f"Test sentence number {i}", "category": "conversational", "seed_id": i}
        for i in range(1, n + 1)
    ])


def test_seed_tasks_inserts_rows():
    inserted = contributions.seed_tasks([
        {"source_en": "Hello there", "category": "conversational", "seed_id": 1},
        {"source_en": "How are you", "category": "conversational", "seed_id": 2},
    ])
    assert inserted == 2
    assert contributions.task_count() == 2


def test_seed_tasks_skips_empty_sources():
    inserted = contributions.seed_tasks([
        {"source_en": "good one", "category": "conversational", "seed_id": 1},
        {"source_en": "", "category": "conversational", "seed_id": 2},
        {"source_en": "   ", "category": "conversational", "seed_id": 3},
        {"source_en": "another good one", "category": "conversational", "seed_id": 4},
    ])
    assert inserted == 2


def test_next_task_returns_a_task_when_pool_has_rows():
    _seed(5)
    task = contributions.next_task(_h())
    assert task is not None
    assert isinstance(task["id"], int)
    assert task["source_en"].startswith("Test sentence")


def test_next_task_returns_none_when_pool_is_empty():
    assert contributions.next_task(_h()) is None


def test_next_task_excludes_already_submitted_tasks_for_same_contributor():
    _seed(3)
    h = _h()
    # Submit all three for one user
    for _ in range(3):
        task = contributions.next_task(h)
        assert task is not None
        contributions.save_contribution(h, task["id"], "Oshindonga", "test translation")
    # Now the pool has no UNSEEN tasks for this contributor
    assert contributions.next_task(h) is None
    # But a different contributor still gets tasks
    h2 = contributions.hash_msisdn("264819999999")
    assert contributions.next_task(h2) is not None


# ── save_contribution ──────────────────────────────────────────────


def test_save_contribution_writes_row_with_dialect():
    _seed(1)
    h = _h()
    task = contributions.next_task(h)
    result = contributions.save_contribution(h, task["id"], "Oshindonga", "ondi ya nawa")
    assert result["contribution_id"] == 1
    assert result["total_for_contributor"] == 1


def test_save_contribution_runs_pii_sanitize_on_translation():
    _seed(1)
    h = _h()
    task = contributions.next_task(h)
    # PII-laden translation — email should get redacted
    contributions.save_contribution(
        h, task["id"], "Oshindonga", "Email me at user@example.com please"
    )
    with contributions._conn() as c:
        row = c.execute(
            "SELECT target_translation FROM contributions ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert "user@example.com" not in row["target_translation"]
    assert "[REDACTED:email]" in row["target_translation"]


def test_save_contribution_rejects_invalid_dialect():
    _seed(1)
    h = _h()
    task = contributions.next_task(h)
    with pytest.raises(ValueError, match="invalid dialect"):
        contributions.save_contribution(h, task["id"], "Spanish", "hola")


def test_save_contribution_rejects_nonexistent_task():
    h = _h()
    with pytest.raises(ValueError, match="does not exist"):
        contributions.save_contribution(h, 99999, "Oshindonga", "test")


def test_save_contribution_rejects_empty_translation_after_sanitisation():
    _seed(1)
    h = _h()
    task = contributions.next_task(h)
    with pytest.raises(ValueError, match="empty translation"):
        contributions.save_contribution(h, task["id"], "Oshindonga", "")


def test_save_contribution_increments_task_counters():
    _seed(1)
    h = _h()
    task = contributions.next_task(h)
    contributions.save_contribution(h, task["id"], "Oshindonga", "test")
    with contributions._conn() as c:
        row = c.execute(
            "SELECT times_served, times_submitted FROM tasks WHERE id = ?",
            (task["id"],),
        ).fetchone()
    assert row["times_served"] == 1     # incremented by next_task
    assert row["times_submitted"] == 1  # incremented by save_contribution


def test_save_contribution_creates_contributor_row_on_first_submission():
    _seed(1)
    h = _h()
    task = contributions.next_task(h)
    contributions.save_contribution(h, task["id"], "Oshindonga", "test")
    contrib = contributions.get_contributor(h)
    assert contrib is not None
    assert contrib["total_contributions"] == 1


def test_save_contribution_increments_contributor_counter_on_repeat():
    _seed(3)
    h = _h()
    for _ in range(3):
        task = contributions.next_task(h)
        contributions.save_contribution(h, task["id"], "Oshindonga", "test")
    contrib = contributions.get_contributor(h)
    assert contrib["total_contributions"] == 3


# ── pending-save state ────────────────────────────────────────────


def test_get_pending_save_returns_none_for_new_contributor():
    assert contributions.get_pending_save(_h()) is None


def test_set_pending_save_then_get_returns_state():
    _seed(1)
    h = _h()
    # Need a real task_id (FK is loose but realistic test path)
    task = contributions.next_task(h)
    contributions.set_pending_save(h, task["id"], "Oshindonga")
    pending = contributions.get_pending_save(h)
    assert pending is not None
    assert pending["task_id"] == task["id"]
    assert pending["dialect"] == "Oshindonga"
    assert pending["set_at"]  # timestamp present


def test_set_pending_save_rejects_invalid_dialect():
    with pytest.raises(ValueError):
        contributions.set_pending_save(_h(), 1, "Spanish")


def test_set_pending_save_overwrites_previous():
    _seed(2)
    h = _h()
    contributions.set_pending_save(h, 1, "Oshindonga")
    contributions.set_pending_save(h, 2, "Oshikwanyama")
    pending = contributions.get_pending_save(h)
    assert pending["task_id"] == 2
    assert pending["dialect"] == "Oshikwanyama"


def test_clear_pending_save_zeroes_state():
    _seed(1)
    h = _h()
    contributions.set_pending_save(h, 1, "Oshindonga")
    contributions.clear_pending_save(h)
    assert contributions.get_pending_save(h) is None


def test_next_task_sets_pending_indirectly_via_tool_path():
    """The tool side calls set_pending after next; verify the helper
    alone doesn't set pending (we don't want next_task to imply
    pending state — that's the tool's job)."""
    _seed(1)
    h = _h()
    task = contributions.next_task(h)
    assert task is not None
    # contributions.next_task itself does NOT set pending — only the
    # tool's 'next' branch does, because pending requires a dialect.
    assert contributions.get_pending_save(h) is None


def test_save_contribution_clears_pending_save_atomically():
    _seed(1)
    h = _h()
    contributions.set_dialect(h, "Oshindonga")
    task = contributions.next_task(h)
    contributions.set_pending_save(h, task["id"], "Oshindonga")
    assert contributions.get_pending_save(h) is not None
    contributions.save_contribution(h, task["id"], "Oshindonga", "ondi ya nawa")
    assert contributions.get_pending_save(h) is None


# ── set_dialect / whoami ───────────────────────────────────────────


def test_whoami_returns_new_for_unknown_contributor():
    assert contributions.whoami(_h()) == "new"


def test_set_dialect_then_whoami_returns_known():
    h = _h()
    contributions.set_dialect(h, "Oshindonga")
    assert contributions.whoami(h) == "known:Oshindonga"


def test_set_dialect_can_be_changed():
    h = _h()
    contributions.set_dialect(h, "Oshindonga")
    contributions.set_dialect(h, "Oshikwanyama")
    assert contributions.whoami(h) == "known:Oshikwanyama"


def test_set_dialect_rejects_invalid_dialect():
    with pytest.raises(ValueError):
        contributions.set_dialect(_h(), "Spanish")


def test_whoami_returns_unset_for_decliner_who_never_set_dialect():
    h = _h()
    contributions.record_decline(h)
    # Decliner has a contributor row but no preferred_dialect
    assert contributions.whoami(h) == "unset"


# ── decline cooldown ───────────────────────────────────────────────


def test_recently_declined_false_for_unknown_contributor():
    assert contributions.recently_declined(_h()) is False


def test_recently_declined_true_just_after_decline():
    h = _h()
    contributions.record_decline(h)
    assert contributions.recently_declined(h) is True


def test_recently_declined_false_after_cooldown_window():
    h = _h()
    # Manually backdate the decline by 10 days
    contributions.record_decline(h)
    with contributions._conn() as c:
        c.execute(
            "UPDATE contributors SET last_declined_at = ? WHERE contributor_hash = ?",
            ("2025-01-01T00:00:00+00:00", h),
        )
    assert contributions.recently_declined(h) is False


# ── stats ──────────────────────────────────────────────────────────


def test_total_contributions_unfiltered():
    _seed(3)
    h = _h()
    for _ in range(3):
        task = contributions.next_task(h)
        contributions.save_contribution(h, task["id"], "Oshindonga", "x")
    assert contributions.total_contributions() == 3


def test_total_contributions_filtered_by_dialect():
    _seed(4)
    h = _h()
    for _ in range(2):
        task = contributions.next_task(h)
        contributions.save_contribution(h, task["id"], "Oshindonga", "x")
    for _ in range(2):
        task = contributions.next_task(h)
        contributions.save_contribution(h, task["id"], "Oshikwanyama", "x")
    assert contributions.total_contributions("Oshindonga") == 2
    assert contributions.total_contributions("Oshikwanyama") == 2


def test_stats_summary_returns_all_required_fields():
    _seed(2)
    h = _h()
    task = contributions.next_task(h)
    contributions.save_contribution(h, task["id"], "Oshindonga", "test")
    s = contributions.stats_summary()
    assert s["total_contributions"] == 1
    assert s["by_dialect"]["Oshindonga"] == 1
    assert s["total_contributors"] == 1
    assert s["total_tasks"] == 2
