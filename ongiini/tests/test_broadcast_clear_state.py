"""Tests for the contribute-state clearing helper used by proactive sends.

Why this exists: when we inject a proactive assistant turn into a user's
short-term memory, the classifier doesn't see that turn (it reads
contribute_state from SQLite instead). If the user has stale
`awaiting_followup` set, their reply gets misrouted through the
contribute force-tool. The helper clears that state at every
proactive send so the classifier doesn't misroute.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("CONTRIBUTIONS_HASH_SALT", "test-salt")


@pytest.fixture
def temp_data_dir(tmp_path: Path, monkeypatch):
    from ongiini.config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


def test_clear_removes_both_pending_save_and_awaiting_followup(temp_data_dir: Path):
    """Both contribute-state markers get cleared in one call."""
    from ongiini import contributions
    from ongiini.broadcast.sender import clear_contribute_state_for_proactive

    contributions.warmup()
    msisdn = "+264811000001"
    h = contributions.hash_msisdn(msisdn)

    # Seed the user with stale contribute state
    contributions.set_awaiting_followup(h)
    assert contributions.is_awaiting_followup(h) is True
    contributions.set_pending_save(h, task_id=1, dialect="Oshindonga")
    # First need a task to exist for set_pending_save to satisfy FK-like checks
    # set_pending_save uses INSERT OR UPDATE on contributors; task_id is just stored
    assert contributions.get_pending_save(h) is not None

    # Clear via the helper
    clear_contribute_state_for_proactive(msisdn)

    assert contributions.is_awaiting_followup(h) is False
    assert contributions.get_pending_save(h) is None


def test_clear_soft_fails_when_salt_missing(temp_data_dir: Path, monkeypatch):
    """Helper must not raise even if the salt env var is missing.
    The proactive send should still proceed; clearing is best-effort."""
    from ongiini.config import settings
    from ongiini.broadcast.sender import clear_contribute_state_for_proactive

    monkeypatch.setattr(settings, "contributions_hash_salt", "")
    # Should not raise even though hash_msisdn would fail
    clear_contribute_state_for_proactive("+264811000099")


def test_clear_is_idempotent_when_no_state_exists(temp_data_dir: Path):
    """No-op safe — clearing for a user who has never had contribute
    state doesn't fail."""
    from ongiini import contributions
    from ongiini.broadcast.sender import clear_contribute_state_for_proactive

    contributions.warmup()
    clear_contribute_state_for_proactive("+264811000002")  # never seeded
    # No exception = pass
