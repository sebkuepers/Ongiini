"""Tests for the broadcast opt-out sqlite store."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# Ensure the contributions salt is set so hash_msisdn doesn't raise.
# Set BEFORE importing the module under test so settings picks it up.
os.environ.setdefault("CONTRIBUTIONS_HASH_SALT", "test-salt")


@pytest.fixture
def temp_data_dir(tmp_path: Path, monkeypatch):
    """Point settings.data_dir at a per-test tmp dir so each test has
    a fresh sqlite. Required because the module derives its db path
    from settings at call time."""
    from ongiini.config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


def test_warmup_creates_db(temp_data_dir: Path):
    from ongiini.broadcast import opt_outs
    opt_outs.warmup()
    assert (temp_data_dir / "broadcast_opt_outs.sqlite").exists()


def test_record_inserts_only_once(temp_data_dir: Path):
    from ongiini.broadcast import opt_outs
    opt_outs.warmup()
    assert opt_outs.record("+264811111111") is True
    assert opt_outs.record("+264811111111") is False  # idempotent
    assert opt_outs.count() == 1


def test_is_opted_out_roundtrip(temp_data_dir: Path):
    from ongiini.broadcast import opt_outs
    opt_outs.warmup()
    assert opt_outs.is_opted_out("+264822222222") is False
    opt_outs.record("+264822222222")
    assert opt_outs.is_opted_out("+264822222222") is True
    # Different msisdn — separate hash
    assert opt_outs.is_opted_out("+264833333333") is False


def test_all_opted_out_hashes_set(temp_data_dir: Path):
    from ongiini.broadcast import opt_outs
    from ongiini.contributions import hash_msisdn
    opt_outs.warmup()
    msisdns = ["+264811000001", "+264811000002", "+264811000003"]
    for m in msisdns:
        opt_outs.record(m)
    hashes = opt_outs.all_opted_out_hashes()
    assert isinstance(hashes, set)
    assert hashes == {hash_msisdn(m) for m in msisdns}


def test_records_with_source_metadata(temp_data_dir: Path):
    """Source column lets us distinguish STOP-keyword opt-outs from
    manual CLI additions later if we audit."""
    import sqlite3
    from ongiini.broadcast import opt_outs
    opt_outs.warmup()
    opt_outs.record("+264844444444", source="cli")
    db = sqlite3.connect(opt_outs._db_path())
    row = db.execute("SELECT source FROM opt_outs").fetchone()
    db.close()
    assert row[0] == "cli"


# ── Architectural guard ────────────────────────────────────────────


def test_no_looks_like_stop_helper_exists():
    """We intentionally do NOT expose a regex STOP pre-filter. All
    opt-out handling MUST go through the classifier verdict +
    force_tool path, mirroring the contribute flow. Re-introducing a
    regex helper would re-create the api/main.py intercept anti-pattern
    we explicitly removed in 2026-05-25."""
    from ongiini.broadcast import opt_outs
    assert not hasattr(opt_outs, "looks_like_stop")
    assert not hasattr(opt_outs, "_STOP_KEYWORDS")


# ── Self-heal on missing schema ────────────────────────────────────


def test_record_self_heals_when_table_missing(temp_data_dir: Path):
    """If startup warmup soft-failed, the first call must still work
    rather than throwing 'no such table' into the broad except in the
    tool layer."""
    import sqlite3
    from ongiini.broadcast import opt_outs

    # Skip warmup; simulate the soft-failed-startup case by creating
    # the file but not the table.
    db_path = opt_outs._db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite3.connect(db_path).close()
    # No CREATE TABLE here — schema absent.

    # record() must still succeed by self-healing the schema
    assert opt_outs.record("+264800999999") is True
    assert opt_outs.is_opted_out("+264800999999") is True
