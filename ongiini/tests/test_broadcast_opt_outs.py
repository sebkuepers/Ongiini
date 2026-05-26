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


# ── STOP keyword detection ─────────────────────────────────────────


class TestLooksLikeStop:
    def test_bare_stop(self):
        from ongiini.broadcast.opt_outs import looks_like_stop
        assert looks_like_stop("STOP") is True
        assert looks_like_stop("stop") is True
        assert looks_like_stop("Stop") is True

    def test_stop_with_trailing_punctuation(self):
        from ongiini.broadcast.opt_outs import looks_like_stop
        assert looks_like_stop("STOP.") is True
        assert looks_like_stop("stop!") is True
        assert looks_like_stop("stop?") is True

    def test_unsubscribe_variants(self):
        from ongiini.broadcast.opt_outs import looks_like_stop
        assert looks_like_stop("unsubscribe") is True
        assert looks_like_stop("opt out") is True
        assert looks_like_stop("optout") is True
        assert looks_like_stop("opt-out") is True

    def test_does_not_fire_on_unrelated_use_of_stop(self):
        from ongiini.broadcast.opt_outs import looks_like_stop
        # Embedded in a longer message — let the classifier handle it
        assert looks_like_stop("can you stop the war for me") is False
        assert looks_like_stop("the stop sign was red") is False

    def test_empty_or_none(self):
        from ongiini.broadcast.opt_outs import looks_like_stop
        assert looks_like_stop("") is False
        assert looks_like_stop("   ") is False
