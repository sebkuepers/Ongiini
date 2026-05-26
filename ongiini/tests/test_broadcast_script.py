"""Tests for scripts/broadcast.py — recipient enumeration + filtering."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("CONTRIBUTIONS_HASH_SALT", "test-salt")

# Ensure repo root is on sys.path so we can import scripts.broadcast
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def temp_data_dir(tmp_path: Path, monkeypatch):
    from ongiini.config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    # Allow-list for tests — accept any 264 number
    monkeypatch.setattr(settings, "whitelist", set())
    return tmp_path


def _seed_user_file(data_dir: Path, msisdn: str) -> None:
    (data_dir / f"{msisdn}.json").write_text('[]')


def test_enumerate_includes_all_user_files(temp_data_dir: Path):
    from ongiini.broadcast import opt_outs
    from scripts.broadcast import enumerate_recipients
    opt_outs.warmup()

    _seed_user_file(temp_data_dir, "264811000001")
    _seed_user_file(temp_data_dir, "264811000002")
    _seed_user_file(temp_data_dir, "264811000003")

    recipients = enumerate_recipients()
    assert set(recipients) == {"264811000001", "264811000002", "264811000003"}


def test_enumerate_excludes_opted_out(temp_data_dir: Path):
    from ongiini.broadcast import opt_outs
    from scripts.broadcast import enumerate_recipients
    opt_outs.warmup()

    _seed_user_file(temp_data_dir, "264811000010")
    _seed_user_file(temp_data_dir, "264811000020")
    _seed_user_file(temp_data_dir, "264811000030")

    opt_outs.record("264811000020")
    recipients = enumerate_recipients()
    assert "264811000020" not in recipients
    assert set(recipients) == {"264811000010", "264811000030"}


def test_enumerate_filters_by_allowed_country(temp_data_dir: Path):
    """is_allowed enforces Namibian +264 prefix. Stray files for
    non-Namibian numbers (test artifacts, edge cases) must be skipped."""
    from ongiini.broadcast import opt_outs
    from scripts.broadcast import enumerate_recipients
    opt_outs.warmup()

    _seed_user_file(temp_data_dir, "264811000099")   # ok
    _seed_user_file(temp_data_dir, "491588888888")   # German number — not Namibian

    recipients = enumerate_recipients()
    assert "264811000099" in recipients
    assert "491588888888" not in recipients


def test_only_msisdn_overrides_enumeration(temp_data_dir: Path):
    """Smoke-test path: --only-msisdn skips the /data glob."""
    from ongiini.broadcast import opt_outs
    from scripts.broadcast import enumerate_recipients
    opt_outs.warmup()

    # No user files seeded — would normally return []
    recipients = enumerate_recipients(only_msisdn=["+264811000001"])
    # normalize() strips the +
    assert recipients == ["264811000001"]


def test_only_msisdn_still_respects_opt_outs(temp_data_dir: Path):
    """If you accidentally include an opted-out msisdn in --only-msisdn,
    we still skip them. STOP is sacred."""
    from ongiini.broadcast import opt_outs
    from scripts.broadcast import enumerate_recipients
    opt_outs.warmup()
    opt_outs.record("264811000001")

    recipients = enumerate_recipients(only_msisdn=["+264811000001"])
    assert recipients == []
