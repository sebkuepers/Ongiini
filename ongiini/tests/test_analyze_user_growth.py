"""Unit tests for scripts/analyze_user_growth.py.

The script's own internal logic is small and pure — these tests pin
the FB-pre-fill / bare-greeting / organic classifier so the patterns
can't silently drift the next time someone adjusts them.

Imports are pulled in via importlib so the test doesn't depend on the
scripts/ directory being on sys.path in production.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_user_growth.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_user_growth", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def aug():
    return _load_module()


# ── classify_first_msg ──────────────────────────────────────────────


class TestFbAdPrefillDetection:
    """The Meta click-to-chat pre-fill — high-confidence ad signal."""

    @pytest.mark.parametrize("text", [
        "Hello! Can I get more info on this?",
        "hello, can I get more info on this?",
        "HELLO! CAN I GET MORE INFORMATION ON THIS",
        "Hello. Can I get more info about this?",
        "Hi, I'd like to know more",
        "Hello, how does this work?",
        "Hi can you help me?",
    ])
    def test_canonical_meta_prefills_match(self, aug, text):
        assert aug.classify_first_msg(text) == "fb_ad_prefill"


class TestBareGreetingDetection:
    """Single greeting words — could be ad or organic, kept separate."""

    @pytest.mark.parametrize("text", [
        "hi", "Hi", "hi!", "Hi.", "Hello", "hello!",
        "hey", "Hey!", "ongiini", "Ongiini?",
        "good morning", "Good Morning",
        "howzit",
        "moro 👋🏾",
        "hi 🇳🇦",
    ])
    def test_bare_greetings_match(self, aug, text):
        assert aug.classify_first_msg(text) == "bare_greeting"


class TestOrganicSpecificDetection:
    """Anything with a real question or topic counts as organic."""

    @pytest.mark.parametrize("text", [
        "help me with my CV",
        "what is photosynthesis",
        "how do I register a business in Namibia",
        "explain debits and credits",
        "Hoekom hou my mielieblare op groei",
        "I need help with grade 9 math",
        "translate 'good morning' to Oshiwambo",
        "what AI model are you running on",  # identity Q is organic-specific
    ])
    def test_specific_questions_match_organic(self, aug, text):
        assert aug.classify_first_msg(text) == "organic_specific"


class TestEdgeCases:
    def test_empty_string_is_unknown(self, aug):
        assert aug.classify_first_msg("") == "unknown"

    def test_whitespace_only_is_unknown(self, aug):
        assert aug.classify_first_msg("   \n  ") == "unknown"

    def test_none_input_is_unknown(self, aug):
        assert aug.classify_first_msg(None) == "unknown"


# ── mask ───────────────────────────────────────────────────────────


def test_mask_preserves_country_code_and_last_three(aug):
    assert aug.mask("264811234567") == "264***567"


def test_mask_handles_short_input(aug):
    assert aug.mask("abc") == "abc"


# ── build_report end-to-end on synthetic data ──────────────────────


def test_build_report_synthetic(aug, tmp_path):
    """Drive the script end-to-end against a fake /data layout."""
    trace = tmp_path / "trace.jsonl"
    # 3 users, varying first-seen timestamps
    rows = [
        {"ts": "2026-05-24T10:00:00+00:00", "msisdn": "264811111111"},
        {"ts": "2026-05-24T11:00:00+00:00", "msisdn": "264811111111"},  # 2nd turn, same user
        {"ts": "2026-05-24T12:00:00+00:00", "msisdn": "264822222222"},
        {"ts": "2026-05-24T13:00:00+00:00", "msisdn": "264833333333"},
        {"ts": "2026-05-23T10:00:00+00:00", "msisdn": "999000000000"},  # non-Namibian, ignored
    ]
    trace.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    # Per-user JSON: FB ad, bare greeting, organic
    (tmp_path / "264811111111.json").write_text(json.dumps([
        {"role": "user", "content": "Hello! Can I get more info on this?"},
        {"role": "assistant", "content": "Ongiini! ..."},
    ]))
    (tmp_path / "264822222222.json").write_text(json.dumps([
        {"role": "user", "content": "hi"},
    ]))
    (tmp_path / "264833333333.json").write_text(json.dumps([
        {"role": "user", "content": "help me with my CV"},
    ]))

    rep = aug.build_report(tmp_path, count=100, window_days=None)

    assert rep["cohort_size"] == 3
    assert rep["buckets"]["fb_ad_prefill"] == 1
    assert rep["buckets"]["bare_greeting"] == 1
    assert rep["buckets"]["organic_specific"] == 1
    assert rep["buckets"]["unknown"] == 0
    # 1 of 3 = 33.3% organic strict
    assert rep["ratios"]["organic_pct"] == pytest.approx(33.3, abs=0.5)
    # Newest first
    assert rep["per_user"][0]["id_masked"] == "264***333"


def test_build_report_window_filter(aug, tmp_path):
    """Window filter restricts to users first-seen within last N days."""
    trace = tmp_path / "trace.jsonl"
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    long_ago = (now - timedelta(days=30)).isoformat(timespec="seconds")
    yesterday = (now - timedelta(days=1)).isoformat(timespec="seconds")
    rows = [
        {"ts": long_ago, "msisdn": "264811111111"},
        {"ts": yesterday, "msisdn": "264822222222"},
    ]
    trace.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "264811111111.json").write_text(json.dumps([{"role": "user", "content": "hi"}]))
    (tmp_path / "264822222222.json").write_text(json.dumps([{"role": "user", "content": "hi"}]))

    rep = aug.build_report(tmp_path, count=100, window_days=7)
    assert rep["cohort_size"] == 1   # only the yesterday user
    assert rep["per_user"][0]["id_masked"] == "264***222"


def test_build_report_missing_trace(aug, tmp_path):
    rep = aug.build_report(tmp_path, count=100, window_days=None)
    assert "error" in rep
    assert "trace.jsonl" in rep["error"]
