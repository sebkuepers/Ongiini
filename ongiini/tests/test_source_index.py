"""Unit tests for the source_index storage module.

Round-tripping, dedup behaviour, cap enforcement, atomic write, and
the formatter shape. No external dependencies — uses a tmp_path
override for ``settings.data_dir``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ongiini.config import settings
from ongiini.memory import source_index


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Point settings.data_dir at a tmp location for the test."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


# ---------- load ----------

def test_load_missing_file_returns_empty(tmp_data_dir):
    assert source_index.load("+264811234567") == []


def test_load_corrupted_file_returns_empty(tmp_data_dir):
    p = tmp_data_dir / "source_index" / "+264811234567.json"
    p.parent.mkdir()
    p.write_text("{not valid json")
    assert source_index.load("+264811234567") == []


def test_load_filters_entries_without_url(tmp_data_dir):
    # ``normalize`` strips the "+" — file name is digits only.
    p = tmp_data_dir / "source_index" / "264811234567.json"
    p.parent.mkdir()
    p.write_text(json.dumps([
        {"url": "https://a.example", "ts": "2026-05-23T00:00:00+00:00"},
        {"ts": "2026-05-23T00:01:00+00:00"},        # no url — drop
        {"url": 42},                                # non-string — drop
        "not a dict",                               # not a dict — drop
    ]))
    out = source_index.load("+264811234567")
    assert len(out) == 1
    assert out[0]["url"] == "https://a.example"


# ---------- append ----------

def test_append_creates_directory_and_file(tmp_data_dir):
    source_index.append("+264811234567", ["https://a.example"])
    # ``normalize`` strips the "+", so the file lands at the digits-only path.
    p = tmp_data_dir / "source_index" / "264811234567.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert data[0]["url"] == "https://a.example"
    assert data[0]["ts"]    # any non-empty ts


def test_append_dedupes_by_url(tmp_data_dir):
    source_index.append("+264811234567", ["https://a.example", "https://b.example"])
    source_index.append("+264811234567", ["https://a.example", "https://c.example"])
    out = source_index.load("+264811234567")
    urls = [e["url"] for e in out]
    # All three URLs present exactly once.
    assert sorted(urls) == ["https://a.example", "https://b.example", "https://c.example"]


def test_append_caps_at_max_entries(tmp_data_dir):
    # _MAX_ENTRIES is 30; append 35 distinct URLs.
    urls = [f"https://example.com/page-{i}" for i in range(35)]
    source_index.append("+264811234567", urls)
    out = source_index.load("+264811234567")
    assert len(out) == 30


def test_append_skips_blank_and_non_string(tmp_data_dir):
    source_index.append("+264811234567", ["https://a.example", "", None, 42, "  "])
    out = source_index.load("+264811234567")
    # Blank strings ARE filtered out at the entry level; the non-empty
    # whitespace-only one isn't (we trust the hook's URL extraction).
    # Anything that's not str-and-truthy is dropped.
    assert [e["url"] for e in out] == ["https://a.example", "  "]


def test_append_empty_list_is_noop(tmp_data_dir):
    source_index.append("+264811234567", [])
    p = tmp_data_dir / "source_index" / "+264811234567.json"
    assert not p.exists()


def test_append_atomic_write_leaves_no_tmp(tmp_data_dir):
    """Crash-safe pattern uses os.replace; verify no .tmp residue."""
    source_index.append("+264811234567", ["https://a.example"])
    leftover = list((tmp_data_dir / "source_index").glob("*.tmp"))
    assert leftover == []


# ---------- delete ----------

def test_delete_existing_file_returns_true(tmp_data_dir):
    source_index.append("+264811234567", ["https://a.example"])
    assert source_index.delete("+264811234567") is True
    assert source_index.load("+264811234567") == []


def test_delete_missing_file_returns_false(tmp_data_dir):
    assert source_index.delete("+264811234567") is False


# ---------- format_for_injection ----------

def test_format_empty_returns_empty_string():
    assert source_index.format_for_injection([]) == ""


def test_format_includes_url_per_line():
    entries = [
        {"url": "https://a.example/article", "ts": "2026-05-23T00:00:00+00:00"},
        {"url": "https://b.example/post", "ts": "2026-05-22T00:00:00+00:00"},
    ]
    out = source_index.format_for_injection(entries)
    assert "https://a.example/article" in out
    assert "https://b.example/post" in out
    # Header tells the model what to do with this block.
    assert "re-list" in out.lower() or "sources" in out.lower()


def test_format_caps_at_inject_limit():
    entries = [{"url": f"https://example.com/{i}", "ts": "x"} for i in range(20)]
    out = source_index.format_for_injection(entries)
    # _INJECT_LIMIT is 10 — only first 10 should appear.
    assert out.count("https://example.com/") == 10
    assert "https://example.com/0" in out
    assert "https://example.com/15" not in out


# ---------- path safety ----------

def test_msisdn_with_path_separators_is_rejected(tmp_data_dir):
    """``normalize`` raises ``InvalidMsisdn`` on anything that's not
    6-18 digits, so a malicious ``../`` msisdn can never reach the
    filesystem. Same path-traversal safety contract as short_term."""
    from ongiini.filters import InvalidMsisdn
    with pytest.raises(InvalidMsisdn):
        source_index.append("../escape", ["https://a.example"])
    with pytest.raises(InvalidMsisdn):
        source_index.load("../escape")
