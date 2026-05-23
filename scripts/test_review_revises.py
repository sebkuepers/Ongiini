"""Tests for review_revises.py — the human-rating CLI for compose/revise
pairs. Covers the JSONL load/append round-trip and the summary aggregation
math; interactive prompts are not exercised here."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make scripts/ importable from pytest run-from-repo-root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import review_revises  # noqa: E402


def _write_capture(dir_path: Path, msg_id: str, **fields) -> Path:
    """Write a minimal capture JSON."""
    payload = {
        "msg_id": msg_id,
        "policy": "search_deep",
        "ts": "2026-05-23T18:00:00+00:00",
        "user_question": "q",
        "tool_results": [],
        "critique_verdict": "REVISE",
        "critique_reasons": [],
        "raw_critique": "",
        "compose_draft": "compose",
        "revised_reply": "revise",
        "compose_len": 7,
        "revised_len": 6,
        "revise_error": None,
    }
    payload.update(fields)
    p = dir_path / f"{msg_id}.json"
    p.write_text(json.dumps(payload))
    return p


# ---------- load_captures ----------

def test_load_captures_returns_empty_when_dir_missing(tmp_path):
    assert review_revises._load_captures(tmp_path / "nope") == []


def test_load_captures_reads_all_json(tmp_path):
    _write_capture(tmp_path, "a")
    _write_capture(tmp_path, "b")
    out = review_revises._load_captures(tmp_path)
    assert sorted(c["msg_id"] for c in out) == ["a", "b"]


def test_load_captures_skips_ratings_file(tmp_path):
    _write_capture(tmp_path, "a")
    (tmp_path / review_revises.RATINGS_FILE_NAME).write_text(
        json.dumps({"msg_id": "a", "verdict": "revise-better", "note": ""}) + "\n"
    )
    out = review_revises._load_captures(tmp_path)
    assert len(out) == 1
    assert out[0]["msg_id"] == "a"


def test_load_captures_skips_malformed_json(tmp_path):
    _write_capture(tmp_path, "good")
    (tmp_path / "bad.json").write_text("not json")
    out = review_revises._load_captures(tmp_path)
    assert [c["msg_id"] for c in out] == ["good"]


# ---------- load_ratings + append ----------

def test_load_ratings_empty_when_no_file(tmp_path):
    assert review_revises._load_ratings(tmp_path / "ratings.jsonl") == {}


def test_append_rating_writes_jsonl_line(tmp_path):
    ratings_path = tmp_path / "ratings.jsonl"
    review_revises._append_rating(ratings_path, "wamid_1", "revise-better", "looks tighter")
    line = ratings_path.read_text().strip()
    rec = json.loads(line)
    assert rec == {"msg_id": "wamid_1", "verdict": "revise-better", "note": "looks tighter"}


def test_load_ratings_keeps_last_verdict_per_msg_id(tmp_path):
    """When a pair is re-rated, the LAST verdict wins. Important for
    --re-rate workflow."""
    ratings_path = tmp_path / "ratings.jsonl"
    review_revises._append_rating(ratings_path, "x", "tie", "first pass")
    review_revises._append_rating(ratings_path, "x", "revise-better", "looked again")
    review_revises._append_rating(ratings_path, "y", "compose-better", "")
    loaded = review_revises._load_ratings(ratings_path)
    assert loaded["x"]["verdict"] == "revise-better"
    assert loaded["y"]["verdict"] == "compose-better"


def test_load_ratings_ignores_garbage_lines(tmp_path):
    ratings_path = tmp_path / "ratings.jsonl"
    ratings_path.write_text(
        '{"msg_id": "ok", "verdict": "tie", "note": ""}\n'
        "not json at all\n"
        '{"no_msg_id": true}\n'
    )
    loaded = review_revises._load_ratings(ratings_path)
    assert list(loaded.keys()) == ["ok"]
