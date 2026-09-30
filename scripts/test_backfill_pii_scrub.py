"""Tests for the PII backfill over stored short-term history and mem0 history."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_pii_scrub import backfill_history, backfill_short_term, scrub_value  # noqa: E402


def _write(path: Path, turns: list) -> None:
    path.write_text(json.dumps(turns))


def test_scrub_value_handles_strings_and_multipart():
    assert scrub_value("call 081 234 5678") == "call [REDACTED:phone]"
    parts = [{"type": "text", "text": "me: +264812345678"}, {"type": "image_url"}]
    assert scrub_value(parts)[0]["text"] == "me: [REDACTED:phone]"
    assert scrub_value(parts)[1] == {"type": "image_url"}
    assert scrub_value(None) is None


def test_short_term_dry_run_counts_without_writing(tmp_path):
    f = tmp_path / "264811111111.json"
    turns = [{"role": "user", "content": "my cell is 0812345678"},
             {"role": "assistant", "content": "Call BIPA on 061 374 400."}]
    _write(f, turns)
    stats = backfill_short_term(tmp_path, dry_run=True)
    assert stats["messages_changed"] == 1
    assert json.loads(f.read_text()) == turns


def test_short_term_apply_scrubs_only_what_matches(tmp_path):
    f = tmp_path / "264811111111.json"
    _write(f, [{"role": "user", "content": "my cell is 0812345678"},
               {"role": "assistant", "content": "Call BIPA on 061 374 400."}])
    stats = backfill_short_term(tmp_path, dry_run=False)
    out = json.loads(f.read_text())
    assert stats == {"files": 1, "files_changed": 1, "messages_changed": 1, "skipped_busy": 0}
    assert out[0]["content"] == "my cell is [REDACTED:phone]"
    assert out[1]["content"] == "Call BIPA on 061 374 400."
    assert not list(tmp_path.glob("*.scrub-tmp"))


def test_history_rows_rewritten(tmp_path):
    db = tmp_path / "h.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE history (id TEXT, old_memory TEXT, new_memory TEXT)")
    con.executemany("INSERT INTO history VALUES (?, ?, ?)", [
        ("1", None, "[PROFILE] Phone number is 081 234 5678"),
        ("2", "[PROFILE] Lives in Oshakati", "[PROFILE] Lives in Ondangwa"),
    ])
    con.commit()
    con.close()
    assert backfill_history(db, dry_run=False)["rows_changed"] == 1
    rows = dict(sqlite3.connect(db).execute("SELECT id, new_memory FROM history"))
    assert rows["1"] == "[PROFILE] Phone number is [REDACTED:phone]"
    assert rows["2"] == "[PROFILE] Lives in Ondangwa"
