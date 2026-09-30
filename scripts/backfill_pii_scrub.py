#!/usr/bin/env python3
"""Re-apply the current PII scrubber to data stored before a pattern existed.

`ongiini.pii.sanitize` only protects new writes. When a pattern is added
(mobile phone numbers, September 2026), everything already on disk keeps
the old text. This script rewrites three stores with the current
scrubber:

  1. short-term history  /data/264*.json   — every message content string
  2. mem0 facts          /data/qdrant       — via mem0's own update(), so
                                              the vector is re-embedded
  3. mem0 history        /data/mem0_history.db — old/new memory text

Prints counts only, never content. --dry-run changes nothing. Stores 2
and 3 need exclusive access to the embedded qdrant, so stop the webhook
first and run this in a one-off container from the same image (scripts/
is not baked into the image, so copy it to data/private/ first):

    cp scripts/backfill_pii_scrub.py data/private/
    docker compose stop webhook
    docker compose run --rm --no-deps webhook \\
        python3 /data/private/backfill_pii_scrub.py --dry-run
    docker compose run --rm --no-deps webhook \\
        python3 /data/private/backfill_pii_scrub.py
    docker compose up -d webhook

Short-term files are replaced atomically and skipped if they changed
while being processed.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if Path("/app/ongiini").is_dir():   # inside the webhook image
    sys.path.insert(0, "/app")
from ongiini import pii  # noqa: E402


def scrub_value(value):
    """Sanitise a message content: plain string or multipart list."""
    if isinstance(value, str):
        return pii.sanitize(value)
    if isinstance(value, list):
        return [{**part, "text": pii.sanitize(part["text"])}
                if isinstance(part, dict) and isinstance(part.get("text"), str)
                else part for part in value]
    return value


def backfill_short_term(data_dir: Path, dry_run: bool) -> dict:
    stats = {"files": 0, "files_changed": 0, "messages_changed": 0, "skipped_busy": 0}
    for path in sorted(data_dir.glob("264*.json")):
        stats["files"] += 1
        before = path.stat().st_mtime_ns
        try:
            turns = json.loads(path.read_text())
        except Exception:                                   # noqa: BLE001
            continue
        if not isinstance(turns, list):
            continue
        changed = 0
        for turn in turns:
            if isinstance(turn, dict) and "content" in turn:
                new = scrub_value(turn["content"])
                if new != turn["content"]:
                    turn["content"] = new
                    changed += 1
        if not changed:
            continue
        stats["files_changed"] += 1
        stats["messages_changed"] += changed
        if dry_run:
            continue
        if path.stat().st_mtime_ns != before:
            stats["skipped_busy"] += 1
            continue
        tmp = path.with_suffix(".json.scrub-tmp")
        tmp.write_text(json.dumps(turns, ensure_ascii=False))
        os.replace(tmp, path)
    return stats


def backfill_mem0(dry_run: bool) -> dict:
    from ongiini.memory import long_term

    m = long_term._client()
    client = m.vector_store.client
    collection = m.vector_store.collection_name
    stats = {"facts": 0, "facts_changed": 0}
    offset = None
    todo = []
    while True:
        points, offset = client.scroll(collection, limit=256, offset=offset,
                                       with_payload=True, with_vectors=False)
        for p in points:
            stats["facts"] += 1
            text = (p.payload or {}).get("data")
            if isinstance(text, str) and pii.sanitize(text) != text:
                todo.append((str(p.id), pii.sanitize(text)))
        if offset is None:
            break
    stats["facts_changed"] = len(todo)
    if not dry_run:
        for memory_id, text in todo:
            m.update(memory_id, text)
    return stats


def backfill_history(db: Path, dry_run: bool) -> dict:
    stats = {"rows": 0, "rows_changed": 0}
    con = sqlite3.connect(str(db))
    rows = con.execute("SELECT id, old_memory, new_memory FROM history").fetchall()
    stats["rows"] = len(rows)
    updates = []
    for row_id, old, new in rows:
        s_old = pii.sanitize(old) if isinstance(old, str) else old
        s_new = pii.sanitize(new) if isinstance(new, str) else new
        if (s_old, s_new) != (old, new):
            updates.append((s_old, s_new, row_id))
    stats["rows_changed"] = len(updates)
    if not dry_run and updates:
        with con:
            con.executemany(
                "UPDATE history SET old_memory = ?, new_memory = ? WHERE id = ?", updates)
    con.close()
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("/data"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-mem0", action="store_true",
                    help="Only short-term history + mem0 history db "
                         "(no qdrant lock needed)")
    args = ap.parse_args(argv)

    mode = "dry run" if args.dry_run else "APPLY"
    print(f"[{mode}] short-term:", backfill_short_term(args.data_dir, args.dry_run))
    if not args.skip_mem0:
        print(f"[{mode}] mem0 facts:", backfill_mem0(args.dry_run))
    db = args.data_dir / "mem0_history.db"
    if db.exists():
        print(f"[{mode}] mem0 history:", backfill_history(db, args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
