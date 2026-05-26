"""Broadcast opt-out store.

Standalone sqlite at /data/broadcast_opt_outs.sqlite, mirroring the
isolated-table pattern of contributions.py. One table, salted-hash
identifiers, no PII at rest.

A msisdn that appears here is permanently excluded from proactive
broadcast sends until the row is deleted. The store is consulted
once at the top of every broadcast run; the per-msisdn check is a
hash lookup so 10k recipients cost a few ms.

Distinct from delete_my_data (which wipes chat history): opting out
of broadcasts does NOT delete conversation data. The user can keep
chatting; we just won't proactively message them.
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ..config import settings
from ..contributions import hash_msisdn

log = logging.getLogger("ongiini.broadcast.opt_outs")


def _db_path() -> Path:
    return settings.data_dir / "broadcast_opt_outs.sqlite"


_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS opt_outs (
    msisdn_hash    TEXT PRIMARY KEY,
    opted_out_at   TEXT NOT NULL,
    source         TEXT NOT NULL DEFAULT 'stop_keyword'
);
"""


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    """Yield a sqlite connection with the schema guaranteed to exist.

    Self-heals if startup warmup soft-failed: api/main.py wraps
    warmup() in try/except per CLAUDE.md, which means we can be
    invoked against a missing schema. Running CREATE TABLE IF NOT
    EXISTS on every connection is cheap and removes a footgun where
    `is_opted_out` raises "no such table" and the broad except in
    the opt_out tool turns that into a generic apology.
    """
    _db_path().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_db_path(), isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA_DDL)
        yield conn
    finally:
        conn.close()


def warmup() -> None:
    """Create the table if not present. Called from FastAPI lifespan.
    Idempotent — safe to call repeatedly. `_conn()` also self-heals
    the schema, so warmup is technically redundant; we keep it so
    startup logs a clear "broadcast opt-outs sqlite warmed at X"
    line operators can grep for."""
    with _conn() as _:
        pass
    log.info("broadcast opt-outs sqlite warmed at %s", _db_path())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record(msisdn: str, *, source: str = "stop_keyword") -> bool:
    """Mark this msisdn as opted-out of broadcasts. Idempotent — if
    already opted out, this is a no-op (returns False). Returns True
    if a new row was inserted."""
    h = hash_msisdn(msisdn)
    now = _now_iso()
    with _conn() as c:
        # INSERT OR IGNORE: only inserts if not already present. Then
        # we check rowcount to know whether the insert actually fired.
        cur = c.execute(
            "INSERT OR IGNORE INTO opt_outs (msisdn_hash, opted_out_at, source) "
            "VALUES (?, ?, ?)",
            (h, now, source),
        )
        return cur.rowcount > 0


def is_opted_out(msisdn: str) -> bool:
    """Lookup by raw msisdn. Re-hashes; cheap (single sqlite SELECT)."""
    h = hash_msisdn(msisdn)
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM opt_outs WHERE msisdn_hash = ?", (h,)
        ).fetchone()
    return row is not None


def all_opted_out_hashes() -> set[str]:
    """Return every opted-out msisdn hash as a set. Used by the
    broadcast script to filter recipients in one batch instead of
    issuing N per-msisdn lookups."""
    with _conn() as c:
        rows = c.execute("SELECT msisdn_hash FROM opt_outs").fetchall()
    return {r["msisdn_hash"] for r in rows}


def count() -> int:
    with _conn() as c:
        return int(c.execute("SELECT COUNT(*) AS n FROM opt_outs").fetchone()["n"])


# Note: there is intentionally no `looks_like_stop` regex helper here.
# All STOP / UNSUBSCRIBE handling MUST flow through the classifier
# verdict OPT_OUT_BROADCAST → force_tool("opt_out_broadcast") path.
# A regex pre-filter would re-introduce the api/main.py intercept
# anti-pattern we removed in 2026-05-25 for the contribute flow.


# ── CLI ────────────────────────────────────────────────────────────


def _cli_stats() -> None:
    print(f"opt-outs recorded: {count()}")


def _cli_check(msisdn: str) -> None:
    print(f"{msisdn}: {'OPTED OUT' if is_opted_out(msisdn) else 'not opted out'}")


def _cli_add(msisdn: str) -> None:
    inserted = record(msisdn, source="cli")
    print(f"{msisdn}: {'inserted' if inserted else 'already opted out'}")


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Broadcast opt-outs CLI")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("stats", help="Show count")
    cp = sub.add_parser("check", help="Check one msisdn")
    cp.add_argument("msisdn")
    ap = sub.add_parser("add", help="Manually opt out one msisdn")
    ap.add_argument("msisdn")
    args = p.parse_args(argv)
    warmup()
    if args.cmd == "stats":
        _cli_stats()
    elif args.cmd == "check":
        _cli_check(args.msisdn)
    elif args.cmd == "add":
        _cli_add(args.msisdn)
    else:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
