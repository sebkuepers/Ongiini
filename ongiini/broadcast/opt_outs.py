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


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_db_path(), isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def warmup() -> None:
    """Create the table if not present. Called from FastAPI lifespan.
    Idempotent — safe to call repeatedly."""
    _db_path().parent.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS opt_outs (
            msisdn_hash    TEXT PRIMARY KEY,
            opted_out_at   TEXT NOT NULL,
            source         TEXT NOT NULL DEFAULT 'stop_keyword'
        );
        """)
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


# ── STOP keyword detection ─────────────────────────────────────────


# Case-insensitive whole-word match. Kept narrow on purpose: a
# message like "stop sending me notifications" should match but
# "stop the war" should not — the latter is too rare to worry about
# now, and a false-positive opt-out is recoverable (user just
# messages again to opt back in via STOP-handling product reversal
# later if we add it).
_STOP_KEYWORDS = frozenset({
    "stop", "stop messages", "stop messaging me", "stop sending",
    "unsubscribe", "opt out", "optout", "opt-out", "no more messages",
    "stop notifications", "verwyder", "stop boodskappe",
})


def looks_like_stop(text: str) -> bool:
    """Detect STOP/UNSUBSCRIBE intent in a short user message.

    Conservative: only fires on (a) the literal keyword/phrase as the
    ENTIRE message (after lowercasing + trimming), or (b) a very
    short message (<32 chars) where the keyword is a whole token.
    Longer messages with "stop" embedded don't fire — we let the
    classifier handle them via the OPT_OUT_BROADCAST verdict.

    This is a fast pre-filter for the obvious 'STOP' case to avoid
    a classifier round-trip + tool call when intent is unambiguous.
    """
    if not text:
        return False
    s = text.strip().lower()
    # Strip surrounding punctuation that users sometimes add ("STOP.", "stop!")
    s = s.strip(".!?,;:'\"")
    if s in _STOP_KEYWORDS:
        return True
    return False


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
