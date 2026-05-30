"""Community contribution database — Oshiwambo translation pairs
collected from native-speaker users via the contribute_translation
tool.

Standalone sqlite at /data/contributions.sqlite, intentionally
separate from the per-user chat memory tier (mem0 + short-term JSON)
so contributions can be exported, audited, and eventually published
as an open dataset without touching personal chat data.

The contract is unusual for this codebase:
- Contributions are PERMANENT additions to a public-good dataset.
  Unlike chat history (cleared by delete_my_data), contributions stay.
- Users are informed upfront via the invitation message — submitting
  is the consent act.
- The contributor's msisdn is stored only as a salted SHA-256 hash;
  the dataset publication carries no identifying information.

CLI: ``python -m ongiini.contributions {stats|sample|reset}``
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import random
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from .config import settings
from .pii import sanitize as pii_sanitize

log = logging.getLogger("ongiini.contributions")


# Dialect labels we accept. Anything else is rejected at write time.
DIALECT_OSHINDONGA = "Oshindonga"
DIALECT_OSHIKWANYAMA = "Oshikwanyama"
VALID_DIALECTS = {DIALECT_OSHINDONGA, DIALECT_OSHIKWANYAMA}


# Decline cool-down — if a user said no to the invitation, don't ask
# again for 7 days. Stops the bot from nagging.
DECLINE_COOLDOWN_DAYS = 7


def _db_path() -> Path:
    """The sqlite path. Lives next to other /data files; bind-mounted
    from the host so it survives container restarts."""
    return settings.data_dir / "contributions.sqlite"


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    """Per-call connection; sqlite handles concurrent readers fine and
    we only ever have one writer (the webhook). Keeps the connection
    pool simple — no globals to manage across event-loop reloads."""
    conn = sqlite3.connect(_db_path(), isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def warmup() -> None:
    """Create tables if not present. Called from the FastAPI lifespan
    on startup. Idempotent — safe to call repeatedly."""
    _db_path().parent.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            source_en       TEXT NOT NULL,
            category        TEXT,
            seed_id         INTEGER,
            created_at      TEXT NOT NULL,
            times_served    INTEGER NOT NULL DEFAULT 0,
            times_submitted INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS contributions (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id              INTEGER NOT NULL REFERENCES tasks(id),
            contributor_hash     TEXT NOT NULL,
            target_dialect       TEXT NOT NULL,
            target_translation   TEXT NOT NULL,
            submitted_at         TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_contributions_contributor
            ON contributions(contributor_hash);
        CREATE INDEX IF NOT EXISTS idx_contributions_task
            ON contributions(task_id);
        CREATE INDEX IF NOT EXISTS idx_contributions_dialect
            ON contributions(target_dialect);

        CREATE TABLE IF NOT EXISTS contributors (
            contributor_hash       TEXT PRIMARY KEY,
            preferred_dialect      TEXT,
            first_contributed_at   TEXT NOT NULL,
            last_contributed_at    TEXT NOT NULL,
            last_declined_at       TEXT,
            total_contributions    INTEGER NOT NULL DEFAULT 0
        );
        """)
        # Schema migration: add pending-save columns if they don't exist.
        # Used by the runtime save-forcing mechanism — when contribute_
        # translation 'next' is called, we mark the contributor as
        # pending the matching save. The next user message is then
        # save-forced by api/main.py before the model loop runs, so the
        # model can't skip the save by improvising.
        cols = {r["name"] for r in c.execute("PRAGMA table_info(contributors)").fetchall()}
        if "pending_task_id" not in cols:
            c.execute("ALTER TABLE contributors ADD COLUMN pending_task_id INTEGER")
        if "pending_dialect" not in cols:
            c.execute("ALTER TABLE contributors ADD COLUMN pending_dialect TEXT")
        if "pending_set_at" not in cols:
            c.execute("ALTER TABLE contributors ADD COLUMN pending_set_at TEXT")
        # awaiting_followup_at marks "we just force-saved a contribution
        # and asked 'want another?' — the user's NEXT short reply is a
        # yes/no decision the force-followup path interprets directly
        # instead of letting the model improvise another fake 'next'."
        if "awaiting_followup_at" not in cols:
            c.execute("ALTER TABLE contributors ADD COLUMN awaiting_followup_at TEXT")
    log.info("contributions sqlite warmed at %s", _db_path())


def hash_msisdn(msisdn: str) -> str:
    """Salted SHA-256 hash of a phone number. The salt lives in env;
    without it the hash is meaningless. This is how we identify a
    contributor across submissions without storing their phone number
    in the dataset."""
    salt = settings.contributions_hash_salt
    if not salt:
        raise RuntimeError(
            "CONTRIBUTIONS_HASH_SALT is not set. "
            "Cannot hash contributor msisdn without it."
        )
    return hashlib.sha256(f"{salt}:{msisdn}".encode()).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Task management ────────────────────────────────────────────────


def seed_tasks(seeds: list[dict]) -> int:
    """Bulk-insert seed task rows. Called by the seed script after
    the paraphrase generation completes. Returns count inserted.

    Expects each dict to have: source_en, category, seed_id.
    Skips rows with empty source_en. Doesn't dedup — caller decides."""
    inserted = 0
    now = _now_iso()
    with _conn() as c:
        for s in seeds:
            src = (s.get("source_en") or "").strip()
            if not src:
                continue
            c.execute(
                "INSERT INTO tasks (source_en, category, seed_id, created_at) "
                "VALUES (?, ?, ?, ?)",
                (src, s.get("category"), s.get("seed_id"), now),
            )
            inserted += 1
    return inserted


def task_count() -> int:
    """How many tasks are in the pool right now."""
    with _conn() as c:
        row = c.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()
    return int(row["n"])


def next_task(contributor_hash: str, exclude_task_ids: list[int] | None = None) -> dict | None:
    """Pick a task to serve to this contributor.

    Strategy: random task they HAVEN'T already submitted for AND
    isn't in ``exclude_task_ids`` (used by the skip path to avoid
    immediately re-serving the just-rejected task). Tie-break order:
    fewest submissions globally first, then fewest serves to spread
    coverage across the pool, then random."""
    excluded = tuple(exclude_task_ids or ())
    # Dynamically build the NOT IN clause for exclusions; sqlite needs
    # one ? per element.
    excl_clause = ""
    params: list = [contributor_hash]
    if excluded:
        placeholders = ",".join("?" * len(excluded))
        excl_clause = f" AND t.id NOT IN ({placeholders})"
        params.extend(excluded)
    with _conn() as c:
        row = c.execute(
            f"""
            SELECT t.id, t.source_en, t.category
              FROM tasks t
             WHERE t.id NOT IN (
                SELECT task_id FROM contributions
                 WHERE contributor_hash = ?
             ){excl_clause}
             ORDER BY t.times_submitted ASC, t.times_served ASC, RANDOM()
             LIMIT 1
            """,
            params,
        ).fetchone()
        if not row:
            return None
        # Bump times_served — counts how many times we showed this
        # sentence to a contributor, regardless of whether they submitted
        c.execute(
            "UPDATE tasks SET times_served = times_served + 1 WHERE id = ?",
            (row["id"],),
        )
        return {"id": int(row["id"]), "source_en": row["source_en"],
                "category": row["category"]}


# ── Pending-save state (runtime save-forcing) ─────────────────────────


def set_pending_save(contributor_hash: str, task_id: int, dialect: str) -> None:
    """Mark this contributor as 'expecting to submit a translation for
    task_id in dialect'. Called when contribute_translation 'next'
    succeeds; cleared when 'save' or 'decline' fires.

    Used by api/main.py to force a save call on the user's next inbound
    message instead of relying on the model to call save itself. Upserts
    a contributor row if none exists yet so we never lose pending state
    on a brand-new contributor."""
    if dialect not in VALID_DIALECTS:
        raise ValueError(f"invalid dialect {dialect!r}")
    now = _now_iso()
    with _conn() as c:
        c.execute(
            """
            INSERT INTO contributors
              (contributor_hash, first_contributed_at, last_contributed_at,
               pending_task_id, pending_dialect, pending_set_at)
              VALUES (?, ?, ?, ?, ?, ?)
              ON CONFLICT(contributor_hash) DO UPDATE SET
                pending_task_id = excluded.pending_task_id,
                pending_dialect = excluded.pending_dialect,
                pending_set_at  = excluded.pending_set_at
            """,
            (contributor_hash, now, now, int(task_id), dialect, now),
        )


def clear_pending_save(contributor_hash: str) -> None:
    """Clear the pending-save markers for this contributor. Called by
    save_contribution on success, and by the decline path."""
    with _conn() as c:
        c.execute(
            """
            UPDATE contributors
               SET pending_task_id = NULL,
                   pending_dialect = NULL,
                   pending_set_at  = NULL
             WHERE contributor_hash = ?
            """,
            (contributor_hash,),
        )


def set_awaiting_followup(contributor_hash: str) -> None:
    """Mark this contributor as having JUST received a force-save
    confirmation ('Want another sentence, or done for now?'). The
    next inbound message is interpreted as a yes/no for that prompt.
    Cleared as soon as we act on the next message."""
    now = _now_iso()
    with _conn() as c:
        c.execute(
            """
            INSERT INTO contributors
              (contributor_hash, first_contributed_at, last_contributed_at, awaiting_followup_at)
              VALUES (?, ?, ?, ?)
              ON CONFLICT(contributor_hash) DO UPDATE SET
                awaiting_followup_at = excluded.awaiting_followup_at
            """,
            (contributor_hash, now, now, now),
        )


def clear_awaiting_followup(contributor_hash: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE contributors SET awaiting_followup_at = NULL "
            "WHERE contributor_hash = ?",
            (contributor_hash,),
        )


def is_awaiting_followup(contributor_hash: str, window_minutes: int = 30) -> bool:
    """True if the contributor was force-saved within the last
    ``window_minutes`` and hasn't yet responded. After the window
    expires the flag is ignored so old state doesn't haunt a future
    conversation."""
    with _conn() as c:
        row = c.execute(
            "SELECT awaiting_followup_at FROM contributors "
            "WHERE contributor_hash = ?",
            (contributor_hash,),
        ).fetchone()
    if not row or not row["awaiting_followup_at"]:
        return False
    try:
        marked = datetime.fromisoformat(row["awaiting_followup_at"])
    except (ValueError, TypeError):
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
    return marked >= cutoff


def get_pending_save(contributor_hash: str) -> dict | None:
    """Return the pending-save state for this contributor, or None if
    no pending save is queued. Used by api/main.py to detect when a
    user's inbound message should be force-saved as a translation."""
    with _conn() as c:
        row = c.execute(
            "SELECT pending_task_id, pending_dialect, pending_set_at "
            "FROM contributors WHERE contributor_hash = ?",
            (contributor_hash,),
        ).fetchone()
    if not row or row["pending_task_id"] is None:
        return None
    return {
        "task_id": int(row["pending_task_id"]),
        "dialect": row["pending_dialect"],
        "set_at": row["pending_set_at"],
    }


def save_contribution(
    contributor_hash: str,
    task_id: int,
    target_dialect: str,
    target_translation_raw: str,
) -> dict:
    """Persist a contributor's translation. PII-sanitises before write.
    Returns a dict with the new contribution row + updated contributor
    stats. Raises ValueError if dialect is invalid or task_id doesn't
    exist."""
    if target_dialect not in VALID_DIALECTS:
        raise ValueError(f"invalid dialect {target_dialect!r}")
    cleaned = pii_sanitize(target_translation_raw or "").strip()
    if not cleaned:
        raise ValueError("empty translation after sanitisation")
    now = _now_iso()
    with _conn() as c:
        # Verify task exists
        task = c.execute(
            "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not task:
            raise ValueError(f"task {task_id} does not exist")
        # Write the contribution row
        cur = c.execute(
            "INSERT INTO contributions "
            "(task_id, contributor_hash, target_dialect, target_translation, submitted_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, contributor_hash, target_dialect, cleaned, now),
        )
        contribution_id = cur.lastrowid
        # Bump task counter
        c.execute(
            "UPDATE tasks SET times_submitted = times_submitted + 1 WHERE id = ?",
            (task_id,),
        )
        # Upsert contributor row + bump counter. UPSERT on the hash key:
        # if contributor exists, increment total + update last_contributed_at;
        # if not, insert.
        c.execute(
            """
            INSERT INTO contributors
              (contributor_hash, first_contributed_at, last_contributed_at, total_contributions)
              VALUES (?, ?, ?, 1)
              ON CONFLICT(contributor_hash) DO UPDATE SET
                last_contributed_at = excluded.last_contributed_at,
                total_contributions = total_contributions + 1
            """,
            (contributor_hash, now, now),
        )
        # Read back the contributor's current totals
        contrib = c.execute(
            "SELECT total_contributions, preferred_dialect FROM contributors "
            "WHERE contributor_hash = ?",
            (contributor_hash,),
        ).fetchone()
        # Clear the pending-save marker now that the save has landed.
        # Done in the same connection so it's atomic with the insert.
        c.execute(
            "UPDATE contributors SET pending_task_id = NULL, "
            "pending_dialect = NULL, pending_set_at = NULL "
            "WHERE contributor_hash = ?",
            (contributor_hash,),
        )
    return {
        "contribution_id": contribution_id,
        "total_for_contributor": int(contrib["total_contributions"]),
        "contributor_dialect": contrib["preferred_dialect"],
    }


def save_orphan(
    contributor_hash: str,
    target_dialect: str,
    target_translation_raw: str,
) -> dict:
    """Persist a translation when no pending task was active. Creates a
    placeholder task row so the contribution still satisfies the NOT NULL
    foreign key, then writes the contribution. Used by the contribute_save
    tool as a safety net when the classifier fires SAVE but no task has
    been served — the user's effort never gets dropped.

    Returns the same shape as save_contribution plus orphan_task_id.
    PII-sanitises the translation. Accepts dialect 'unknown' for cases
    where the contributor hasn't declared one yet.
    """
    cleaned = pii_sanitize(target_translation_raw or "").strip()
    if not cleaned:
        raise ValueError("empty translation after sanitisation")
    # Allow 'unknown' alongside the normal dialects — orphan saves often
    # happen before dialect is confirmed.
    if target_dialect not in VALID_DIALECTS and target_dialect != "unknown":
        raise ValueError(f"invalid dialect {target_dialect!r}")
    now = _now_iso()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO tasks (source_en, category, created_at, "
            "times_served, times_submitted) VALUES (?, ?, ?, 0, 1)",
            (
                "[orphan: classifier fired SAVE without a served task]",
                "orphan_no_pending",
                now,
            ),
        )
        orphan_task_id = cur.lastrowid
        cur = c.execute(
            "INSERT INTO contributions "
            "(task_id, contributor_hash, target_dialect, target_translation, submitted_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (orphan_task_id, contributor_hash, target_dialect, cleaned, now),
        )
        contribution_id = cur.lastrowid
        c.execute(
            """
            INSERT INTO contributors
              (contributor_hash, first_contributed_at, last_contributed_at, total_contributions)
              VALUES (?, ?, ?, 1)
              ON CONFLICT(contributor_hash) DO UPDATE SET
                last_contributed_at = excluded.last_contributed_at,
                total_contributions = total_contributions + 1
            """,
            (contributor_hash, now, now),
        )
        contrib = c.execute(
            "SELECT total_contributions FROM contributors "
            "WHERE contributor_hash = ?",
            (contributor_hash,),
        ).fetchone()
    return {
        "contribution_id": contribution_id,
        "orphan_task_id": orphan_task_id,
        "total_for_contributor": int(contrib["total_contributions"]),
    }


# ── Contributor management ──────────────────────────────────────────


def get_contributor(contributor_hash: str) -> dict | None:
    """Look up a contributor row by hash. Returns dict or None."""
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM contributors WHERE contributor_hash = ?",
            (contributor_hash,),
        ).fetchone()
    return dict(row) if row else None


def set_dialect(contributor_hash: str, dialect: str) -> None:
    """Store the contributor's preferred dialect. Idempotent — overwrites
    if they want to switch. Creates the contributor row if it doesn't
    exist yet (e.g., they declare dialect before submitting)."""
    if dialect not in VALID_DIALECTS:
        raise ValueError(f"invalid dialect {dialect!r}")
    now = _now_iso()
    with _conn() as c:
        c.execute(
            """
            INSERT INTO contributors
              (contributor_hash, preferred_dialect, first_contributed_at, last_contributed_at)
              VALUES (?, ?, ?, ?)
              ON CONFLICT(contributor_hash) DO UPDATE SET
                preferred_dialect = excluded.preferred_dialect
            """,
            (contributor_hash, dialect, now, now),
        )


def whoami(contributor_hash: str) -> str:
    """Return one of: 'new', 'unset', 'known:Oshindonga', 'known:Oshikwanyama'.
    The skill reads this to decide whether to ask the dialect question."""
    c_row = get_contributor(contributor_hash)
    if c_row is None:
        return "new"
    d = c_row.get("preferred_dialect")
    if not d:
        return "unset"
    return f"known:{d}"


def record_decline(contributor_hash: str) -> None:
    """Mark that the contributor said no to the invitation. We won't
    re-ask them within DECLINE_COOLDOWN_DAYS. Creates the contributor
    row if it doesn't exist yet."""
    now = _now_iso()
    with _conn() as c:
        c.execute(
            """
            INSERT INTO contributors
              (contributor_hash, first_contributed_at, last_contributed_at, last_declined_at)
              VALUES (?, ?, ?, ?)
              ON CONFLICT(contributor_hash) DO UPDATE SET
                last_declined_at = excluded.last_declined_at
            """,
            (contributor_hash, now, now, now),
        )


def recently_declined(contributor_hash: str, days: int = DECLINE_COOLDOWN_DAYS) -> bool:
    """True if the contributor declined within the last `days` days."""
    row = get_contributor(contributor_hash)
    if not row or not row.get("last_declined_at"):
        return False
    try:
        last = datetime.fromisoformat(row["last_declined_at"])
    except (ValueError, TypeError):
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return last >= cutoff


# ── Stats ──────────────────────────────────────────────────────────


def total_contributions(dialect: str | None = None) -> int:
    """How many contributions in total. Optionally filtered by dialect."""
    with _conn() as c:
        if dialect is None:
            row = c.execute("SELECT COUNT(*) AS n FROM contributions").fetchone()
        else:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM contributions WHERE target_dialect = ?",
                (dialect,),
            ).fetchone()
    return int(row["n"])


def contributor_total(contributor_hash: str) -> int:
    """How many translations this specific contributor has submitted."""
    row = get_contributor(contributor_hash)
    if not row:
        return 0
    return int(row.get("total_contributions") or 0)


def stats_summary() -> dict:
    """One-shot summary for the contribute_translation(action='stats') call.
    Returns total count + per-dialect breakdown + contributor count."""
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) AS n FROM contributions").fetchone()["n"]
        per_dialect = c.execute(
            "SELECT target_dialect, COUNT(*) AS n FROM contributions "
            "GROUP BY target_dialect"
        ).fetchall()
        contributors = c.execute(
            "SELECT COUNT(*) AS n FROM contributors WHERE total_contributions > 0"
        ).fetchone()["n"]
        tasks_n = c.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
    return {
        "total_contributions": int(total),
        "by_dialect": {row["target_dialect"]: int(row["n"]) for row in per_dialect},
        "total_contributors": int(contributors),
        "total_tasks": int(tasks_n),
    }


# ── CLI ────────────────────────────────────────────────────────────


def _cli_stats() -> None:
    s = stats_summary()
    print(f"tasks:         {s['total_tasks']}")
    print(f"contributors:  {s['total_contributors']}")
    print(f"contributions: {s['total_contributions']}")
    if s["by_dialect"]:
        print("by dialect:")
        for k, v in s["by_dialect"].items():
            print(f"  {k:14s} {v}")
    else:
        print("(no contributions yet)")


def _cli_sample(limit: int) -> None:
    with _conn() as c:
        rows = c.execute(
            """
            SELECT c.id, c.target_dialect, c.submitted_at,
                   t.source_en, t.category
              FROM contributions c
              JOIN tasks t ON c.task_id = t.id
             ORDER BY c.id DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
    if not rows:
        print("(no contributions yet)")
        return
    for r in rows:
        print(f"#{r['id']:>4} [{r['target_dialect']:12s}] {r['category']:15s} {r['source_en'][:60]}")


def _cli_reset(confirm: bool) -> None:
    if not confirm:
        print("Will wipe all tables. Re-run with --confirm to actually do it.")
        return
    with _conn() as c:
        c.executescript(
            "DROP TABLE IF EXISTS contributions; "
            "DROP TABLE IF EXISTS contributors; "
            "DROP TABLE IF EXISTS tasks;"
        )
    warmup()
    print("Reset complete. All three tables wiped and re-created.")


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Contributions DB CLI")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("stats", help="Show count summary")

    s = sub.add_parser("sample", help="Show last N contributions")
    s.add_argument("--limit", type=int, default=20)

    r = sub.add_parser("reset", help="Wipe all tables")
    r.add_argument("--confirm", action="store_true",
                   help="Actually perform the reset (without it, dry-runs)")

    args = p.parse_args(argv)

    if args.cmd == "stats":
        warmup()
        _cli_stats()
    elif args.cmd == "sample":
        warmup()
        _cli_sample(args.limit)
    elif args.cmd == "reset":
        _cli_reset(args.confirm)
    else:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
