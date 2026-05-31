"""One-shot cleanup of `orphan_no_pending` rows produced by the
short-lived orphan-fallback in `contribute_save` (2026-05-30 evening
deploy, reverted next morning).

Analysis of the 18 rows polluted overnight showed three categories:

  - **11 Afrikaans-practice rows** from users chatting in Afrikaans
    with no contribute-flow context. Classifier mis-classified
    non-English latin text as Oshiwambo. Not contributions — delete.
  - **3 garbage rows**: a clinic-name + date, "In oshiwambo" (user
    asking a question), and a student's multiple-choice exam answer.
    Not contributions — delete.
  - **4 real Oshindonga rows** with dialect='unknown' because the
    user hadn't declared a dialect. The translations themselves look
    legitimate. Keep, but reclassify the placeholder task's category
    so they're easy to find for manual dialect curation.

Idempotent: re-running finds nothing to clean.

Usage:
    python3 scripts/cleanup_orphan_pollution.py --dry-run     # print plan
    python3 scripts/cleanup_orphan_pollution.py               # execute
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Allow running from repo root with /app sys.path inside the container.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/app")

from ongiini import contributions  # noqa: E402


# Contribution IDs to DELETE (Afrikaans practice + garbage).
# Each row's matching placeholder task (category='orphan_no_pending')
# is also deleted in the same transaction.
DELETE_IDS = [
    # Afrikaans practice — user 8c84cb87
    209, 210, 211, 212,
    # Afrikaans practice — user 12754b00
    216, 217, 219, 220, 221, 222, 223,
    # Garbage
    208,   # "23 -03-2025\nOutapi clinic"
    214,   # "In oshiwambo" (question, not contribution)
    229,   # "Sunlight\noption 1.B..." (student's chemistry answer)
]

# Contribution IDs to KEEP (real Oshindonga, dialect=unknown).
# Their placeholder tasks get reclassified so manual curation can
# find them later.
KEEP_IDS = [213, 218, 224, 225]


NEW_CATEGORY = "real_oshindonga_recovered_2026-05-31"
OLD_CATEGORY = "orphan_no_pending"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(contributions._db_path())
    c.row_factory = sqlite3.Row
    return c


def _summarise_remaining(c: sqlite3.Connection) -> dict[str, int]:
    return {
        "orphan_no_pending_total": c.execute(
            "SELECT COUNT(*) FROM contributions c "
            "JOIN tasks t ON t.id = c.task_id "
            "WHERE t.category = ?", (OLD_CATEGORY,)
        ).fetchone()[0],
        "real_recovered_total": c.execute(
            "SELECT COUNT(*) FROM contributions c "
            "JOIN tasks t ON t.id = c.task_id "
            "WHERE t.category = ?", (NEW_CATEGORY,)
        ).fetchone()[0],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit without changes.")
    args = p.parse_args()

    conn = _conn()

    # Show current state
    before = _summarise_remaining(conn)
    print(f"BEFORE: {before}")
    print()

    # Lookup the rows we plan to touch, plus their tasks
    targets = []
    for cid in DELETE_IDS + KEEP_IDS:
        row = conn.execute(
            "SELECT c.id, c.task_id, c.contributor_hash, c.target_dialect, "
            "       c.target_translation, t.category "
            "FROM contributions c JOIN tasks t ON t.id = c.task_id "
            "WHERE c.id = ?", (cid,)
        ).fetchone()
        if row is None:
            targets.append((cid, None, "missing — already cleaned?"))
            continue
        action = "DELETE" if cid in DELETE_IDS else f"RECLASSIFY task → {NEW_CATEGORY}"
        # Sanity: only touch rows still in the old category
        if row["category"] != OLD_CATEGORY and cid in DELETE_IDS:
            action = f"SKIP (already category={row['category']!r})"
        if row["category"] == NEW_CATEGORY and cid in KEEP_IDS:
            action = "SKIP (already reclassified)"
        targets.append((cid, dict(row), action))

    print("PLAN:")
    for cid, row, action in targets:
        if row is None:
            print(f"  contrib {cid}: {action}")
            continue
        preview = row["target_translation"][:60].replace("\n", " ")
        print(f"  contrib {cid:>3}: {action}")
        print(f"             hash={row['contributor_hash'][:8]} "
              f"dialect={row['target_dialect']} text={preview!r}")
    print()

    if args.dry_run:
        print("--dry-run: not applying changes.")
        return 0

    # Apply
    conn.execute("BEGIN")
    try:
        deleted = 0
        reclassified = 0
        for cid, row, action in targets:
            if row is None:
                continue
            if cid in DELETE_IDS and action.startswith("DELETE"):
                # Delete the contribution row first, then its placeholder task
                conn.execute("DELETE FROM contributions WHERE id = ?", (cid,))
                conn.execute("DELETE FROM tasks WHERE id = ?",
                             (row["task_id"],))
                deleted += 1
            elif cid in KEEP_IDS and action.startswith("RECLASSIFY"):
                conn.execute(
                    "UPDATE tasks SET category = ? WHERE id = ?",
                    (NEW_CATEGORY, row["task_id"]),
                )
                reclassified += 1
        # Recompute contributor totals so deleted rows aren't counted
        # against users any more.
        for cid in DELETE_IDS:
            # We need the hash to recompute — fetch from the targets list
            row = next((r for c, r, _ in targets if c == cid and r), None)
            if not row:
                continue
            h = row["contributor_hash"]
            new_total = conn.execute(
                "SELECT COUNT(*) FROM contributions WHERE contributor_hash = ?",
                (h,),
            ).fetchone()[0]
            conn.execute(
                "UPDATE contributors SET total_contributions = ? WHERE contributor_hash = ?",
                (new_total, h),
            )
        conn.commit()
        print(f"APPLIED: {deleted} deleted, {reclassified} reclassified.")
    except Exception as exc:                       # noqa: BLE001
        conn.rollback()
        print(f"FAILED: {exc}")
        return 1

    after = _summarise_remaining(conn)
    print(f"AFTER:  {after}")
    print()
    if after["orphan_no_pending_total"] != 0:
        print("⚠️  orphan_no_pending rows remain. Re-run with --dry-run to "
              "inspect what wasn't cleaned (likely new rows generated since "
              "this script's hard-coded IDs were captured).")
        return 1
    print("✓ All orphan_no_pending rows removed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
