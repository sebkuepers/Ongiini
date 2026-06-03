#!/usr/bin/env python3
"""Reactivate contributors who are still in their WhatsApp 24h-window.

Goal: send a personalized free-text in-window message to each prior
contributor today, asking if they have 2 minutes to translate one more
sentence. Personalises by their dialect + their previous contribution
count. Pre-fetches a fresh task from the queue for each user and pre-
sets their `pending_task_id` so when they reply, the existing
contribute_save_translation tool flow handles it cleanly.

Run inside the webhook container:

    docker exec -i ongiini-webhook python3 /data/reactivate_contributors.py \
        --dry-run

Then if happy:

    docker exec -i ongiini-webhook python3 /data/reactivate_contributors.py
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, "/app")
from ongiini.contributions import hash_msisdn   # uses the same salt
from ongiini.whatsapp import send_text
from ongiini.memory import short_term as memory

log = logging.getLogger("reactivate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATA_DIR = Path("/data")
CONTRIB_DB = DATA_DIR / "contributions.sqlite"
USAGE_LOG = DATA_DIR / "usage.log"
SNAPSHOT_PATH = DATA_DIR / "reactivation_batch.json"

USAGE_LINE = re.compile(
    r"^(?P<ts>\S+)\s\|\s(?P<msi>\d+)\s\|.*kind=(?P<kind>\w+)"
)


def build_msisdn_lookup() -> dict[str, str]:
    """Walk /data/*.json (per-user memories) and build {salted_hash: msisdn}.
    Includes ALL phone-number-shaped filenames (not just 264*) so we
    catch foreign test numbers etc."""
    lookup: dict[str, str] = {}
    for p in sorted(DATA_DIR.glob("*.json")):
        stem = p.stem
        # Must be all digits (phone-number-shaped), 9+ chars
        if not stem.isdigit() or len(stem) < 9:
            continue
        try:
            h = hash_msisdn(stem)
        except Exception:
            continue
        lookup[h] = stem
    return lookup


def find_last_router_event_per_msisdn() -> dict[str, datetime]:
    """Scan usage.log for each msisdn's most recent kind=router event
    (user-message arrival). Used for 24h-window check."""
    last: dict[str, datetime] = {}
    with USAGE_LOG.open() as f:
        for line in f:
            m = USAGE_LINE.match(line.strip())
            if not m or m.group("kind") != "router":
                continue
            msi = m.group("msi")
            try:
                ts = datetime.fromisoformat(m.group("ts")).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if msi not in last or ts > last[msi]:
                last[msi] = ts
    return last


def get_contributors(conn: sqlite3.Connection) -> list[dict]:
    """Return list of {hash, preferred_dialect, total_contributions, last_contributed_at, msisdns_done_task_ids}.

    Each contributor row also carries a list of task_ids they've already
    contributed to (so we don't re-serve them an old task).
    """
    # Use TWO cursors so the inner queries don't consume the outer iteration
    outer = conn.cursor()
    inner = conn.cursor()
    rows = []
    for r in outer.execute("""
        SELECT contributor_hash, preferred_dialect, total_contributions,
               last_contributed_at, pending_task_id
        FROM contributors
    """):
        h, dialect, n, last, pending_task = r
        # Tasks already done in their preferred dialect
        done = set()
        for tr in inner.execute(
            "SELECT task_id FROM contributions "
            "WHERE contributor_hash = ? AND target_dialect = ?",
            (h, dialect),
        ):
            done.add(tr[0])
        rows.append({
            "contributor_hash": h,
            "preferred_dialect": dialect,
            "total_contributions": n,
            "last_contributed_at": last,
            "pending_task_id": pending_task,
            "done_task_ids": done,
        })
    return rows


MAX_WORDS_PREFERRED = 12   # short enough for 2-minute commitment
MAX_WORDS_HARD = 20        # absolute ceiling


def pick_fresh_task(
    conn: sqlite3.Connection, done_task_ids: set, category_pref: list[str] | None = None
) -> dict | None:
    """Pick a fresh task (not yet done by this contributor) — prefer
    short conversational categories first, then any. Hard cap at
    MAX_WORDS_HARD; soft prefer MAX_WORDS_PREFERRED."""
    cur = conn.cursor()
    if category_pref is None:
        category_pref = [
            "greeting", "ack", "rapport", "closing",
            "chat", "vocab", "education", "health",
        ]
    placeholders = ",".join("?" * len(done_task_ids)) if done_task_ids else "0"
    done_filter = f"AND id NOT IN ({placeholders})" if done_task_ids else ""
    args_done = list(done_task_ids) if done_task_ids else []

    def _looks_like_fragment(text: str) -> bool:
        """Reject obvious fragments of numbered lists or parenthetical
        asides (e.g. '(In a house, in space, in a school?) 3.')."""
        t = text.strip()
        if t.startswith("(") and "?)" in t:
            return True
        if re.match(r"^\d+\.\s", t):
            return True
        if re.search(r"\s\d+\.\s*$", t):
            return True
        return False

    def _try(category: str | None, max_words: int):
        cat_clause = "AND category = ?" if category else ""
        args = ([category] if category else []) + args_done
        q = (
            f"SELECT id, source_en, category, times_served "
            f"FROM tasks "
            f"WHERE 1=1 {cat_clause} {done_filter} "
            f"ORDER BY times_served ASC, RANDOM() LIMIT 30"
        )
        rows = cur.execute(q, args).fetchall()
        for row in rows:
            wc = len(str(row[1]).split())
            if wc > max_words:
                continue
            if _looks_like_fragment(str(row[1])):
                continue
            return {"id": row[0], "source_en": row[1],
                    "category": row[2], "times_served": row[3],
                    "word_count": wc}
        return None

    # Pass 1: preferred categories, preferred word cap
    for cat in category_pref:
        r = _try(cat, MAX_WORDS_PREFERRED)
        if r:
            return r
    # Pass 2: preferred categories, hard ceiling
    for cat in category_pref:
        r = _try(cat, MAX_WORDS_HARD)
        if r:
            return r
    # Pass 3: any category, hard ceiling
    return _try(None, MAX_WORDS_HARD)


def build_message(en: str, dialect: str, prev_count: int) -> str:
    """Build the in-window reactivation message. Personalised by dialect + count."""
    greeting = "Tangi unene" if dialect == "Oshindonga" else "Tangi unene"
    plural = "translations" if prev_count != 1 else "translation"
    return (
        f"{greeting} for your {prev_count} previous {plural} in {dialect}!\n\n"
        f"We're collecting more data this morning. Could you spare 2 minutes "
        f"to translate one short sentence?\n\n"
        f"English: \"{en}\"\n\n"
        f"Just reply with your {dialect} version — or say \"not now\" if "
        f"you're busy. Tangi!"
    )


async def main(args: argparse.Namespace) -> int:
    # Verify env / salt is loaded (will raise RuntimeError if not)
    test_hash = hash_msisdn("264811000000")
    log.info("salt verified (test hash prefix: %s)", test_hash[:12])

    log.info("building msisdn lookup from per-user JSON files…")
    hash_to_msisdn = build_msisdn_lookup()
    log.info("  → %d msisdns hashed", len(hash_to_msisdn))

    log.info("scanning usage.log for last router events per user…")
    last_router = find_last_router_event_per_msisdn()
    log.info("  → %d msisdns with router events", len(last_router))

    conn = sqlite3.connect(CONTRIB_DB)
    contributors = get_contributors(conn)
    log.info("loaded %d contributor records", len(contributors))

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)
    log.info("24h cutoff = %s (anyone with last router event >= this)",
             cutoff.isoformat())

    candidates = []
    for c in contributors:
        h = c["contributor_hash"]
        # Skip contributors who never actually submitted a translation
        # (just started the flow but bailed). Don't ping them as if they
        # had contributed.
        if c["total_contributions"] < 1 or not c["preferred_dialect"]:
            log.info("  %s — no submitted contributions yet, skip", h[:12])
            continue
        msisdn = hash_to_msisdn.get(h)
        if not msisdn:
            log.warning("contributor %s — no matching msisdn (hash not in /data)", h[:12])
            continue
        last_router_ts = last_router.get(msisdn)
        if not last_router_ts:
            log.info("  %s — no router events ever, skip", h[:12])
            continue
        if last_router_ts < cutoff:
            hours_ago = (now - last_router_ts).total_seconds() / 3600
            log.info("  %s — out of window (%.1fh ago), skip", h[:12], hours_ago)
            continue
        # In window — pick a task
        task = pick_fresh_task(conn, c["done_task_ids"])
        if not task:
            log.warning("  %s — no fresh tasks available, skip", h[:12])
            continue
        message = build_message(
            task["source_en"], c["preferred_dialect"], c["total_contributions"],
        )
        candidates.append({
            "contributor_hash": h,
            "msisdn": msisdn,
            "preferred_dialect": c["preferred_dialect"],
            "prev_contributions": c["total_contributions"],
            "last_router_ts": last_router_ts.isoformat(),
            "hours_in_window": round((now - last_router_ts).total_seconds() / 3600, 1),
            "task_id": task["id"],
            "source_en": task["source_en"],
            "category": task["category"],
            "message": message,
        })

    log.info("\n=== %d in-window contributors selected for reactivation ===", len(candidates))
    for c in candidates:
        print(f"\n--- ...{c['msisdn'][-4:]}  ({c['preferred_dialect']}, "
              f"{c['prev_contributions']}× before, "
              f"last msg {c['hours_in_window']}h ago) ---")
        print(f"task #{c['task_id']} ({c['category']}): {c['source_en']}")
        print(f"\nmessage:\n{c['message']}")

    # Snapshot
    snapshot = [{**c, "done_task_ids": None} for c in candidates]
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, default=str, indent=2, ensure_ascii=False))
    log.info("\nwrote snapshot to %s", SNAPSHOT_PATH)

    if args.dry_run:
        log.info("\n[dry-run] no sends performed. Re-run without --dry-run to send.")
        return 0

    log.info("\n=== SENDING ===")
    n_sent = n_fail = 0
    interval = 1.0 / max(0.1, args.rate_per_sec)
    log_path = DATA_DIR / "reactivation.log"

    for i, c in enumerate(candidates, start=1):
        t0 = time.monotonic()
        msisdn = c["msisdn"]
        body = c["message"]
        result = {
            "msisdn_hash6": hashlib.sha256(msisdn.encode()).hexdigest()[:12],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "len": len(body),
            "task_id": c["task_id"],
            "dialect": c["preferred_dialect"],
        }
        try:
            # 1. Write the proactive turn into per-user memory first so
            #    the agent has context if the user replies.
            async with memory.lock_for(msisdn):
                memory.append_synthetic_assistant_turn(msisdn, body)
            # 2. Set their pending_task_id so contribute flow picks
            #    them up on reply. Use the contributions module helper.
            from ongiini.contributions import set_pending_save
            set_pending_save(
                contributor_hash=c["contributor_hash"],
                task_id=c["task_id"],
                dialect=c["preferred_dialect"],
            )
            # 3. Send.
            await send_text(msisdn, body)
            result["status"] = "sent"
            n_sent += 1
        except Exception as exc:                       # noqa: BLE001
            log.warning("send failed for %s: %s", msisdn[-4:], exc)
            result["status"] = "send_failed"
            result["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            n_fail += 1
        with log_path.open("a") as f:
            f.write(json.dumps(result, separators=(",", ":")) + "\n")
        log.info("[%d/%d]  %s — %s", i, len(candidates), msisdn[-4:], result["status"])
        elapsed = time.monotonic() - t0
        if elapsed < interval and i < len(candidates):
            await asyncio.sleep(interval - elapsed)

    log.info("\nDONE: sent=%d failed=%d", n_sent, n_fail)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Identify candidates + show messages, do not send")
    p.add_argument("--rate-per-sec", type=float, default=2.0,
                   help="Send throttle. Default 2/s.")
    args = p.parse_args()
    sys.exit(asyncio.run(main(args)))
