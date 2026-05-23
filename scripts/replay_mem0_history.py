"""Replay surviving mem0 facts from mem0_history.db back into qdrant.

Recovery tool for the 2026-05-23 data-loss bug: mem0 0.1.x's QdrantDB
rmtree'd the qdrant directory on every container restart, destroying
the vector store while leaving the SQLite audit trail intact.

This script reads mem0_history.db, identifies facts that should exist
(ADDed/UPDATEd, never DELETEd), re-embeds their text with the same
sentence-transformer mem0 uses, and inserts them directly into qdrant
via the low-level QdrantDB adapter.

Why direct qdrant insert (not mem0.add()):
- mem0.add() re-runs LLM fact extraction on raw messages
- We already have the extracted facts in mem0_history
- We just need to put them back in qdrant with the correct point IDs
- Going through mem0.add() would duplicate-extract and double-write

USAGE
    # Show what would be recovered, do not modify qdrant
    docker exec ongiini-webhook python3 /app/scripts/replay_mem0_history.py --dry-run

    # Actually replay
    docker exec ongiini-webhook python3 /app/scripts/replay_mem0_history.py

The script must run INSIDE the webhook container (it needs mem0,
qdrant_client, sentence-transformers, and access to /data).
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("replay_mem0")


def _final_state_per_memory(history_db: Path) -> dict[str, dict]:
    """Walk mem0_history.db and resolve each memory_id to its final
    state (text). Returns a dict mapping memory_id → {"text": ...,
    "first_seen": ..., "last_seen": ...}.

    Skips memory_ids whose last event was DELETE (those facts were
    explicitly removed and should stay gone)."""
    h = sqlite3.connect(str(history_db))
    try:
        rows = h.execute(
            "SELECT memory_id, old_memory, new_memory, event, created_at "
            "FROM history ORDER BY created_at ASC"
        ).fetchall()
    finally:
        h.close()

    state: dict[str, dict] = {}
    deleted: set[str] = set()

    for memory_id, old_text, new_text, event, created_at in rows:
        if event == "DELETE":
            state.pop(memory_id, None)
            deleted.add(memory_id)
            continue
        if memory_id in deleted:
            deleted.discard(memory_id)
        if event in ("ADD", "UPDATE"):
            text = new_text or old_text or ""
            if not text.strip():
                continue
            cur = state.get(memory_id, {})
            state[memory_id] = {
                "text": text,
                "first_seen": cur.get("first_seen", created_at),
                "last_seen": created_at,
            }
    return state


def _qdrant_existing_ids(qdrant_path: str = "/data/qdrant") -> set[str]:
    """Enumerate existing point IDs in qdrant via the official client.

    Uses scroll() instead of reading the storage.sqlite directly, so
    we never touch pickle ourselves — qdrant_client owns its
    serialisation format."""
    from qdrant_client import QdrantClient
    try:
        client = QdrantClient(path=qdrant_path)
    except Exception as exc:
        log.warning("could not open qdrant at %s: %s", qdrant_path, exc)
        return set()
    ids: set[str] = set()
    try:
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name="ongiini_memories",
                limit=512,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for p in points:
                ids.add(str(p.id))
            if next_offset is None:
                break
            offset = next_offset
    except Exception as exc:
        log.warning("qdrant scroll failed (collection may not exist yet): %s", exc)
    finally:
        try:
            client.close()
        except Exception:
            pass
    return ids


def _attach_user_id_from_short_term(facts: dict[str, dict]) -> dict[str, dict]:
    """Cross-reference timestamps against short-term JSON file mtimes
    to attribute facts to users. The mem0 history table doesn't store
    user_id directly.

    Best-effort — facts with no match get user_id="unknown" and still
    get inserted (not queryable by user but at least preserved)."""
    from datetime import datetime
    data_dir = Path("/data")
    candidates: list[tuple[float, str]] = []
    for jp in data_dir.glob("*.json"):
        name = jp.stem
        if not name.isdigit() or len(name) != 12:
            continue
        try:
            mtime = jp.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, name))
    candidates.sort()

    def msisdn_at(ts_iso: str) -> str | None:
        try:
            dt = datetime.fromisoformat(ts_iso)
            target = dt.timestamp()
        except Exception:
            return None
        best = None
        best_delta = 120.0  # ±2 min slack
        for mtime, msisdn in candidates:
            delta = abs(mtime - target)
            if delta < best_delta:
                best_delta = delta
                best = msisdn
        return best

    for memory_id, fact in facts.items():
        fact["user_id"] = msisdn_at(fact["last_seen"])
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be recovered; do not modify qdrant.")
    parser.add_argument("--history-db", default="/data/mem0_history.db",
                        help="Path to mem0_history.db (default: /data/mem0_history.db)")
    parser.add_argument("--qdrant-path", default="/data/qdrant",
                        help="Qdrant local-storage path (default: /data/qdrant)")
    args = parser.parse_args()

    history_db = Path(args.history_db)
    if not history_db.exists():
        log.error("mem0 history DB not found: %s", history_db)
        return 1

    log.info("loading mem0 history from %s", history_db)
    surviving = _final_state_per_memory(history_db)
    log.info("surviving facts in history (non-deleted): %d", len(surviving))

    log.info("checking qdrant for existing point IDs…")
    in_qdrant = _qdrant_existing_ids(args.qdrant_path)
    log.info("qdrant currently has %d points", len(in_qdrant))

    missing_ids = [mid for mid in surviving if mid not in in_qdrant]
    log.info("missing from qdrant (recovery candidates): %d", len(missing_ids))

    if not missing_ids:
        log.info("nothing to recover — qdrant matches history.")
        return 0

    log.info("attaching user_ids via short-term file timestamps…")
    to_recover = _attach_user_id_from_short_term(
        {mid: surviving[mid] for mid in missing_ids}
    )

    by_user: dict[str, list[str]] = defaultdict(list)
    for mid, fact in to_recover.items():
        by_user[fact.get("user_id") or "(unknown)"].append(fact["text"])

    log.info("recovery summary:")
    for user, texts in sorted(by_user.items(), key=lambda kv: -len(kv[1])):
        log.info("  %s: %d facts", user, len(texts))
        for t in texts[:3]:
            log.info("    - %s", t[:120])
        if len(texts) > 3:
            log.info("    … and %d more", len(texts) - 3)

    if args.dry_run:
        log.info("--dry-run set — no changes made")
        return 0

    log.info("loading embedder + qdrant adapter…")
    from sentence_transformers import SentenceTransformer
    from mem0.vector_stores.qdrant import Qdrant
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    qdb = Qdrant(
        collection_name="ongiini_memories",
        embedding_model_dims=384,
        path=args.qdrant_path,
        on_disk=True,
    )
    log.info("inserting %d facts into qdrant…", len(missing_ids))
    inserted = 0
    for mid in missing_ids:
        fact = to_recover[mid]
        try:
            vec = embedder.encode([fact["text"]])[0].tolist()
            payload = {
                "user_id": fact.get("user_id") or "unknown",
                "data": fact["text"],
                "hash": "",
                "created_at": fact.get("first_seen") or fact.get("last_seen"),
            }
            qdb.insert(vectors=[vec], payloads=[payload], ids=[mid])
            inserted += 1
        except Exception as exc:
            log.warning("failed to insert %s: %s", mid, exc)
    log.info("recovered %d facts (of %d candidates)", inserted, len(missing_ids))
    return 0 if inserted else 1


if __name__ == "__main__":
    sys.exit(main())
