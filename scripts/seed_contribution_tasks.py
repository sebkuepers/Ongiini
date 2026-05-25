"""Load the curated seed corpus into the contributions task pool.

This is the v2 of seed_contribution_tasks.py. The v1 was a Gemma-
paraphrase pump that inflated 200 seeds × 50 paraphrases → 10K rows;
that approach was scrapped because paraphrasing maximises volume but
not semantic diversity (the model fine-tune would learn ~190 distinct
concepts dressed in 9500 outfits).

The v2 corpus is produced by mining production assistant replies
(scripts/mine_production_sentences.py) and deduplicating
(scripts/dedupe_mined_sentences.py). The pipeline runs OFF this
script — by the time we get here, the corpus is just a JSONL file
of curated sentences ready to write straight into sqlite.

Usage:
    # Quick visual check — no DB writes
    python3 scripts/seed_contribution_tasks.py --seeds /tmp/curated_seeds_v2.jsonl --dry-run

    # Live load — writes ~5.8K rows into /data/contributions.sqlite
    python3 scripts/seed_contribution_tasks.py --seeds /tmp/curated_seeds_v2.jsonl

    # Reset (wipe tasks; full DB reset via contributions CLI)
    python3 scripts/seed_contribution_tasks.py --reset --confirm
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Make ongiini.* importable when run via `python3 scripts/...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ongiini import contributions  # noqa: E402


log = logging.getLogger("seed_contribution_tasks")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per line. Each must have at least
    ``source_en`` and ``category``."""
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{i}: invalid JSON: {e}")
            if "source_en" not in d:
                raise RuntimeError(f"{path}:{i}: missing 'source_en'")
            out.append(d)
    return out


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.reset:
        if not args.confirm:
            print(
                "Will wipe the tasks table. Re-run with --confirm to "
                "actually perform the reset.",
                file=sys.stderr,
            )
            return 1
        contributions.warmup()
        with contributions._conn() as c:
            existing = c.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
            c.execute("DELETE FROM tasks")
        log.info("wiped tasks table (had %d rows)", existing)
        return 0

    seeds_path = Path(args.seeds)
    if not seeds_path.exists():
        log.error("seeds file not found: %s", seeds_path)
        return 2

    seeds = _load_jsonl(seeds_path)
    log.info("loaded %d sentences from %s", len(seeds), seeds_path)

    # Per-category count for visibility
    by_cat: dict[str, int] = {}
    for s in seeds:
        by_cat[s.get("category", "unknown")] = by_cat.get(s.get("category", "unknown"), 0) + 1
    for cat in sorted(by_cat, key=lambda c: -by_cat[c]):
        log.info("  %-18s %d", cat, by_cat[cat])

    if args.dry_run:
        log.info("dry-run — no DB writes")
        return 0

    contributions.warmup()
    existing = contributions.task_count()
    if existing > 0 and not args.force:
        log.error(
            "tasks table already has %d rows. Re-run with --force to "
            "append, or --reset --confirm to wipe first.",
            existing,
        )
        return 3

    # Map each input dict into the seed_tasks contract (source_en,
    # category, seed_id). We forward the provenance id (mining
    # script gave us a stable id) into the seed_id column.
    rows = [
        {
            "source_en": s["source_en"],
            "category": s.get("category"),
            "seed_id": s.get("id"),
        }
        for s in seeds
    ]
    inserted = contributions.seed_tasks(rows)
    log.info("inserted %d rows into tasks (had %d before; now %d)",
             inserted, existing, contributions.task_count())
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", default="/tmp/curated_seeds_v2.jsonl",
                   help="path to curated seeds JSONL")
    p.add_argument("--dry-run", action="store_true",
                   help="show per-category counts without writing")
    p.add_argument("--force", action="store_true",
                   help="allow loading into a non-empty tasks table")
    p.add_argument("--reset", action="store_true",
                   help="wipe the tasks table (requires --confirm)")
    p.add_argument("--confirm", action="store_true",
                   help="actually perform --reset")
    return p


def main(argv: list[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
