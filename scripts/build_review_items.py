#!/usr/bin/env python3
"""Build the items for a full reference review on ongiini.ai/rate/.

Every reference of the translator, per dialect, becomes one "review" item
of the round; an invited reviewer of that dialect (rating_admin.py
add-reviewer) checks all of them — see ongiini/ratings.py. The items
include the blind split, so they never reach the open raters, and the
output stays under the gitignored data/private/.

    python3 scripts/build_review_items.py --translator kaarina --round v1_review
    docker exec -i ongiini-webhook python3 /data/private/rating_admin.py load /data/private/review_items_v1_review.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT / "data/private/oshiwambo_eval_v3.tsv"
DIALECTS = ("oshindonga", "oshikwanyama")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--translator", default="kaarina")
    ap.add_argument("--round", default="v1_review")
    ap.add_argument("--tsv", type=Path, default=TSV)
    args = ap.parse_args(argv)
    items = []
    with args.tsv.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            for d in DIALECTS:
                text = (r.get(f"{args.translator}_{d}") or "").strip()
                if not text:
                    continue
                key = hashlib.sha256(f"{args.round}:{r['id']}:{d}".encode()).hexdigest()[:16]
                items.append({"item_key": key, "round": args.round, "sentence_id": int(r["id"]), "dialect": d,
                              "english": r["english"].strip(), "text": text, "grp": "review", "kind": "ref",
                              "source": args.translator})
    out = ROOT / f"data/private/review_items_{args.round}.json"
    out.write_text(json.dumps(items, ensure_ascii=False, indent=0))
    print(json.dumps({d: sum(it["dialect"] == d for it in items) for d in DIALECTS}), "→", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
