#!/usr/bin/env python3
"""Extend the frozen 423-item v2 set to the 600-item v1.0 composition.

Append-only: every v2 row keeps its id, fields and blind-split flag, so
references already delivered for ids 1..423 stay valid. The 177 new
items come from three v3 seed files and get ids 424..600, grouped by
provenance. 20% of the new items join the blind split (seeded), which
keeps the blind share of the whole set at ~20%.

Provenance labels follow the v1.0 schema: v2's `real_mined` becomes
`mined_paraphrased` (the English was always a paraphrase).

Writes:
  data/oshiwambo_eval_v3.tsv            English sources + baseline columns
                                        (public; no references)
  data/private/oshiwambo_eval_v3.tsv    same rows plus every translator /
                                        extra column carried over from
                                        data/private/oshiwambo_eval_v2.tsv

    python3 scripts/build_eval_v3.py
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path

from build_eval_v2 import length_bucket, load_seeds_markdown

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data/oshiwambo_eval_v2.tsv"
V2_PRIVATE = ROOT / "data/private/oshiwambo_eval_v2.tsv"
OUT = ROOT / "data/oshiwambo_eval_v3.tsv"
OUT_PRIVATE = ROOT / "data/private/oshiwambo_eval_v3.tsv"
SEEDS = [  # (path, provenance) in id order
    (ROOT / "data/oshiwambo_eval_v3_mined_seeds.md", "mined_paraphrased"),
    (ROOT / "data/oshiwambo_eval_v3_crafted_seeds.md", "crafted"),
    (ROOT / "data/oshiwambo_eval_v3_formal_seeds.md", "formal_drafted"),
]
PUBLIC_COLS = [
    "id", "length_bucket", "domain", "phenomenon_tags", "provenance", "english",
    "claude_oshindonga", "gemma_oshindonga", "claude_oshikwanyama",
    "gemma_oshikwanyama", "in_blind_split",
]
PHENOMENA = [
    "negation", "noun_class_agreement", "pronoun_coreference", "tense_aspect",
    "numbers_dates", "named_entities", "code_switch", "politeness_register",
    "idiom_nonliteral", "polysemy", "multi_sentence",
]
TAG_FLOOR = 30


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind-split-pct", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    with V2.open() as f:
        old = list(csv.DictReader(f, delimiter="\t"))
    for r in old:
        if r["provenance"] == "real_mined":
            r["provenance"] = "mined_paraphrased"

    new: list[dict] = []
    for path, prov in SEEDS:
        for it in load_seeds_markdown(path, prov):
            if prov == "mined_paraphrased" and it["domain"] == "challenge":
                it["domain"] = "chat"  # mined items always carry a domain
            new.append(it)
    next_id = max(int(r["id"]) for r in old) + 1
    for n, it in enumerate(new, start=next_id):
        it["id"] = str(n)
        it["length_bucket"] = length_bucket(it["english"])
        it["phenomenon_tags"] = ";".join(it["phenomenon_tags"])

    rng = random.Random(args.seed)
    n_blind = round(len(new) * args.blind_split_pct / 100)
    blind = set(rng.sample([it["id"] for it in new], n_blind))
    for it in new:
        it["in_blind_split"] = "true" if it["id"] in blind else "false"

    rows = old + new
    failures = []
    seen = Counter(norm(r["english"]) for r in rows)
    dupes = [k for k, v in seen.items() if v > 1]
    if dupes:
        failures.append(f"duplicate English: {dupes[:3]}")
    tags = Counter(t for r in rows for t in r["phenomenon_tags"].split(";") if t)
    unknown = set(tags) - set(PHENOMENA)
    if unknown:
        failures.append(f"unknown tags: {sorted(unknown)}")
    thin = {t: tags[t] for t in PHENOMENA if tags[t] < TAG_FLOOR}
    if thin:
        failures.append(f"tags below {TAG_FLOOR}: {thin}")
    if failures:
        for msg in failures:
            print("ERROR: " + msg, file=sys.stderr)
        return 1

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PUBLIC_COLS, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in PUBLIC_COLS} for r in rows)

    if V2_PRIVATE.exists():
        with V2_PRIVATE.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            extra = [c for c in reader.fieldnames if c not in PUBLIC_COLS]
            private = {r["id"]: r for r in reader}
        cols = PUBLIC_COLS[:-1] + extra + ["in_blind_split"]
        with OUT_PRIVATE.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow({**{c: r.get(c, "") for c in PUBLIC_COLS},
                            **{c: private.get(r["id"], {}).get(c, "") for c in extra}})

    n = len(rows)
    lengths = Counter(r["length_bucket"] for r in rows)
    print(f"{n} items ({len(old)} frozen + {len(new)} new, ids {next_id}..{n})",
          file=sys.stderr)
    print("provenance:", dict(Counter(r["provenance"] for r in rows)), file=sys.stderr)
    print("length:", {b: f"{lengths[b]} ({lengths[b] / n:.0%})" for b in "SML"},
          file=sys.stderr)
    print("domain:", dict(Counter(r["domain"] for r in rows)), file=sys.stderr)
    print("phenomena:", {t: tags[t] for t in PHENOMENA}, file=sys.stderr)
    b = sum(r["in_blind_split"] == "true" for r in rows)
    print(f"blind: {b} ({b / n:.1%})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
