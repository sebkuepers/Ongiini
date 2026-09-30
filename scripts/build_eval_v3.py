#!/usr/bin/env python3
"""Extend the frozen 423-item v2 set to the 600-item v1.0 composition.

Append-only for items: every v2 row keeps its id and fields, so
references already delivered for ids 1..423 stay valid. The 177 new
items come from three v3 seed files and get ids 424..600, grouped by
provenance.

The blind split follows the concept paper (§3): 30% of all 600 items
(180), stratified by primary phenomenon × length bucket × domain,
seed 42. It replaces the v0.1 flags (84 of 423, unstratified) — safe
because no references or scores had been published.

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


def stratified_blind(rows: list[dict], share: float, seed: int) -> set[str]:
    """Pick round(share * n) ids so every stratum (primary phenomenon ×
    length × domain) is represented in proportion. Quotas use the
    largest-remainder method; ties and within-stratum picks are seeded."""
    rng = random.Random(seed)
    strata: dict[tuple, list[str]] = {}
    for r in rows:
        primary = r["phenomenon_tags"].split(";")[0] or "none"
        strata.setdefault((primary, r["length_bucket"], r["domain"]), []).append(r["id"])
    target = round(share * len(rows))
    keys = sorted(strata)
    quota = {k: int(share * len(strata[k])) for k in keys}
    rest = sorted(keys, key=lambda k: (-(share * len(strata[k]) - quota[k]), rng.random()))
    for k in rest[: target - sum(quota.values())]:
        quota[k] += 1
    picked: set[str] = set()
    for k in keys:
        ids = sorted(strata[k], key=int)
        rng.shuffle(ids)
        picked.update(ids[: quota[k]])
    return balance_tags(rows, picked, share, rng)


def balance_tags(rows: list[dict], picked: set[str], share: float,
                 rng: random.Random) -> set[str]:
    """Stratifying on the primary tag under-samples secondary tags. Swap
    a dev item carrying a short tag for a blind item of the same length
    bucket until every tag has at least floor(share * n_tag) blind items,
    without pushing any other tag below its floor."""
    tags = {r["id"]: [t for t in r["phenomenon_tags"].split(";") if t] for r in rows}
    length = {r["id"]: r["length_bucket"] for r in rows}
    total = Counter(t for ts in tags.values() for t in ts)
    floor = {t: int(share * n) for t, n in total.items()}
    order = sorted(tags, key=int)
    rng.shuffle(order)
    for _ in range(200):
        count = Counter(t for i in picked for t in tags[i])
        short = [t for t in sorted(floor) if count[t] < floor[t]]
        if not short:
            break
        t = short[0]
        swap = None
        for add in (i for i in order if i not in picked and t in tags[i]):
            for drop in (i for i in order if i in picked and length[i] == length[add]
                         and t not in tags[i]):
                after = count.copy()
                after.update(tags[add])
                after.subtract(tags[drop])
                if all(after[x] >= floor[x] or after[x] >= count[x] for x in floor):
                    swap = (add, drop)
                    break
            if swap:
                break
        if not swap:
            break
        picked = (picked - {swap[1]}) | {swap[0]}
    return picked


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind-split-pct", type=float, default=30.0)
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

    rows = old + new
    blind = stratified_blind(rows, args.blind_split_pct / 100, args.seed)
    for r in rows:
        r["in_blind_split"] = "true" if r["id"] in blind else "false"
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
    bt = Counter(t for r in rows if r["in_blind_split"] == "true"
                 for t in r["phenomenon_tags"].split(";") if t)
    print("blind per phenomenon:", {t: bt[t] for t in PHENOMENA}, file=sys.stderr)
    print("blind per length:", dict(Counter(r["length_bucket"] for r in rows
                                            if r["in_blind_split"] == "true")), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
