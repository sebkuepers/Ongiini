#!/usr/bin/env python3
"""Build the final v2 eval-set TSV by combining all source buckets.

Inputs (all under data/):
  - eval_v2_retained_v1.tsv             ~150-191 items from v1 curation
  - eval_v2_real_candidates.tsv         ~286 mined; filter to sebastian_approved=y
  - oshiwambo_eval_v2_challenge_seeds.md  ~110 crafted phenomenon items
  - oshiwambo_eval_v2_formal_seeds.md     ~20 crafted formal items

Output:
  - data/oshiwambo_eval_v2.tsv          the final v2 dataset

What this script does:
  1. Loads + parses each source.
  2. Computes length_bucket from word count (S ≤6, M ≤18, L >18).
  3. Assigns stable IDs 1..N (sorted: provenance then within-source order).
  4. Sets in_blind_split=true for a deterministic random 20% (seed=42).
  5. Writes the full v2 TSV schema with empty translation columns.
  6. Validates: phenomenon tags ≥10 each, no duplicate EN, no digit-leak PII.
     Validation failures print to stderr — script exits non-zero.

Usage:
  python3 scripts/build_eval_v2.py [--target 400] [--out data/oshiwambo_eval_v2.tsv]
  python3 scripts/build_eval_v2.py --include-unreviewed   # for preview before
                                                            Sebastian reviews
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

RETAINED_PATH = DATA / "eval_v2_retained_v1.tsv"
MINED_PATH = DATA / "eval_v2_real_candidates.tsv"
CHALLENGE_PATH = DATA / "oshiwambo_eval_v2_challenge_seeds.md"
FORMAL_PATH = DATA / "oshiwambo_eval_v2_formal_seeds.md"
DEFAULT_OUT = DATA / "oshiwambo_eval_v2.tsv"

# Final v2 TSV column order.
V2_COLUMNS = [
    "id",
    "length_bucket",
    "domain",
    "phenomenon_tags",
    "provenance",
    "english",
    "claude_oshindonga",
    "gemma_oshindonga",
    "claude_oshikwanyama",
    "gemma_oshikwanyama",
    "elizabeth_oshindonga",
    "elizabeth_oshindonga_notes",
    "elizabeth_oshikwanyama",
    "elizabeth_oshikwanyama_notes",
    "claude_odg_score_1to5",
    "gemma_odg_score_1to5",
    "claude_okw_score_1to5",
    "gemma_okw_score_1to5",
    "in_blind_split",
]


def length_bucket(text: str) -> str:
    wc = len(text.split())
    if wc <= 6:
        return "S"
    if wc <= 18:
        return "M"
    return "L"


# ── Source parsers ───────────────────────────────────────────────


# v1 category → v2 domain. Most v1 categories represent the TOPIC, not the
# register — and the register on WhatsApp is overwhelmingly conversational.
# So cv_jobs / education / health / tech etc. all become "chat" (informal
# user register asking about those topics). Only gov_legal is genuinely
# formal-register.
V1_CATEGORY_TO_DOMAIN = {
    "greeting": "chat", "rapport": "chat", "closing": "chat",
    "ack": "chat", "refusal": "chat",
    "cv_jobs": "chat", "education": "chat", "health": "chat",
    "money": "chat", "tech": "chat",
    "gov_legal": "formal",
    "vocab": "challenge",
    "family": "community",
    "religion": "religious",
}


def load_retained_v1(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    with path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            en = (row.get("english") or "").strip()
            if not en:
                continue
            v1_cat = (row.get("v1_category") or "").strip()
            domain = V1_CATEGORY_TO_DOMAIN.get(v1_cat, "chat")
            items.append({
                "english": en,
                "domain": domain,
                "phenomenon_tags": [],
                "provenance": "v1_retained",
            })
    return items


def load_real_candidates(path: Path, include_unreviewed: bool) -> list[dict]:
    if not path.exists():
        return []
    items = []
    skipped_unreviewed = 0
    with path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            en = (row.get("english") or "").strip()
            if not en:
                continue
            approved = (row.get("sebastian_approved") or "").strip().lower()
            if not include_unreviewed and approved not in ("y", "yes", "1", "true"):
                skipped_unreviewed += 1
                continue
            items.append({
                "english": en,
                "domain": row.get("domain", "chat"),
                "phenomenon_tags": [],
                "provenance": "real_mined",
            })
    if skipped_unreviewed:
        print(
            f"  (skipped {skipped_unreviewed} unreviewed real candidates; "
            f"pass --include-unreviewed to preview with them)",
            file=sys.stderr,
        )
    return items


# Markdown source-file format used by challenge + formal seeds:
#   ```
#   EN text | tag1, tag2 | word_count
#   ```
# inside fenced code blocks. Parse those.

SEED_LINE = re.compile(
    r"^(?P<en>.+?)\s*\|\s*(?P<tags>[^|]+?)\s*\|\s*(?P<wc>\d+)\s*$"
)


def load_seeds_markdown(path: Path, default_provenance: str) -> list[dict]:
    if not path.exists():
        return []
    items: list[dict] = []
    in_block = False
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            in_block = not in_block
            continue
        if not in_block or not line.strip():
            continue
        m = SEED_LINE.match(line.strip())
        if not m:
            continue
        en = m.group("en").strip()
        tags = [t.strip() for t in m.group("tags").split(",") if t.strip()]
        # First non-domain tag stays a phenomenon tag; any tag that
        # names a domain (chat/formal/religious/community) goes into
        # the domain field.
        DOMAIN_NAMES = {"chat", "formal", "religious", "community"}
        phenomenon_tags = [t for t in tags if t not in DOMAIN_NAMES]
        # Default domain for challenge items is "challenge"; formal
        # items get "formal". An inline domain tag overrides.
        explicit_domain = next(
            (t for t in tags if t in DOMAIN_NAMES), None,
        )
        if explicit_domain:
            domain = explicit_domain
        elif default_provenance == "formal_drafted":
            domain = "formal"
        else:
            domain = "challenge"
        items.append({
            "english": en,
            "domain": domain,
            "phenomenon_tags": phenomenon_tags,
            "provenance": default_provenance,
        })
    return items


# ── Validation ───────────────────────────────────────────────────


# Rough PII heuristic — long digit sequences that survived scrubbing.
# 7+ digits in a row usually means a phone number or ID slipped through.
DIGIT_LEAK = re.compile(r"\d{7,}")


def validate(items: list[dict]) -> list[str]:
    failures: list[str] = []
    # Duplicate EN strings (case-insensitive, whitespace-normalised)
    norm_counts = Counter(
        re.sub(r"\s+", " ", i["english"].lower().strip()) for i in items
    )
    dupes = [k for k, v in norm_counts.items() if v > 1]
    if dupes:
        failures.append(f"{len(dupes)} duplicate EN strings (showing first 3): "
                        f"{dupes[:3]}")
    # Length distribution within ±8% of 20/50/30 target.
    # (Tolerance is generous because v1's short-heavy legacy can't be
    # fully neutralised without throwing away usable retained items.)
    lb_counts = Counter(i["length_bucket"] for i in items)
    total = max(len(items), 1)
    pct = {b: lb_counts[b] / total * 100 for b in ("S", "M", "L")}
    target = {"S": 20, "M": 50, "L": 30}
    for b in target:
        delta = abs(pct[b] - target[b])
        if delta > 8:
            failures.append(
                f"length bucket {b} is {pct[b]:.1f}% — outside ±8% of {target[b]}%"
            )
    # Domain distribution — relaxed from the original 60/15/10/10/5 because
    # the phenomenon-coverage floor of ≥10 per tag forces ≥110 challenge
    # items, and we don't have enough community/religious items in real
    # production data to hit 10% each. These are the realistic targets:
    dm_counts = Counter(i["domain"] for i in items)
    dpct = {d: dm_counts.get(d, 0) / total * 100 for d in
            ("chat", "formal", "religious", "community", "challenge")}
    target_dm = {"chat": 50, "formal": 15, "religious": 3,
                 "community": 7, "challenge": 25}
    for d, want in target_dm.items():
        if abs(dpct[d] - want) > 10:
            failures.append(
                f"domain {d} is {dpct[d]:.1f}% — outside ±10% of {want}%"
            )
    # Phenomenon tags ≥10 each
    tag_counts: Counter = Counter()
    for i in items:
        for t in i["phenomenon_tags"]:
            tag_counts[t] += 1
    for tag, n in tag_counts.items():
        if n < 10:
            failures.append(f"phenomenon tag '{tag}' only has {n} items (need ≥10)")
    # Digit leak
    leakers = [i["english"] for i in items if DIGIT_LEAK.search(i["english"])]
    if leakers:
        # Allow some — N$ amounts, NIDs in deliberate items. Just warn.
        print(
            f"  (info) {len(leakers)} items contain long digit sequences "
            f"(may be intentional N$/ID; verify): e.g. {leakers[0][:80]!r}",
            file=sys.stderr,
        )
    return failures


# ── Main ──────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--blind-split-pct", type=float, default=20.0,
                    help="Percentage of items to mark in_blind_split=true.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-unreviewed", action="store_true",
                    help="Include unreviewed real candidates (for preview only)")
    ap.add_argument("--target", type=int, default=400,
                    help="Soft target — script reports overage/underage but "
                         "does NOT auto-trim. Trim by adjusting source files.")
    args = ap.parse_args(argv)

    print("loading sources …", file=sys.stderr)
    retained = load_retained_v1(RETAINED_PATH)
    print(f"  retained-v1:    {len(retained):>3}", file=sys.stderr)
    mined = load_real_candidates(MINED_PATH, args.include_unreviewed)
    print(f"  real-mined:     {len(mined):>3}", file=sys.stderr)
    challenge = load_seeds_markdown(CHALLENGE_PATH, "crafted")
    print(f"  challenge:      {len(challenge):>3}", file=sys.stderr)
    formal = load_seeds_markdown(FORMAL_PATH, "formal_drafted")
    print(f"  formal:         {len(formal):>3}", file=sys.stderr)

    # Compute length_bucket up-front so trim logic can use it
    for src in (retained, mined, challenge, formal):
        for i in src:
            i["length_bucket"] = length_bucket(i["english"])

    # Per-provenance targets. The original 80 for crafted broke
    # phenomenon coverage (≥10 per tag); bumped to 110 to preserve
    # coverage on the 11 tags. Other targets stay close to plan.
    PROVENANCE_TARGETS = {
        "v1_retained": 150,
        "real_mined": 145,
        "crafted": 110,
        "formal_drafted": 20,
    }

    def trim(items: list[dict], target: int, prov: str) -> list[dict]:
        if len(items) <= target:
            return items
        # crafted: phenomenon-coverage constraint takes precedence.
        # We use reverse-greedy: start with ALL items, repeatedly drop
        # the LOWEST-quality item whose removal does NOT drop any
        # phenomenon tag below 10. Stop when len == target or no
        # more safe removals are possible.
        if prov == "crafted":
            FLOOR = 10
            # Count tag occurrences
            from collections import Counter as _Counter
            tag_counts: _Counter = _Counter()
            for it in items:
                for t in it["phenomenon_tags"]:
                    tag_counts[t] += 1
            # Rank items lowest-quality first (eligible for removal)
            def quality(it):
                wc = len(it["english"].split())
                return (
                    0 if 12 <= wc <= 28 else 1,
                    int("\n" in it["english"]),
                    -wc,
                )
            ranked_desc = sorted(items, key=quality, reverse=True)
            kept = list(items)
            for victim in ranked_desc:
                if len(kept) <= target:
                    break
                # Would removing this drop any tag below FLOOR?
                safe = all(
                    tag_counts[t] - 1 >= FLOOR
                    for t in victim["phenomenon_tags"]
                )
                if safe:
                    kept.remove(victim)
                    for t in victim["phenomenon_tags"]:
                        tag_counts[t] -= 1
            return kept
        else:
            # v1_retained: prefer longer items (v1 is dominated by S)
            ranked = sorted(items, key=lambda x: (
                {"L": 0, "M": 1, "S": 2}[x["length_bucket"]],
                -len(x["english"].split()),
            ))
            return ranked[:target]

    retained = trim(retained, PROVENANCE_TARGETS["v1_retained"], "v1_retained")
    mined = trim(mined, PROVENANCE_TARGETS["real_mined"], "real_mined")
    challenge = trim(challenge, PROVENANCE_TARGETS["crafted"], "crafted")
    formal = trim(formal, PROVENANCE_TARGETS["formal_drafted"], "formal_drafted")

    print(f"\nafter trimming to targets:", file=sys.stderr)
    print(f"  retained-v1:    {len(retained):>3}", file=sys.stderr)
    print(f"  real-mined:     {len(mined):>3}", file=sys.stderr)
    print(f"  challenge:      {len(challenge):>3}", file=sys.stderr)
    print(f"  formal:         {len(formal):>3}", file=sys.stderr)

    all_items = retained + mined + challenge + formal

    # Assign deterministic IDs (provenance-grouped, stable order)
    PROVENANCE_ORDER = ["v1_retained", "real_mined", "crafted", "formal_drafted"]
    all_items.sort(key=lambda i: (
        PROVENANCE_ORDER.index(i["provenance"]),
    ))
    for n, i in enumerate(all_items, start=1):
        i["id"] = n

    # Deterministic 20% blind split
    rng = random.Random(args.seed)
    n_blind = max(1, int(len(all_items) * args.blind_split_pct / 100))
    blind_ids = set(rng.sample([i["id"] for i in all_items], n_blind))
    for i in all_items:
        i["in_blind_split"] = "true" if i["id"] in blind_ids else "false"

    # Report distribution
    print(f"\nv2 totals:  {len(all_items)} items "
          f"(target {args.target}, delta {len(all_items)-args.target:+d})",
          file=sys.stderr)
    by_prov = Counter(i["provenance"] for i in all_items)
    print(f"by provenance:", file=sys.stderr)
    for p in PROVENANCE_ORDER:
        print(f"  {p:18s} {by_prov[p]:>3}", file=sys.stderr)

    by_len = Counter(i["length_bucket"] for i in all_items)
    print(f"by length_bucket:", file=sys.stderr)
    for b in ("S", "M", "L"):
        pct = by_len[b] / len(all_items) * 100 if all_items else 0
        print(f"  {b}  {by_len[b]:>3}  ({pct:.0f}%)", file=sys.stderr)

    by_dom = Counter(i["domain"] for i in all_items)
    print(f"by domain:", file=sys.stderr)
    for d in ("chat", "formal", "religious", "community", "challenge"):
        pct = by_dom.get(d, 0) / len(all_items) * 100 if all_items else 0
        print(f"  {d:12s} {by_dom.get(d, 0):>3}  ({pct:.0f}%)", file=sys.stderr)

    tag_counts: Counter = Counter()
    for i in all_items:
        for t in i["phenomenon_tags"]:
            tag_counts[t] += 1
    if tag_counts:
        print(f"phenomenon tag coverage:", file=sys.stderr)
        for t, n in sorted(tag_counts.items()):
            marker = " ✓" if n >= 10 else " ⚠ (<10)"
            print(f"  {t:25s} {n:>3}{marker}", file=sys.stderr)

    blind_n = sum(1 for i in all_items if i["in_blind_split"] == "true")
    print(f"blind split: {blind_n} items ({blind_n/len(all_items)*100:.0f}%)",
          file=sys.stderr)

    # Validate
    failures = validate(all_items)
    if failures:
        print(f"\n=== VALIDATION FAILURES ({len(failures)}) ===", file=sys.stderr)
        for f in failures:
            print(f"  ✗ {f}", file=sys.stderr)
    else:
        print(f"\nall validations passed ✓", file=sys.stderr)

    # Write TSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=V2_COLUMNS, delimiter="\t")
        writer.writeheader()
        for i in all_items:
            row = {col: "" for col in V2_COLUMNS}
            row.update({
                "id": i["id"],
                "length_bucket": i["length_bucket"],
                "domain": i["domain"],
                "phenomenon_tags": ";".join(i["phenomenon_tags"]),
                "provenance": i["provenance"],
                "english": i["english"],
                "in_blind_split": i["in_blind_split"],
            })
            writer.writerow(row)
    print(f"\nwrote {len(all_items)} rows to {out_path}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
