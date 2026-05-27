#!/usr/bin/env python3
"""Mine real WhatsApp user messages as eval-set candidates.

Walks /data/*.json (per-user memories) on Spark, extracts user-role
messages, applies PII scrubbing, filters for medium-long length
(5-40 words), stratifies by topic, deduplicates, caps per-user
contribution to keep variety, outputs ~300 candidates as TSV.

The output is for SEBASTIAN's manual review — he fills the
`sebastian_approved` column with y/n. Only y rows feed into v2.

This is the unblocker for the "real WhatsApp samples" bucket
(~150 items) in the v2 eval set composition.

Run inside the webhook container:

    docker exec -i ongiini-webhook python3 /data/mine_eval_candidates.py
        --target 300
        --out /data/eval_v2_real_candidates.tsv

(The script is copied to /data because the container has read-only
rootfs and /data is the writable bind-mount.)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# Make ongiini.pii importable when run inside the container (where
# /app holds the package) and when run standalone from the worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if "/app" not in sys.path and Path("/app/ongiini").is_dir():
    sys.path.insert(0, "/app")
try:
    from ongiini.pii import sanitize as pii_sanitize
except ImportError:
    def pii_sanitize(t: str) -> str:
        return t


# ── Filters ───────────────────────────────────────────────────────

MIN_WORDS = 5
MAX_WORDS = 40

# Marker strings from canned welcome / system replies that occasionally
# leak into user-role buckets in older memory files. Skip these
# silently — they're noise.
CANNED_FRAGMENT_MARKERS = (
    "I'm an AI helper",
    "Happy you found us",
    "Hi! What can I help",
    "Ek is 'n KI-helper",
)


# English heuristic — at least N common English function words must appear.
# Calibrated to be permissive: real WhatsApp English with abbreviations
# and slang still passes. Code-switched messages where Eng is dominant
# also pass; messages dominated by Oshiwambo / Afrikaans get rejected
# below.
ENGLISH_FUNC = re.compile(
    r"\b(the|and|is|are|you|your|my|me|i|to|for|with|have|how|what|"
    r"when|where|can|could|would|please|thank|help|need|want|am|on|"
    r"in|at|of|that|this|do|did|will|now|today|tomorrow|yesterday|"
    r"about|from|but|not|no|yes|so|because|if|or|some|any|other|"
    r"there|here|just|also|then|than|all|good|like|been|get|got)\b",
    re.IGNORECASE,
)

OSHIWAMBO_MARKERS = re.compile(
    r"\b(tangi|kala po|ongiini|ohandi|kuume|kandi|kwafe|kalunga|"
    r"ombili|eewa|owa lala|onawa|nawa|tate|kuku|mukainwa|mwanyume|"
    r"omu na|ondi|ohau|otami|otandi|aame|onga|aanegumbo)\b",
    re.IGNORECASE,
)

AFRIKAANS_MARKERS = re.compile(
    r"\b(jy|jou|ek|sien|nie|hierdie|asseblief|baie|dankie|moet|kan|"
    r"vir|wanneer|sou|wees|kry|gee|maak|ons|hulle|julle|met|hoe|"
    r"die|sodat|omdat|natuurlik|natuurlike|hoekom|waaroor)\b",
    re.IGNORECASE,
)


# Domain stratification — first matching pattern wins. Order matters:
# religious + community are checked before formal because a sentence
# can have multiple signals.
DOMAIN_KEYWORDS = {
    "religious": re.compile(
        r"\b(God|Kalunga|pray|prayer|amen|blessing|blessed|church|faith|"
        r"scripture|bible|holy|spirit|worship|sermon|pastor)\b",
        re.IGNORECASE,
    ),
    "community": re.compile(
        r"\b(family|mother|father|sister|brother|son|daughter|wife|husband|"
        r"village|neighbour|neighbor|community|grandparent|grandmother|"
        r"grandfather|child|children|relatives|cousin|tribe|clan)\b",
        re.IGNORECASE,
    ),
    "formal": re.compile(
        r"\b(ministry|application|register|registration|certificate|"
        r"document|form|office|government|tax|VAT|BIPA|MOHSS|MEFT|MURD|"
        r"affidavit|approval|approved|enquire|kindly|sincerely|"
        r"hospital|clinic|polic[ey]|school|teacher|university|college|"
        r"scholarship|invoice|payment|bank|account|loan)\b",
        re.IGNORECASE,
    ),
}


def word_count(text: str) -> int:
    return len(text.split())


def is_likely_english(text: str) -> bool:
    return len(ENGLISH_FUNC.findall(text)) >= 2


def is_dominantly_other_language(text: str) -> bool:
    """True if the message has more Oshiwambo or Afrikaans markers
    than English ones (i.e. would be a bad EN source for translation)."""
    en = len(ENGLISH_FUNC.findall(text))
    osh = len(OSHIWAMBO_MARKERS.findall(text))
    afr = len(AFRIKAANS_MARKERS.findall(text))
    return (osh > en) or (afr > en)


def stratify(text: str) -> str:
    """Return one of: religious | community | formal | chat.
    "chat" is the default for everyday conversational messages."""
    for domain, pat in DOMAIN_KEYWORDS.items():
        if pat.search(text):
            return domain
    return "chat"


def looks_like_image_caption(text: str) -> bool:
    s = text.lstrip()
    return s.startswith("[image:") or s.startswith("[image ")


def normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


# ── Main ──────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Mine real WhatsApp messages as eval-set candidates",
    )
    ap.add_argument(
        "--data-dir", default="/data",
        help="Per-user JSON dir (default /data inside container).",
    )
    ap.add_argument(
        "--out", default="/data/eval_v2_real_candidates.tsv",
        help="Output TSV path.",
    )
    ap.add_argument(
        "--target", type=int, default=300,
        help="Target candidate count. Default 300 (Sebastian picks ~150).",
    )
    ap.add_argument(
        "--per-user-cap", type=int, default=3,
        help="Max candidates from any single user — keeps variety.",
    )
    ap.add_argument(
        "--min-words", type=int, default=MIN_WORDS,
        help="Min word count for a candidate. Default 5.",
    )
    ap.add_argument(
        "--max-words", type=int, default=MAX_WORDS,
        help="Max word count for a candidate. Default 40.",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducible selection.",
    )
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    files = sorted(data_dir.glob("264*.json"))
    print(f"Scanning {len(files)} per-user JSON files…", file=sys.stderr)

    seen_norm: set[str] = set()
    per_user_count: dict[str, int] = defaultdict(int)
    by_domain: dict[str, list[dict]] = defaultdict(list)

    counters = {
        "scanned": 0, "user_turns": 0, "kept": 0,
        "skip_short": 0, "skip_long": 0, "skip_not_en": 0,
        "skip_other_lang": 0, "skip_image": 0, "skip_canned": 0,
        "skip_dupe": 0, "skip_per_user_cap": 0, "skip_pii_dense": 0,
    }

    for f in files:
        counters["scanned"] += 1
        try:
            turns = json.loads(f.read_text())
        except Exception:                              # noqa: BLE001
            continue
        if not isinstance(turns, list):
            continue
        # Hash the msisdn so we don't carry it in the output. Salt-free
        # because we just need a per-user collision-free tag for the
        # per-user cap; the hash is opaque to the consumer.
        user_hash = hashlib.sha256(f.stem.encode()).hexdigest()[:12]
        for i, t in enumerate(turns):
            if not isinstance(t, dict):
                continue
            if t.get("role") != "user":
                continue
            counters["user_turns"] += 1
            text = (t.get("content") or "").strip()
            if not text:
                continue
            if looks_like_image_caption(text):
                counters["skip_image"] += 1
                continue
            if any(m in text for m in CANNED_FRAGMENT_MARKERS):
                counters["skip_canned"] += 1
                continue
            wc = word_count(text)
            if wc < args.min_words:
                counters["skip_short"] += 1
                continue
            if wc > args.max_words:
                counters["skip_long"] += 1
                continue
            if not is_likely_english(text):
                counters["skip_not_en"] += 1
                continue
            if is_dominantly_other_language(text):
                counters["skip_other_lang"] += 1
                continue

            scrubbed = pii_sanitize(text)
            # If PII scrubber redacted 3+ things, the item is too
            # personally specific even after scrub. Skip.
            if scrubbed.count("[REDACTED:") >= 3:
                counters["skip_pii_dense"] += 1
                continue

            norm = normalize_for_dedup(scrubbed)
            if norm in seen_norm:
                counters["skip_dupe"] += 1
                continue
            seen_norm.add(norm)

            if per_user_count[user_hash] >= args.per_user_cap:
                counters["skip_per_user_cap"] += 1
                continue
            per_user_count[user_hash] += 1

            domain = stratify(scrubbed)
            by_domain[domain].append({
                "english": scrubbed,
                "domain": domain,
                "word_count": wc,
                "src_user_hash": user_hash,
                "src_msg_idx": i,
            })
            counters["kept"] += 1

    # ── Report intake stats ──────────────────────────────────────
    print("\n=== INTAKE FUNNEL ===", file=sys.stderr)
    for k, v in counters.items():
        print(f"  {k:20s}  {v}", file=sys.stderr)

    print(f"\n=== CANDIDATE POOL BY DOMAIN ===", file=sys.stderr)
    for d, items in sorted(by_domain.items()):
        print(f"  {d:12s}  {len(items)}", file=sys.stderr)

    # ── Pick to target distribution ──────────────────────────────
    # Mirrors v2 plan's "real WhatsApp" share of the final 400-item set:
    # chat dominates, formal next, religious + community equal small.
    # The crafted challenge + formal-drafted buckets fill the rest in v2.
    target_pct = {
        "chat":      0.65,
        "formal":    0.20,
        "religious": 0.075,
        "community": 0.075,
    }
    targets = {d: int(args.target * pct) for d, pct in target_pct.items()}

    rng = random.Random(args.seed)
    selected: list[dict] = []
    for domain, want in targets.items():
        pool = list(by_domain.get(domain, []))
        rng.shuffle(pool)
        # Bias toward medium-long: stable-sort by word_count desc, cap
        # at 25 so we don't over-prioritise 40-word monologues.
        pool.sort(key=lambda x: -min(x["word_count"], 25))
        selected.extend(pool[:want])

    print(f"\n=== SELECTED ({len(selected)} of target {args.target}) ===", file=sys.stderr)
    for d in target_pct:
        n = sum(1 for c in selected if c["domain"] == d)
        print(f"  {d:12s}  {n:3d}  (target {targets[d]})", file=sys.stderr)

    # ── Write TSV ────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "id", "english", "domain", "word_count",
            "src_user_hash", "src_msg_idx", "sebastian_approved",
        ])
        for i, c in enumerate(selected, start=1):
            w.writerow([
                i,
                c["english"],
                c["domain"],
                c["word_count"],
                c["src_user_hash"],
                c["src_msg_idx"],
                "",
            ])
    print(f"\nwrote {len(selected)} rows to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
