#!/usr/bin/env python3
"""Curate the 286 mined real WhatsApp candidates → ~150 approved.

Sebastian delegated this thoroughly to me. I do two passes:

  PASS 1 — automated hard-reject regexes catch obvious PII patterns
  (full names, phone numbers, contact-card artifacts, specific small
  businesses/villages/churches/schools, garbled voice transcriptions,
  truncation, abuse, non-English-dominant items, list-only items).

  PASS 2 — explicit per-ID manual decisions for items where regex
  alone can't judge. Each ID I want to reject is listed below with
  the actual reason (so the curation is auditable).

What survives both passes goes into a quality ranking that selects to
the target distribution (~90 chat / 35 formal / 15 community / 5
religious).

Modifies data/eval_v2_real_candidates.tsv in place, filling
`sebastian_approved` with "y" for keeps.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

TSV = Path("/Users/sebkuepers/dev/Ongiini/data/eval_v2_real_candidates.tsv")


# ── PASS 1: hard-reject regex patterns ─────────────────────────────


HARD_REJECT_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Personal-name signals
    (re.compile(r"\bMy name is [A-Z]\w+\s+[A-Z]\w+", re.I), "name: My name is X Y"),
    (re.compile(r"\bI(?:’|')?m\s+[A-Z]\w+\s+[A-Z]\w+", re.I), "name: I'm X Y"),
    (re.compile(r"\bI am [A-Z]\w+\s+[A-Z]\w+\s+from", re.I), "name: I am X Y from"),
    (re.compile(r"\bSurname:\s*\w+", re.I), "CV with surname"),
    (re.compile(r"\bD\.O\.B\b", re.I), "CV with DOB"),
    (re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b"), "three-name sequence"),
    # Phone numbers
    (re.compile(r"\b0[678]\s*\d{2,3}\s*\d{3,4}\s*\d{3,4}\b"), "phone number"),
    (re.compile(r"\+\s*264"), "phone number (+264)"),
    # Redacted-tag artifacts (contact-card detritus)
    (re.compile(r"\[REDACTED:email\]"), "contact card"),
    (re.compile(r"\[REDACTED:id\]"), "contact card"),
    # Specific small businesses / private schools / NGOs
    (re.compile(r"\b\w+'?s\s+(crochet|electrician|investment\s+cc|Eatery|"
                r"Kindergarten|kindergarten|Daycare|private school|"
                r"private College|combined school)\b"),
     "specific small business / school"),
    (re.compile(r"\b\w+\s+(investment cc|Pty Ltd|Trading)\b", re.I), "company"),
    (re.compile(r"\bJungle vibe investment\b", re.I), "company"),
    (re.compile(r"\b(Onandjokwe|Oshagwena|HTS Windhoek technical|"
                r"Nanghonda combined|Mercy Joy|Sunshine private|"
                r"Oshikunde combined|Onandjokwe hospital)\b", re.I),
     "specific small institution"),
    # Specific villages / localities
    (re.compile(r"\b[A-Z][a-z]+ Village\b"), "specific village name"),
    (re.compile(r"\b(Uukwanandjenga|Tsandi Constituency|Atusheni|"
                r"Ongha-Etenda|Endola Circuit|Ruacana)\b", re.I),
     "specific village/locality"),
    # Specific church figures / churches
    (re.compile(r"\bArch Bishop \w", re.I), "specific church figure"),
    (re.compile(r"\b(united church|revival church|Ecclesia global)\b", re.I),
     "specific church"),
    # Abuse / profanity
    (re.compile(r"\bfuck you\b", re.I), "abusive"),
    # Garbled voice
    (re.compile(r"my the number my the number"), "garbled voice"),
    (re.compile(r"biggie biggie"), "garbled voice"),
    (re.compile(r"^\[voice note\] Mechanical feet"), "garbled voice"),
    (re.compile(r"\bMeteor AI\b"), "abusive voice"),
    # Specific named research / mine / circuit
    (re.compile(r"\bnguni breeds in namibia\b", re.I), "very specific research"),
    (re.compile(r"\bENDOLA CIRCUIT\b", re.I), "specific circuit"),
    (re.compile(r"\bSelma JNO Ndafoluma\b", re.I), "specific person"),
    (re.compile(r"\bNdapewa Mani\b", re.I), "specific person"),
    (re.compile(r"\bIyaloo shili\b"), "Oshiwambo content"),
    (re.compile(r"\bP\.D\.K\b"), "specific street"),
    (re.compile(r"\bhockland park\b", re.I), "specific neighbourhood"),
    (re.compile(r"\bBlack street\b", re.I), "specific street"),
    (re.compile(r"\bKQR Namibia Navachab\b", re.I), "specific mine"),
    # Truncation
    (re.compile(r"\.{3,}\s*$"), "truncated (ellipsis)"),
    # Non-English-dominant items
    (re.compile(r"\b(omugandjimayele|pangundu|nopaedhilaadhilo|"
                r"Shama aike wa penge|okwa landa|Omusamane okwa)\b", re.I),
     "Oshiwambo-dominant"),
]


# ── PASS 2: explicit per-ID manual rejections ──────────────────────
#
# Items where regex CAN'T tell — but reading the actual content I
# decided to reject. Each entry is (id, reason) so the call is auditable.
# Other IDs get the "ok" treatment from Pass 1 and progress to ranking.

MANUAL_REJECT: dict[int, str] = {
    # Multi-line list/CV content (1-newline survivors that are still listy
    # or have embedded contact info I can read but regex can't classify)
    3:   "multi-line + emoji-only ack",
    18:  "CV with specific employer (state veterinary in otjiwarongo)",
    25:  "specific employer (Volpes The Home of Linen)",
    26:  "multi-line + asks about specific organization session",
    76:  "multi-line CV scaffolding",
    89:  "multi-line meta question",
    124: "specific employer (solitaire country lodge)",
    142: "feels like distress signal, deserves human reply not eval item",
    145: "multi-line presentation prep",
    150: "multi-line birthday-toast triviality",
    163: "list-style (1. 2. 3.)",
    174: "multi-line meta question",
    181: "multi-line list of place names to translate",
    187: "multi-line CV scaffold",
    195: "multi-line + location info",
    196: "multi-line + specific course/Nust combo",
    197: "multi-line + specific college (Sunshine)",
    201: "specific NGO (tara nawa) tied to user role",
    202: "specific employer (Cosdef gobabis)",
    207: "multi-line list",
    219: "specific institute (i care health training)",
    226: "list-style",
    233: "multi-line + relates to specific personal call",
    245: "specific training org (Gez SmE) tied to user CV",
    252: "multi-line trivial",
    265: "multi-line CV list",
    272: "list-style with mother-tongue Rukwangali",
    # Intimate personal content that could embarrass/identify even
    # though no names — eval set must not feel exploitative
    6:   "intimate cheating story",
    14:  "intimate medical (C-section + ethnic identifier)",
    17:  "intimate fertility (couple ages + reproductive status)",
    50:  "intimate breakup ruminations",
    52:  "personal relationship venting",
    94:  "contraceptive + pregnancy uncertainty",
    138: "sexual content involving a minor",
    # Garbled voice notes that survived regex
    16:  "garbled voice (Ulanium Lawsing)",
    87:  "garbled voice (Vierge Vierge Var)",
    132: "voice note: cryptic, unclear meaning",
    # Items that read fine but reference very specific things
    44:  "user admits to theft (real but might be flagged sensitive)",
    77:  "voice note asking about Ongiini origins (meta noise)",
    78:  "meta/political about Ongiini, niche",
    149: "voice note: unclear ask",
    152: "voice note: short ambiguous question",
    161: "off-channel canned reply leaked through",
    36:  "meta about AI development context",
    37:  "asking about plagiarism detection",
    # Items with very specific course codes / institutional minutiae
    162: "specific course codes (BMI511S1)",
    # Items that are essentially Oshindonga-grammar mini-lessons
    # (bot-style, not user-style) but came through user-role
    70:  "bot-style grammar explanation (not user query)",
    96:  "Oshindonga grammar critique (chunk of bot/user back-and-forth)",
    194: "Oshindonga embedded with quote-marks artifact",
    # Phone/email-card leaks regex missed
    97:  "implicit business advert (chips/Russian/fat cakes)",
}


# ── PASS 1 evaluation ─────────────────────────────────────────────


def is_list_only(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    listy = sum(
        1 for ln in lines
        if re.match(r"^\s*(?:\d+[\.\)]\s*|\-\s*|\*\s*|\(\d+\))", ln)
    )
    return listy >= 2 and listy / len(lines) >= 0.6


def pass1_evaluate(row: dict) -> tuple[bool, str]:
    en = row["english"]
    for pat, reason in HARD_REJECT_PATTERNS:
        if pat.search(en):
            return False, f"regex: {reason}"
    if is_list_only(en):
        return False, "list-only"
    # 2+ newlines = almost always list/CV
    if en.count("\n") >= 2:
        return False, "multi-line (≥2 newlines)"
    return True, "ok"


# ── Selection rank ────────────────────────────────────────────────


def rank(item: dict) -> tuple:
    wc = int(item["word_count"])
    # Sweet spot: 12-28 words → priority 0; 7-35 → 1; else 2
    if 12 <= wc <= 28:
        wc_score = 0
    elif 7 <= wc <= 35:
        wc_score = 1
    else:
        wc_score = 2
    has_newline = "\n" in item["english"]
    return (wc_score, int(has_newline), -wc)


# ── Main ──────────────────────────────────────────────────────────


def main() -> int:
    with TSV.open() as f:
        cols = f.readline().rstrip("\n").split("\t")
    with TSV.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    print(f"loaded {len(rows)} candidates from {TSV.name}", file=sys.stderr)

    reasons = Counter()
    survivors_after_pass1: list[dict] = []
    for row in rows:
        keep, reason = pass1_evaluate(row)
        reasons[reason] += 1
        if keep:
            survivors_after_pass1.append(row)

    print("\n=== PASS 1 (regex + structure) ===", file=sys.stderr)
    for r, n in reasons.most_common():
        print(f"  {n:>3}  {r}", file=sys.stderr)
    print(f"\n→ Pass 1 survivors: {len(survivors_after_pass1)}", file=sys.stderr)

    # PASS 2: manual per-ID rejects
    manual_kept = [
        r for r in survivors_after_pass1
        if int(r["id"]) not in MANUAL_REJECT
    ]
    print(f"\n=== PASS 2 (per-ID manual review) ===", file=sys.stderr)
    print(f"  manually rejected: {len(MANUAL_REJECT)} ids", file=sys.stderr)
    print(f"→ Pass 2 survivors: {len(manual_kept)}", file=sys.stderr)
    by_domain = Counter(r["domain"] for r in manual_kept)
    print(f"  by domain: {dict(by_domain)}", file=sys.stderr)

    # Selection to target distribution
    targets = {"chat": 90, "formal": 35, "community": 15, "religious": 5}
    selected: list[dict] = []
    for domain, want in targets.items():
        pool = [r for r in manual_kept if r["domain"] == domain]
        pool.sort(key=rank)
        chosen = pool[:want]
        if len(chosen) < want:
            print(
                f"  WARN: domain '{domain}' only has {len(chosen)} kept "
                f"(wanted {want})",
                file=sys.stderr,
            )
        selected.extend(chosen)

    selected_ids = {r["id"] for r in selected}
    print(f"\n=== FINAL SELECTION: {len(selected)} items ===", file=sys.stderr)
    sel_by_dom = Counter(r["domain"] for r in selected)
    for d in ("chat", "formal", "community", "religious"):
        print(
            f"  {d:12s}  {sel_by_dom.get(d, 0):>3}  (target {targets.get(d, 0)})",
            file=sys.stderr,
        )
    sel_wc = [int(r["word_count"]) for r in selected]
    if sel_wc:
        from statistics import median, mean
        print(
            f"  word_count median={median(sel_wc):.0f} mean={mean(sel_wc):.1f} "
            f"min={min(sel_wc)} max={max(sel_wc)}",
            file=sys.stderr,
        )

    # Write back
    for row in rows:
        row["sebastian_approved"] = "y" if row["id"] in selected_ids else ""

    with TSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nupdated {TSV} (selected={len(selected)} of {len(rows)})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
