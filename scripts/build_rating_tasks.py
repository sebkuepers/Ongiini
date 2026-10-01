#!/usr/bin/env python3
"""Build the item list for rating round 1 ("reference check", plan v3).

Development split only; the blind split stays unseen by raters. Per
dialect (defaults ≈ a 150-judgment budget):

  random_ref     45  references drawn at random (outside the flagged top) —
                     the unbiased estimate of reference quality
  flagged_ref    25  references back-translation flagged "major", most
                     severe first (negation > number/name/missing > rest),
                     2 raters each; the other flagged ones stay
                     back-translation-only for now
  claude_pair    30  Claude Opus 5 for 20 random + 10 flagged sentences;
                     ratings.py hands each to the rater who judged that
                     reference, at least PAIR_GAP tasks later
  gemma          10  Gemma 4 26B for random sentences, to raters who did
                     not see that reference
  control_error  12  a reference with a planted, English-checkable error:
                     a changed number, a swapped Namibian place name, or a
                     dropped final clause/sentence (expected: wrong)
  control_wrong   3  the reference of a different sentence (attention)
  practice        2  onboarding examples (number changed, clause dropped)
                     with feedback; never scored

Derailed model outputs (retry_derailed_baselines.is_derailed) are never
used. Output (private): data/private/rating_items_<round>.json

    python3 scripts/build_rating_tasks.py [--round r1]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retry_derailed_baselines import is_derailed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT / "data/private/oshiwambo_eval_v3.tsv"
BACK = ROOT / "data/private/qa/backtranslation.jsonl"
BASE = ROOT / "data/private/export/data/baselines"
DIALECTS = ("oshindonga", "oshikwanyama")
PLACES = ["Windhoek", "Oshakati", "Ondangwa", "Ongwediva", "Rundu", "Katima Mulilo", "Walvis Bay",
          "Swakopmund", "Otjiwarongo", "Tsumeb", "Grootfontein", "Keetmanshoop", "Gobabis",
          "Opuwo", "Outapi", "Eenhana", "Okahandja", "Rehoboth", "Mariental", "Lüderitz"]
SEVERITY = [(3, r"negat|\bnot\b|never|flipp|opposite"), (2, r"number|digit|date|time|amount|\d"),
            (2, r"\bname|place|town"), (2, r"missing|omit|dropp|absent|lost|left out"),
            (1, r"added|extra|wrong (person|subject)")]


def load_outputs(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    return {json.loads(l)["id"]: json.loads(l)["translation"].strip()
            for l in path.read_text().splitlines() if l}


def key(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:14]


def severity(issue: str) -> int:
    t = (issue or "").lower()
    return max([s for s, pat in SEVERITY if re.search(pat, t)], default=0)


# ── planted errors (checkable against the English without knowing the language)

def err_number(en: str, ref: str, rng: random.Random):
    nums = [n for n in re.findall(r"\d+", ref) if n in re.findall(r"\d+", en)]
    if not nums:
        return None
    n = rng.choice(nums)
    digits = list(n)
    i = rng.randrange(len(digits))
    digits[i] = rng.choice([d for d in "123456789" if d != digits[i]])
    return ref.replace(n, "".join(digits), 1), "number", "a number does not match the English."


def err_place(en: str, ref: str, rng: random.Random):
    here = [p for p in PLACES if p in ref and p in en]
    if not here:
        return None
    p = rng.choice(here)
    return (ref.replace(p, rng.choice([q for q in PLACES if q not in en])), "place",
            "a place name does not match the English.")


def err_clause(en: str, ref: str, rng: random.Random):
    split = lambda s: [x for x in re.split(r"(?<=[.!?])\s+", s.strip()) if x]
    sr, se = split(ref), split(en)
    if len(sr) >= 2 and len(se) >= 2:
        return " ".join(sr[:-1]), "clause", "the last sentence of the English is missing."
    if "," in ref and "," in en:
        head, tail = ref.rsplit(",", 1)
        if len(tail.split()) >= 3 and len(head.split()) >= 3:
            return head.strip() + ".", "clause", "the end of the sentence is missing."
    return None


def plant(en, ref, rng, kinds=("number", "place", "clause")):
    makers = {"number": err_number, "place": err_place, "clause": err_clause}
    for k in kinds:
        out = makers[k](en, ref, rng)
        if out and out[0].strip() and out[0].strip() != ref.strip():
            return out
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--translator", default="kaarina")
    ap.add_argument("--round", default="r1")
    ap.add_argument("--random", type=int, default=45)
    ap.add_argument("--flagged", type=int, default=25)
    ap.add_argument("--pairs-random", type=int, default=20)
    ap.add_argument("--pairs-flagged", type=int, default=10)
    ap.add_argument("--gemma", type=int, default=10)
    ap.add_argument("--controls", type=int, default=12)
    ap.add_argument("--attention", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    out = args.out or ROOT / f"data/private/rating_items_{args.round}.json"

    with TSV.open() as f:
        rows = {int(r["id"]): r for r in csv.DictReader(f, delimiter="\t")}
    back = {}
    for line in BACK.read_text().splitlines():
        o = json.loads(line)
        back[(o["id"], o["dialect"])] = o

    rng = random.Random(args.seed)
    items, summary = [], {}
    for d in DIALECTS:
        ref = {i: r[f"{args.translator}_{d}"].strip() for i, r in rows.items()
               if r["in_blind_split"] == "false" and r[f"{args.translator}_{d}"].strip()}
        en = {i: rows[i]["english"].strip() for i in ref}
        claude = {i: t for i, t in load_outputs(BASE / f"claude-opus-5_{d}.jsonl").items()
                  if i in ref and t and not is_derailed(en[i], t)}
        gemma = {i: t for i, t in load_outputs(BASE / f"gemma-4-26b-api_{d}.jsonl").items()
                 if i in ref and t and not is_derailed(en[i], t)}

        def add(i, grp, kind, text, source, **kw):
            items.append({"item_key": key(args.round, d, grp, i, source), "round": args.round,
                          "sentence_id": i, "dialect": d, "english": en[i], "text": text,
                          "grp": grp, "kind": kind, "source": source, **kw})

        ids = sorted(ref)
        flagged_all = [i for i in ids if back.get((i, d), {}).get("verdict") == "major"]
        rng.shuffle(flagged_all)
        flagged_all.sort(key=lambda i: -severity(back[(i, d)].get("issue", "")))
        flagged = flagged_all[: args.flagged]

        # practice + controls come from sentences that are NOT rated as references
        pool = [i for i in ids if i not in flagged]
        rng.shuffle(pool)
        practice = []
        for kind in ("number", "clause"):
            for i in pool:
                if i in {p for p, _ in practice}:
                    continue
                if (e := plant(en[i], ref[i], rng, (kind,))):
                    practice.append((i, e))
                    break
        used = {i for i, _ in practice}
        controls = []
        for i in pool:
            if i in used or len(controls) >= args.controls:
                continue
            if (e := plant(en[i], ref[i], rng)):
                controls.append((i, e))
                used.add(i)
        randoms = [i for i in pool if i not in used][: args.random]
        used |= set(randoms)

        for i, (text, how, why) in practice:
            add(i, "practice", "practice", text, f"{args.translator}+err:{how}", expected="wrong",
                explanation=f"This one is ✗ No: {why}")
        for i in randoms:
            add(i, "random_ref", "ref", ref[i], args.translator)
        for i in flagged:
            add(i, "flagged_ref", "ref", ref[i], args.translator, target=2,
                severity=severity(back[(i, d)].get("issue", "")))
        pair_ids = ([i for i in randoms if i in claude][: args.pairs_random]
                    + [i for i in flagged if i in claude][: args.pairs_flagged])
        for i in pair_ids:
            grp_of_ref = "flagged_ref" if i in flagged else "random_ref"
            add(i, "claude_pair", "model", claude[i], "claude-opus-5",
                pair_of=key(args.round, d, grp_of_ref, i, args.translator))
        gem_ids = [i for i in randoms if i in gemma and i not in pair_ids][: args.gemma]
        for i in gem_ids:
            add(i, "gemma", "model", gemma[i], "gemma-4-26b")
        for i, (text, how, _) in controls:
            add(i, "control_error", "control", text, f"{args.translator}+err:{how}", expected="wrong",
                target=99)
        rest = [i for i in ids if i not in used and i not in flagged]
        for i in rest[: args.attention]:
            j = rng.choice([k for k in ids if k != i])
            add(i, "control_wrong", "control", ref[j], f"{args.translator}-ref-of-{j}",
                expected="wrong", target=99)

        summary[d] = {"random_ref": len(randoms), "flagged_ref": len(flagged),
                      "flagged_major_total": len(flagged_all), "claude_pair": len(pair_ids),
                      "gemma": len(gem_ids), "control_error": len(controls),
                      "control_error_kinds": sorted({h for _, (_, h, _) in controls}),
                      "control_wrong": min(args.attention, len(rest)), "practice": len(practice)}
    out.write_text(json.dumps(items, ensure_ascii=False, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {out} ({len(items)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
