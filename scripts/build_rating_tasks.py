#!/usr/bin/env python3
"""Build the task list for rating round 1 ("Kaarina check").

Development split only (the blind split stays unseen by raters). Per
dialect, in priority order — so whatever the volunteers manage first is
the most useful:

  1. the translator's references that back-translation flagged "major"
     (2 ratings each)
  2. a random sample of her other references (1 rating each)
  3. Claude Opus 5 and Gemma 4 26B for the same sentences (1 each) —
     the contrast raters need to use the scale
  controls: the translator's reference of a *different* sentence —
     fluent, correct-language, wrong meaning (expected verdict: wrong)

Candidates are shuffled into opaque task ids; raters never learn which
is human, which is a model and which is a control.

Output (private): data/private/rating_tasks_round1.json

    python3 scripts/build_rating_tasks.py [--sample 30] [--controls 12]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT / "data/private/oshiwambo_eval_v3.tsv"
BACK = ROOT / "data/private/qa/backtranslation.jsonl"
BASE = ROOT / "data/private/export/data/baselines"
OUT = ROOT / "data/private/rating_tasks_round1.json"
DIALECTS = ("oshindonga", "oshikwanyama")
MODELS = [("claude-opus-5", BASE / "claude-opus-5_{d}.jsonl"),
          ("gemma-4-26b", BASE / "gemma-4-26b-api_{d}.jsonl")]


def outputs(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    return {json.loads(l)["id"]: json.loads(l)["translation"] for l in path.read_text().splitlines() if l}


def tid(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--translator", default="kaarina")
    ap.add_argument("--sample", type=int, default=30, help="Random references per dialect")
    ap.add_argument("--controls", type=int, default=12, help="Controls per dialect")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    with TSV.open() as f:
        rows = {int(r["id"]): r for r in csv.DictReader(f, delimiter="\t")}
    flags = {}
    for line in BACK.read_text().splitlines():
        o = json.loads(line)
        flags[(o["id"], o["dialect"])] = o["verdict"]

    rng = random.Random(args.seed)
    tasks, summary = [], {}
    for d in DIALECTS:
        ref = lambda i: rows[i][f"{args.translator}_{d}"].strip()
        dev = [i for i, r in rows.items() if r["in_blind_split"] == "false" and ref(i)]
        models = {name: outputs(Path(str(p).format(d=d))) for name, p in MODELS}
        usable = [i for i in dev if all(models[m].get(i) for m in models)]
        flagged = sorted(i for i in usable if flags.get((i, d)) == "major")
        rest = [i for i in usable if i not in flagged]
        sample = sorted(rng.sample(rest, min(args.sample, len(rest))))
        chosen = flagged + sample
        for i in chosen:
            en = rows[i]["english"].strip()
            tasks.append({"task_id": tid("r1", d, i, "ref"), "round": "r1", "item_id": i, "dialect": d,
                          "english": en, "text": ref(i), "kind": "ref", "source": args.translator,
                          "priority": 1 if i in flagged else 2, "target": 2 if i in flagged else 1})
            for name in models:
                tasks.append({"task_id": tid("r1", d, i, name), "round": "r1", "item_id": i,
                              "dialect": d, "english": en, "text": models[name][i].strip(),
                              "kind": "model", "source": name, "priority": 3, "target": 1})
        pool = [i for i in dev if i not in chosen]
        for i in rng.sample(pool, min(args.controls, len(pool))):
            j = rng.choice([k for k in dev if k != i])
            tasks.append({"task_id": tid("r1", d, i, "control", j), "round": "r1", "item_id": i,
                          "dialect": d, "english": rows[i]["english"].strip(), "text": ref(j),
                          "kind": "control", "source": f"{args.translator}-ref-of-{j}",
                          "priority": 9, "target": 99, "expected": "wrong"})
        summary[d] = {"flagged_major": len(flagged), "random": len(sample),
                      "model_candidates": len(chosen) * len(models), "controls": args.controls,
                      "ratings_needed": 2 * len(flagged) + len(sample) + len(chosen) * len(models)}
    OUT.write_text(json.dumps(tasks, ensure_ascii=False, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)} ({len(tasks)} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
