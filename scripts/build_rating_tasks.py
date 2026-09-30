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
is human, which is a model and which is a control. Sentences where a
model output is derailed (looping or narrating, see
retry_derailed_baselines.is_derailed) are skipped: an obviously broken
answer teaches nothing and costs scarce rater time. Derailment is already
reported by the automatic scores.

Output (private): data/private/rating_tasks_round1.json — one screen per
sentence with the reference and both models side by side; on every 8th
screen the last model is swapped for a control.

    python3 scripts/build_rating_tasks.py [--sample 60]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retry_derailed_baselines import is_derailed  # noqa: E402

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
    ap.add_argument("--sample", type=int, default=60, help="Random references per dialect")
    ap.add_argument("--control-every", type=int, default=8,
                    help="On every n-th screen one model candidate is replaced by a control")
    ap.add_argument("--round", default="r1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    with TSV.open() as f:
        rows = {int(r["id"]): r for r in csv.DictReader(f, delimiter="\t")}
    flags = {}
    for line in BACK.read_text().splitlines():
        o = json.loads(line)
        flags[(o["id"], o["dialect"])] = o["verdict"]

    rng = random.Random(args.seed)
    screens, summary = [], {}
    for d in DIALECTS:
        ref = lambda i: rows[i][f"{args.translator}_{d}"].strip()
        dev = [i for i, r in rows.items() if r["in_blind_split"] == "false" and ref(i)]
        models = {name: outputs(Path(str(p).format(d=d))) for name, p in MODELS}
        usable = [i for i in dev if all(models[m].get(i) and not is_derailed(rows[i]["english"], models[m][i])
                                        for m in models)]
        flagged = sorted(i for i in usable if flags.get((i, d)) == "major")
        rest = [i for i in usable if i not in flagged]
        chosen = flagged + sorted(rng.sample(rest, min(args.sample, len(rest))))
        controls = 0
        for n, i in enumerate(chosen, start=1):
            sid = tid(args.round, d, i)
            cands = [{"cand_id": tid(sid, "ref"), "text": ref(i), "kind": "ref", "source": args.translator}]
            names = list(models)
            if n % args.control_every == 0:          # swap the last model for a control
                j = rng.choice([k for k in dev if k != i])
                cands.append({"cand_id": tid(sid, "ctl", j), "text": ref(j), "kind": "control",
                              "source": f"{args.translator}-ref-of-{j}", "expected": "no"})
                names = names[:-1]
                controls += 1
            for name in names:
                cands.append({"cand_id": tid(sid, name), "text": models[name][i].strip(),
                              "kind": "model", "source": name})
            screens.append({"screen_id": sid, "round": args.round, "item_id": i, "dialect": d,
                            "english": rows[i]["english"].strip(),
                            "priority": 1 if i in flagged else 2,
                            "target": 2 if i in flagged else 1, "candidates": cands})
        summary[d] = {"screens": len(chosen), "flagged_major": len(flagged),
                      "random": len(chosen) - len(flagged), "with_control": controls,
                      "ratings_to_fill": len(chosen) + len(flagged)}
    args.out.write_text(json.dumps(screens, ensure_ascii=False, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.out} ({len(screens)} screens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
