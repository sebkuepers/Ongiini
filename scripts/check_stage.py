#!/usr/bin/env python3
"""Plausibility checks between pipeline stages (deploy/train/pipeline_bt_curve.sh).

"The program exited 0" is not a result. Each check reads what a stage wrote
and fails (exit 1, reason on stdout) when the numbers cannot be right, so the
pipeline stops and alerts instead of building on a broken stage.

    python3 scripts/check_stage.py train data/private/lora/B50k_12b_r16
    python3 scripts/check_stage.py bench gemma-4-12b-B50k
    python3 scripts/check_stage.py retention gemma-4-12b-B50k [--judged]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

EXP = Path("data/private/experiments")


def check_train(out: str) -> list[str]:
    run = json.loads(Path(out, "run.json").read_text())
    loss = run.get("eval_loss")
    if loss is None or not math.isfinite(loss):
        return [f"eval_loss missing or not finite: {loss}"]
    return [f"eval_loss {loss:.2f} > 2.0 (A 1.57, B10k 1.34)"] if loss > 2.0 else []


def check_bench(label: str, base: str = "gemma-4-12b-base") -> list[str]:
    bad = []
    for d in ("oshindonga", "oshikwanyama"):
        rows = [json.loads(l) for l in (EXP / "lora" / f"{label}_{d}.jsonl").read_text().splitlines()]
        empty = sum(not r["translation"].strip() for r in rows)
        if len(rows) < 423:
            bad.append(f"{d}: only {len(rows)} of 423 outputs")
        if empty > 0.02 * len(rows):
            bad.append(f"{d}: {empty} empty outputs")
    scores = json.loads((EXP / "lora" / "scores.json").read_text())
    if label not in scores:
        return bad + ["not in scores.json"]
    got, ref = scores[label]["oshindonga_dev"]["chrf++"], scores[base]["oshindonga_dev"]["chrf++"]
    if got < ref + 5:
        bad.append(f"Ndonga dev chrF++ {got} not clearly above base {ref}")
    return bad


def check_retention(label: str, judged: bool, preflight: bool = False) -> list[str]:
    res = json.loads((EXP / "retention" / f"{label}.json").read_text())
    bad = []
    if not 1 < res["en_perplexity"] < 100:
        bad.append(f"English perplexity {res['en_perplexity']} implausible")
    if not preflight and res["gsm8k"] < 50:  # 3 items in a preflight
        bad.append(f"GSM8K {res['gsm8k']} < 50 (base 93)")
    if judged:
        t = res.get("english_pairwise_vs_base")
        if not t:
            bad.append("no judge result")
        elif t["unparsed"] or sum(t.values()) == 0:
            bad.append(f"judge incomplete: {t}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "bench", "retention"])
    ap.add_argument("target")
    ap.add_argument("--judged", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args(argv)
    try:
        bad = {"train": lambda: check_train(args.target), "bench": lambda: check_bench(args.target),
               "retention": lambda: check_retention(args.target, args.judged, args.preflight)}[args.stage]()
    except (OSError, KeyError, ValueError) as exc:
        bad = [f"cannot read results: {exc!r}"]
    if bad:
        print(f"CHECK {args.stage} {args.target} FAILED: " + "; ".join(bad))
        return 1
    print(f"check {args.stage} {args.target} ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
