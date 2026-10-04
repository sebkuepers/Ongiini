#!/usr/bin/env python3
"""Plausibility checks between pipeline stages (deploy/train/pipeline_bt_curve.sh).

"The program exited 0" is not a result. Each check reads what a stage wrote
and fails (exit 1, reason on stdout) when the stage is broken, so the pipeline
stops and alerts instead of building on it. Every threshold must be validated
against measured runs (A, B10k) before use — run the checks on them first.

    python3 scripts/check_stage.py train data/private/lora/B50k_12b_r16
    python3 scripts/check_stage.py cpt data/private/lora/C1_cpt_12b_r64
    python3 scripts/check_stage.py bench gemma-4-12b-B50k
    python3 scripts/check_stage.py retention gemma-4-12b-B50k [--judged]
"""
from __future__ import annotations

import argparse
import json
import os
import math
import sys
from pathlib import Path

EXP = Path(os.environ.get("ONGIINI_EXP_DIR", "data/private/experiments"))  # override: pipeline tests


def check_train(out: str) -> list[str]:
    run = json.loads(Path(out, "run.json").read_text())
    loss = run.get("eval_loss")
    if loss is None or not math.isfinite(loss):
        return [f"eval_loss missing or not finite: {loss}"]
    # Defect = did not learn. Measured on this val set: untrained base ~4.2,
    # trained A 1.57, B10k 1.32 — 3.0 sits clearly between.
    return [f"eval_loss {loss:.2f} > 3.0 (base ~4.2, A 1.57, B10k 1.32)"] if loss > 3.0 else []


def check_cpt(out: str) -> list[str]:
    """CPT defect = did not learn: the held-out Oshiwambo loss must drop."""
    run = json.loads(Path(out, "run.json").read_text())
    before, after = run.get("held_out_loss_before"), run.get("held_out_loss_after")
    if after is None or not math.isfinite(after):
        return [f"held-out loss missing or not finite: {after}"]
    if before is not None and after > before - 0.05:
        return [f"held-out loss did not drop: {before:.3f} -> {after:.3f}"]
    return []


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
    """Only defects stop the pipeline: a missing result or a judge that could
    not run. The retention numbers themselves are measurements, reported in the
    WhatsApp message and compared with the base model there — never gates.
    (Gemma 4 12B-it has raw-text WikiText perplexity ~1400 with <bos>, measured
    2026-10-04; a guessed absolute threshold of 100 stopped the run.)"""
    res = json.loads((EXP / "retention" / f"{label}.json").read_text())
    bad = [f"{k} missing" for k in ("en_perplexity", "gsm8k", "vocab_en_to_ndo") if k not in res]
    if judged:
        t = res.get("english_pairwise_vs_base")
        if not t or sum(t.values()) == 0:
            bad.append(f"judge did not run: {t}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "cpt", "bench", "retention"])
    ap.add_argument("target")
    ap.add_argument("--judged", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args(argv)
    try:
        bad = {"train": lambda: check_train(args.target), "cpt": lambda: check_cpt(args.target), "bench": lambda: check_bench(args.target),
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
