#!/usr/bin/env python3
"""Prompt-robustness runs for Ongiini-Eval-OW (docs/eval-protocol.md §9).

Runs six systems with the two extra templates from eval_prompts.py on the
development items that have references, into baselines/robustness/ as
<label>__<template>_<dialect>.jsonl. run_evaluation.py picks them up and
reports chrF++ per template and Kendall τ between the system rankings.
Costs API credit (≈ 36 $ estimated 2026-10-01) — start deliberately.

    ~/.venvs/ongiini-eval/bin/python scripts/prompt_robustness.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/private/export/data/baselines"
OUT = BASE / "robustness"
EVAL_SET = ROOT / "data/private/export/data/eval_set.jsonl"
TEMPLATES = ("robustness-r23-zeroshot", "robustness-min-zeroshot")
SYSTEMS = [  # label, model id, extra runner args
    ("claude-opus-5", "anthropic/claude-opus-5", []),
    ("gemini-3.1-pro", "google/gemini-3.1-pro-preview", []),
    ("gpt-6-astra", "openai/gpt-6-astra", []),
    ("deepseek-v4.1-chat", "deepseek/deepseek-v4.1-flash", ["--reasoning", "off"]),
    ("gemma-4-26b-api", "google/gemma-4-26b-a4b-it", []),
    ("mistral-medium-3.5", "mistralai/mistral-medium-3-5", []),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated labels")
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in EVAL_SET.read_text().splitlines() if l.strip()]
    ids = sorted(int(r["id"]) for r in rows if not r["in_blind_split"]
                 and r["oshindonga_reference"].strip() and r["oshikwanyama_reference"].strip())
    OUT.mkdir(parents=True, exist_ok=True)
    ids_file = OUT / "dev_ids.json"
    ids_file.write_text(json.dumps(ids))
    only = set(filter(None, args.only.split(",")))
    for label, model, extra in SYSTEMS:
        if only and label not in only:
            continue
        for t in TEMPLATES:
            cmd = [sys.executable, "scripts/run_openrouter_baseline.py", "--model", model,
                   "--label", f"{label}__{t}", "--template", t, "--out-dir", str(OUT),
                   "--ids-file", str(ids_file), "--concurrency", "6", *extra]
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=False, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
