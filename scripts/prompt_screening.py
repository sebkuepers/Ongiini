#!/usr/bin/env python3
"""Prompt screening for the robustness check (docs/eval-protocol.md §9).

Exploratory pre-study: each variant in eval_prompts.SCREEN changes one
property of the paper prompt (plus the literature prompts MIN and R23 and
the May legacy prompt). Cheap systems spanning model size — Gemma 4
26B (small, the Spark's production model), Gemma 4 31B, Qwen 3.5 122B (mid, reasoning off), DeepSeek V4.1 (large,
reasoning off) — on 100 development items with both references (seed 42).
It decides which properties the robustness check tests; it never changes
the leaderboard prompt.

    ~/.venvs/ongiini-eval/bin/python scripts/prompt_screening.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "data/private/export/data/eval_set.jsonl"
OUT = ROOT / "data/private/experiments/prompt_screen"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_prompts import LEGACY_ID, MIN_ID, PAPER_ZEROSHOT_ID, R23_ID, SCREEN  # noqa: E402

TEMPLATES = [PAPER_ZEROSHOT_ID, *SCREEN, MIN_ID, R23_ID, LEGACY_ID]
SYSTEMS = [  # label, model id, extra runner args
    ("gemma-4-26b", "google/gemma-4-26b-a4b-it", []),
    ("gemma-4-31b", "google/gemma-4-31b-it", []),
    ("qwen3.5-122b-nothink", "qwen/qwen3.5-122b-a10b", ["--reasoning", "off"]),
    ("deepseek-v4.1-nothink", "deepseek/deepseek-v4.1-flash", ["--reasoning", "off"]),
]
N_ITEMS = 100


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated system labels")
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in EVAL_SET.read_text().splitlines() if l.strip()]
    pool = sorted(int(r["id"]) for r in rows if not r["in_blind_split"]
                  and r["oshindonga_reference"].strip() and r["oshikwanyama_reference"].strip())
    ids = sorted(random.Random(42).sample(pool, N_ITEMS))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ids.json").write_text(json.dumps(ids))
    only = set(filter(None, args.only.split(",")))
    for label, model, extra in SYSTEMS:
        if only and label not in only:
            continue
        for t in TEMPLATES:
            cmd = [sys.executable, "scripts/run_openrouter_baseline.py", "--model", model,
                   "--label", f"{label}__{t}", "--template", t, "--out-dir", str(OUT),
                   "--ids-file", str(OUT / "ids.json"), "--concurrency", "8", *extra]
            print(" ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=False, cwd=ROOT, stdout=subprocess.DEVNULL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
