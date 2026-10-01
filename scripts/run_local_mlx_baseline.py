#!/usr/bin/env python3
"""Translate the eval-set English sources with a local model via mlx-lm.

For open models small enough to run on an Apple-silicon laptop (e.g.
Okalai's OkaLM 1B/3B/8B). Two protocols:

  zeroshot  The paper's zero-shot template (Appendix C; eval_prompts.
            PAPER_ZEROSHOT, id ongiini-eval-ow-v1-zeroshot) — the same
            prompt every API baseline gets. Base models get it as raw
            text; the continuation after "<Language>:" is the
            translation. This is the comparable leaderboard row.
  fewshot   Supplementary: k fixed development-split pairs as a
            completion prompt ("English: … / <Language>: …"). Needs
            references, so it reads the private TSV; the example ids are
            written next to the output so they can be excluded from
            scoring. Never uses blind-split items.

Decoding is greedy; output stops at the first newline. Writes one
submission-schema JSONL line per item and dialect.

    ~/.venvs/ongiini-eval/bin/python scripts/run_local_mlx_baseline.py \\
        --model okalai-ai/okalm-1b --dialect oshikwanyama --protocol zeroshot \\
        --out data/private/export/data/baselines/okalm-1b_zeroshot_oshikwanyama.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_prompts import PAPER_ZEROSHOT_ID, render  # noqa: E402
from fill_baseline_translations import LANGUAGE_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data/oshiwambo_eval_v3.tsv"
PRIVATE = ROOT / "data/private/oshiwambo_eval_v3.tsv"


def fewshot_examples(dialect: str, k: int, seed: int, translator: str) -> list[dict]:
    """k development-split pairs with references, spread over S/M/L."""
    with PRIVATE.open() as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t")
                if r["in_blind_split"] == "false" and r[f"{translator}_{dialect}"].strip()]
    rng = random.Random(seed)
    by_len = {b: [r for r in rows if r["length_bucket"] == b] for b in "SML"}
    picks, order = [], ["S", "M", "M", "L", "S", "M", "L"]
    for b in (order * k)[:k]:
        pool = [r for r in by_len[b] if r not in picks]
        picks.append(rng.choice(pool))
    return picks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face repo id or local path")
    ap.add_argument("--dialect", choices=sorted(LANGUAGE_NAMES), required=True)
    ap.add_argument("--protocol", choices=("zeroshot", "fewshot"), default="zeroshot")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--translator", default="kaarina")
    ap.add_argument("--max-id", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0, help="Only the first N items (smoke test)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    lang = LANGUAGE_NAMES[args.dialect]
    with SOURCES.open() as f:
        items = [r for r in csv.DictReader(f, delimiter="\t") if int(r["id"]) <= args.max_id]
    if args.limit:
        items = items[: args.limit]

    prefix = ""
    prompt_id = PAPER_ZEROSHOT_ID           # paper Appendix C (was the v0 legacy text until 2026-10-01)
    excluded: list[int] = []
    if args.protocol == "fewshot":
        shots = fewshot_examples(args.dialect, args.shots, args.seed, args.translator)
        excluded = [int(r["id"]) for r in shots]
        prefix = "".join(f"English: {r['english']}\n{lang}: {r[f'{args.translator}_{args.dialect}']}\n\n"
                         for r in shots)
        prompt_id = f"ongiini-eval-ow-v1-{args.shots}shot-dev-seed{args.seed}"

    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Resumable: rows already written with the same template are kept and skipped.
    done = set()
    if args.out.exists():
        done = {json.loads(l)["id"] for l in args.out.read_text().splitlines()
                if l.strip() and json.loads(l).get("prompt_template_id") == prompt_id}
    t0 = time.time()
    with args.out.open("a" if done else "w") as f:
        for n, r in enumerate(items, start=1):
            if int(r["id"]) in done:
                continue
            en = r["english"].strip()
            if args.protocol == "zeroshot":
                prompt = render(PAPER_ZEROSHOT_ID, args.dialect, en)
            else:
                prompt = prefix + f"English: {en}\n{lang}:"
            raw = generate(model, tokenizer, prompt=prompt, sampler=sampler,
                           max_tokens=min(400, 12 + 4 * len(en.split())), verbose=False)
            translation = raw.strip().split("\n")[0].strip().strip('"').strip()
            f.write(json.dumps({
                "id": int(r["id"]), "dialect": args.dialect, "model_id": args.model,
                "prompt_template_id": prompt_id, "translation": translation,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "seed": args.seed, "raw_response": raw,
            }, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(items):
                print(f"{n}/{len(items)}  {time.time() - t0:.0f}s", file=sys.stderr)
    if excluded:
        args.out.with_suffix(".fewshot_ids.json").write_text(json.dumps(excluded))
        print(f"few-shot example ids (exclude from scoring): {excluded}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
