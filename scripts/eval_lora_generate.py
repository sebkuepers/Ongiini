#!/usr/bin/env python3
"""Translate the benchmark with a local Gemma 4, with or without a LoRA adapter (paper 3).

Same weights, code, prompt (paper zero-shot template via the model's chat
template) and greedy decoding for the base run and every adapter run, so
the difference is the adapter alone. Writes baselines-style JSONL
(``<label>_<dialect>.jsonl``) that run_evaluation / eval_scoring can read.
Runs in the training image on the Spark.

    python3 scripts/eval_lora_generate.py --model /models/gemma-4-12b-it-bf16 \\
        --label gemma-4-12b-base --out data/private/experiments/lora
    python3 scripts/eval_lora_generate.py --model /models/gemma-4-12b-it-bf16 \\
        --adapter data/private/lora/A_parallel_ndo_12b --label gemma-4-12b-A --out ...
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_prompts import PAPER_ZEROSHOT_ID, render  # noqa: E402
from train_lora import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data/oshiwambo_eval_v3.tsv"
IDS = ROOT / "data/private/export/data/referenced_ids.json"
DIALECTS = ("oshindonga", "oshikwanyama")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--four-bit", action="store_true", help="4-bit base (31B: bf16 does not fit next to production)")
    args = ap.parse_args(argv)

    ids = set(json.loads(IDS.read_text()))
    with SOURCES.open() as f:
        items = [(int(r["id"]), r["english"].strip()) for r in csv.DictReader(f, delimiter="\t") if int(r["id"]) in ids]
    if args.limit:
        items = items[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = load_model(args.model, four_bit=args.four_bit)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    t0 = time.time()
    for d in DIALECTS:
        path = out_dir / f"{args.label}_{d}.jsonl"
        done = {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()} if path.exists() else set()
        todo = sorted((it for it in items if it[0] not in done), key=lambda it: len(it[1]))
        for b in range(0, len(todo), args.batch):
            batch = todo[b: b + args.batch]
            prompts = [tok.apply_chat_template([{"role": "user", "content": render(PAPER_ZEROSHOT_ID, d, en)}],
                                               tokenize=False, add_generation_prompt=True) for _, en in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            max_new = min(400, 20 + 3 * max(len(en.split()) for _, en in batch))
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with path.open("a") as f:
                for (iid, _), text in zip(batch, texts):
                    t = text.strip().split("\n")[0].strip()
                    f.write(json.dumps({"id": iid, "dialect": d, "model_id": args.model, "adapter": args.adapter or None,
                                        "prompt_template_id": PAPER_ZEROSHOT_ID, "translation": t,
                                        "raw_response": text, "timestamp": stamp, "seed": None}, ensure_ascii=False) + "\n")
            print(f"{d} {b + len(batch)}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
