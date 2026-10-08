#!/usr/bin/env python3
"""Quality scores for back-translated pairs (paper 3, data-quality step for T4).

The 200k back-translated pairs (English by DeepSeek from real Oshiwambo) vary in
quality (Claude judge on a sample: 77 % good). Score each pair with a trained
adapter by forced decoding: the mean negative log-likelihood of the Oshiwambo
side given the English in the benchmark prompt. Pairs whose English does not say
what the Oshiwambo says get a high loss. Normalised per token; written with the
pair's row index so a selection can be built afterwards.

    python3 scripts/score_bt_pairs.py --model /models/gemma-4-12b-it-bf16 \\
        --adapter data/private/lora/T2_12b --pairs data/private/corpus/osheng_v1/bt_ndo_dict.jsonl \\
        --out data/private/corpus/osheng_v1/bt_scores_T2.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_prompts import PAPER_ZEROSHOT_ID, render  # noqa: E402
from train_lora import load_model  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="", help="empty = base model (tiny tests)")
    ap.add_argument("--pairs", required=True, help="jsonl with English 'en' and Oshiwambo 'text' (bt output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-batch-tokens", type=int, default=4096, help="padded tokens per batch")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(args.pairs) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out)
    done = {json.loads(l)["row"] for l in open(out)} if out.exists() else set()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, four_bit=False)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    pad = tok.pad_token_id
    seqs = {}
    for i, r in enumerate(rows):
        if i in done:
            continue
        en = (r.get("en") or r.get("english") or "").strip()
        ow = (r.get("text") or r.get("ow") or "").strip()
        prompt = tok.apply_chat_template([{"role": "user", "content": render(PAPER_ZEROSHOT_ID, "oshindonga", en)}],
                                         tokenize=False, add_generation_prompt=True)
        p = tok(prompt, add_special_tokens=False).input_ids
        a = tok(ow + "<turn|>", add_special_tokens=False).input_ids
        seqs[i] = ((p + a)[:512], min(len(p), 511))
    # Length-sorted batches under a token budget: the logits over Gemma's 262k
    # vocabulary dominate memory (batch 32 padded to 512 tokens tripped the memory
    # brake on 2026-10-08), and float32 is taken only over the answer tokens.
    order = sorted(seqs, key=lambda i: len(seqs[i][0]))
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * len(seqs[i][0]) > args.max_batch_tokens:
            batches.append(cur); cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    print(f"to score: {len(order)} pairs in {len(batches)} batches (already done: {len(done)})", flush=True)
    with out.open("a") as f:
        for nb, idx in enumerate(batches):
            L = max(len(seqs[i][0]) for i in idx)
            ids = torch.full((len(idx), L), pad, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for k, i in enumerate(idx):
                s = seqs[i][0]
                ids[k, : len(s)] = torch.tensor(s); att[k, : len(s)] = 1
            ids, att = ids.to(model.device), att.to(model.device)
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=att).logits
                for k, i in enumerate(idx):
                    s, st = seqs[i]
                    lp = torch.log_softmax(logits[k, st - 1: len(s) - 1].float(), -1)
                    nll = -lp.gather(-1, ids[k, st: len(s), None])[..., 0].sum().item() / max(1, len(s) - st)
                    f.write(json.dumps({"row": i, "nll": round(nll, 4), "tokens": len(s) - st}) + "\n")
            del logits
            if nb % 50 == 0:
                print(f"scored {len(done) + sum(len(b) for b in batches[: nb + 1])}/{len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
