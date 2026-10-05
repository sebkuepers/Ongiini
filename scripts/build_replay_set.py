#!/usr/bin/env python3
"""Replay data against catastrophic forgetting (paper 3, experiment C2).

The CPT adapter C1 translates very well but lost general abilities (English
answers degenerate into looping Oshiwambo-like text). C2 continues it with the
translation pairs mixed with general tasks answered by Gemma 4 itself, so the
adapter relearns to behave like the base model outside translation.

Backends: --backend hf (default) = the 12B base model in this container, a
normal GPU job (self-distillation). --backend vllm = the production server —
NEVER while a training runs: on 2026-10-05 four long generations next to a
training stalled vLLM for 47 minutes (Ongiini down).

Prompts: databricks-dolly-15k (CC BY-SA 3.0; all categories, context included)
and GSM8K *train* (MIT; the retention suite uses the test split). Prompts that
match a retention-suite item are dropped. Pairs longer than --max-tokens
(prompt + answer, Gemma tokenizer) are dropped, so nothing is truncated in SFT.

    python3 scripts/build_replay_set.py --out data/private/corpus/replay_v1/replay.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset
from openai import AsyncOpenAI
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retention_suite as RS  # noqa: E402

VLLM = "http://localhost:8124/v1"


def norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def retention_prompts() -> set[str]:
    data = RS.DATA
    items = list(RS.QUESTIONS) + list(RS.CHAT) + [q for q, _ in RS.TOOL_CASES]
    items += [g["q"] for g in json.loads((data / "gsm8k_100.json").read_text())]
    return {norm(t) for t in items}


def prompts(n_dolly: int, n_gsm: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    banned = retention_prompts()
    out = []
    dolly = list(load_dataset("databricks/databricks-dolly-15k", split="train"))
    rng.shuffle(dolly)
    for r in dolly:
        q = r["instruction"].strip()
        if r.get("context"):
            q = f"{q}\n\n{r['context'].strip()}"
        if norm(q) in banned or len(q) > 1200:
            continue
        out.append({"prompt": q, "source": f"dolly:{r['category']}", "license": "CC-BY-SA-3.0"})
        if len(out) >= n_dolly:
            break
    gsm = list(load_dataset("openai/gsm8k", "main", split="train"))
    rng.shuffle(gsm)
    for r in gsm[:n_gsm]:
        if norm(r["question"]) not in banned:
            out.append({"prompt": r["question"].strip(), "source": "gsm8k:train", "license": "MIT"})
    rng.shuffle(out)
    return out


async def answer_all(items: list[dict], concurrency: int, max_new: int) -> None:
    client = AsyncOpenAI(base_url=VLLM, api_key="none", timeout=300)
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(it):
        nonlocal done
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model="gemma-4-26b", temperature=0.7, max_tokens=max_new,
                    messages=[{"role": "user", "content": it["prompt"]}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                it["answer"] = (r.choices[0].message.content or "").strip()
                it["finish"] = r.choices[0].finish_reason
            except Exception as exc:  # noqa: BLE001 — a failed item is just dropped
                it["answer"], it["finish"] = "", f"error {exc!r}"[:80]
        done += 1
        if done % 200 == 0:
            print(f"answered {done}/{len(items)}", flush=True)

    await asyncio.gather(*(one(it) for it in items))


def answer_local(items: list[dict], model_path: str, batch: int, max_new: int) -> None:
    """Self-distillation: the base 12B answers, greedy, sorted by length for batching."""
    import torch
    from train_lora import load_model
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    model = load_model(model_path, four_bit=False).eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i]["prompt"]))
    for b in range(0, len(order), batch):
        idx = order[b: b + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": items[i]["prompt"]}], tokenize=False,
                                         add_generation_prompt=True) for i in idx]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        new = g[:, enc["input_ids"].shape[1]:]
        for i, ids, text in zip(idx, new, tok.batch_decode(new, skip_special_tokens=True)):
            items[i]["answer"] = text.strip()
            n_real = int((ids != tok.pad_token_id).sum())
            items[i]["finish"] = "length" if n_real >= max_new else "stop"
        print(f"answered {min(b + batch, len(order))}/{len(order)}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="/models/gemma-4-12b-it-bf16", help="also the model for --backend hf")
    ap.add_argument("--dolly", type=int, default=4000)
    ap.add_argument("--gsm", type=int, default=1300)
    ap.add_argument("--concurrency", type=int, default=3, help="keep low: this is the production server")
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--max-tokens", type=int, default=380, help="prompt + answer, chat template included")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--keep-length", action="store_true", help="keep answers that hit --max-new (tiny tests only)")
    ap.add_argument("--batch", type=int, default=32, help="hf backend")
    args = ap.parse_args(argv)

    items = prompts(args.dolly, args.gsm, args.seed)
    print(f"prompts: {len(items)}", flush=True)
    if args.backend == "vllm":
        asyncio.run(answer_all(items, args.concurrency, args.max_new))
    else:
        answer_local(items, args.tokenizer, args.batch, args.max_new)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    kept, why = [], {"empty/error": 0, "hit max_new": 0, "too long": 0}
    for it in items:
        if not it.get("answer") or str(it.get("finish", "")).startswith("error"):
            why["empty/error"] += 1
            continue
        if it["finish"] == "length" and not args.keep_length:
            why["hit max_new"] += 1
            continue
        msgs = [{"role": "user", "content": it["prompt"]}, {"role": "assistant", "content": it["answer"]}]
        text = tok.apply_chat_template(msgs, tokenize=False)
        if len(tok(text, add_special_tokens=False).input_ids) > args.max_tokens:
            why["too long"] += 1
            continue
        kept.append({"messages": msgs, "source": it["source"], "license": it["license"]})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    src = {}
    for r in kept:
        k = r["source"].split(":")[0]
        src[k] = src.get(k, 0) + 1
    print(json.dumps({"kept": len(kept), "dropped": why, "by_source": src}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
