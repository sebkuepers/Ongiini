#!/usr/bin/env python3
"""Continued pretraining (CPT) of Gemma 4 on monolingual Oshiwambo text (paper 3).

LoRA on all linear layers of the language model (attention + MLP, rank 64):
word knowledge needs more capacity than the translation task, and the
vision/audio towers stay untouched. The base weights stay frozen, so the
adapter can later continue as an SFT adapter (train_lora.py --init-adapter)
and be switched on per request in vLLM like the others.

Data: data/private/corpus/osheng_v1/mono.jsonl, sentences regrouped into their
articles (doc, i), Oshiwambo only (lid). Each article = <bos> text <eos>;
everything concatenated and cut into blocks of --block tokens (packing: every
token is trained, no padding). 0.5 % of the articles are held out; their loss
is Oshiwambo perplexity. The tokenised blocks are cached next to the corpus.
The eval set was guarded out of the corpus by prepare_osheng_corpus.py.

    python3 scripts/train_cpt.py --model /models/gemma-4-12b-it-bf16 \\
        --out data/private/lora/C1_cpt_12b_r64
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from train_lora import TARGETS_ALL, load_model

CORPUS = Path("data/private/corpus/osheng_v1/mono.jsonl")
DIALECTS = {"Oshindonga", "Oshikwanyama"}


def articles(path: Path, limit_docs: int = 0) -> tuple[list[str], list[str]]:
    """Articles as text (sentences in order), split into train / held-out by doc hash."""
    docs: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("lid") in DIALECTS and r.get("doc") is not None:
                docs[r["doc"]].append((int(r.get("i", 0)), r["text"].strip()))
    train, held = [], []
    for k in sorted(docs):
        text = " ".join(t for _, t in sorted(docs[k]) if t)
        (held if int(hashlib.sha1(k.encode()).hexdigest(), 16) % 200 == 0 else train).append(text)
        if limit_docs and len(train) >= limit_docs:
            break
    return train, held


def blocks(texts: list[str], tok, size: int) -> Dataset:
    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(lambda b: {"ids": [[tok.bos_token_id] + tok(t, add_special_tokens=False).input_ids
                                   + [tok.eos_token_id] for t in b["text"]]},
                batched=True, remove_columns=["text"], num_proc=8)

    def group(b):
        flat = [t for ids in b["ids"] for t in ids]
        n = len(flat) // size * size
        return {"input_ids": [flat[i: i + size] for i in range(0, n, size)]}
    return ds.map(group, batched=True, batch_size=2000, remove_columns=["ids"], num_proc=8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after N optimizer steps (preflight)")
    ap.add_argument("--limit-docs", type=int, default=0, help="only the first N training articles (tests)")
    ap.add_argument("--save-steps", type=int, default=100)
    args = ap.parse_args(argv)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    cache = Path(args.corpus).with_name(f"cpt_blocks_{args.block}" + (f"_lim{args.limit_docs}" if args.limit_docs else ""))
    if (cache / "train").exists():
        train_ds, held_ds = load_from_disk(str(cache / "train")), load_from_disk(str(cache / "held"))
    else:
        train_txt, held_txt = articles(Path(args.corpus), args.limit_docs)
        train_ds, held_ds = blocks(train_txt, tok, args.block), blocks(held_txt, tok, args.block)
        train_ds.save_to_disk(str(cache / "train")); held_ds.save_to_disk(str(cache / "held"))
        print(f"articles: train {len(train_txt)}, held-out {len(held_txt)}", flush=True)
    if len(held_ds) == 0:  # tiny test corpora: hold out the last 2 % of training blocks instead
        k = max(1, len(train_ds) // 50)
        held_ds, train_ds = train_ds.select(range(len(train_ds) - k, len(train_ds))), train_ds.select(range(len(train_ds) - k))
    held_ds = held_ds.select(range(min(200, len(held_ds))))
    print(f"blocks of {args.block}: train {len(train_ds)} ({len(train_ds) * args.block / 1e6:.1f}M tokens), "
          f"held-out {len(held_ds)}; prepared in {time.time() - t0:.0f}s", flush=True)

    model = load_model(args.model, four_bit=False)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=TARGETS_ALL, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    cfg = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, lr_scheduler_type="cosine", warmup_steps=30,
        logging_steps=10, eval_strategy="steps", eval_steps=args.save_steps, save_strategy="steps",
        save_steps=args.save_steps, save_total_limit=2, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, report_to=[], seed=42,
        remove_unused_columns=False)
    trainer = Trainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=held_ds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    last = get_last_checkpoint(args.out) if Path(args.out).is_dir() else None
    if last:
        print(f"resuming from {last}", flush=True)
    before = trainer.evaluate()["eval_loss"] if not last else None
    tr = trainer.train(resume_from_checkpoint=last)
    trainer.save_model(args.out)
    after = trainer.evaluate()["eval_loss"]
    res = {**vars(args), "held_out_loss_before": before, "held_out_loss_after": after,
           "held_out_ppl_before": math.exp(before) if before is not None else None,
           "held_out_ppl_after": math.exp(after), "train_blocks": len(train_ds),
           "steps": trainer.state.global_step,
           "sec_per_step": round(tr.metrics["train_runtime"] / max(1, trainer.state.global_step), 2),
           "steps_per_epoch": math.ceil(len(train_ds) / (args.batch * args.accum)),
           "minutes": round((time.time() - t0) / 60, 1)}
    Path(args.out, "run.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
