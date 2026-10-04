#!/usr/bin/env python3
"""LoRA fine-tuning of Gemma 4 for English→Oshiwambo (paper 3).

Runs inside the training image on the DGX Spark (deploy/train/Dockerfile),
next to production vLLM. Data: chat-format JSONL ({"messages": [user,
assistant]}) built from data/private/corpus/. Loss is computed on the
Oshiwambo answer only (prompt/completion format). The adapter targets the attention projections, which
vLLM can load as a LoRA on the production model.

    python3 scripts/train_lora.py --model /models/gemma-4-26b-a4b-it-bf16 \\
        --train data/private/corpus/osheng_v1/sft_A_parallel_ndo_train.jsonl \\
        --val data/private/corpus/osheng_v1/sft_A_parallel_ndo_val.jsonl \\
        --out data/private/lora/A_parallel_ndo_r16
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def load_model(path: str, four_bit: bool):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    kw = {"dtype": torch.bfloat16, "device_map": {"": 0}, "attn_implementation": "sdpa"}
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    try:
        return AutoModelForCausalLM.from_pretrained(path, **kw)
    except (ValueError, KeyError):
        return AutoModelForImageTextToText.from_pretrained(path, **kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0, help="train rows (smoke test)")
    ap.add_argument("--bf16-base", action="store_true", help="no 4-bit quantisation of the base")
    ap.add_argument("--group-by-length", action="store_true",
                    help="batch examples of similar length (less padding on short sentences)")
    ap.add_argument("--chat-format", choices=["trl", "rendered"], default="trl",
                    help="trl = learning-curve recipe; rendered = prompt exactly as at inference")
    args = ap.parse_args(argv)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, four_bit=not args.bf16_base)
    print(f"loaded in {time.time() - t0:.0f}s; GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    ds = load_dataset("json", data_files={"train": args.train, "val": args.val})
    # Loss on the Oshiwambo answer only. "trl" = trl's conversational
    # prompt/completion format (A, B10k and the rest of the learning curve).
    # Gemma 4's generation prompt ends with an empty thought block that the
    # rendered prompt+answer lacks, so trl's prefix mask drops the first answer
    # tokens (~4) from the loss and trains on a prompt the model never sees at
    # inference. "rendered" fixes both: prompt = the inference prompt as text.
    def conversational(r):
        return {"prompt": r["messages"][:-1], "completion": r["messages"][-1:]}

    def rendered(r):
        prompt = tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(r["messages"], tokenize=False)
        answer = r["messages"][-1]["content"].strip()
        end = full[full.rindex(answer) + len(answer):]  # end-of-turn marker after the answer
        return {"prompt": prompt, "completion": answer + end}
    ds = ds.map(rendered if args.chat_format == "rendered" else conversational,
                remove_columns=ds["train"].column_names)
    if args.limit:
        ds["train"] = ds["train"].select(range(min(args.limit, len(ds["train"]))))

    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, lr_scheduler_type="cosine", warmup_steps=20,
        logging_steps=10, eval_strategy="steps", eval_steps=100, save_strategy="steps",
        save_steps=200, save_total_limit=2, bf16=True, gradient_checkpointing=True,
        max_length=args.max_len, completion_only_loss=True, report_to=[], seed=42,
        train_sampling_strategy="group_by_length" if args.group_by_length else "random")
    peft_cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                          target_modules=TARGETS, task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["val"],
                         processing_class=tok, peft_config=peft_cfg)
    trainer.model.print_trainable_parameters()
    try:  # show what the loss sees for one example (diagnostic only, never fatal)
        ex = trainer.train_dataset[0]
        if "labels" in ex:
            print("loss on:", repr(tok.decode([t for t in ex["labels"] if t != -100])), flush=True)
        elif "completion_mask" in ex:
            print("loss on:", repr(tok.decode([i for i, m in zip(ex["input_ids"], ex["completion_mask"]) if m])), flush=True)
        else:
            print("loss on: columns", list(ex), flush=True)
        print("input:", repr(tok.decode(ex["input_ids"])[-160:]), flush=True)
    except Exception as exc:  # noqa: BLE001
        print("loss-on diagnostic failed:", repr(exc), flush=True)
    trainer.train()
    trainer.save_model(args.out)
    metrics = trainer.evaluate()
    Path(args.out, "run.json").write_text(json.dumps({**vars(args), **metrics,
                                                       "minutes": round((time.time() - t0) / 60, 1)}, indent=2))
    print(json.dumps(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
