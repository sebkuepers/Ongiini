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
import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_guard import DivergenceGuard, retry_argv  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
# All linear layers of the language model (not the vision/audio towers), as in
# the CPT adapter (scripts/train_cpt.py): attention + MLP.
TARGETS_ALL = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


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
    ap.add_argument("--alpha", type=float, default=0, help="LoRA alpha; 0 = 2 x rank (all runs before T1)")
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
    ap.add_argument("--targets", choices=["attn", "all"], default="attn",
                    help="attn = attention projections (curve); all = attention + MLP, as the CPT adapter")
    ap.add_argument("--init-adapter", default="",
                    help="continue training this LoRA adapter (SFT after CPT); rank/targets come from it")
    ap.add_argument("--divergence-retries", type=int, default=1,
                    help="after a divergence, restart from scratch at half the lr this many times")
    args = ap.parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, four_bit=not args.bf16_base)
    print(f"loaded in {time.time() - t0:.0f}s; GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    # Load the files separately and keep only "messages": mixes built by
    # build_sft_templates.py have no doc/source columns, the val file does.
    from datasets import Dataset, DatasetDict
    if args.chat_format == "rendered":
        # Rows are rendered here; pre-rendered {"prompt", "completion"} rows (native
        # tool-call replay, scripts/build_tool_replay.py) pass through unchanged.
        # Each row travels as one JSON string, so mixed row shapes and extra
        # columns (doc/source) never meet the datasets schema inference.
        def load_rows(f):
            return Dataset.from_list([{"pre": l.strip()} for l in open(f) if l.strip()])
        ds = DatasetDict({k: load_rows(f) for k, f in (("train", args.train), ("val", args.val))})
    else:
        ds = DatasetDict({k: load_dataset("json", data_files=f)["train"].select_columns(["messages"])
                          for k, f in (("train", args.train), ("val", args.val))})
    # Loss on the Oshiwambo answer only. "trl" = trl's conversational
    # prompt/completion format (A, B10k and the rest of the learning curve).
    # Gemma 4's generation prompt ends with an empty thought block that the
    # rendered prompt+answer lacks, so trl's prefix mask drops the first answer
    # tokens (~4) from the loss and trains on a prompt the model never sees at
    # inference. "rendered" fixes both: prompt = the inference prompt as text.
    def conversational(r):
        return {"prompt": r["messages"][:-1], "completion": r["messages"][-1:]}

    def rendered(r):
        r = json.loads(r["pre"])
        if "messages" not in r:
            return {"prompt": r["prompt"], "completion": r["completion"]}
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
        # non-reentrant checkpointing works with an already-wrapped PeftModel
        # (--init-adapter) without input grads; same maths as before.
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.init_adapter else None,
        train_sampling_strategy="group_by_length" if args.group_by_length else "random")
    if args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        peft_cfg = None
    else:
        peft_cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha or 2 * args.rank, lora_dropout=0.05,
                              target_modules=TARGETS_ALL if args.targets == "all" else TARGETS,
                              task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["val"],
                         processing_class=tok, peft_config=peft_cfg)
    if hasattr(trainer.model, "print_trainable_parameters"):
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
    from transformers import TrainerCallback
    guard = DivergenceGuard(limit=float(os.environ.get("DIVERGENCE_LIMIT", "4")),  # overrides only for tests
                            after=int(os.environ.get("DIVERGENCE_AFTER", "50")))

    class _Stop(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            logs = logs or {}
            had_warning = bool(guard.warning)
            if not guard.tripped and guard.check(state.global_step, logs.get("loss"), logs.get("grad_norm")):
                print(f"DIVERGED: {guard.reason} — stopping", flush=True)
            if guard.warning and not had_warning:
                print(f"UNSTABLE (warning only): {guard.warning}", flush=True)
            if guard.tripped:
                control.should_training_stop = True
    trainer.add_callback(_Stop())
    from train_cpt import last_complete_checkpoint  # same rule: skip a checkpoint cut off mid-save
    last = last_complete_checkpoint(args.out)
    if last:
        print(f"resuming from {last}", flush=True)
    trainer.train(resume_from_checkpoint=last)
    history = json.loads(os.environ.get("DIVERGENCE_HISTORY", "[]"))
    if guard.tripped:
        if args.divergence_retries <= 0:  # no run.json: the pipeline treats the stage as failed
            return 1
        # keep the diverged run, then start over in a fresh process (clean GPU memory)
        # at half the learning rate; the pipeline sees one stage that took longer
        kept = Path(f"{args.out}.diverged-lr{args.lr:g}")
        if kept.exists():
            kept = Path(f"{kept}-{int(time.time())}")
        Path(args.out).rename(kept)
        history.append({"lr": args.lr, "reason": guard.reason, "kept": str(kept)})
        os.environ["DIVERGENCE_HISTORY"] = json.dumps(history)
        print(f"RETRY: from scratch at lr {args.lr / 2:g} (diverged run kept in {kept})", flush=True)
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()),
                                  *retry_argv(argv, args.lr, args.divergence_retries)])
    trainer.save_model(args.out)
    metrics = trainer.evaluate()
    Path(args.out, "run.json").write_text(json.dumps({**vars(args), **metrics,
                                                       "minutes": round((time.time() - t0) / 60, 1),
                                                       "unstable_warning": guard.warning,
                                                       "diverged_attempts": history}, indent=2))
    print(json.dumps(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
