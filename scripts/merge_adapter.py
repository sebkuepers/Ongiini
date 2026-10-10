#!/usr/bin/env python3
"""Merge a LoRA adapter into the base weights and save a full model (paper 3, variant M).

C1-T4 continued the r64 CPT adapter through the T4 SFT and lost instruction following. If
the CPT adapter alone is intact (A1), the loss comes from the SFT moving a large adapter:
variant M merges C1 into Gemma 4 and trains the T4 recipe as a FRESH r32 adapter on top of
the merged weights. The merged model is a new base directory (bf16, ~24 GB) that
train_lora.py, the eval scripts and the eval vLLM (EVAL_MODEL_DIR) use like the original.
The tokenizer, chat template and processor files are copied from the base.

    python3 scripts/merge_adapter.py --model /models/gemma-4-12b-it-bf16 \\
        --adapter data/private/lora/C1_cpt_12b --out data/private/models/gemma-4-12b-C1merged
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lora import load_model  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    from peft import PeftModel
    model = load_model(args.model, four_bit=False)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    for f in Path(args.model).iterdir():  # tokenizer, chat template, processor, generation config
        if f.is_file() and not f.name.startswith("model") and f.name != "config.json":
            shutil.copy(f, out / f.name)
    (out / "merged_from.json").write_text(json.dumps({"base": args.model, "adapter": args.adapter}, indent=2))
    print(f"merged {args.adapter} into {args.model} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
