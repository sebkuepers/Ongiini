#!/usr/bin/env python3
"""Build a tiny random Gemma 4 with the real 12B's architecture and tokenizer.

For end-to-end pipeline tests (deploy/train/pipeline_cpt.sh with TINY=1):
same model class, module names, chat template and tokenizer as
gemma-4-12b-it, but a few MB of random weights, so every script path
(LoRA targets, CPT, SFT on a CPT adapter, generation, retention) runs in
seconds. Outputs are garbage by design.

    python3 scripts/make_tiny_model.py /models/gemma-4-12b-it-bf16 data/private/models-tiny/gemma4-tiny
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText


def main(argv=None) -> int:
    src, dst = (argv or sys.argv[1:])[:2]
    cfg = AutoConfig.from_pretrained(src)
    t = cfg.text_config
    # keep the pattern of sliding / full attention layers (5:1), just 6 layers
    pattern = list(t.layer_types[:6])
    t.num_hidden_layers, t.layer_types = 6, pattern
    t.hidden_size, t.intermediate_size = 64, 128
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 32
    for attr in ("global_head_dim", "num_global_key_value_heads"):
        if getattr(t, attr, None):
            setattr(t, attr, 32 if attr == "global_head_dim" else 1)
    for sub in ("vision_config", "audio_config"):
        c = getattr(cfg, sub, None)
        if c is None:
            continue
        for attr, val in (("num_hidden_layers", 1), ("hidden_size", 32), ("intermediate_size", 64),
                          ("num_attention_heads", 2), ("num_key_value_heads", 1), ("head_dim", 16)):
            if getattr(c, attr, None):
                setattr(c, attr, val)
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(cfg).to(torch.bfloat16)
    n = sum(p.numel() for p in model.parameters())
    print(f"tiny model: {n / 1e6:.1f}M parameters", flush=True)
    model.save_pretrained(dst)
    for f in Path(src).iterdir():
        if f.suffix in (".json", ".jinja") and f.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(f, Path(dst) / f.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
