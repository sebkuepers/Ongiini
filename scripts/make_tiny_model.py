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

import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText


def main(argv=None) -> int:
    src, dst = (argv or sys.argv[1:])[:2]
    # Edit the JSON, not the config object: Gemma 4 derives per-layer values
    # (bigger heads in the full-attention layers) from these fields itself.
    d = json.loads((Path(src) / "config.json").read_text())
    t = d["text_config"]
    t["layer_types"] = t["layer_types"][:6]  # keeps the 5:1 sliding/full pattern
    t.update(num_hidden_layers=6, hidden_size=64, intermediate_size=128, num_attention_heads=2,
             num_key_value_heads=1, head_dim=32, global_head_dim=64, num_global_key_value_heads=1)
    t.pop("per_layer_config", None)
    for sub in ("vision_config", "audio_config"):
        c = d.get(sub)
        if not isinstance(c, dict):
            continue
        for attr, val in (("num_hidden_layers", 1), ("hidden_size", 32), ("intermediate_size", 64),
                          ("num_attention_heads", 2), ("num_key_value_heads", 1), ("head_dim", 16)):
            if c.get(attr):
                c[attr] = val
    cfg = type(AutoConfig.from_pretrained(src)).from_dict(d)
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
