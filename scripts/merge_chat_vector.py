#!/usr/bin/env python3
"""Chat model from an all-weights CPT of the pretrained Gemma 4 (paper 3, variant CF, 2026-10-16).

The OkaLM recipe (Nakashole, LREC 2026) continues pretraining the PRETRAINED model on all weights
at lr 1e-5; it never had to stay a chat assistant. We keep the assistant with a chat vector
(Huang et al. 2024): merged = IT + scale x (CPT - pretrained), tensor by tensor. Every tensor
the CPT trained (train_cpt.py --full: lm_weights-*.safetensors, checkpoint names) gets its delta;
all other tensors (embeddings, vision, audio) stay exactly the IT model's. The output is a
normal model directory (IT layout and files) that train_lora.py, the eval scripts and the eval
vLLM (EVAL_MODEL_DIR) use like the original. One tensor at a time, so memory stays small.

    python3 scripts/merge_chat_vector.py --base /models/gemma-4-12b-base-bf16 \\
        --it /models/gemma-4-12b-it-bf16 --cpt data/private/lora/CF_cpt_12b \\
        --out data/private/models/CF_cv_12b
"""
from __future__ import annotations

import argparse
import json
import shutil
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def tensor_files(d: Path, pattern: str) -> dict[str, Path]:
    """Tensor name -> file holding it."""
    where = {}
    for f in sorted(d.glob(pattern)):
        with safe_open(str(f), framework="pt") as h:
            where.update({k: f for k in h.keys()})
    return where


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the model the CPT started from (pretrained)")
    ap.add_argument("--it", required=True, help="the instruct model whose layout and files the output keeps")
    ap.add_argument("--cpt", required=True, help="train_cpt.py --full output (lm_weights-*.safetensors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="share of the CPT delta added to the IT model")
    ap.add_argument("--shard-gb", type=float, default=4.0)
    args = ap.parse_args(argv)
    base, it, cpt, out = Path(args.base), Path(args.it), Path(args.cpt), Path(args.out)
    w_it, w_base = tensor_files(it, "model*.safetensors"), tensor_files(base, "model*.safetensors")
    w_cpt = tensor_files(cpt, "lm_weights-*.safetensors")
    if not w_cpt:
        raise SystemExit(f"no lm_weights-*.safetensors in {cpt}")
    unknown = sorted(set(w_cpt) - set(w_it))
    if unknown:
        raise SystemExit(f"{len(unknown)} CPT tensors not in the IT checkpoint, e.g. {unknown[:3]}")
    missing = sorted(k for k in w_cpt if k not in w_base)
    if missing:
        raise SystemExit(f"{len(missing)} CPT tensors not in the base checkpoint, e.g. {missing[:3]}")
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("model*.safetensors"):
        f.unlink()
    index: dict[str, str] = {}
    shard: dict[str, torch.Tensor] = {}
    size = k = 0
    delta_sq = it_sq = 0.0
    with ExitStack() as stack:
        handles: dict[Path, object] = {}

        def get(where: dict[str, Path], name: str) -> torch.Tensor:
            f = where[name]
            if f not in handles:
                handles[f] = stack.enter_context(safe_open(str(f), framework="pt"))
            return handles[f].get_tensor(name)

        def flush():
            nonlocal shard, size, k
            fn = f"model-{k:05d}.safetensors"
            save_file(shard, str(out / fn), metadata={"format": "pt"})
            index.update({n: fn for n in shard})
            shard, size, k = {}, 0, k + 1

        for name in sorted(w_it):
            t = get(w_it, name)
            if name in w_cpt:
                a, b = get(w_cpt, name).float(), get(w_base, name).float()
                if a.shape != t.shape or b.shape != t.shape:
                    raise SystemExit(f"shape mismatch for {name}: it {tuple(t.shape)} cpt {tuple(a.shape)} base {tuple(b.shape)}")
                d = a.sub_(b).mul_(args.scale)
                delta_sq += float(d.square().sum()); it_sq += float(t.float().square().sum())
                t = t.float().add_(d).to(t.dtype)
            shard[name] = t.contiguous()
            size += t.numel() * t.element_size()
            if size >= args.shard_gb * 1e9:
                flush()
        if shard:
            flush()
    if k > 1:  # k = number of shards written
        (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": index}, indent=1))
    else:  # a single shard: the plain file name loaders expect
        (out / "model-00000.safetensors").rename(out / "model.safetensors")
    for f in it.iterdir():  # config, tokenizer, chat template, processor, generation config
        if f.is_file() and not f.name.startswith("model"):
            shutil.copy(f, out / f.name)
    info = {"base": str(base), "it": str(it), "cpt": str(cpt), "scale": args.scale, "tensors": len(w_it),
            "tensors_changed": len(w_cpt), "relative_delta_norm": round((delta_sq / max(it_sq, 1e-30)) ** 0.5, 6)}
    (out / "merged_from.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
