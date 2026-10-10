#!/usr/bin/env python3
"""Weaken a LoRA adapter by a factor (paper 3, round 6 "Regler-Test").

The LoRA update is delta_W = (alpha / r) * B @ A; multiplying every lora_B matrix by s
scales delta_W by s exactly, so s = 1 is the trained adapter and s = 0 the base model.
C1-T4 (CPT + T4) gained +1.7 chrF over T4b but lost instruction following (IFEval strict
88.0 -> 82.8, p 0.002); scaling shows whether some s keeps most of the gain and the
abilities. Copies the adapter config and weights; nothing is retrained.

    python3 scripts/scale_adapter.py data/private/lora/C1-T4_12b 0.7 data/private/lora/scaled/C1-T4_s0.7
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


def main(argv=None) -> int:
    src, s, dst = (argv or sys.argv[1:])[:3]
    s, src, dst = float(s), Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    w = load_file(str(src / "adapter_model.safetensors"))
    n = 0
    for k in w:
        if "lora_B" in k:
            w[k] = (w[k].float() * s).to(w[k].dtype)
            n += 1
    if not n:
        raise SystemExit(f"no lora_B tensors in {src}")
    save_file(w, str(dst / "adapter_model.safetensors"))
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")
    (dst / "scaled_from.json").write_text(json.dumps({"source": str(src), "scale": s, "lora_B_tensors": n}))
    print(f"{dst}: {n} lora_B tensors x {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
