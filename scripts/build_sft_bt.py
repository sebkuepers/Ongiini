#!/usr/bin/env python3
"""Build LoRA training sets for the back-translation learning curve (paper 3, experiment B).

Each set = the LAC/ASSAR human pairs of experiment A + the first N
back-translated pairs in sampling order (so the 10k set is a subset of the
50k set, which is a subset of the 200k set). Pairs are filtered: English
present, not a copy of the source, word-length ratio English/Oshiwambo
within 0.5–3.0. Validation stays the 300 LAC/ASSAR pairs of experiment A,
so validation loss is comparable across the curve.

    ~/.venvs/ongiini-eval/bin/python scripts/build_sft_bt.py --n 10000 50000 200000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtranslate_corpus import sample  # noqa: E402
from eval_prompts import PAPER_ZEROSHOT_ID, render  # noqa: E402

D = Path(__file__).resolve().parents[1] / "data/private/corpus/osheng_v1"


def ok(r: dict) -> bool:
    en, ow = (r.get("en") or "").strip(), r["text"].strip()
    if not en or en.lower() == ow.lower() or en.lower().startswith("oshindonga"):
        return False
    ratio = len(en.split()) / max(1, len(ow.split()))
    return 0.5 <= ratio <= 3.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument("--bt", type=Path, default=D / "bt_ndo_dict.jsonl")
    args = ap.parse_args(argv)
    order = {(r["doc"], r["i"]): k for k, r in enumerate(sample(max(args.n), "Oshindonga", 42))}
    rows = [r for r in map(json.loads, args.bt.open()) if (r["doc"], r["i"]) in order]
    rows.sort(key=lambda r: order[(r["doc"], r["i"])])
    good = [r for r in rows if ok(r)]
    lac = [json.loads(l) for l in (D / "sft_A_parallel_ndo_train.jsonl").open()]
    print(f"back-translated {len(rows)}, kept {len(good)} after filters; LAC/ASSAR {len(lac)}", file=sys.stderr)
    for n in sorted(args.n):
        pick = [r for r in good if order[(r["doc"], r["i"])] < n]
        out = D / f"sft_B{n // 1000}k_train.jsonl"
        with out.open("w") as f:
            for row in lac:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            for r in pick:
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": render(PAPER_ZEROSHOT_ID, "oshindonga", r["en"])},
                    {"role": "assistant", "content": r["text"]}],
                    "source": f"bt:{r['source']}", "doc": r["doc"]}, ensure_ascii=False) + "\n")
        print(f"{out.name}: {len(lac)} LAC + {len(pick)} back-translated", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
