#!/usr/bin/env python3
"""English replay text for continued pretraining (paper 3, round 6b "CPT with replay").

C1-T4 (CPT on Oshiwambo, then T4) gained +1.7 chrF over T4b but lost instruction following
and open-task quality; scaling the adapter down did not help (round 6a). Replay of general
text during CPT is the standard remedy (Ibrahim et al. 2024; Lugha-Llama mixes English
educational text). This takes documents from FineWeb-Edu (HuggingFaceFW/fineweb-edu,
sample-10BT, ODC-By 1.0) in stream order until --tokens Gemma tokens are collected, and writes
{"text", "id", "url"} lines that train_cpt.py --replay-corpus reads.

    python3 scripts/build_cpt_replay_corpus.py --tokenizer /models/gemma-4-12b-it-bf16 \\
        --tokens 4000000 --out data/private/corpora/replay_en/fineweb_edu_4M.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tokens", type=int, default=4_000_000)
    ap.add_argument("--max-doc-tokens", type=int, default=4000, help="skip very long documents")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    out, total, n = Path(args.out), 0, 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in ds:
            k = len(tok(r["text"], add_special_tokens=False).input_ids)
            if k > args.max_doc_tokens:
                continue
            f.write(json.dumps({"text": r["text"], "id": r.get("id"), "url": r.get("url")}, ensure_ascii=False) + "\n")
            total += k
            n += 1
            if total >= args.tokens:
                break
    print(json.dumps({"docs": n, "tokens": total, "source": "HuggingFaceFW/fineweb-edu sample-10BT (ODC-By 1.0)"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
