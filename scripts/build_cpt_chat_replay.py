#!/usr/bin/env python3
"""Chat-format replay text for continued pretraining (paper 3, variant C1r).

The literature (Mix-CPT, Knowledge-Instruct, Sunflower, Thinking Machines' on-policy
distillation notes) mixes instruction/chat-format data INTO the CPT stage so the instruct
model keeps its template and behaviour; replay only at SFT time was too late for us (C1).
This renders Gemma 4's own answers — the self-distilled replay sets (Dolly/GSM8K prompts,
scripts/build_replay_set.py) and the native tool-call replay (scripts/build_tool_replay.py) —
with the instruct chat template into plain text lines that train_cpt.py --replay-corpus packs
like any other document (the leading <bos> is stripped; blocks() adds its own).

    python3 scripts/build_cpt_chat_replay.py --tokenizer /models/gemma-4-12b-it-bf16 \\
        --out data/private/corpora/replay_en/gemma_chat_replay.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

REPLAY = [f"data/private/corpus/replay_v{i}/replay.jsonl" for i in (1, 2, 3, 4)]
TOOLS = "data/private/corpus/replay_tools_v1/replay.jsonl"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="the INSTRUCT model (chat template)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    bos = tok.bos_token or "<bos>"
    out, n, toks, seen = [], 0, 0, set()
    for p in REPLAY:
        if not Path(p).exists():
            continue
        for l in open(p):
            r = json.loads(l)
            key = r["messages"][0]["content"][:200]
            if key in seen:
                continue
            seen.add(key)
            text = tok.apply_chat_template(r["messages"], tokenize=False)
            out.append({"text": text.removeprefix(bos), "source": r.get("source", "replay")})
    if Path(TOOLS).exists():
        for l in open(TOOLS):
            r = json.loads(l)
            out.append({"text": (r["prompt"] + r["completion"]).removeprefix(bos), "source": r.get("source", "toolreplay")})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            toks += len(tok(r["text"], add_special_tokens=False).input_ids)
            n += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"docs": n, "tokens": toks, "sources": sorted({r["source"].split(":")[0] for r in out})}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
