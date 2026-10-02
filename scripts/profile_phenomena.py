#!/usr/bin/env python3
"""Grammatical profile of the training data vs the benchmark (paper 3).

An LLM labels the English side of a sample of back-translated training pairs
and of all benchmark sources with person, sentence type, tense, passive,
negation and register. Comparing the two distributions shows which
constructions the (mostly newspaper) corpus lacks — e.g. first/second
person, questions, requests — and therefore what targeted selection or
synthetic dialogues must add.

    ~/.venvs/ongiini-eval/bin/python scripts/profile_phenomena.py --n 1500
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_openrouter_baseline import api_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BT = ROOT / "data/private/corpus/osheng_v1/bt_ndo_dict.jsonl"
EVAL = ROOT / "data/private/export/data/eval_set.jsonl"
OUT = ROOT / "data/private/corpus/osheng_v1/phenomena_profile.json"
FIELDS = {
    "person": ["1", "2", "3", "none"],
    "type": ["statement", "question", "imperative", "request", "exclamation"],
    "tense": ["past", "present", "future", "modal", "none"],
    "passive": [True, False],
    "negation": [True, False],
    "register": ["news", "formal", "informal", "chat", "religious"],
}
PROMPT = """\
For each numbered English sentence give a JSON object with:
"person": main grammatical person of the speaker/addressee perspective: "1" (I/we), "2" (you), "3" (he/she/they/it), "none";
"type": "statement" | "question" | "imperative" | "request" | "exclamation";
"tense": "past" | "present" | "future" | "modal" | "none";
"passive": true/false; "negation": true/false;
"register": "news" | "formal" | "informal" | "chat" | "religious".
Return ONLY a JSON array with one object per sentence, in order.

{rows}"""


async def label(sents: list[str], model: str) -> list[dict]:
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key("OPENROUTER_API_KEY"))
    sem, out = asyncio.Semaphore(16), [None] * len(sents)

    async def one(start: int) -> None:
        chunk = sents[start:start + 20]
        rows = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(chunk))
        async with sem:
            for wait in (0, 5, 20):
                await asyncio.sleep(wait)
                try:
                    r = await client.chat.completions.create(
                        model=model, temperature=0, max_tokens=3000,
                        messages=[{"role": "user", "content": PROMPT.format(rows=rows)}],
                        extra_body={"reasoning": {"enabled": False}})
                    m = re.search(r"\[.*\]", r.choices[0].message.content or "", re.S)
                    arr = json.loads(m.group(0)) if m else []
                    if len(arr) == len(chunk):
                        for k, a in enumerate(arr):
                            out[start + k] = a
                        return
                except Exception as exc:                    # noqa: BLE001
                    print(f"  retry: {exc}", file=sys.stderr)

    await asyncio.gather(*(one(i) for i in range(0, len(sents), 20)))
    return [o for o in out if isinstance(o, dict)]


def dist(rows: list[dict]) -> dict:
    res = {}
    for f, vals in FIELDS.items():
        c = Counter(str(r.get(f)).lower() for r in rows)
        res[f] = {str(v).lower(): round(100 * c[str(v).lower()] / len(rows), 1) for v in vals}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    args = ap.parse_args(argv)
    bt = [json.loads(l)["en"] for l in BT.open() if l.strip()]
    random.Random(42).shuffle(bt)
    bench = [json.loads(l)["english"] for l in EVAL.open() if l.strip()]
    tr = asyncio.run(label(bt[: args.n], args.model))
    be = asyncio.run(label(bench, args.model))
    res = {"training_sample": {"n": len(tr), **dist(tr)}, "benchmark": {"n": len(be), **dist(be)}}
    OUT.write_text(json.dumps(res, indent=2))
    for f in FIELDS:
        print(f"{f}:")
        for v in res["benchmark"][f]:
            print(f"   {v:12s} training {res['training_sample'][f][v]:5.1f} %   benchmark {res['benchmark'][f][v]:5.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
