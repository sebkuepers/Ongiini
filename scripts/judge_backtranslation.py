#!/usr/bin/env python3
"""LLM spot check of back-translated pairs (paper 3, experiment B).

A strong model (default Claude Opus 5) rates, for a seeded sample, whether
the English sentence says the same as the Oshiwambo one: 5 same meaning,
4 small detail off, 3 partly, 2 mostly wrong, 1 unrelated / not a
translation. Ten pairs per call. The ratings estimate the quality of the
synthetic data and calibrate the automatic filters (two-teacher
agreement, length ratio); they are not training data.

    OPENROUTER_API_KEY=… ~/.venvs/ongiini-eval/bin/python scripts/judge_backtranslation.py \\
        data/private/corpus/osheng_v1/bt_pilot1k.jsonl --n 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_openrouter_baseline import api_key  # noqa: E402

PROMPT = """\
You are checking machine translations from {name} (an Oshiwambo language of \
northern Namibia) into English. For each numbered pair, rate whether the \
English says the same as the {name} sentence:
5 = same meaning; 4 = a small detail off; 3 = partly right; 2 = mostly wrong; \
1 = unrelated or not a translation.

Answer with one line per pair: "<number>. <score> | <at most 8 words on the main problem, or ok>".

{pairs}"""


async def run(rows: list[dict], model: str, conc: int) -> list[dict]:
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key("OPENROUTER_API_KEY"))
    sem = asyncio.Semaphore(conc)
    out: list[dict] = []

    async def one(chunk: list[dict]) -> None:
        name = chunk[0]["lid"]
        pairs = "\n".join(f"{k + 1}. {name}: {r['text']}\n   English: {r['en']}" for k, r in enumerate(chunk))
        async with sem:
            for wait in (0, 5, 20):
                await asyncio.sleep(wait)
                try:
                    resp = await client.chat.completions.create(
                        model=model, temperature=0, max_tokens=4000,
                        messages=[{"role": "user", "content": PROMPT.format(name=name, pairs=pairs)}],
                        # thinking would eat the budget and cut the list off
                        extra_body={"reasoning": {"enabled": False}})
                    text = resp.choices[0].message.content or ""
                    break
                except Exception as exc:                    # noqa: BLE001
                    print(f"  retry: {exc}", file=sys.stderr)
            else:
                return
        got = {int(m[1]): (int(m[2]), m[3].strip()) for m in re.finditer(r"(?m)^\W*(\d+)[.):]\s*\**\s*([1-5])\b\**\s*[|:\-–—]*\s*(.*)$", text)}
        for k, r in enumerate(chunk):
            if k + 1 in got:
                out.append({**r, "judge": got[k + 1][0], "judge_note": got[k + 1][1], "judge_model": model})

    chunks = [rows[i:i + 10] for i in range(0, len(rows), 10)]
    await asyncio.gather(*(one(c) for c in chunks))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", type=Path)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="anthropic/claude-opus-5")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in args.pairs.open() if l.strip()]
    rows = [r for r in rows if r.get("en")]
    random.Random(args.seed).shuffle(rows)
    judged = asyncio.run(run(rows[: args.n], args.model, args.concurrency))
    out = args.pairs.with_name(args.pairs.stem + ".judged.jsonl")
    with out.open("w") as f:
        for r in judged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(judged)} ratings to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
