#!/usr/bin/env python3
"""Blind pairwise LLM judgement of English answers, base vs adapter (retention suite, test 6).

For each of the 40 questions the judge sees the two answers in random order
and says which is better or that they are equal. Reports the adapter's
win / tie / loss counts.

    python3 scripts/retention_judge.py gemma-4-12b-base gemma-4-12b-A
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from openai import AsyncOpenAI, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_openrouter_baseline import api_key  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data/private/experiments/retention"
PROMPT = """\
A user asked an AI assistant for people in Namibia:

{q}

Answer 1:
{a1}

Answer 2:
{a2}

Which answer is more helpful, correct and clear? Reply with exactly one word: 1, 2 or tie."""


async def judge(base: dict, cand: dict, model: str) -> dict:
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key("OPENROUTER_API_KEY"))
    rng, sem = random.Random(42), asyncio.Semaphore(3)
    tally = {"adapter_wins": 0, "ties": 0, "base_wins": 0, "unparsed": 0}

    async def one(q: str) -> None:
        flip = rng.random() < 0.5
        a1, a2 = (cand[q], base[q]) if flip else (base[q], cand[q])
        async with sem:
            for attempt in range(8):  # OpenRouter limits new accounts to 20 requests/min per model
                try:
                    r = await client.chat.completions.create(
                        model=model, temperature=0, max_tokens=10, extra_body={"reasoning": {"enabled": False}},
                        messages=[{"role": "user", "content": PROMPT.format(q=q, a1=a1, a2=a2)}])
                    break
                except RateLimitError:
                    if attempt == 7:
                        raise
                    await asyncio.sleep(15 * (attempt + 1))
        v = (r.choices[0].message.content or "").strip().lower()
        if v.startswith("tie"):
            tally["ties"] += 1
        elif v[:1] and v[:1] in "12":
            adapter_won = (v[:1] == "1") == flip
            tally["adapter_wins" if adapter_won else "base_wins"] += 1
        else:
            tally["unparsed"] += 1

    await asyncio.gather(*(one(q) for q in base if q in cand))
    return tally


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("candidate")
    ap.add_argument("--model", default="anthropic/claude-opus-5")
    args = ap.parse_args(argv)
    base = json.loads((OUT / f"{args.base}_answers.json").read_text())
    cand = json.loads((OUT / f"{args.candidate}_answers.json").read_text())
    tally = asyncio.run(judge(base, cand, args.model))
    res_path = OUT / f"{args.candidate}.json"
    res = json.loads(res_path.read_text())
    res["english_pairwise_vs_base"] = tally
    res_path.write_text(json.dumps(res, indent=2))
    print(json.dumps(tally))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
