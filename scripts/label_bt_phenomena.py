#!/usr/bin/env python3
"""Label every back-translated pair with grammatical features (paper 3, selection experiment).

Same labels and prompt as scripts/profile_phenomena.py (person, sentence
type, tense, passive, negation, register), on the English side of all pairs
in bt_ndo_dict.jsonl. Resumable; writes one row per pair keyed by
(doc, i) to bt_labels.jsonl. Feeds build_sft_selected.py --strategy grammar.

    python3 scripts/label_bt_phenomena.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_phenomena import BT, PROMPT  # noqa: E402
from run_openrouter_baseline import api_key  # noqa: E402

OUT = BT.with_name("bt_labels.jsonl")


async def run(rows: list[dict], model: str, conc: int) -> None:
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key("OPENROUTER_API_KEY"))
    sem, lock = asyncio.Semaphore(conc), asyncio.Lock()

    async def one(chunk: list[dict]) -> None:
        text = "\n".join(f"{k + 1}. {r['en']}" for k, r in enumerate(chunk))
        async with sem:
            for wait in (0, 5, 20, 60):
                await asyncio.sleep(wait)
                try:
                    r = await client.chat.completions.create(
                        model=model, temperature=0, max_tokens=3000,
                        messages=[{"role": "user", "content": PROMPT.format(rows=text)}],
                        extra_body={"reasoning": {"enabled": False}})
                    m = re.search(r"\[.*\]", r.choices[0].message.content or "", re.S)
                    arr = json.loads(m.group(0)) if m else []
                    if len(arr) == len(chunk) and all(isinstance(a, dict) for a in arr):
                        break
                except Exception as exc:                    # noqa: BLE001
                    print(f"  retry: {exc}", file=sys.stderr)
            else:
                return
        async with lock:
            with OUT.open("a") as f:
                for row, a in zip(chunk, arr):
                    f.write(json.dumps({"doc": row["doc"], "i": row["i"], **a}, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(rows[k:k + 20]) for k in range(0, len(rows), 20)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args(argv)
    done = {(json.loads(l)["doc"], json.loads(l)["i"]) for l in OUT.open()} if OUT.exists() else set()
    rows = [r for r in map(json.loads, BT.open()) if r.get("en") and (r["doc"], r["i"]) not in done]
    print(f"{len(rows)} pairs to label ({len(done)} done)", file=sys.stderr)
    asyncio.run(run(rows, args.model, args.concurrency))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
