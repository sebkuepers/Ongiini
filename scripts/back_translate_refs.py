#!/usr/bin/env python3
"""Triage the translator references by blind back-translation.

We cannot read Oshiwambo ourselves, so this produces a *flag list* for
native-speaker review — not a verdict. Two steps per reference:

  1. back-translate  the Oshiwambo reference into English, WITHOUT the
                     English source (so the model cannot fill in what was
                     meant and hide an error)
  2. compare         the English source with the back-translation — English
                     only — and classify the meaning difference as
                     same / minor / major, with a one-line reason

Model: Claude Opus 5 via OpenRouter, pinned to Anthropic as the only
provider with data collection denied (references are unreleased and
include the blind split), reasoning off, temperature 0. A misreading by
the model shows up as a false flag; humans decide.

Output (private): data/private/qa/backtranslation.jsonl, resumable.

    ~/.venvs/ongiini-eval/bin/python scripts/back_translate_refs.py [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_openrouter_baseline import api_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT / "data/private/oshiwambo_eval_v3.tsv"
OUT = ROOT / "data/private/qa/backtranslation.jsonl"
MODEL = "anthropic/claude-opus-5"
PROVIDER = {"order": ["anthropic"], "allow_fallbacks": False, "data_collection": "deny"}
DIALECTS = {"oshindonga": "Oshindonga", "oshikwanyama": "Oshikwanyama"}

BACK = """Translate the following {dialect} text (an Oshiwambo language of northern \
Namibia) into English. Translate faithfully — keep negations, numbers, names and \
who-does-what exactly as written; do not smooth over anything you cannot read. \
Output only the English translation.

{dialect}: {text}
English:"""

COMPARE = """Two English sentences should mean the same thing. The first is an \
original; the second is a machine back-translation of a human translation of it.

Original: {source}
Back-translation: {back}

Ignore wording, style, word order and small grammar slips. Judge only meaning: \
missing or added content, negation flipped, wrong numbers/dates/names, wrong \
person or subject, wrong sense of a word.

Answer with JSON only: {{"verdict": "same" | "minor" | "major", "issue": "<one short \
phrase, empty if same>"}}"""


async def ask(client, prompt: str, max_tokens: int) -> str:
    for wait in (0, 5, 15, 45):
        await asyncio.sleep(wait)
        try:
            r = await client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"provider": PROVIDER, "reasoning": {"enabled": False}})
            return (r.choices[0].message.content or "").strip()
        except Exception as exc:                            # noqa: BLE001
            last = exc
    raise last


async def main_async(args) -> int:
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key("OPENROUTER_API_KEY"))
    with TSV.open() as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {(json.loads(l)["id"], json.loads(l)["dialect"]) for l in OUT.read_text().splitlines() if l}
    jobs = [(r, d) for r in rows for d in DIALECTS
            if r[f"{args.translator}_{d}"].strip() and (int(r["id"]), d) not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    sem, lock = asyncio.Semaphore(6), asyncio.Lock()
    print(f"{len(jobs)} references to check", file=sys.stderr)

    async def one(r, d):
        async with sem:
            ref = r[f"{args.translator}_{d}"].strip()
            try:
                back = await ask(client, BACK.format(dialect=DIALECTS[d], text=ref), 400)
                raw = await ask(client, COMPARE.format(source=r["english"].strip(), back=back), 120)
                m = re.search(r"\{.*\}", raw, re.S)
                verdict = json.loads(m.group(0)) if m else {"verdict": "unparsed", "issue": raw[:80]}
            except Exception as exc:                        # noqa: BLE001
                print(f"  error {r['id']} {d}: {exc}", file=sys.stderr)
                return
            rec = {"id": int(r["id"]), "dialect": d, "in_blind_split": r["in_blind_split"] == "true",
                   "back_translation": back, "verdict": verdict.get("verdict"),
                   "issue": verdict.get("issue", ""), "notes_present": bool(r[f"{args.translator}_{d}_notes"].strip())}
            async with lock:
                with OUT.open("a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(r, d) for r, d in jobs))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--translator", default="kaarina")
    ap.add_argument("--limit", type=int, default=0)
    return asyncio.run(main_async(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
