#!/usr/bin/env python3
"""Digitise a scanned bilingual dictionary page by page with a vision model.

For the Viljoen/Amakali/Namuandi Oshindonga–English dictionary (Gamsberg
Macmillan 1984; scans without a text layer; permission for external OCR,
training and model release — see data/private/corpora/raw/nph-viljoen-1984/README.md).
Each page image goes to Gemini via OpenRouter with a fixed schema; the reply is
one JSON object per page with the page kind and its entries. Resumable: pages
already in the output are skipped. Output stays in data/private (never commit).

    python3 scripts/digitize_dictionary.py --pdf data/private/corpora/raw/nph-viljoen-1984/viljoen-amakali-namuandi-1984.pdf \\
        --out data/private/corpora/raw/nph-viljoen-1984/pages.jsonl --pages 70,100
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_openrouter_baseline import api_key  # noqa: E402

PROMPT = """This is one scanned page of an Oshindonga–English / English–Oshindonga dictionary
(Viljoen, Amakali, Namuandi). Transcribe it exactly into JSON. Do not invent, correct or
complete anything; keep the original spelling, including italics content.

Layout conventions of this book:
- Oshindonga→English entries: an optional noun-class prefix printed to the LEFT of the bold
  stem (e.g. "omu vali (aa-) parent" = prefix "omu", stem "vali", plural class "aa-").
  Italic words after the stem are synonyms/variants (often with their own class in brackets).
  A letter in brackets after a verb, e.g. "(e)" or "(u)", is a grammatical marker — keep it.
  Example sentences are in italics (Oshindonga) followed by their English translation.
- English→Oshindonga entries: an English headword followed by Oshindonga equivalents.
- Some pages are introduction / grammar tables / title pages.

Return ONLY a JSON object:
{"page_kind": "ow_en" | "en_ow" | "grammar" | "other",
 "printed_page_number": "<number printed on the page or null>",
 "entries": [
   {"prefix": "<noun-class prefix printed left of the stem, or ''>",
    "stem": "<bold headword as printed>",
    "full_form": "<prefix+stem as one word, e.g. omuvali; for English headwords the English word>",
    "plural": "<text inside the brackets right after the headword, e.g. 'aa-' or 'oma-', or ''>",
    "marker": "<other bracketed marker like (e) or (u), or ''>",
    "variants": ["<italic synonyms / variant forms with their bracket info>"],
    "gloss": "<the English meaning (ow_en) or the Oshindonga equivalents (en_ow), as printed>",
    "examples": [{"ow": "<Oshindonga example>", "en": "<English translation>"}]}
 ],
 "text": "<for grammar/other pages: the full page text in reading order; else ''>"}
Read the two columns left column first, top to bottom, then the right column."""


def render(pdf: str, page: int, dpi: int) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi), "-png", pdf,
                        f"{d}/p"], check=True)
        return next(Path(d).glob("p*.png")).read_bytes()


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


async def run(args) -> None:
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key("OPENROUTER_API_KEY"))
    out = Path(args.out)
    done = {json.loads(l)["page"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    pages = [p for p in args.pages if p not in done]
    sem = asyncio.Semaphore(args.concurrency)
    cost = 0.0

    async def one(p: int):
        nonlocal cost
        img = base64.b64encode(await asyncio.to_thread(render, args.pdf, p, args.dpi)).decode()
        async with sem:
            for attempt in range(3):
                try:
                    r = await client.chat.completions.create(
                        model=args.model, temperature=0, max_tokens=16000,
                        extra_body={"reasoning": {"effort": "low"}, "usage": {"include": True}},
                        messages=[{"role": "user", "content": [
                            {"type": "text", "text": PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}]}])
                    data = parse(r.choices[0].message.content or "")
                    if not data:
                        raise ValueError("no JSON in reply")
                    cost += float(getattr(r, "usage", None) and getattr(r.usage, "cost", 0) or 0)
                    with out.open("a") as f:
                        f.write(json.dumps({"page": p, **data}, ensure_ascii=False) + "\n")
                    print(f"page {p}: {data.get('page_kind')} {len(data.get('entries') or [])} entries", flush=True)
                    return
                except Exception as exc:  # noqa: BLE001 — retry, then record the failure
                    err = repr(exc)[:200]
            print(f"page {p}: FAILED {err}", flush=True)

    await asyncio.gather(*(one(p) for p in pages))
    print(f"done; approx. cost ${cost:.2f}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", default="1-127", help="e.g. 70,100 or 1-127")
    ap.add_argument("--model", default="google/gemini-3.1-pro-preview")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args(argv)
    pages = []
    for part in args.pages.split(","):
        a, _, b = part.partition("-")
        pages += list(range(int(a), int(b or a) + 1))
    args.pages = pages
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
