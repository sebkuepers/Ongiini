#!/usr/bin/env python3
"""Build an Oshindonga→English dictionary of the corpus's frequent word forms with Gemini (paper 3).

For each word form: up to 3 short corpus sentences as context; the model
returns lemma, part of speech, noun class (nouns), 1–3 English senses and
its confidence. The entries are a tool (glossary for the back-translation
teacher and, reversed, for Gemma), not training text. Validation is
automatic: agreement with Okalai's MIT noun lexicon where they overlap, the
model's own confidence, and the downstream effect.

Example sentences come from The Namibian scrape and stay in the private
working file only; ``export`` writes a dictionary without them.

    ~/.venvs/ongiini-eval/bin/python scripts/build_corpus_dictionary.py pilot --n 500
    ~/.venvs/ongiini-eval/bin/python scripts/build_corpus_dictionary.py build --top 20000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_openrouter_baseline import api_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MONO = ROOT / "data/private/corpus/osheng_v1/mono.jsonl"
OUT = ROOT / "data/private/corpus/dictionary_v1"
NCR = ROOT / "data/private/lexicon/okalai_ncr"
WORD = re.compile(r"[A-Za-zÀ-ÿ]+(?:['’-][A-Za-zÀ-ÿ]+)*")
BATCH, N_CONTEXT = 25, 3

PROMPT = """\
You are a lexicographer for Oshindonga (an Oshiwambo language of northern \
Namibia). For each Oshindonga word form below, using the example sentences, \
give a dictionary entry.

Return ONLY a JSON array, one object per word, in the same order:
{{"form": "<the form>", "lemma": "<dictionary form or stem>", \
"pos": "noun|verb|adjective|adverb|pronoun|preposition|conjunction|particle|numeral|interjection|other", \
"noun_class": <number or null>, "senses": ["<English, at most 6 words>", ...], \
"confidence": "high|medium|low"}}

Give 1–3 senses, most common first; for grammatical words describe the \
function (e.g. "is (copula)", "that (complementiser)"). Use "low" confidence \
when unsure — do not guess silently.

{words}"""


def frequencies():
    low, cap, total = Counter(), Counter(), 0
    contexts: dict[str, list[str]] = defaultdict(list)
    for r in map(json.loads, MONO.open()):
        if r["lid"] != "Oshindonga":
            continue
        toks = WORD.findall(r["text"])
        for k, t in enumerate(toks):
            w = t.lower()
            low[w] += 1
            total += 1
            if k > 0 and t[0].isupper():
                cap[w] += 1
            if 5 <= len(toks) <= 25 and len(contexts[w]) < 12:
                contexts[w].append(r["text"])
    proper = {w for w, c in low.items() if c >= 5 and cap[w] / c > 0.6}
    ranked = [w for w, _ in low.most_common() if w not in proper and len(w) > 1]
    return ranked, low, contexts


async def ask(words: list[str], contexts, model: str, effort: str, conc: int) -> list[dict]:
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key("OPENROUTER_API_KEY"))
    sem, out, cost = asyncio.Semaphore(conc), [], [0.0]

    async def one(batch: list[str]) -> None:
        block = "\n\n".join(f"{k + 1}. {w}\n" + "\n".join(f"   - {s}" for s in sorted(contexts[w], key=len)[:N_CONTEXT])
                            for k, w in enumerate(batch))
        async with sem:
            for wait in (0, 5, 20):
                await asyncio.sleep(wait)
                try:
                    r = await client.chat.completions.create(
                        model=model, temperature=0, max_tokens=12000,
                        messages=[{"role": "user", "content": PROMPT.format(words=block)}],
                        extra_body={"reasoning": {"effort": effort}, "usage": {"include": True}})
                    text = r.choices[0].message.content or ""
                    m = re.search(r"\[.*\]", text, re.S)
                    entries = json.loads(m.group(0)) if m else []
                    cost[0] += float(((r.usage.model_extra or {}) if r.usage else {}).get("cost") or 0)
                    break
                except Exception as exc:                    # noqa: BLE001
                    print(f"  retry: {exc}", file=sys.stderr)
            else:
                return
        by_form = {str(e.get("form", "")).lower(): e for e in entries if isinstance(e, dict)}
        for w in batch:
            if w in by_form:
                out.append({**by_form[w], "form": w, "model": model, "effort": effort})

    await asyncio.gather(*(one(words[i:i + BATCH]) for i in range(0, len(words), BATCH)))
    print(f"model cost {cost[0]:.2f} $ for {len(words)} words", file=sys.stderr)
    return out


def ncr_agreement(entries: list[dict]) -> dict:
    """Share of entries whose form is in Okalai's Ndonga noun lexicon and
    where one of the senses shares a content word with Okalai's English."""
    ncr = defaultdict(set)
    for split in ("train", "val", "test"):
        for line in (NCR / f"{split}_NCR_NDONGA.txt").read_text(encoding="utf-8").splitlines():
            p = [x.strip() for x in line.split("|||")]
            if len(p) == 5:
                ncr[p[3].lower()].add(p[0].lower())
    stop = {"a", "an", "the", "of", "to", "in", "or", "and", "s"}
    hits = [e for e in entries if e["form"] in ncr]
    agree = 0
    for e in hits:
        gem = {t for s in e.get("senses", []) for t in re.findall(r"[a-z]+", s.lower())} - stop
        okl = {t for en in ncr[e["form"]] for t in re.findall(r"[a-z]+", en)} - stop
        agree += bool(gem & okl or any(g.rstrip("s") in {o.rstrip("s") for o in okl} for g in gem))
    return {"overlap_with_okalai": len(hits), "agree": agree,
            "agree_pct": round(100 * agree / len(hits), 1) if hits else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("pilot", "build", "export"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--top", type=int, default=20000)
    ap.add_argument("--model", default="google/gemini-3.1-pro-preview")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / ("pilot.jsonl" if args.mode == "pilot" else "entries.jsonl")

    if args.mode == "export":
        rows = [json.loads(l) for l in (OUT / "entries.jsonl").open()]
        keep = ("form", "lemma", "pos", "noun_class", "senses", "confidence", "freq", "rank")
        with (OUT / "dictionary_ndo_en.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps({k: r.get(k) for k in keep}, ensure_ascii=False) + "\n")
        return 0

    ranked, freq, contexts = frequencies()
    if args.mode == "pilot":
        rng = random.Random(42)
        words = (rng.sample(ranked[:1000], int(args.n * 0.3)) + rng.sample(ranked[1000:5000], int(args.n * 0.3))
                 + rng.sample(ranked[5000:20000], args.n - 2 * int(args.n * 0.3)))
    else:
        words = ranked[: args.top]
    done = {json.loads(l)["form"] for l in work.open()} if work.exists() else set()
    todo = [w for w in words if w not in done]
    rank = {w: i + 1 for i, w in enumerate(ranked)}
    entries = asyncio.run(ask(todo, contexts, args.model, args.effort, args.concurrency))
    with work.open("a") as f:
        for e in entries:
            e |= {"freq": freq[e["form"]], "rank": rank.get(e["form"]),
                  "contexts": sorted(contexts[e["form"]], key=len)[:N_CONTEXT]}
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    rows = [json.loads(l) for l in work.open()]
    report = {"entries": len(rows), "requested": len(words),
              "confidence": dict(Counter(r.get("confidence") for r in rows)),
              "pos": dict(Counter(r.get("pos") for r in rows).most_common(8)), **ncr_agreement(rows)}
    for lo, hi in ((1, 1000), (1001, 5000), (5001, 20000)):
        band = [r for r in rows if r.get("rank") and lo <= r["rank"] <= hi]
        if band:
            report[f"low_conf_rank_{lo}_{hi}"] = round(100 * sum(r.get("confidence") == "low" for r in band) / len(band), 1)
    (OUT / f"{work.stem}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
