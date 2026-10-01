#!/usr/bin/env python3
"""Grammar test, level 1: do models know Oshiwambo noun classes?

Noun class drives agreement in Bantu languages (subject concord, adjectives,
possessives), so knowing a noun's class is a precondition for grammatical
sentences. Items: the test split of Okalai's MIT-licensed DictionaryNCR
(github.com/okalai-ai/moimoe; Ndonga 1,500, Kwanyama 300), each an
Oshiwambo noun with its English gloss and class number.

The class is often guessable from the noun prefix, so every model is
compared with a prefix rule learned on the train split (majority class
of the longest prefix seen ≥ 5 times). Only accuracy above that rule is
knowledge beyond the surface form.

Knowledge *about* grammar, not its use in sentences (that is level 2–3,
see the grammar-test notes). Results stay in data/private/experiments/.

    OPENROUTER_API_KEY=… ~/.venvs/ongiini-eval/bin/python scripts/exp_noun_class.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_dictionary_prompting import LEX_DIR, LEX_LANG, lexicon  # noqa: E402  (downloads the files)
from run_openrouter_baseline import api_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/private/experiments/noun_class"
NAMES = {"oshindonga": "Oshindonga", "oshikwanyama": "Oshikwanyama"}
MODELS = {  # label → (OpenRouter id, extra body)
    "gemma-4-26b": ("google/gemma-4-26b-a4b-it", {}),
    "deepseek-v4.1": ("deepseek/deepseek-v4.1-flash", {"reasoning": {"enabled": False}}),
    "claude-opus-5": ("anthropic/claude-opus-5", {}),
    "gemini-3.1-pro": ("google/gemini-3.1-pro-preview", {}),
}
BATCH = 25

PROMPT = """\
Below are {name} nouns (an Oshiwambo language of northern Namibia), each with \
its English meaning. For each noun, give its noun class number in the \
standard Bantu numbering used in Oshiwambo grammars (1, 2, 3, … 15).

Answer with one line per noun, exactly "<number>. <class>", nothing else.

{rows}"""


def split(dialect: str, name: str) -> list[dict]:
    rows = []
    for line in (LEX_DIR / f"{name}_NCR_{LEX_LANG[dialect]}.txt").read_text(encoding="utf-8").splitlines():
        p = [x.strip() for x in line.split("|||")]
        if len(p) == 5 and p[3] and p[4]:
            rows.append({"en": p[0], "word": p[3], "cls": p[4]})
    return rows


def prefix_rule(train: list[dict]):
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in train:
        w = r["word"].lower()
        for k in range(1, 6):
            counts[w[:k]][r["cls"]] += 1
    overall = Counter(r["cls"] for r in train).most_common(1)[0][0]

    def predict(word: str) -> str:
        w = word.lower()
        for k in range(5, 0, -1):
            c = counts.get(w[:k])
            if c and sum(c.values()) >= 5:
                return c.most_common(1)[0][0]
        return overall
    return predict


async def ask(client, model: str, extra: dict, dialect: str, batch: list[dict]) -> dict[int, str]:
    rows = "\n".join(f"{i + 1}. {r['word']} ({r['en']})" for i, r in enumerate(batch))
    msg = PROMPT.format(name=NAMES[dialect], rows=rows)
    for wait in (0, 5, 20, 60):
        await asyncio.sleep(wait)
        try:
            r = await client.chat.completions.create(model=model, temperature=0, max_tokens=8192,
                                                     messages=[{"role": "user", "content": msg}],
                                                     extra_body=extra)
            text = (r.choices[0].message.content or "") if r.choices else ""
            got = {int(m[1]) - 1: m[2] for m in re.finditer(r"(?m)^\s*(\d+)\.\s*(?:class\s*)?(\d+)", text)}
            if got:
                return got
        except Exception as exc:                      # noqa: BLE001
            print(f"  retry {model}: {exc}", file=sys.stderr)
    return {}


async def run_model(label: str, data: dict[str, list[dict]], conc: int) -> None:
    model, extra = MODELS[label]
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key("OPENROUTER_API_KEY"))
    sem = asyncio.Semaphore(conc)
    out = OUT / f"{label}.jsonl"
    done = {(json.loads(l)["dialect"], json.loads(l)["i"]) for l in out.read_text().splitlines()} if out.exists() else set()

    async def one(d: str, start: int) -> None:
        batch = data[d][start:start + BATCH]
        if all((d, start + k) in done for k in range(len(batch))):
            return
        async with sem:
            got = await ask(client, model, extra, d, batch)
        with out.open("a") as f:
            for k, r in enumerate(batch):
                f.write(json.dumps({"dialect": d, "i": start + k, "word": r["word"], "gold": r["cls"],
                                    "pred": got.get(k)}, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(d, s) for d in data for s in range(0, len(data[d]), BATCH)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    data, rules = {}, {}
    for d in NAMES:
        lexicon(d)                                    # make sure the files are there
        data[d] = split(d, "test")
        rules[d] = prefix_rule(split(d, "train") + split(d, "val"))
    labels = [m for m in args.models.split(",") if m]
    if not args.score_only:
        async def all_models():
            await asyncio.gather(*(run_model(m, data, args.concurrency) for m in labels))
        asyncio.run(all_models())

    report = {}
    for d in NAMES:
        rows = data[d]
        base = sum(rules[d](r["word"]) == r["cls"] for r in rows) / len(rows)
        hard = [k for k, r in enumerate(rows) if rules[d](r["word"]) != r["cls"]]
        rep = {"n": len(rows), "prefix_rule": round(100 * base, 1), "n_prefix_rule_wrong": len(hard),
               "majority_class": round(100 * Counter(r["cls"] for r in rows).most_common(1)[0][1] / len(rows), 1)}
        for m in labels:
            p = OUT / f"{m}.jsonl"
            if not p.exists():
                continue
            pred = {}
            for l in p.read_text().splitlines():
                x = json.loads(l)
                if x["dialect"] == d:
                    pred[x["i"]] = x["pred"]
            ok = [pred.get(k) == r["cls"] for k, r in enumerate(rows)]
            rep[m] = {"accuracy": round(100 * sum(ok) / len(rows), 1),
                      "on_prefix_rule_wrong": round(100 * sum(ok[k] for k in hard) / max(1, len(hard)), 1),
                      "unanswered": sum(pred.get(k) is None for k in range(len(rows)))}
        report[d] = rep
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
