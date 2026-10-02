#!/usr/bin/env python3
"""Back-translate Oshiwambo corpus sentences into English (paper 3, experiment B).

Samples sentences from data/private/corpus/osheng_v1/mono.jsonl (seeded),
translates them Oshiwambo→English with an OpenRouter model (default
DeepSeek V4.1 Flash, MIT-licensed weights, reasoning off), and optionally a
second, local teacher (Meyabase ng-en) for agreement filtering. Resumable;
each row keeps source, document, LID label, both English versions and the
agreement score.

    OPENROUTER_API_KEY=… ~/.venvs/ongiini-eval/bin/python scripts/backtranslate_corpus.py \\
        --n 1000 --out data/private/corpus/osheng_v1/bt_pilot1k.jsonl --second-teacher
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_scoring as E  # noqa: E402
from run_openrouter_baseline import api_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MONO = ROOT / "data/private/corpus/osheng_v1/mono.jsonl"
PROMPT = ("Translate the following {name} sentence (an Oshiwambo language of northern Namibia) "
          "into natural English. Only output the English translation.\n\n{name}: {text}\nEnglish:")
NAMES = {"Oshindonga": "Oshindonga", "Oshikwanyama": "Oshikwanyama"}
PROMPT_GLOSSARY = (
    "Translate the following {name} sentence (an Oshiwambo language of northern Namibia) into natural English.\n\n"
    "Glossary of words in the sentence, from a corpus dictionary (may be incomplete or list several senses; "
    "pick the sense that fits the sentence, ignore entries that do not fit):\n{glossary}\n\n"
    "Only output the English translation.\n\n{name}: {text}\nEnglish:")
SKIP_TOP = 300            # the most frequent forms are function words the teacher already knows
MAX_ENTRIES = 12


def load_dictionary(path: Path) -> dict[str, dict]:
    return {e["form"]: e for e in map(json.loads, path.open()) if e.get("senses")}


def glossary_for(text: str, dictionary: dict[str, dict]) -> str:
    import re
    seen, lines = set(), []
    words = re.findall(r"[A-Za-zÀ-ÿ]+(?:['’-][A-Za-zÀ-ÿ]+)*", text)
    hits = [dictionary[w.lower()] for w in words if w.lower() in dictionary]
    hits = [e for e in hits if (e.get("rank") or 0) > SKIP_TOP]
    for e in sorted(hits, key=lambda e: -(e.get("rank") or 0)):            # rarest first
        if e["form"] in seen:
            continue
        seen.add(e["form"])
        pos = e.get("pos") or ""
        lines.append(f"- {e['form']} ({pos}{', lemma ' + e['lemma'] if e.get('lemma') and e['lemma'] != e['form'] else ''}): "
                     + "; ".join(e["senses"][:3]))
        if len(lines) >= MAX_ENTRIES:
            break
    return "\n".join(lines)


def sample(n: int, dialect: str, seed: int) -> list[dict]:
    rows = [r for r in map(json.loads, MONO.open()) if r["lid"] == dialect]
    random.Random(seed).shuffle(rows)
    return rows[:n]


async def teacher(rows: list[dict], out: Path, model: str, conc: int, reasoning: str = "off",
                  dictionary: dict | None = None) -> float:
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key("OPENROUTER_API_KEY"))
    sem, lock, cost = asyncio.Semaphore(conc), asyncio.Lock(), [0.0]

    def prompt_for(r: dict) -> str:
        g = glossary_for(r["text"], dictionary) if dictionary else ""
        if g:
            return PROMPT_GLOSSARY.format(name=NAMES[r["lid"]], text=r["text"], glossary=g)
        return PROMPT.format(name=NAMES[r["lid"]], text=r["text"])

    async def one(r: dict) -> None:
        async with sem:
            for wait in (0, 5, 20, 60):
                await asyncio.sleep(wait)
                try:
                    resp = await client.chat.completions.create(
                        model=model, temperature=0, seed=42,
                        max_tokens=512 if reasoning == "off" else 8192,
                        messages=[{"role": "user", "content": prompt_for(r)}],
                        extra_body={"usage": {"include": True}, **(
                            {} if reasoning == "default" else {"reasoning": {"enabled": reasoning == "on"}})})
                    if resp.choices:
                        break
                except Exception as exc:                    # noqa: BLE001
                    print(f"  retry: {exc}", file=sys.stderr)
            else:
                return
            en = (resp.choices[0].message.content or "").strip().split("\n")[0].strip().strip('"')
            cost[0] += float(((resp.usage.model_extra or {}) if resp.usage else {}).get("cost") or 0)
        async with lock:
            with out.open("a") as f:
                f.write(json.dumps({**r, "en": en, "teacher": model}, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(r) for r in rows))
    return cost[0]


def second_teacher(out: Path, batch: int = 32) -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    rows = [json.loads(l) for l in out.open()]
    todo = [i for i, r in enumerate(rows) if "en2" not in r and r["lid"] == "Oshindonga"]
    if not todo:
        return
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("meyabase/ng-en-translation")
    m = AutoModelForSeq2SeqLM.from_pretrained("meyabase/ng-en-translation").to(dev).eval()
    for b in range(0, len(todo), batch):
        idx = todo[b: b + batch]
        enc = tok([rows[i]["text"] for i in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(dev)
        with torch.no_grad():
            g = m.generate(**enc, num_beams=2, max_new_tokens=256)
        for i, t in zip(idx, tok.batch_decode(g, skip_special_tokens=True)):
            rows[i]["en2"] = t
            rows[i]["agree_chrf"] = round(E.METRICS["chrf"].sentence_score(rows[i]["en"], [t]).score, 1)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--dialect", default="Oshindonga")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--second-teacher", action="store_true")
    ap.add_argument("--reasoning", choices=("off", "on", "default"), default="off")
    ap.add_argument("--dictionary", type=Path, help="corpus dictionary JSONL for per-sentence glossaries")
    args = ap.parse_args(argv)
    rows = sample(args.n, args.dialect, args.seed)
    done = {(json.loads(l)["doc"], json.loads(l)["i"]) for l in args.out.open()} if args.out.exists() else set()
    todo = [r for r in rows if (r["doc"], r["i"]) not in done]
    print(f"{len(todo)} of {len(rows)} to translate", file=sys.stderr)
    dictionary = load_dictionary(args.dictionary) if args.dictionary else None
    cost = asyncio.run(teacher(todo, args.out, args.model, args.concurrency, args.reasoning, dictionary))
    print(f"teacher cost {cost:.3f} $", file=sys.stderr)
    if args.second_teacher:
        second_teacher(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
