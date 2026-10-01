#!/usr/bin/env python3
"""Paper-3 experiment 1: does a glossary in the prompt help Gemma 4?

For every development item with a reference, the English nouns that occur
in Okalai's MIT-licensed noun lexicon (github.com/okalai-ai/moimoe,
data/DictionaryNCR) are looked up and listed in the prompt above the
sentence; everything else is the paper's zero-shot template and the
baseline decoding. Scored against the existing baseline run of the same
model on the same items: chrF++ with paired bootstrap CI of the
difference and approximate randomisation, overall and split by whether
the item got any glossary entry.

Diagnostic, development split only — never the blind split, and the
lexicon is never tuned on the eval set.

    OPENROUTER_API_KEY=… ~/.venvs/ongiini-eval/bin/python scripts/exp_dictionary_prompting.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_scoring as E  # noqa: E402
from eval_prompts import DIALECT_NAMES, PAPER_ZEROSHOT, PAPER_ZEROSHOT_ID  # noqa: E402
from retry_derailed_baselines import is_derailed  # noqa: E402
from run_openrouter_baseline import MAX_ATTEMPTS, api_key, clean  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "data/private/export/data/eval_set.jsonl"
BASE = ROOT / "data/private/export/data/baselines"
LEX_DIR = ROOT / "data/private/lexicon/okalai_ncr"
OUT = ROOT / "data/private/experiments/dictionary"
LEX_URL = "https://raw.githubusercontent.com/okalai-ai/moimoe/main/data/DictionaryNCR/{L}/{s}_NCR_{L}.txt"
LEX_LANG = {"oshindonga": "NDONGA", "oshikwanyama": "KWANYAMA"}
TEMPLATE_ID = "exp-glossary-okalai-ncr-v1"
MAX_ENTRIES, MAX_PER_WORD = 12, 2


# ── lexicon ──────────────────────────────────────────────────────────

def lexicon(dialect: str) -> dict[str, list[tuple[str, str]]]:
    """English noun (lower case) → [(Oshiwambo word, noun class)], file order."""
    L = LEX_LANG[dialect]
    LEX_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[tuple[str, str]]] = {}
    for split in ("train", "val", "test"):
        p = LEX_DIR / f"{split}_NCR_{L}.txt"
        if not p.exists():
            urllib.request.urlretrieve(LEX_URL.format(L=L, s=split), p)
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = [x.strip() for x in line.split("|||")]
            if len(parts) != 5 or not parts[0] or not parts[3]:
                continue
            en, _, _, word, cls = parts
            entries = out.setdefault(en.lower(), [])
            if (word, cls) not in entries:
                entries.append((word, cls))
    return out


def _variants(tok: str) -> list[str]:
    """The token and its likely singular forms (the lexicon lists singular nouns)."""
    v = [tok]
    if tok.endswith("ies") and len(tok) > 4:
        v.append(tok[:-3] + "y")
    if tok.endswith("es") and len(tok) > 3:
        v.append(tok[:-2])
    if tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
        v.append(tok[:-1])
    return v


def glossary(english: str, lex: dict[str, list[tuple[str, str]]]) -> list[tuple[str, str, str]]:
    """(English, Oshiwambo, class) for nouns of the sentence found in the
    lexicon — two-word and hyphenated compounds first, then single words."""
    toks = re.findall(r"[a-z]+(?:[-'][a-z]+)*", english.lower().replace("’", "'"))
    toks = [t.removesuffix("'s") for t in toks]
    found: list[tuple[str, str, str]] = []
    used: set[int] = set()
    for i in range(len(toks) - 1):                       # compounds
        for key in (f"{toks[i]} {toks[i + 1]}", f"{toks[i]}-{toks[i + 1]}"):
            for v in _variants(key):
                if v in lex:
                    found += [(v, w, c) for w, c in lex[v][:MAX_PER_WORD]]
                    used |= {i, i + 1}
                    break
    for i, t in enumerate(toks):
        if i in used or len(t) < 3:
            continue
        for v in _variants(t):
            if v in lex and all(f[0] != v for f in found):
                found += [(v, w, c) for w, c in lex[v][:MAX_PER_WORD]]
                break
    return found[:MAX_ENTRIES]


def prompt(dialect: str, english: str, gloss: list[tuple[str, str, str]]) -> str:
    name = DIALECT_NAMES[dialect]
    base = PAPER_ZEROSHOT.format(dialect=name, source=english)
    if not gloss:
        return base
    lines = "\n".join(f"- {en}: {w} (noun class {c})" for en, w, c in gloss)
    block = (f"Glossary of {name} nouns from a dictionary (use them where they fit; "
             f"inflect them as the sentence needs):\n{lines}\n\n")
    return base.replace("English: ", block + "English: ", 1)


# ── run ──────────────────────────────────────────────────────────────

async def translate(args, jobs: list[dict]) -> None:
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key("OPENROUTER_API_KEY"))
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    body = {} if args.reasoning == "default" else {"reasoning": {"enabled": args.reasoning == "on"}}

    async def one(job: dict) -> None:
        async with sem:
            text, raw, attempts = "", "", 0
            for attempts in range(1, MAX_ATTEMPTS + 1):
                r = None
                for wait in (0, 5, 15, 30, 60, 120):
                    await asyncio.sleep(wait)
                    try:
                        r = await client.chat.completions.create(
                            model=args.model, max_tokens=args.max_tokens, temperature=0, seed=42,
                            messages=[{"role": "user", "content": job["prompt"]}],
                            extra_body=body)
                        if r.choices:
                            break
                    except Exception as exc:              # noqa: BLE001
                        print(f"  retry {job['id']} {job['dialect']}: {exc}", file=sys.stderr)
                    r = None
                if r is None:
                    print(f"  error {job['id']} {job['dialect']}", file=sys.stderr)
                    return
                msg = r.choices[0].message
                raw = msg.content or ""
                thinking = (msg.model_extra or {}).get("reasoning") or ""
                finish = r.choices[0].finish_reason
                text = clean(raw, job["dialect"])
                if text and not is_derailed(job["english"], text):
                    break
        rec = {"id": job["id"], "dialect": job["dialect"], "model_id": args.model,
               "prompt_template_id": TEMPLATE_ID if job["gloss"] else PAPER_ZEROSHOT_ID, "translation": text, "raw_response": raw,
               "attempts": attempts, "finish_reason": finish, "glossary": job["gloss"],
               "reasoning_setting": args.reasoning, "max_tokens": args.max_tokens}
        async with lock:
            with (OUT / f"{args.label}_{job['dialect']}.jsonl").open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if thinking:
                with (OUT / f"{args.label}_{job['dialect']}.reasoning.jsonl").open("a") as f:
                    f.write(json.dumps({"id": job["id"], "dialect": job["dialect"], "english": job["english"],
                                        "glossary": job["gloss"], "reasoning": thinking},
                                       ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(j) for j in jobs))


# ── score ────────────────────────────────────────────────────────────

def compare(items: dict, dialect: str, ids: list[int], base: dict[int, str], exp: dict[int, str]) -> dict:
    refs = [E.normalise(items[i][f"{dialect}_reference"]) for i in ids]
    a = E.SystemStats.build([E.clean_hypothesis(base[i], dialect) for i in ids], refs)
    b = E.SystemStats.build([E.clean_hypothesis(exp[i], dialect) for i in ids], refs)
    sa, sb = E.corpus_scores(a)["chrf++"], E.corpus_scores(b)["chrf++"]
    boot = E.bootstrap_indices(len(ids))
    diffs = np.sort([E.score_from("chrf++", b.stats["chrf++"][ix].sum(0)) -
                     E.score_from("chrf++", a.stats["chrf++"][ix].sum(0)) for ix in boot])
    return {"n": len(ids), "baseline": round(sa, 1), "glossary": round(sb, 1),
            "delta": round(sb - sa, 1),
            "delta_ci95": [round(float(diffs[25]), 1), round(float(diffs[974]), 1)],
            "p_ar": round(E.approx_randomisation(a, b), 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-26b-a4b-it")
    ap.add_argument("--label", default="gemma-glossary", help="output stem in data/private/experiments/dictionary")
    ap.add_argument("--baseline", default="gemma-4-26b-api",
                    help="what to compare with: a stem in the experiment dir, else in baselines/")
    ap.add_argument("--reasoning", choices=("default", "on", "off"), default="default")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--no-glossary", action="store_true", help="same run without the glossary (a fresh baseline)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args(argv)

    items = E.load_items(EVAL_SET)
    OUT.mkdir(parents=True, exist_ok=True)
    plan, report = {}, {"template_id": TEMPLATE_ID, "lexicon": "okalai-ai/moimoe DictionaryNCR (MIT)",
                        "model": args.model, "baseline": args.baseline, "dialects": {}}
    for d in E.DIALECTS:
        lex = lexicon(d)
        ids = sorted(i for i, r in items.items() if not r["in_blind_split"]
                     and (r.get(f"{d}_reference") or "").strip())
        gl = {i: glossary(items[i]["english"], lex) for i in ids}          # for the split, always
        run_gl = {i: [] for i in ids} if args.no_glossary else gl
        plan[d] = (ids, gl, run_gl)
        report["dialects"][d] = {"lexicon_entries": sum(map(len, lex.values())),
                                 "items_with_glossary": sum(bool(g) for g in gl.values()),
                                 "mean_entries": round(float(np.mean([len(g) for g in gl.values()])), 2)}
    if not args.score_only:
        jobs = []
        for d, (ids, gl, run_gl) in plan.items():
            p = OUT / f"{args.label}_{d}.jsonl"
            done = {json.loads(l)["id"] for l in p.read_text().splitlines() if l.strip()} if p.exists() else set()
            jobs += [{"id": i, "dialect": d, "english": items[i]["english"].strip(), "gloss": run_gl[i],
                      "prompt": prompt(d, items[i]["english"].strip(), run_gl[i])}
                     for i in ids if i not in done]
        print(f"{len(jobs)} translations to run", file=sys.stderr)
        asyncio.run(translate(args, jobs))

    def outputs(stem: str, d: str) -> dict[int, str]:
        p = OUT / f"{stem}_{d}.jsonl"
        p = p if p.exists() else BASE / f"{stem}_{d}.jsonl"
        return {int(r["id"]): r.get("translation") or "" for r in E.load_jsonl(p)}

    report |= {"label": args.label, "reasoning": args.reasoning, "max_tokens": args.max_tokens,
               "glossary": not args.no_glossary}
    for d, (ids, gl, _) in plan.items():
        base, exp = outputs(args.baseline, d), outputs(args.label, d)
        ids = [i for i in ids if i in base]
        ids = [i for i in ids if i in exp]
        with_g = [i for i in ids if gl[i]]
        without = [i for i in ids if not gl[i]]
        rd = report["dialects"][d]
        rd["all"] = compare(items, d, ids, base, exp)
        if len(with_g) >= 20:
            rd["with_glossary"] = compare(items, d, with_g, base, exp)
        if len(without) >= 20:
            rd["without_glossary"] = compare(items, d, without, base, exp)
    (OUT / f"report_{args.label}_vs_{args.baseline}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
