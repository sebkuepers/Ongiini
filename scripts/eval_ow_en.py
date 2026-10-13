#!/usr/bin/env python3
"""Does the model UNDERSTAND Oshiwambo? Translate the benchmark's human references into English (paper 3).

The en→ow benchmark measures writing; the experimental.ongiini.ai ratings (2026-10-13) showed
the bigger weakness is reading: greetings answered in the wrong form, questions misread, user
corrections ignored. This is the reverse direction on the same 423 items: the source is the
human Oshindonga / Oshikwanyama reference, the reference is the original English. chrF++ against
the English, per dialect and split, with a paired bootstrap CI against a baseline label.
Runs against any OpenAI-compatible vLLM (greedy, the same prompt for every label).

    python3 scripts/eval_ow_en.py generate --model gemma-4-12b --label base12b
    python3 scripts/eval_ow_en.py generate --model t4b --label T4b --server http://172.17.0.1:8201/v1
    python3 scripts/eval_ow_en.py score --base base12b --labels T4b C1r
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_scoring as E  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / os.environ.get("ONGIINI_EXP_DIR", "data/private/experiments") / "ow_en"
NAMES = {"oshindonga": "Oshindonga", "oshikwanyama": "Oshikwanyama"}
PROMPT = "Translate the following {lang} text into English. Reply with the English translation only.\n\n{text}"


def items_by_split() -> tuple[dict, dict[str, list[int]]]:
    items = E.load_items(ROOT / "data/private/export/data/eval_set.jsonl")
    splits = {s: sorted(i for i, r in items.items() if r["in_blind_split"] == (s == "blind")
                        and all((r.get(f"{d}_reference") or "").strip() for d in E.DIALECTS))
              for s in ("dev", "blind")}
    return items, splits


async def generate(args) -> None:
    from openai import AsyncOpenAI
    items, splits = items_by_split()
    client = AsyncOpenAI(base_url=args.server, api_key="none", timeout=600)
    sem = asyncio.Semaphore(args.concurrency)
    OUT.mkdir(parents=True, exist_ok=True)
    for d in E.DIALECTS:
        path = OUT / f"{args.label}_{d}.jsonl"
        done = {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()} if path.exists() else set()
        todo = [i for s in ("dev", "blind") for i in splits[s] if i not in done][: args.limit or None]
        errors = 0

        async def one(i: int):
            nonlocal errors
            src = items[i][f"{d}_reference"].strip()
            async with sem:
                try:
                    r = await client.chat.completions.create(
                        model=args.model, temperature=0, max_tokens=min(400, 40 + 3 * len(src.split())),
                        messages=[{"role": "user", "content": PROMPT.format(lang=NAMES[d], text=src)}])
                    text = r.choices[0].message.content or ""
                except Exception as e:  # noqa: BLE001 — counted, the item is redone on the next run
                    errors += 1
                    print(f"{d} {i}: {type(e).__name__}", flush=True)
                    return
            with path.open("a") as f:
                f.write(json.dumps({"id": i, "dialect": d, "model": args.model, "translation": text.strip()},
                                   ensure_ascii=False) + "\n")

        await asyncio.gather(*(one(i) for i in todo))
        print(f"{args.label} {d}: {len(todo) - errors} written, {errors} errors", flush=True)
        if errors:
            raise SystemExit(1)


def hyp(text: str) -> str:
    return E.normalise(text.strip().split("\n")[0].strip().strip('"'))


def score(args) -> None:
    items, splits = items_by_split()
    load = lambda lab, d: {r["id"]: r["translation"] for r in map(json.loads, (OUT / f"{lab}_{d}.jsonl").read_text().splitlines())}
    rep: dict = {}
    for lab in [args.base, *args.labels]:
        cells = {}
        for d in E.DIALECTS:
            hx, hb = load(lab, d), load(args.base, d)
            for s, ids in splits.items():
                refs = [E.normalise(items[i]["english"]) for i in ids]
                x = E.SystemStats.build([hyp(hx.get(i, "")) for i in ids], refs)
                cell = {"n": len(ids), "missing": sum(i not in hx for i in ids), "chrf++": round(E.corpus_scores(x)["chrf++"], 1)}
                if lab != args.base:
                    b = E.SystemStats.build([hyp(hb.get(i, "")) for i in ids], refs)
                    diffs = np.sort([E.score_from("chrf++", x.stats["chrf++"][ix].sum(0)) - E.score_from("chrf++", b.stats["chrf++"][ix].sum(0))
                                     for ix in E.bootstrap_indices(len(ids))])
                    cell |= {"delta": round(cell["chrf++"] - E.corpus_scores(b)["chrf++"], 1),
                             "ci95": [round(float(diffs[25]), 1), round(float(diffs[974]), 1)]}
                cells[f"{d}_{s}"] = cell
        rep[lab] = cells
    path = OUT / f"report_{'_'.join(args.labels)}_vs_{args.base}.json"
    path.write_text(json.dumps(rep, indent=2))
    for lab, c in rep.items():
        print(f"{lab:14s} " + "  ".join(f"{k.replace('oshi', '')} {v['chrf++']:5.1f}" + (f" ({v['delta']:+.1f} {v['ci95']})" if "delta" in v else "")
                                         for k, v in c.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True, help="served model name")
    g.add_argument("--label", required=True)
    g.add_argument("--server", default=os.environ.get("ONGIINI_EVAL_SERVER", "http://localhost:8200/v1"))
    g.add_argument("--concurrency", type=int, default=32)
    g.add_argument("--limit", type=int, default=0, help="first N items per dialect (tests)")
    s = sub.add_parser("score")
    s.add_argument("--base", required=True)
    s.add_argument("--labels", nargs="+", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        asyncio.run(generate(args))
    else:
        score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
