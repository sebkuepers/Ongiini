#!/usr/bin/env python3
"""Inference-time levers for an adapter, no retraining (paper 3, experiment I1).

Through the eval vLLM (deploy/eval/serve_12b_lora.sh) on the Ndonga dev split:
  greedy          the benchmark prompt, temperature 0 (reference point)
  mbr             N sampled candidates + greedy; pick the candidate with the
                  highest mean chrF++ against the others (MBR with chrF as utility)
  fewshot         k most similar training pairs (word overlap with the English
                  source) as examples in the few-shot template T1/T2 were trained on
  fewshot_mbr     both
Example pool: LAC/ASSAR + back-translated pairs (B50kG) and the Viljoen dictionary
example sentences — never the eval set. Paired bootstrap vs greedy.

    python3 scripts/exp_inference_tricks.py --model t2 --out data/private/experiments/inference_tricks
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import sacrebleu
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_prompts import PAPER_ZEROSHOT_ID, render  # noqa: E402

CHRF = sacrebleu.metrics.CHRF(word_order=2)
STOP = set("the a an of to in on and or is are was were be for with at by it this that i you he she we they my your".split())


def words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", t.lower()) if w not in STOP and len(w) > 2}


def pool() -> list[tuple[str, str]]:
    pairs = []
    for l in open("data/private/corpus/osheng_v1/sft_B50kG_train.jsonl"):
        m = json.loads(l)["messages"]
        s = re.search(r"English: (.*)\nOshindonga:\s*$", m[0]["content"], re.S)
        if s:
            pairs.append((s.group(1).strip(), m[-1]["content"].strip()))
    for l in open("data/private/corpora/dictionaries/viljoen-1984/examples.jsonl"):
        x = json.loads(l)
        pairs.append((x["en"], x["ow"]))
    return pairs


class Retriever:
    """IDF-weighted word overlap — small and deterministic."""

    def __init__(self, pairs):
        self.pairs = pairs
        self.sets = [words(en) for en, _ in pairs]
        df = Counter(w for s in self.sets for w in s)
        self.idf = {w: math.log(len(pairs) / (1 + c)) for w, c in df.items()}
        self.index: dict[str, list[int]] = {}
        for i, s in enumerate(self.sets):
            for w in s:
                self.index.setdefault(w, []).append(i)

    def top(self, en: str, k: int):
        q = words(en)
        score = Counter()
        for w in q:
            for i in self.index.get(w, [])[:5000]:
                score[i] += self.idf.get(w, 0)
        return [self.pairs[i] for i, _ in score.most_common(k)]


def fewshot_prompt(en: str, shots) -> str:
    ex = "".join(f"English: {s}\nOshindonga: {t}\n\n" for s, t in shots)
    return f"Translate English into Oshindonga.\n\n{ex}English: {en}\nOshindonga:"


def mbr(cands: list[str]) -> str:
    cands = [c for c in cands if c] or [""]
    best, best_u = cands[0], -1.0
    for i, c in enumerate(cands):
        others = [o for j, o in enumerate(cands) if j != i] or [c]
        u = sum(CHRF.sentence_score(c, [o]).score for o in others) / len(others)
        if u > best_u:
            best, best_u = c, u
    return best


async def main_async(args) -> None:
    ev = {json.loads(l)["id"]: json.loads(l) for l in open("data/private/export/data/eval_set.jsonl")}
    ids = json.load(open("data/private/export/data/referenced_ids.json"))
    src = {int(r["id"]): r["english"].strip() for r in csv.DictReader(open("data/oshiwambo_eval_v3.tsv"), delimiter="\t")}
    dev = [i for i in ids if ev[i].get("oshindonga_reference") and not ev[i]["in_blind_split"]]
    if args.limit:
        dev = dev[: args.limit]
    refs = [ev[i]["oshindonga_reference"] for i in dev]
    evtexts = {ev[i]["english"].strip().lower() for i in ev}
    ret = Retriever([p for p in pool() if p[0].strip().lower() not in evtexts])
    client = AsyncOpenAI(base_url=args.server, api_key="x", timeout=600)
    sem = asyncio.Semaphore(args.concurrency)

    async def gen(prompt: str, n: int, temp: float, max_new: int) -> list[str]:
        async with sem:
            r = await client.chat.completions.create(
                model=args.model, temperature=temp, n=n, max_tokens=max_new, top_p=0.95 if temp else 1.0,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return [(c.message.content or "").strip().split("\n")[0].strip() for c in r.choices]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    results = {}
    for cond in args.conditions:
        async def item(i):
            en = src[i]
            max_new = min(400, 20 + 3 * len(en.split()))
            p = fewshot_prompt(en, ret.top(en, args.k)) if cond.startswith("fewshot") else render(PAPER_ZEROSHOT_ID, "oshindonga", en)
            greedy = (await gen(p, 1, 0, max_new))[0]
            if cond.endswith("mbr"):
                return mbr([greedy] + await gen(p, args.n, args.temp, max_new))
            return greedy
        hyp = await asyncio.gather(*(item(i) for i in dev))
        results[cond] = hyp
        with (out / f"{args.model}_{cond}.jsonl").open("w") as f:
            for i, h in zip(dev, hyp):
                f.write(json.dumps({"id": i, "translation": h}, ensure_ascii=False) + "\n")
        print(f"{cond}: chrF++ {CHRF.corpus_score(hyp, [refs]).score:.1f}", flush=True)
    rep, rng = {"model": args.model, "items": len(dev), "n": args.n, "k": args.k}, random.Random(1)
    base = results.get("greedy")
    for cond, hyp in results.items():
        rep[cond] = {"chrf": round(CHRF.corpus_score(hyp, [refs]).score, 2)}
        if base is not None and cond != "greedy":
            ds = []
            for _ in range(1000):
                s = [rng.randrange(len(dev)) for _ in dev]
                ds.append(CHRF.corpus_score([hyp[k] for k in s], [[refs[k] for k in s]]).score
                          - CHRF.corpus_score([base[k] for k in s], [[refs[k] for k in s]]).score)
            ds.sort()
            rep[cond]["delta_vs_greedy"] = round(rep[cond]["chrf"] - CHRF.corpus_score(base, [refs]).score, 2)
            rep[cond]["ci95"] = [round(ds[25], 2), round(ds[974], 2)]
    (out / f"report_{args.model}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="t2")
    ap.add_argument("--server", default="http://localhost:8200/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", nargs="*", default=["greedy", "mbr", "fewshot", "fewshot_mbr"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args(argv)
    asyncio.run(main_async(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
