#!/usr/bin/env python3
"""Selection experiment (paper 3): 50k back-translated pairs chosen by grammar or by vocabulary.

Same pool (the 200k back-translated Ndonga pairs), same size, same 6,655
LAC/ASSAR pairs, same training — only *which* sentences differs:

* ``grammar``: prefer the constructions the newspaper corpus lacks and the
  benchmark/assistant needs (1st/2nd person, questions, imperatives,
  requests, present/modal/future, informal/chat register, active voice),
  scored from bt_labels.jsonl; ties and the remainder filled at random.
* ``vocab``: greedy multi-cover of the corpus dictionary — each dictionary
  form counts until it appears K times; rarer forms weigh more.

Compared with the random B50k of the learning curve.

    python3 scripts/build_sft_selected.py --strategy grammar --n 50000
    python3 scripts/build_sft_selected.py --strategy vocab --n 50000
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sft_bt import D, ok  # noqa: E402
from eval_prompts import PAPER_ZEROSHOT_ID, render  # noqa: E402

DICT = D.parent / "dictionary_v1/entries.jsonl"
WORD = re.compile(r"[A-Za-zÀ-ÿ]+(?:['’-][A-Za-zÀ-ÿ]+)*")
K_COVER = 10


def grammar_score(lab: dict) -> float:
    s = 0.0
    s += 2.0 * (str(lab.get("person")) in ("1", "2"))
    s += 2.0 * (lab.get("type") in ("question", "imperative", "request"))
    s += 1.0 * (lab.get("tense") in ("present", "modal", "future"))
    s += 1.0 * (lab.get("register") in ("informal", "chat"))
    s += 0.5 * (lab.get("passive") in (False, "false"))
    return s


def select_grammar(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    labels = {(l["doc"], l["i"]): l for l in map(json.loads, (D / "bt_labels.jsonl").open())}
    scored = [(grammar_score(labels.get((r["doc"], r["i"]), {})), rng.random(), r) for r in pool]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [r for _, _, r in scored[:n]]


def select_vocab(pool: list[dict], n: int) -> list[dict]:
    rank = {e["form"]: e["rank"] for e in map(json.loads, DICT.open()) if e.get("rank")}
    weight = {f: 1.0 + math.log(max(r, 300) / 300) for f, r in rank.items()}
    forms = [{w.lower() for w in WORD.findall(r["text"])} & rank.keys() for r in pool]
    count: Counter = Counter()

    def gain(k: int) -> float:
        return sum(weight[f] for f in forms[k] if count[f] < K_COVER)

    heap = [(-gain(k), k) for k in range(len(pool))]
    heapq.heapify(heap)
    chosen: list[int] = []
    while heap and len(chosen) < n:
        neg, k = heapq.heappop(heap)
        g = gain(k)                                          # lazy greedy: refresh stale gains
        if heap and g < -heap[0][0] - 1e-9:
            heapq.heappush(heap, (-g, k))
            continue
        chosen.append(k)
        for f in forms[k]:
            count[f] += 1
    return [pool[k] for k in chosen]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=("grammar", "vocab"), required=True)
    ap.add_argument("--n", type=int, default=50000)
    args = ap.parse_args(argv)
    rng = random.Random(42)
    pool = [r for r in map(json.loads, (D / "bt_ndo_dict.jsonl").open()) if ok(r)]
    pick = select_grammar(pool, args.n, rng) if args.strategy == "grammar" else select_vocab(pool, args.n)
    lac = [json.loads(l) for l in (D / "sft_A_parallel_ndo_train.jsonl").open()]
    tag = {"grammar": "G", "vocab": "V"}[args.strategy]
    out = D / f"sft_B{args.n // 1000}k{tag}_train.jsonl"
    with out.open("w") as f:
        for row in lac:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for r in pick:
            f.write(json.dumps({"messages": [
                {"role": "user", "content": render(PAPER_ZEROSHOT_ID, "oshindonga", r["en"])},
                {"role": "assistant", "content": r["text"]}],
                "source": f"bt:{r['source']}:{args.strategy}", "doc": r["doc"]}, ensure_ascii=False) + "\n")
    stats = {"pool": len(pool), "picked": len(pick), "file": out.name}
    if args.strategy == "grammar":
        labels = {(l["doc"], l["i"]): l for l in map(json.loads, (D / "bt_labels.jsonl").open())}
        sel = [labels.get((r["doc"], r["i"]), {}) for r in pick]
        for fld in ("person", "type", "register"):
            c = Counter(str(x.get(fld)).lower() for x in sel)
            stats[fld] = {k: round(100 * v / len(sel), 1) for k, v in c.most_common(6)}
    else:
        rank = {e["form"] for e in map(json.loads, DICT.open())}
        cov = Counter(w.lower() for r in pick for w in WORD.findall(r["text"]) if w.lower() in rank)
        stats["dictionary_forms_seen"] = len(cov)
        stats["forms_seen_10x"] = sum(v >= 10 for v in cov.values())
    (out.with_suffix(".stats.json")).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
