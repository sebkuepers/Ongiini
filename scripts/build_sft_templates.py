#!/usr/bin/env python3
"""Many prompt templates for the translation SFT data (paper 3, goal candidate T1).

One fixed template teaches the model the FORMAT instead of the task ("format
specialization", Wang et al. 2022, ProMoT): our r64 adapters answered prompts
seen in training but produced "the the the" for everything else. Tower (2024)
used many zero- and few-shot templates per task plus general instruction data.

This re-renders translation pairs (built with the benchmark template) with ~30
phrasings — instructions, chat-style questions, a system prompt, few-shot
examples, different names for the language — keeping the benchmark template for
--keep-share of the rows, and mixes in general replay rows (each used once).

    python3 scripts/build_sft_templates.py --pairs data/private/corpus/osheng_v1/sft_B50kG_train.jsonl \\
        --n-pairs 12000 --replay data/private/corpus/replay_v1/replay.jsonl data/private/corpus/replay_v2/replay.jsonl \\
        --out data/private/corpus/osheng_v1/sft_T1_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

NAMES = {"oshindonga": ["Oshindonga", "Ndonga", "Oshiwambo (Ndonga)", "Oshindonga (Oshiwambo)"],
         "oshikwanyama": ["Oshikwanyama", "Kwanyama", "Oshiwambo (Kwanyama)", "Oshikwanyama (Oshiwambo)"]}
ZERO = [
    "Translate into {L}: {s}",
    "Translate the following English text into {L}.\n\n{s}",
    "Please translate this into {L}:\n{s}",
    "How do you say this in {L}? \"{s}\"",
    "Can you put this into {L} for me? {s}",
    "{s}\n\nTranslate the sentence above into {L}. Reply with the translation only.",
    "English: {s}\n{L}:",
    "I need this in {L}, please: {s}",
    "Write this in {L}: {s}",
    "My grandmother only reads {L}. How would I write this for her? {s}",
    "Translate to {L} (no explanation): {s}",
    "What is the {L} translation of: \"{s}\"",
    "Give me the {L} version of this message:\n\n{s}",
    "EN → {L}\n{s}",
    "Kindly render the following in {L}.\n{s}",
    "Translate from English to {L}:\n\"{s}\"",
    "Help me say this in {L}: {s}",
    "{L} translation please:\n{s}",
    "Translate this sentence to {L} and only output the result: {s}",
    "Could you translate \"{s}\" into {L}?",
]
SYSTEMS = ["You are a helpful assistant for people in Namibia.",
           "You are a translator for English and Oshiwambo.",
           "You are Ongiini, a friendly AI assistant."]
FEWSHOT = "Translate English into {L}.\n\n{shots}English: {s}\n{L}:"


def parse(row: dict):
    """English source and dialect from a pair rendered with the benchmark template."""
    user, answer = row["messages"][0]["content"], row["messages"][-1]["content"]
    m = re.search(r"English: (.*)\n(Oshindonga|Oshikwanyama):\s*$", user, re.S)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).lower(), answer.strip(), row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--n-pairs", type=int, default=12000)
    ap.add_argument("--replay", nargs="*", default=[])
    ap.add_argument("--keep-share", type=float, default=0.25, help="rows that keep the benchmark template")
    ap.add_argument("--fewshot-share", type=float, default=0.1)
    ap.add_argument("--system-share", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    parsed = [p for p in (parse(json.loads(l)) for l in open(args.pairs) if l.strip()) if p]
    rng.shuffle(parsed)
    pool, rows, kinds = parsed[args.n_pairs:], [], {"benchmark": 0, "zero": 0, "fewshot": 0, "system": 0}
    for src, dia, ans, orig in parsed[: args.n_pairs]:
        L = rng.choice(NAMES[dia])
        r = rng.random()
        if r < args.keep_share:
            msgs, k = orig["messages"], "benchmark"
        elif r < args.keep_share + args.fewshot_share and pool:
            shots = "".join(f"English: {s}\n{L}: {a}\n\n" for s, _, a, _ in rng.sample(pool, rng.choice([1, 2, 3])))
            msgs, k = [{"role": "user", "content": FEWSHOT.format(L=L, shots=shots, s=src)}], "fewshot"
            msgs = msgs + [{"role": "assistant", "content": ans}]
        else:
            msgs = [{"role": "user", "content": rng.choice(ZERO).format(L=L, s=src)},
                    {"role": "assistant", "content": ans}]
            k = "zero"
            if rng.random() < args.system_share / max(1e-9, 1 - args.keep_share - args.fewshot_share):
                msgs = [{"role": "system", "content": rng.choice(SYSTEMS)}] + msgs
                k = "system"
        kinds[k] += 1
        rows.append({"messages": msgs})
    replay, seen = [], set()
    for path in args.replay:
        for l in open(path):
            if not l.strip():
                continue
            m = json.loads(l)["messages"]
            key = m[0]["content"][:200]
            if key not in seen:  # each general prompt once: repeated replay gets memorised
                seen.add(key)
                replay.append({"messages": m})
    rows += replay
    rng.shuffle(rows)
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps({"translation": args.n_pairs, "templates": kinds, "replay_unique": len(replay),
                      "replay_share": round(len(replay) / len(rows), 3), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
