#!/usr/bin/env python3
"""Consensus distillation data (paper 3, variant T7, 2026-10-19).

Choosing, per sentence, the output that agrees most with the other finished systems (MBR over
systems, chrF++ utility, reference-free) reached Ndonga dev 49.3 / blind 50.7 with six 12B
adapters, against 47.0 / 49.3 for the best single one (data/private/tmp/combo.py). Serving six
adapters per message is not practical, so we distil: the ensemble translates new English
sentences, the consensus output becomes the target of an ordinary training pair, and one model
is trained on those pairs next to the T5 mix.

    sources   English sentences from the English chat replay (prompts and answers, the eval's chat
              register), 5-35 words, no markup, deduplicated, no 6-gram shared with the eval set
    generate  each adapter translates them (benchmark zero-shot prompt, greedy) through the eval
              vLLM (deploy/eval/serve_12b_lora.sh, one adapter name per system); resumable
    select    consensus pick per sentence plus its agreement (mean chrF++ against the others);
              pairs below --min-agreement are dropped (the systems disagree: unreliable target)

    python3 scripts/consensus_distill.py sources --n 30000
    python3 scripts/consensus_distill.py generate --models c1t4 cbrt4 t5 r64rep t4b c1rt4
    python3 scripts/consensus_distill.py select --models c1t4 cbrt4 t5 r64rep t4b c1rt4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

import sacrebleu
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sft_t4 import grams  # noqa: E402
from eval_prompts import PAPER_ZEROSHOT_ID, render  # noqa: E402

OUT = Path("data/private/corpus/consensus_v1")
CHRF = sacrebleu.metrics.CHRF(word_order=2)
REPLAY = Path("data/private/corpora/replay_en/gemma_chat_replay.jsonl")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def sources(args) -> None:
    ev = [json.loads(l) for l in open("data/private/export/data/eval_set.jsonl")]
    evg = set().union(*(grams(r["english"]) for r in ev if r.get("english")))
    seen, out = set(), []
    for l in open(REPLAY):
        for turn in re.split(r"<\|turn>(?:user|model)\n", json.loads(l)["text"]):
            turn = turn.replace("<turn|>", " ")
            for line in turn.split("\n"):
                if re.search(r"[*#`$|<>\[\]{}=_\\]|https?:|^\s*(\d+\.|-)\s", line):
                    continue  # markdown, maths, code, lists
                for s in SENT.split(line.strip()):
                    s = s.strip()
                    n = len(s.split())
                    if not (5 <= n <= 35) or not s[0].isupper() or s[-1] not in ".!?" or not s.isascii():
                        continue
                    k = s.lower()
                    if k in seen or grams(s) & evg:
                        continue
                    seen.add(k)
                    out.append(s)
    random.Random(args.seed).shuffle(out)
    out = out[: args.n]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sources.jsonl").write_text("".join(json.dumps({"i": i, "en": s}) + "\n" for i, s in enumerate(out)))
    print(json.dumps({"candidates": len(seen), "sources": len(out)}))


async def generate(args) -> None:
    src = [json.loads(l) for l in open(OUT / "sources.jsonl")]
    client = AsyncOpenAI(base_url=args.server, api_key="none")
    sem = asyncio.Semaphore(args.concurrency)
    for m in args.models:
        path = OUT / f"hyp_{m}.jsonl"
        done = {json.loads(l)["i"] for l in open(path)} if path.exists() else set()
        todo = [r for r in src if r["i"] not in done]

        async def one(r, f):
            async with sem:
                for attempt in range(4):
                    try:
                        c = await client.chat.completions.create(
                            model=m, temperature=0, max_tokens=256,
                            messages=[{"role": "user", "content": render(PAPER_ZEROSHOT_ID, "oshindonga", r["en"])}])
                        t = (c.choices[0].message.content or "").strip().split("\n")[0].strip()
                        f.write(json.dumps({"i": r["i"], "ow": t}, ensure_ascii=False) + "\n")
                        return
                    except Exception:  # noqa: BLE001 — server restart or overload: retry, then leave for resume
                        await asyncio.sleep(5 * (attempt + 1))

        with path.open("a") as f:
            await asyncio.gather(*(one(r, f) for r in todo))
        print(f"{m}: {len(done) + len(todo)} requested, file {sum(1 for _ in open(path))} rows", flush=True)


def select(args) -> None:
    src = {json.loads(l)["i"]: json.loads(l)["en"] for l in open(OUT / "sources.jsonl")}
    hyp = {m: {json.loads(l)["i"]: json.loads(l)["ow"] for l in open(OUT / f"hyp_{m}.jsonl")} for m in args.models}
    rows, dropped = [], 0
    for i, en in src.items():
        c = [hyp[m][i] for m in args.models if hyp[m].get(i)]
        if len(c) < 3:
            dropped += 1
            continue
        sc = [sum(CHRF.sentence_score(c[a], [c[b]]).score for b in range(len(c)) if b != a) / (len(c) - 1)
              for a in range(len(c))]
        a = max(range(len(c)), key=lambda k: sc[k])
        n_en, n_ow = len(en.split()), len(c[a].split())
        copied = CHRF.sentence_score(c[a], [en]).score > 60  # the systems left it (mostly) in English
        if sc[a] < args.min_agreement or copied or not (0.5 <= n_ow / n_en <= 2.5):
            dropped += 1
            continue
        rows.append({"en": en, "text": c[a], "agreement": round(sc[a], 1)})
    rows.sort(key=lambda r: -r["agreement"])
    (OUT / "consensus.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    ag = [r["agreement"] for r in rows]
    print(json.dumps({"pairs": len(rows), "dropped": dropped,
                      "agreement_median": ag[len(ag) // 2] if ag else None, "models": args.models}))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sources")
    s.add_argument("--n", type=int, default=30000)
    s.add_argument("--seed", type=int, default=5)
    g = sub.add_parser("generate")
    g.add_argument("--models", nargs="+", required=True)
    g.add_argument("--server", default="http://127.0.0.1:8200/v1")
    g.add_argument("--concurrency", type=int, default=48)
    c = sub.add_parser("select")
    c.add_argument("--models", nargs="+", required=True)
    c.add_argument("--min-agreement", type=float, default=40.0)
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        asyncio.run(generate(args))
    else:
        {"sources": sources, "select": select}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
