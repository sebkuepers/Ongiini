#!/usr/bin/env python3
"""Regenerate derailed Gemma baseline outputs with the original settings.

Gemma 4 26B sometimes derails on the zero-shot translation prompt: it
loops a token until max_tokens ("okumanya okumanya …") or narrates
self-corrections ("Wait, let me correct that …"). Those outputs say
nothing about translation ability, so this script re-asks the same
prompt with the same decoding settings (fill_baseline_translations.py:
temperature 0.2, max_tokens 600) until the output is clean or
--max-attempts is reached. The last attempt is kept either way.

Protocol note for any reported numbers: derailed first-pass outputs are
regenerated up to N attempts; report the first-pass derailment rate and
the final one alongside chrF. The attempts log makes this auditable.

Only English source text is sent to the model. Runs inside the webhook
container, like fill_baseline_translations.py:

    docker exec -i ongiini-webhook python3 /data/retry_derailed_baselines.py \\
        --tsv /data/oshiwambo_eval_v2.tsv \\
        --cache /data/baseline_cache_gemma.json \\
        --out /data/baseline_cache_gemma_retried.json
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from fill_baseline_translations import LANGUAGE_NAMES, translate_via_vllm

META = re.compile(r"\b(wait|let me|actually|note:|translation|translates)\b", re.I)


def is_derailed(english: str, output: str) -> bool:
    """Runaway length or English meta-commentary. First-pass outputs are
    bimodal: clean ones stay near the source length, derailed ones run
    to max_tokens (>5x) — 3x sits safely in the gap."""
    return len(output) > 3 * max(len(english), 20) or bool(META.search(output))


def key(model: str, lang: str, text: str) -> str:
    return hashlib.sha256(f"{model}|{lang}|{text}".encode("utf-8")).hexdigest()


async def main_async(args: argparse.Namespace) -> int:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=args.vllm_base_url, api_key="not-needed")
    cache = json.loads(Path(args.cache).read_text())
    with Path(args.tsv).open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    todo = []
    for r in rows:
        en = r["english"].strip()
        for lang in LANGUAGE_NAMES:
            k = key(args.model, lang, en)
            if k not in cache:
                print(f"ERROR: id {r['id']} {lang} missing from cache", file=sys.stderr)
                return 1
            if is_derailed(en, cache[k]):
                todo.append((r["id"], lang, en, k))
    total = len(rows) * len(LANGUAGE_NAMES)
    print(f"first pass: {len(todo)}/{total} derailed", file=sys.stderr)

    out = dict(cache)
    log = []
    sem = asyncio.Semaphore(args.concurrency)

    async def retry(item_id: str, lang: str, en: str, k: str) -> None:
        async with sem:
            for attempt in range(1, args.max_attempts + 1):
                text = await translate_via_vllm(client, en, lang, args.model)
                bad = is_derailed(en, text)
                if not bad or attempt == args.max_attempts:
                    out[k] = text
                    log.append({"id": int(item_id), "lang": lang,
                                "attempts": attempt, "derailed": bad})
                    return

    await asyncio.gather(*(retry(*t) for t in todo))

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=0))
    log.sort(key=lambda x: (x["id"], x["lang"]))
    Path(args.out).with_suffix(".attempts.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in log))
    still = sum(1 for x in log if x["derailed"])
    print(f"after retries: {still}/{total} derailed "
          f"(max {args.max_attempts} attempts)", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", required=True, help="TSV with the English sources")
    p.add_argument("--cache", required=True, help="First-pass Gemma cache (read-only)")
    p.add_argument("--out", required=True, help="Where to write the retried cache")
    p.add_argument("--model", default="gemma-4-26b")
    p.add_argument("--vllm-base-url", default="http://host.docker.internal:8124/v1")
    p.add_argument("--max-attempts", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=4,
                   help="Parallel requests; keep low when chat traffic is live")
    return asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
