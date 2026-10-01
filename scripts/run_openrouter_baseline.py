#!/usr/bin/env python3
"""Translate the eval-set English sources with an API model (OpenRouter or any
OpenAI-compatible endpoint, e.g. Meta's Model API for Muse Spark).

Leaderboard protocol (documented in the submission guide):
  * prompt: the paper's zero-shot template (eval_prompts.PAPER_ZEROSHOT),
    sent as a single user message, no system prompt, no examples
  * temperature 0 and seed 42 where the model accepts them, else the
    provider default; reasoning models keep their default effort
    (--reasoning on/off only for models the paper lists in both modes)
  * max 1024 output tokens; 4096 when the model reasons, so hidden
    reasoning cannot eat the whole budget and leave an empty answer
  * derailed outputs (runaway length or meta-commentary) are re-asked
    with the same settings, up to 5 attempts, every attempt logged —
    the same rule as for Gemma
  * only the English source leaves the machine

Writes submission-schema JSONL (one file per dialect) plus a sidecar
with token usage, cost and attempt counts. Resumable: ids already in the
output file are skipped.

    OPENROUTER_API_KEY=… ~/.venvs/ongiini-eval/bin/python \\
        scripts/run_openrouter_baseline.py --model anthropic/claude-opus-5 \\
        --label claude-opus-5 --limit 10
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_prompts import PAPER_ZEROSHOT_ID, render  # noqa: E402
from retry_derailed_baselines import is_derailed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data/oshiwambo_eval_v3.tsv"
OUT_DIR = ROOT / "data/private/export/data/baselines"
DIALECTS = ("oshindonga", "oshikwanyama")
MAX_ATTEMPTS = 5


def api_key(name: str) -> str:
    key = os.environ.get(name)
    if not key and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith(f"{name}="):
                key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit(f"{name} missing")
    return key


def clean(text: str, dialect: str) -> str:
    text = (text or "").strip()
    label = dialect.capitalize() + ":"
    if text.lower().startswith(label.lower()):
        text = text[len(label):].strip()
    return text.strip('"').strip()


async def main_async(args: argparse.Namespace) -> int:
    client = AsyncOpenAI(base_url=args.base_url, api_key=api_key(args.key_env))
    openrouter = "openrouter.ai" in args.base_url
    if openrouter:
        models = (await client.models.list()).data
        params = next((m.model_extra or {}).get("supported_parameters", [])
                      for m in models if m.id == args.model)
    else:  # other OpenAI-compatible APIs: declare what the provider documents
        params = args.params.split(",")
    extra: dict = {}
    body: dict = {"usage": {"include": True}} if openrouter else {}
    if "temperature" in params:
        extra["temperature"] = 0
    if "seed" in params:
        extra["seed"] = 42
    if args.reasoning == "low":                 # for models whose reasoning cannot be disabled (GLM 5.3)
        body["reasoning"] = {"effort": "low"}
    elif args.reasoning != "default":
        body["reasoning"] = {"enabled": args.reasoning == "on"}
    reasons = "reasoning" in params and args.reasoning != "off"
    max_tokens = args.max_tokens or (4096 if reasons else 1024)

    with SOURCES.open() as f:
        items = list(csv.DictReader(f, delimiter="\t"))
    if args.ids_file:
        keep = set(json.loads(Path(args.ids_file).read_text()))
        items = [it for it in items if int(it["id"]) in keep]
    if args.limit:
        items = items[: args.limit]

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = {d: out_dir / f"{args.label}_{d}.jsonl" for d in DIALECTS}
    if args.redo_empty:
        args.empty_once = True
    if args.redo_empty and not args.continue_redo:
        # Move records with an empty translation to an archive, so exactly
        # those items are regenerated (protocol: one redo with a larger
        # budget when hidden reasoning used up max_tokens).
        arch = out_dir / "_discarded"
        arch.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for d, p in outs.items():
            if not p.exists():
                continue
            rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            empty = [r for r in rows if not (r.get("translation") or "").strip()]
            if empty:
                (arch / f"{args.label}_{d}.empty-{stamp}.jsonl").write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in empty))
                p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows
                                     if (r.get("translation") or "").strip()))
            print(f"{args.label} {d}: {len(empty)} empty records to redo", file=sys.stderr)
    done = {d: {json.loads(l)["id"] for l in p.read_text().splitlines() if l.strip()}
            if p.exists() else set() for d, p in outs.items()}
    usage_path = out_dir / f"{args.label}.usage.jsonl"
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    totals = {"calls": 0, "cost": 0.0, "prompt": 0, "completion": 0, "reasoning": 0,
              "derailed_first": 0, "derailed_final": 0, "errors": 0}

    async def one(item: dict, dialect: str) -> None:
        en = item["english"].strip()
        prompt = render(args.template, dialect, en)
        async with sem:
            text, attempts, raw = "", 0, ""
            for attempts in range(1, MAX_ATTEMPTS + 1):
                r = None
                for wait in (0, 5, 15, 30, 60, 120, 240):   # rate limits / 5xx
                    await asyncio.sleep(wait)
                    try:
                        r = await client.chat.completions.create(
                            model=args.model, max_tokens=max_tokens,
                            messages=[{"role": "user", "content": prompt}],
                            extra_body=body, **extra)
                        if not r.choices:               # provider returned no completion
                            raise RuntimeError("response without choices")
                        break
                    except Exception as exc:                # noqa: BLE001
                        last, r = exc, None
                if r is None:
                    totals["errors"] += 1
                    print(f"  error id {item['id']} {dialect}: {last}", file=sys.stderr)
                    return
                u = r.usage.model_extra if r.usage else {}
                totals["calls"] += 1
                totals["cost"] += float((u or {}).get("cost") or 0)
                totals["prompt"] += r.usage.prompt_tokens if r.usage else 0
                totals["completion"] += r.usage.completion_tokens if r.usage else 0
                details = getattr(r.usage, "completion_tokens_details", None)
                totals["reasoning"] += getattr(details, "reasoning_tokens", 0) or 0
                raw = r.choices[0].message.content or ""
                text = clean(raw, dialect)
                bad = is_derailed(en, text) or not text
                if attempts == 1 and bad:
                    totals["derailed_first"] += 1
                if not bad:
                    break
                if not text and args.empty_once:
                    break          # protocol §3: an empty output is regenerated once, not retried
            else:
                totals["derailed_final"] += 1
        record = {"id": int(item["id"]), "dialect": dialect, "model_id": args.model,
                  "prompt_template_id": args.template, "translation": text,
                  "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "seed": extra.get("seed"), "raw_response": raw}
        async with lock:
            with outs[dialect].open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            with usage_path.open("a") as f:
                f.write(json.dumps({"id": int(item["id"]), "dialect": dialect,
                                    "attempts": attempts, "max_tokens": max_tokens,
                                    "redo_empty": bool(args.redo_empty or args.continue_redo)}) + "\n")

    jobs = [one(it, d) for it in items for d in DIALECTS if int(it["id"]) not in done[d]]
    print(f"{args.label}: {len(jobs)} translations to run "
          f"(decoding {extra or 'provider default'}, reasoning {args.reasoning}, "
          f"max_tokens {max_tokens})",
          file=sys.stderr)
    await asyncio.gather(*jobs)
    n = max(1, totals["calls"])
    print(json.dumps({"label": args.label, **{k: round(v, 4) if isinstance(v, float) else v
                                              for k, v in totals.items()},
                      "cost_per_call": round(totals["cost"] / n, 6)}), flush=True)
    return 1 if totals["errors"] else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="OpenRouter model id")
    ap.add_argument("--label", required=True, help="Output file stem")
    ap.add_argument("--reasoning", choices=("default", "on", "off", "low"), default="default")
    ap.add_argument("--limit", type=int, default=0, help="Only the first N items")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1",
                    help="Any OpenAI-compatible endpoint, e.g. https://api.meta.ai/v1")
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY",
                    help="Env / .env variable holding the API key")
    ap.add_argument("--redo-empty", action="store_true",
                    help="Regenerate only records whose translation is empty (archived first)")
    ap.add_argument("--max-tokens", type=int, default=0, help="Override the output budget")
    ap.add_argument("--template", default=PAPER_ZEROSHOT_ID,
                    help="Prompt template id from eval_prompts (robustness checks only)")
    ap.add_argument("--out-dir", default="", help="Write outputs here instead of baselines/")
    ap.add_argument("--ids-file", default="", help="JSON list of item ids to run")
    ap.add_argument("--continue-redo", action="store_true",
                    help="Resume an interrupted --redo-empty run (no re-archiving)")
    ap.add_argument("--empty-once", action="store_true",
                    help="Do not retry an empty output (set by --redo-empty)")
    ap.add_argument("--params", default="temperature,reasoning",
                    help="Supported parameters for non-OpenRouter APIs")
    return asyncio.run(main_async(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
