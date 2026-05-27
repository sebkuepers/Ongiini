#!/usr/bin/env python3
"""Fill claude_* and gemma_* baseline translation columns in the v2 TSV.

For each row, calls Anthropic API (Claude) and vLLM (Gemma) to translate
the EN source into Oshindonga and Oshikwanyama. Caches each response so
re-runs only pay for new items.

Runs inside the webhook container (vLLM is reachable at
host.docker.internal:8124). Anthropic API runs from anywhere with
ANTHROPIC_API_KEY in env.

Usage:
  # Spark side (Gemma via vLLM):
  docker exec -i ongiini-webhook python3 /data/fill_baseline_translations.py \
      --tsv /data/oshiwambo_eval_v2.tsv \
      --target gemma --rate-per-sec 0.5

  # Laptop side (Claude via Anthropic API):
  python3 scripts/fill_baseline_translations.py \
      --tsv data/oshiwambo_eval_v2.tsv \
      --target claude

Caching:
  Each translation is keyed by sha256(text|target_lang|model_id) and
  stored in --cache (default data/baseline_translation_cache.json).
  Wiping the cache forces re-translation.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Literal

log = logging.getLogger("fill_baselines")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

LangCode = Literal["oshindonga", "oshikwanyama"]


PROMPT_TEMPLATE = """\
Translate the following English text into {language_name}.

Output ONLY the translation in {language_name}. Do not include the original \
English, any explanation, quotation marks, or romanisation notes.

English: {text}

{language_name}:"""

LANGUAGE_NAMES = {
    "oshindonga": "Oshindonga",
    "oshikwanyama": "Oshikwanyama",
}


# ── Cache ────────────────────────────────────────────────────────


class TranslationCache:
    """Tiny on-disk cache keyed by sha256(text|lang|model)."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, str] = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
            except Exception:                       # noqa: BLE001
                log.warning("could not read cache at %s, starting fresh", path)
                self.data = {}

    def key(self, text: str, lang: LangCode, model: str) -> str:
        return hashlib.sha256(
            f"{model}|{lang}|{text}".encode("utf-8")
        ).hexdigest()

    def get(self, text: str, lang: LangCode, model: str) -> str | None:
        return self.data.get(self.key(text, lang, model))

    def set(self, text: str, lang: LangCode, model: str, value: str) -> None:
        self.data[self.key(text, lang, model)] = value

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=0))
        tmp.replace(self.path)


# ── Translators ──────────────────────────────────────────────────


async def translate_via_anthropic(
    client, text: str, lang: LangCode, model: str
) -> str:
    """Call Anthropic API. Uses messages.create."""
    prompt = PROMPT_TEMPLATE.format(
        language_name=LANGUAGE_NAMES[lang], text=text,
    )
    resp = await client.messages.create(
        model=model,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    # SDK returns a Message with content blocks; take first text block
    out = ""
    for block in resp.content:
        if hasattr(block, "text"):
            out += block.text
    return out.strip().strip('"').strip()


async def translate_via_vllm(
    client, text: str, lang: LangCode, model: str
) -> str:
    """Call vLLM via openai-compatible API."""
    prompt = PROMPT_TEMPLATE.format(
        language_name=LANGUAGE_NAMES[lang], text=text,
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600,
    )
    out = resp.choices[0].message.content or ""
    return out.strip().strip('"').strip()


# ── Main ──────────────────────────────────────────────────────────


async def main_async(args: argparse.Namespace) -> int:
    tsv_path = Path(args.tsv)
    cache = TranslationCache(Path(args.cache))

    # Read TSV
    with tsv_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = reader.fieldnames or []
        rows = list(reader)
    log.info("loaded %d rows from %s (%d columns)", len(rows), tsv_path, len(cols))

    # Set up client
    if args.target == "claude":
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            log.error("anthropic SDK not installed — run: pip install anthropic")
            return 2
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            log.error("ANTHROPIC_API_KEY not set in env")
            return 2
        client = AsyncAnthropic(api_key=api_key)
        model = args.claude_model
        translate = translate_via_anthropic
        odg_col = "claude_oshindonga"
        okw_col = "claude_oshikwanyama"
    else:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            log.error("openai SDK not installed")
            return 2
        client = AsyncOpenAI(
            base_url=args.vllm_base_url,
            api_key="not-needed",
        )
        model = args.gemma_model
        translate = translate_via_vllm
        odg_col = "gemma_oshindonga"
        okw_col = "gemma_oshikwanyama"

    log.info("target=%s, model=%s, langs=odg+okw", args.target, model)
    interval = 1.0 / max(0.01, args.rate_per_sec)

    # For each row, fill the two target columns if empty
    n_filled = n_cached = n_skipped = n_failed = 0
    save_every = 25
    last_save = time.time()
    for idx, row in enumerate(rows, start=1):
        en = (row.get("english") or "").strip()
        if not en:
            n_skipped += 1
            continue
        for col, lang in ((odg_col, "oshindonga"), (okw_col, "oshikwanyama")):
            if (row.get(col) or "").strip():
                # Already filled by a prior run — skip
                n_skipped += 1
                continue
            cached = cache.get(en, lang, model)
            if cached:
                row[col] = cached
                n_cached += 1
                continue
            try:
                t0 = time.monotonic()
                out = await translate(client, en, lang, model)
                row[col] = out
                cache.set(en, lang, model, out)
                n_filled += 1
                elapsed = time.monotonic() - t0
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)
            except Exception as exc:                # noqa: BLE001
                log.warning("translate(%d, %s) failed: %s",
                            idx, lang, exc)
                n_failed += 1

        # Periodic checkpoint of cache + TSV write-back
        if idx % save_every == 0:
            cache.save()
            _write_tsv(tsv_path, cols, rows)
            log.info(
                "progress: %d/%d  (filled=%d cached=%d skipped=%d failed=%d)",
                idx, len(rows), n_filled, n_cached, n_skipped, n_failed,
            )

    cache.save()
    _write_tsv(tsv_path, cols, rows)
    log.info(
        "DONE: filled=%d cached=%d skipped=%d failed=%d",
        n_filled, n_cached, n_skipped, n_failed,
    )
    return 1 if n_failed > 0 else 0


def _write_tsv(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Atomic-ish write so we don't lose data if killed mid-run."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    shutil.move(str(tmp), str(path))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", required=True, help="Path to v2 TSV (will be updated in-place)")
    p.add_argument("--cache", default="data/baseline_translation_cache.json")
    p.add_argument("--target", choices=("claude", "gemma"), required=True,
                   help="Which baseline to fill: claude (Anthropic API) or "
                        "gemma (vLLM via openai-compatible API)")
    p.add_argument("--claude-model", default="claude-sonnet-4-6",
                   help="Anthropic model id")
    p.add_argument("--gemma-model", default="gemma-4-26b",
                   help="vLLM model name")
    p.add_argument("--vllm-base-url", default="http://host.docker.internal:8124/v1",
                   help="vLLM base URL (when run inside webhook container)")
    p.add_argument("--rate-per-sec", type=float, default=2.0,
                   help="Per-translation rate cap. Set to 0.5 inside container "
                        "to be nice to live chat traffic.")
    args = p.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
