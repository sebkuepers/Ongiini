"""Seed the contributions task pool: for each of the 200 source seeds,
ask Gemma (via vLLM on Spark) to produce ~50 paraphrases, write all of
them as rows in the ``tasks`` sqlite table.

Why this script exists:
The community contribution loop needs ~10K English sentences to serve
to native-speaker volunteers. Hand-crafting 10K isn't feasible; we
start with 200 carefully-curated seeds (mined from real Ongiini
replies + crafted to match the 4 concentrated domains per the Tan
2024 finding) and let the local Gemma 4 27B instance produce
paraphrases.

This runs ON the Spark host (or via docker exec into the webhook
container) so it talks to vLLM at the same in-cluster URL the
webhook uses. It is **throttled**: production users are still being
served by the same vLLM instance.

Usage:
    # Quick visual check — 5 seeds, no DB writes
    python3 scripts/seed_contribution_tasks.py --limit 5 --dry-run

    # Small live run — 25 seeds, writes ~1250 rows
    python3 scripts/seed_contribution_tasks.py --limit 25

    # Full run — 200 seeds, writes ~10K rows. ~3-4 hours throttled.
    nohup python3 scripts/seed_contribution_tasks.py > /tmp/seed_run.log 2>&1 &

    # Reset the tasks table only (full reset via contributions CLI)
    python3 scripts/seed_contribution_tasks.py --reset --confirm

Throttling:
    --sleep:       seconds to wait between seeds (default 6s)
    --queue-pause: vLLM queue-depth above which we pause 30s before
                   issuing the next paraphrase batch (default 3)

Sentinel abort:
    Touch ``/data/contributions_seed.abort`` (or pass --abort-file)
    to make the script stop cleanly between seeds.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import runpy
import sys
import time
from pathlib import Path
from typing import Any

# Make the ongiini package importable when run via `python3 scripts/...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from ongiini import contributions  # noqa: E402
from ongiini.config import settings  # noqa: E402


log = logging.getLogger("seed_contribution_tasks")


DEFAULT_PARAPHRASES_PER_SEED = 50
DEFAULT_ABORT_FILE = "/data/contributions_seed.abort"

PARAPHRASE_PROMPT_TEMPLATE = (
    "Give me {n} different ways to express this English sentence, as a "
    "Namibian person speaking on WhatsApp might phrase it. Vary length "
    "(short / medium / long), formality (casual / friendly / formal), "
    "and word choice — but preserve the meaning EXACTLY. Do not change "
    "the meaning, do not add information not in the source, do not "
    "drop information from the source, do not switch domain (a CV "
    "phrase must stay a CV phrase). Each paraphrase must be a complete, "
    "self-contained sentence. "
    "Source sentence: {sentence}"
)


# ── Seeds loading ─────────────────────────────────────────────────


def _load_seeds(seeds_path: Path) -> list[dict[str, Any]]:
    """Load the SEEDS list from a Python file. Uses runpy so the file
    runs in its own namespace and we don't pollute this module."""
    ns = runpy.run_path(str(seeds_path))
    seeds = ns.get("SEEDS")
    if not isinstance(seeds, list):
        raise RuntimeError(f"{seeds_path} did not define a SEEDS list")
    return seeds


# ── Paraphrase filtering ──────────────────────────────────────────


def filter_paraphrases(items: list[str], source_sentence: str) -> list[str]:
    """Apply quality gates to the structured JSON list returned by
    Gemma: drop empties, near-duplicates of the source, and items
    outside a sensible word-count range. The structured-output mode
    already guarantees we get a list of strings, so no regex parsing."""
    out: list[str] = []
    seen: set[str] = set()
    src_norm = source_sentence.strip().lower()
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().strip('"').strip("'")
        if not cleaned:
            continue
        wc = len(cleaned.split())
        if not (3 <= wc <= 200):
            continue
        norm = cleaned.lower()
        if norm == src_norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(cleaned)
    return out


# ── vLLM client ───────────────────────────────────────────────────


async def _vllm_queue_depth(client: httpx.AsyncClient, base_url: str) -> int | None:
    """Best-effort vllm:num_requests_waiting from /metrics. None on any
    failure — caller treats None as 'unknown, proceed'."""
    metrics_url = base_url.rstrip("/").removesuffix("/v1") + "/metrics"
    try:
        r = await client.get(metrics_url, timeout=2.0)
        if r.status_code != 200:
            return None
        for line in r.text.splitlines():
            if line.startswith("vllm:num_requests_waiting"):
                parts = line.rsplit(" ", 1)
                if len(parts) == 2:
                    try:
                        return int(float(parts[1]))
                    except ValueError:
                        return None
    except (httpx.HTTPError, OSError):
        return None
    return None


async def _call_paraphrase(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    sentence: str,
    n: int,
) -> list[str]:
    """Single OpenAI-compatible /chat/completions call to vLLM asking
    for exactly N paraphrases of `sentence` via structured-output JSON
    schema (minItems=maxItems=n). Returns the parsed list of strings.

    Why structured output: with free-form output Gemma routinely under-
    shoots the requested count (we saw counts of 5 / 7 / 9 against an
    ask of 50). The json_schema response_format forces the model's
    decoder to emit exactly the array shape we specified — no
    undershoot, no markdown chrome, no parsing surface."""
    prompt = PARAPHRASE_PROMPT_TEMPLATE.format(n=n, sentence=sentence)
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        # Each paraphrase averages ~25 tokens. N=50 × 25 + JSON
        # overhead → ~1500 tokens. Headroom for longer ones.
        "max_tokens": 3500,
        "temperature": 0.9,
        "top_p": 0.95,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "paraphrases",
                "schema": {
                    "type": "object",
                    "properties": {
                        "paraphrases": {
                            "type": "array",
                            "minItems": n,
                            "maxItems": n,
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["paraphrases"],
                },
            },
        },
    }
    r = await client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        timeout=300.0,
    )
    r.raise_for_status()
    data = r.json()
    raw = data["choices"][0]["message"]["content"]
    parsed = json.loads(raw)
    items = parsed.get("paraphrases", [])
    if not isinstance(items, list):
        raise ValueError(f"expected a list, got {type(items).__name__}")
    return items


# ── Main loop ─────────────────────────────────────────────────────


async def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log.info("seed_contribution_tasks starting")

    if args.reset:
        if not args.confirm:
            print(
                "About to wipe the tasks table. Re-run with --confirm "
                "to actually perform the reset.",
                file=sys.stderr,
            )
            return 1
        contributions.warmup()
        with contributions._conn() as c:
            existing = c.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
            c.execute("DELETE FROM tasks")
        log.info("wiped tasks table (had %d rows)", existing)
        return 0

    seeds_path = Path(args.seeds_file)
    if not seeds_path.exists():
        log.error("seeds file not found: %s", seeds_path)
        return 2
    seeds = _load_seeds(seeds_path)
    if args.limit:
        seeds = seeds[: args.limit]
    log.info("loaded %d seeds from %s", len(seeds), seeds_path)

    if not args.dry_run:
        contributions.warmup()
        existing = contributions.task_count()
        if existing > 0 and not args.force:
            log.error(
                "tasks table already has %d rows. Re-run with --force "
                "to add more, or --reset --confirm to wipe first.",
                existing,
            )
            return 3

    abort_file = Path(args.abort_file)
    base_url = args.vllm_base_url or settings.vllm_base_url
    model = args.vllm_model or settings.vllm_model
    log.info("vllm base_url=%s model=%s", base_url, model)

    total_written = 0
    seed_count = len(seeds)
    started = time.monotonic()

    async with httpx.AsyncClient() as client:
        for i, seed in enumerate(seeds, start=1):
            if abort_file.exists():
                log.warning(
                    "abort file %s detected — stopping after %d / %d seeds",
                    abort_file, i - 1, seed_count,
                )
                break

            depth = await _vllm_queue_depth(client, base_url)
            if depth is not None and depth > args.queue_pause:
                log.warning(
                    "vllm queue depth %d > %d — pausing 30s",
                    depth, args.queue_pause,
                )
                await asyncio.sleep(30)

            source = seed["source_en"]
            category = seed.get("category")
            seed_id = seed.get("id")
            log.info(
                "[%d/%d] seed_id=%s category=%s source=%s",
                i, seed_count, seed_id, category, source[:80],
            )
            try:
                items = await _call_paraphrase(
                    client, base_url, model, source, args.paraphrases_per_seed,
                )
            except (httpx.HTTPError, KeyError, ValueError, json.JSONDecodeError) as e:
                log.error("seed %s paraphrase call failed: %s", seed_id, e)
                await asyncio.sleep(args.sleep)
                continue

            candidates = filter_paraphrases(items, source)
            log.info("  → %d clean paraphrases (of %d returned)",
                     len(candidates), len(items))

            # Always include the source itself so the original wording
            # is never lost.
            rows = [{
                "source_en": source,
                "category": category,
                "seed_id": seed_id,
            }]
            for c in candidates:
                rows.append({
                    "source_en": c,
                    "category": category,
                    "seed_id": seed_id,
                })

            if args.dry_run:
                for c in candidates[:5]:
                    print(f"    {c}")
                if len(candidates) > 5:
                    print(f"    ... ({len(candidates) - 5} more)")
            else:
                inserted = contributions.seed_tasks(rows)
                total_written += inserted

            if i < seed_count:
                await asyncio.sleep(args.sleep)

    elapsed = time.monotonic() - started
    log.info(
        "DONE — processed %d seeds, wrote %d rows in %.1fs",
        seed_count, total_written, elapsed,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seeds-file", default=str(
        Path(__file__).resolve().parent / "data" / "seeds_v1.py"
    ))
    p.add_argument("--limit", type=int, default=0,
                   help="only process the first N seeds (0 = all)")
    p.add_argument("--paraphrases-per-seed", type=int,
                   default=DEFAULT_PARAPHRASES_PER_SEED)
    p.add_argument("--sleep", type=float, default=6.0,
                   help="seconds between seed batches (throttle)")
    p.add_argument("--queue-pause", type=int, default=3,
                   help="vLLM queue depth above which to pause 30s")
    p.add_argument("--abort-file", default=DEFAULT_ABORT_FILE)
    p.add_argument("--dry-run", action="store_true",
                   help="print paraphrases instead of writing to sqlite")
    p.add_argument("--force", action="store_true",
                   help="allow writing to a non-empty tasks table")
    p.add_argument("--reset", action="store_true",
                   help="wipe the tasks table (requires --confirm)")
    p.add_argument("--confirm", action="store_true",
                   help="actually perform --reset")
    p.add_argument("--vllm-base-url", default="",
                   help="override settings.vllm_base_url")
    p.add_argument("--vllm-model", default="",
                   help="override settings.vllm_model")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
