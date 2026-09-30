#!/usr/bin/env python3
"""Export the internal v2 TSV into publication-ready formats.

Reads:
  data/private/oshiwambo_eval_v3.tsv  the internal working TSV (gitignored;
                                    full schema incl. <translator>_* refs,
                                    claude_*, gemma_*, in_blind_split)

Writes under data/private/export/data/ by default, or — with --release,
only when the dataset goes public — under data/oshiwambo_eval/data/:
  eval_set.tsv             clean public schema (no claude/gemma baselines)
  eval_set.jsonl           one JSON object per item — HF-friendly
  en.txt                   EN sources only, one per line, line N = id N
  oshindonga.txt           Oshindonga refs only, parallel to en.txt
  oshikwanyama.txt         Oshikwanyama refs only, parallel to en.txt
  blind_split.jsonl        the 84 held-back items as JSONL
  development_split.jsonl  the 339 non-held items as JSONL
  baselines/claude.jsonl   Claude's translations (separate from refs!)
  baselines/gemma.jsonl    Gemma 4 26B's translations (separate from refs!)

Why the EN/Oshindonga/Oshikwanyama plaintext files matter: every classic
MT-tooling chain (Moses, fairseq, sentencepiece, sacrebleu) takes
plaintext parallel files. Providing them lets researchers benchmark
their systems with zero glue code.

The references are unreleased and the repo is public: never commit a
--release export before the v1.0 launch, or the blind split leaks.

    python3 scripts/export_eval_set.py                 # private preview
    python3 scripts/export_eval_set.py --release       # launch day only

Idempotent — safe to re-run any time.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/private/oshiwambo_eval_v3.tsv"
PUBLIC_ROOT = ROOT / "data/oshiwambo_eval"
PRIVATE_ROOT = ROOT / "data/private/export"


# Public schema — what goes into eval_set.tsv. claude_* / gemma_*
# columns DO NOT appear here; baselines live in baselines/*.jsonl
PUBLIC_COLS = [
    "id",
    "length_bucket",
    "domain",
    "phenomenon_tags",
    "provenance",
    "english",
    "oshindonga_reference",
    "oshikwanyama_reference",
    "oshindonga_translator_notes",
    "oshikwanyama_translator_notes",
    "in_blind_split",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--translator", default="kaarina",
                    help="Column prefix of the reference translations")
    ap.add_argument("--release", action="store_true",
                    help="Write into the public dataset dir (launch day only)")
    args = ap.parse_args(argv)
    t = args.translator
    OUT_ROOT = PUBLIC_ROOT if args.release else PRIVATE_ROOT
    DATA_DIR = OUT_ROOT / "data"
    BASELINES_DIR = DATA_DIR / "baselines"
    if not args.release:
        # Seed baselines so the keep-populated guard below behaves the
        # same as against the public dir.
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        for f in (PUBLIC_ROOT / "data/baselines").glob("*.jsonl"):
            if not (BASELINES_DIR / f.name).exists():
                (BASELINES_DIR / f.name).write_bytes(f.read_bytes())

    if not SOURCE.exists():
        print(f"ERROR: source TSV missing at {SOURCE}", file=sys.stderr)
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    with SOURCE.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    print(f"loaded {len(rows)} rows from {SOURCE.relative_to(ROOT)}", file=sys.stderr)
    rows.sort(key=lambda r: int(r["id"]))

    # ── eval_set.tsv (public schema) ─────────────────────────────
    public_rows = []
    for r in rows:
        public_rows.append({
            "id": r["id"],
            "length_bucket": r["length_bucket"],
            "domain": r["domain"],
            "phenomenon_tags": r["phenomenon_tags"],
            "provenance": r["provenance"],
            "english": r["english"],
            # Translator-reference columns: the internal TSV stores
            # them per translator (<translator>_*); the public file
            # uses neutral column names.
            "oshindonga_reference": r.get(f"{t}_oshindonga", ""),
            "oshikwanyama_reference": r.get(f"{t}_oshikwanyama", ""),
            "oshindonga_translator_notes": r.get(f"{t}_oshindonga_notes", ""),
            "oshikwanyama_translator_notes": r.get(f"{t}_oshikwanyama_notes", ""),
            "in_blind_split": r["in_blind_split"],
        })
    out_tsv = DATA_DIR / "eval_set.tsv"
    with out_tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PUBLIC_COLS, delimiter="\t")
        w.writeheader()
        for r in public_rows:
            w.writerow(r)
    print(f"  wrote {out_tsv.relative_to(ROOT)}  ({len(public_rows)} rows)",
          file=sys.stderr)

    # ── eval_set.jsonl ───────────────────────────────────────────
    out_jsonl = DATA_DIR / "eval_set.jsonl"
    with out_jsonl.open("w") as f:
        for r in public_rows:
            # Convert phenomenon_tags from ";"-joined → list[str]
            tags = [t for t in r["phenomenon_tags"].split(";") if t]
            obj = {**r, "phenomenon_tags": tags,
                   "in_blind_split": r["in_blind_split"] == "true"}
            obj["id"] = int(obj["id"])
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"  wrote {out_jsonl.relative_to(ROOT)}", file=sys.stderr)

    # ── Per-language plaintext files (parallel) ──────────────────
    # Line N = id N. Empty lines preserve alignment when refs missing.
    def write_lang(col_key: str, fname: str) -> None:
        out = DATA_DIR / fname
        with out.open("w") as f:
            for r in public_rows:
                text = (r.get(col_key) or "").replace("\n", " ").strip()
                f.write(text + "\n")
        n_nonempty = sum(1 for r in public_rows if (r.get(col_key) or "").strip())
        print(
            f"  wrote {out.relative_to(ROOT)}  "
            f"({n_nonempty}/{len(public_rows)} populated)",
            file=sys.stderr,
        )

    write_lang("english", "en.txt")
    write_lang("oshindonga_reference", "oshindonga.txt")
    write_lang("oshikwanyama_reference", "oshikwanyama.txt")

    # ── Split files ──────────────────────────────────────────────
    blind = [r for r in public_rows if r["in_blind_split"] == "true"]
    dev = [r for r in public_rows if r["in_blind_split"] != "true"]
    for split_name, items in (("blind_split", blind), ("development_split", dev)):
        out = DATA_DIR / f"{split_name}.jsonl"
        with out.open("w") as f:
            for r in items:
                tags = [t for t in r["phenomenon_tags"].split(";") if t]
                obj = {**r, "phenomenon_tags": tags,
                       "in_blind_split": r["in_blind_split"] == "true"}
                obj["id"] = int(obj["id"])
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(
            f"  wrote {out.relative_to(ROOT)}  ({len(items)} rows)",
            file=sys.stderr,
        )

    # ── Baselines (Claude + Gemma) — separate files so nobody
    #    confuses them with the gold references ────────────────────
    def write_baseline(prefix: str, fname: str, label: str) -> None:
        out = BASELINES_DIR / fname
        # Baselines may be filled straight into the JSONL (see
        # fill_baseline_translations.py) without touching the TSV —
        # never clobber a populated file with an empty source column.
        if out.exists() and not any(
            (r.get(f"{prefix}_oshindonga") or r.get(f"{prefix}_oshikwanyama") or "").strip()
            for r in rows
        ):
            print(f"  kept {out.relative_to(ROOT)}  (no {prefix}_* data in source TSV)",
                  file=sys.stderr)
            return
        n_filled = 0
        with out.open("w") as f:
            for r in rows:
                odg = (r.get(f"{prefix}_oshindonga") or "").strip()
                okw = (r.get(f"{prefix}_oshikwanyama") or "").strip()
                if odg or okw:
                    n_filled += 1
                obj = {
                    "id": int(r["id"]),
                    "system": label,
                    "oshindonga_output": odg,
                    "oshikwanyama_output": okw,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(
            f"  wrote {out.relative_to(ROOT)}  ({n_filled}/{len(rows)} populated)",
            file=sys.stderr,
        )

    write_baseline("claude", "claude.jsonl", "claude-sonnet-4-6")
    write_baseline("gemma", "gemma.jsonl", "gemma-4-26b")

    print("\nexport complete.", file=sys.stderr)
    print(f"  publish from:  {OUT_ROOT.relative_to(ROOT)}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
