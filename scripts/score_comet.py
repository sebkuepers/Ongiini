#!/usr/bin/env python3
"""Learned-metric scores for Ongiini-Eval-OW (docs/eval-protocol.md §1).

SSA-COMET-MTL (exploratory; encoder has Kwanyama in pretraining) and
COMET-22 (appendix; does not cover Oshiwambo), segment level with source +
reference, on the development items that have references. Segment scores
are cached per (model, system, dialect, id, text hash), so re-runs only
score new or changed outputs. System score = mean of segment scores.

Runs in its own venv because unbabel-comet pins an older transformers:

    ~/.venvs/ongiini-comet/bin/python scripts/score_comet.py --out data/private/results/comet.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from comet import download_model, load_from_checkpoint

import eval_scoring as E

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/private/export/data/baselines"
EVAL_SET = ROOT / "data/private/export/data/eval_set.jsonl"
CACHE = ROOT / "data/private/results/comet_cache.jsonl"
MODELS = {"ssa-comet-mtl": "McGill-NLP/ssa-comet-mtl", "comet-22": "Unbabel/wmt22-comet-da"}


def key(*parts) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:20]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args(argv)
    items = {i: it for i, it in E.load_items(EVAL_SET).items() if not it["in_blind_split"]}
    manifest = json.loads((BASE / "manifest.json").read_text())
    cache = {}
    if CACHE.exists():
        for l in CACHE.read_text().splitlines():
            r = json.loads(l)
            cache[r["k"]] = r["score"]

    jobs = []                                    # (model, label, dialect, id, src, mt, ref, k)
    for s in manifest["systems"]:
        for d, rel in s["files"].items():
            rows = {int(r["id"]): r for r in E.load_jsonl(ROOT / rel)}
            for i, it in items.items():
                ref = E.normalise(it.get(f"{d}_reference"))
                if not ref or i in s.get("exclude_ids", []) or i not in rows:
                    continue
                mt = E.clean_hypothesis(rows[i].get("translation"), d)
                for m in MODELS:
                    k = key(m, s["label"], d, i, mt, ref)
                    jobs.append((m, s["label"], d, i, E.normalise(it["english"]), mt, ref, k))

    for m, repo in MODELS.items():
        todo = [j for j in jobs if j[0] == m and j[7] not in cache]
        print(f"{m}: {len(todo)} segments to score ({len(jobs) // len(MODELS)} total)", flush=True)
        if not todo:
            continue
        model = load_from_checkpoint(download_model(repo))
        data = [{"src": j[4], "mt": j[5], "ref": j[6]} for j in todo]
        out = model.predict(data, batch_size=args.batch, gpus=0, num_workers=1, progress_bar=False)
        with CACHE.open("a") as f:
            for j, sc in zip(todo, out.scores):
                cache[j[7]] = float(sc)
                f.write(json.dumps({"k": j[7], "score": float(sc)}) + "\n")

    result: dict = {"models": MODELS, "systems": {}}
    for m in MODELS:
        for j in (j for j in jobs if j[0] == m):
            result["systems"].setdefault(j[1], {}).setdefault(j[2], {}).setdefault(m, []).append(cache[j[7]])
    for label, ds in result["systems"].items():
        for d, ms in ds.items():
            for m, vals in ms.items():
                ms[m] = {"n": len(vals), "mean": round(100 * sum(vals) / len(vals), 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
