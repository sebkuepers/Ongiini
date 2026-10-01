#!/usr/bin/env python3
"""Build the private HTML results report from a run_evaluation.py results folder.

Reads scores.json (and the manifest inside it); computes nothing except the
OkaMT 15-item spot check in the appendix. Aggregates only — no sentences.

    python3 scripts/build_results_report.py data/private/results/2026-10-01
      → data/private/results/2026-10-01/report.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_scoring as E  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().parent / "results_report_template.html"
BASE = ROOT / "data/private/export/data/baselines"
EVAL_SET = ROOT / "data/private/export/data/eval_set.jsonl"
OKAMT = BASE / "okamt-okalex-demo_sample15_oshikwanyama.jsonl"


def okamt_spot_check(scores: dict) -> dict | None:
    """OkaMT's 15 hand-entered Oshikwanyama outputs vs every main system on the same items."""
    if not OKAMT.exists():
        return None
    items = E.load_items(EVAL_SET)
    rows = {int(r["id"]): r["translation"] for r in E.load_jsonl(OKAMT)}
    ids = sorted(i for i in rows if not items[i]["in_blind_split"]
                 and (items[i].get("oshikwanyama_reference") or "").strip())
    refs = [E.normalise(items[i]["oshikwanyama_reference"]) for i in ids]
    out = {"n": len(ids), "systems": {}}
    def chrf(hyps):
        return round(E.METRICS[E.PRIMARY].corpus_score(hyps, [refs]).score, 1)
    out["systems"]["OkaMT (okalex.org demo)"] = chrf([E.clean_hypothesis(rows[i], "oshikwanyama") for i in ids])
    for s in scores["manifest"]["systems"]:
        if s["role"] != "main" or "oshikwanyama" not in s["files"]:
            continue
        try:
            raw, _ = E.load_system_file(ROOT / s["files"]["oshikwanyama"], "oshikwanyama", ids)
        except E.InputError:
            continue
        out["systems"][s["name"]] = chrf([E.clean_hypothesis(raw[i], "oshikwanyama") for i in ids])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    args = ap.parse_args(argv)
    scores = json.loads((args.results / "scores.json").read_text())
    scores["okamt"] = okamt_spot_check(scores)
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(scores, ensure_ascii=False))
    out = args.results / "report.html"
    out.write_text(html)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
