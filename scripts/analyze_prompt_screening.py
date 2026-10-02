#!/usr/bin/env python3
"""Score the prompt screening (scripts/prompt_screening.py).

Per system and template: chrF++ per dialect on the screening items, the
difference to the paper prompt with a paired bootstrap 95 % CI, and the
share of outputs that are empty or still derailed. Exploratory.

    ~/.venvs/ongiini-eval/bin/python scripts/analyze_prompt_screening.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_scoring as E  # noqa: E402
from prompt_screening import EVAL_SET, OUT, SYSTEMS, TEMPLATES  # noqa: E402
from retry_derailed_baselines import is_derailed  # noqa: E402

PAPER = TEMPLATES[0]


def main() -> int:
    items = E.load_items(EVAL_SET)
    ids = json.loads((OUT / "ids.json").read_text())
    boot = E.bootstrap_indices(len(ids))
    report: dict = {}
    for label, _, _ in SYSTEMS:
        for d in E.DIALECTS:
            refs = [E.normalise(items[i][f"{d}_reference"]) for i in ids]
            stats = {}
            for t in TEMPLATES:
                p = OUT / f"{label}__{t}_{d}.jsonl"
                if not p.exists():
                    continue
                try:
                    raw, _ = E.load_system_file(p, d, ids)
                except E.InputError:
                    continue
                hyps = [E.clean_hypothesis(raw[i], d) for i in ids]
                s = E.SystemStats.build(hyps, refs)
                bad = sum((not h) or is_derailed(items[i]["english"], raw[i]) for i, h in zip(ids, hyps))
                stats[t] = (s, round(100 * bad / len(ids), 1))
            if PAPER not in stats:
                continue
            base = stats[PAPER][0]
            b0 = base.stats["chrf++"]
            for t, (s, bad) in stats.items():
                diffs = np.sort([E.score_from("chrf++", s.stats["chrf++"][ix].sum(0)) -
                                 E.score_from("chrf++", b0[ix].sum(0)) for ix in boot])
                sc = E.corpus_scores(s)["chrf++"]
                report.setdefault(label, {}).setdefault(t, {})[d] = {
                    "chrf++": round(sc, 1), "delta": round(sc - E.corpus_scores(base)["chrf++"], 1),
                    "ci95": [round(float(diffs[25]), 1), round(float(diffs[974]), 1)], "bad_pct": bad}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    for label, rows in report.items():
        print(f"\n{label}")
        for t, by in rows.items():
            cells = "  ".join(f"{d[4:9]} {v['chrf++']:5.1f} ({v['delta']:+.1f} [{v['ci95'][0]:+.1f},{v['ci95'][1]:+.1f}]) bad {v['bad_pct']:4.1f}%"
                              for d, v in by.items())
            print(f"  {t:30s} {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
