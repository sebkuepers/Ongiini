#!/usr/bin/env python3
"""Score local Gemma runs (base vs adapters) on the benchmark (paper 3).

Reads <label>_<dialect>.jsonl from the experiments folder (written by
eval_lora_generate.py) and compares every label with the base label:
chrF++ / spBLEU on the dev and blind splits, paired bootstrap CI of the
chrF++ difference, approximate randomisation p on dev, loop counts, and the
dialect-drift diagnostics (Kwanyama-request outputs vs the Ndonga reference,
similarity of a run's two dialect outputs). Writes scores.json next to the
runs and prints a short table.

    python3 scripts/score_lora_runs.py --base gemma-4-12b-base gemma-4-12b-A gemma-4-12b-B10k
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_scoring as E  # noqa: E402
import system_similarity as SIM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / os.environ.get("ONGIINI_EXP_DIR", "data/private/experiments") / "lora"  # override: pipeline tests


def loops(raw: dict, items: dict, ids: list[int]) -> int:
    return sum(len(raw[i]) > 3 * max(20, len(items[i]["english"])) for i in ids)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--base", required=True)
    args = ap.parse_args(argv)
    items = E.load_items(ROOT / "data/private/export/data/eval_set.jsonl")
    out: dict = {}
    for label in [args.base, *args.labels]:
        if not all((EXP / f"{label}_{d}.jsonl").exists() for d in E.DIALECTS):
            continue
        res: dict = {}
        for d in E.DIALECTS:
            for split in ("dev", "blind"):
                ids = sorted(i for i, r in items.items() if r["in_blind_split"] == (split == "blind")
                             and (r.get(f"{d}_reference") or "").strip())
                refs = [E.normalise(items[i][f"{d}_reference"]) for i in ids]
                raw_b, _ = E.load_system_file(EXP / f"{args.base}_{d}.jsonl", d, ids)
                raw_x, _ = E.load_system_file(EXP / f"{label}_{d}.jsonl", d, ids)
                b = E.SystemStats.build([E.clean_hypothesis(raw_b[i], d) for i in ids], refs)
                x = E.SystemStats.build([E.clean_hypothesis(raw_x[i], d) for i in ids], refs)
                sx = E.corpus_scores(x)
                cell = {"n": len(ids), "chrf++": round(sx["chrf++"], 1), "spbleu": round(sx["spbleu"], 1),
                        "loops": loops(raw_x, items, ids)}
                if label != args.base:
                    boot = E.bootstrap_indices(len(ids))
                    diffs = np.sort([E.score_from("chrf++", x.stats["chrf++"][ix].sum(0))
                                     - E.score_from("chrf++", b.stats["chrf++"][ix].sum(0)) for ix in boot])
                    cell |= {"delta_vs_base": round(sx["chrf++"] - E.corpus_scores(b)["chrf++"], 1),
                             "delta_ci95": [round(float(diffs[25]), 1), round(float(diffs[974]), 1)]}
                    if split == "dev":
                        cell["p_ar"] = round(E.approx_randomisation(b, x), 4)
                res[f"{d}_{split}"] = cell
        both = sorted(i for i, r in items.items() if not r["in_blind_split"]
                      and all((r.get(f"{d}_reference") or "").strip() for d in E.DIALECTS))
        o = {d: E.load_system_file(EXP / f"{label}_{d}.jsonl", d, both)[0] for d in E.DIALECTS}
        hk = [E.clean_hypothesis(o["oshikwanyama"][i], "oshikwanyama") for i in both]
        hn = [E.clean_hypothesis(o["oshindonga"][i], "oshindonga") for i in both]
        ndo_ref = [E.normalise(items[i]["oshindonga_reference"]) for i in both]
        # dev slices by phenomenon tag and domain (small n: read as tendencies)
        sl: dict = {}
        for d in E.DIALECTS:
            ids = sorted(i for i, r in items.items() if not r["in_blind_split"] and (r.get(f"{d}_reference") or "").strip())
            raw_x, _ = E.load_system_file(EXP / f"{label}_{d}.jsonl", d, ids)
            groups: dict = {}
            for i in ids:
                for t in (items[i].get("phenomenon_tags") or []):
                    groups.setdefault(f"tag:{t}", []).append(i)
                groups.setdefault(f"domain:{items[i]['domain']}", []).append(i)
            for g, gi in groups.items():
                h = [E.clean_hypothesis(raw_x[i], d) for i in gi]
                r = [E.normalise(items[i][f"{d}_reference"]) for i in gi]
                sl.setdefault(g, {})[d] = {"n": len(gi), "chrf++": round(E.METRICS["chrf++"].corpus_score(h, [r]).score, 1)}
        res["slices"] = sl
        res["drift"] = {"kua_request_vs_ndo_ref": round(E.METRICS["chrf++"].corpus_score(hk, [ndo_ref]).score, 1),
                        "self_similarity": SIM.dialect_self_similarity(hn, hk),
                        "identical": sum(a == c for a, c in zip(hn, hk))}
        out[label] = res
    (EXP / "scores.json").write_text(json.dumps(out, indent=2))
    for label, res in out.items():
        print(f"{label:22s} ndo dev {res['oshindonga_dev']['chrf++']:5.1f}  blind {res['oshindonga_blind']['chrf++']:5.1f} | "
              f"kua dev {res['oshikwanyama_dev']['chrf++']:5.1f} | loops ndo {res['oshindonga_dev']['loops']} | "
              f"drift kua→ndo {res['drift']['kua_request_vs_ndo_ref']} self-sim {res['drift']['self_similarity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
