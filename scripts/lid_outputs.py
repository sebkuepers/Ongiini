#!/usr/bin/env python3
"""Language identification of system outputs with GlotLID v3 (docs/eval-protocol.md §4).

Labels every output (and the human references, for calibration) with the
GlotLID top label and probability, and groups the labels into the
categories the report uses. GlotLID covers Ndonga (ndo_Latn, F1 0.93)
and Kwanyama (kua_Latn, F1 0.94); calibration on our own references
decides whether dialect confusion can be reported as such.

    ~/.venvs/ongiini-eval/bin/python scripts/lid_outputs.py --out data/private/results/lid.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import fasttext
from huggingface_hub import hf_hub_download

import eval_scoring as E

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/private/export/data/baselines"
EVAL_SET = ROOT / "data/private/export/data/eval_set.jsonl"
CALIBRATION_MIN = 0.85

GROUPS = {
    "ndo_Latn": "Oshindonga", "kua_Latn": "Oshikwanyama",
    "her_Latn": "Herero", "kwn_Latn": "Kwangali", "tsn_Latn": "Tswana",
    "swh_Latn": "Swahili", "swa_Latn": "Swahili", "eng_Latn": "English",
}


def group(label: str) -> str:
    return GROUPS.get(label, "other")


def load_model():
    path = hf_hub_download("cis-lmu/glotlid", "model_v3.bin")
    return fasttext.load_model(path)


def predict(model, texts: list[str]) -> list[tuple[str, float]]:
    out = []
    for t in texts:
        t = E.normalise(t)
        if not t:
            out.append(("empty", 1.0))
            continue
        labels, probs = model.predict(t, k=1)
        out.append((labels[0].replace("__label__", ""), float(probs[0])))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    model = load_model()
    items = E.load_items(EVAL_SET)
    manifest = json.loads((BASE / "manifest.json").read_text())

    result: dict = {"model": "cis-lmu/glotlid model_v3.bin", "calibration": {}, "systems": {}}
    for d in E.DIALECTS:
        refs = {i: it[f"{d}_reference"] for i, it in items.items() if (it.get(f"{d}_reference") or "").strip()}
        preds = predict(model, list(refs.values()))
        c = Counter(group(l) for l, _ in preds)
        expected = "Oshindonga" if d == "oshindonga" else "Oshikwanyama"
        result["calibration"][d] = {"n": len(preds), "groups": dict(c),
                                    "correct_rate": round(c[expected] / len(preds), 3)}
    cal_ok = all(v["correct_rate"] >= CALIBRATION_MIN for v in result["calibration"].values())
    result["dialect_level_ok"] = cal_ok

    for s in manifest["systems"]:
        result["systems"][s["label"]] = {}
        for d, rel in s["files"].items():
            rows = E.load_jsonl(ROOT / rel)
            preds = predict(model, [E.clean_hypothesis(r.get("translation"), d) for r in rows])
            result["systems"][s["label"]][d] = {str(r["id"]): [l, round(p, 3)] for r, (l, p) in zip(rows, preds)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False))
    print(json.dumps(result["calibration"], indent=1), "dialect-level OK:", cal_ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
