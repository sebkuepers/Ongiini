#!/usr/bin/env python3
"""Build the dialect-annotation sheets (docs/rating-protocol.md, "Dialect annotation").

Items, seed 42:
* system outputs — 8 main systems at evenly spaced ranks of development
  chrF++ (mean over both dialects), restricted to systems that take the
  target dialect as an instruction (single-direction MT models return the
  same text for both requests) and have ≤ 10 % empty outputs; 30
  development sentences with both references, each requested in both
  dialects → 480 rows;
* references — the development Oshikwanyama references GlotLID labels
  Oshindonga, plus a random sample of the other development references of
  both dialects, ≤ 120 rows in all;
* wrong language — 10 NLLB-200 → Tswana outputs (expected "not Oshiwambo");
* known controls — the reference translator's 20 clear texts, added with
  --controls once they exist (JSON list of {"text", "dialect"}).

Writes, under data/private/annotation/dialect_v1/:
* sheet_A.csv, sheet_B.csv — the same rows, shuffled differently, only
  row id and Oshiwambo text, plus empty label / comment columns;
* key.jsonl — row id → group, system, requested dialect, sentence id;
* plan.json — the chosen systems and counts (copy the systems into the
  protocol before annotation starts).

    ~/.venvs/ongiini-eval/bin/python scripts/build_dialect_annotation.py \\
        data/private/results/2026-10-01-interim [--controls controls.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_scoring as E  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/private/export/data/baselines"
EVAL_SET = ROOT / "data/private/export/data/eval_set.jsonl"
OUT = ROOT / "data/private/annotation/dialect_v1"
SEED, N_SYSTEMS, N_SENTENCES, MAX_REFS, N_WRONG = 42, 8, 30, 120, 10
SINGLE_OUTPUT = ("mt",)                     # categories that ignore the requested dialect
LABELS = "Oshindonga | Oshikwanyama | mixed | not Oshiwambo | can't tell"


def choose_systems(scores: dict) -> list[str]:
    cands = []
    for s in scores["manifest"]["systems"]:
        if s["role"] != "main" or s["category"] in SINGLE_OUTPUT or len(s.get("files", {})) < 2:
            continue
        per = [scores["dialects"][d]["dev"]["systems"].get(s["label"]) for d in E.DIALECTS]
        if None in per or max(p["empty_pct"] for p in per) > 10:
            continue
        cands.append((sum(p["chrf++"] for p in per) / 2, s["label"]))
    cands.sort(reverse=True)
    if len(cands) <= N_SYSTEMS:
        return [c[1] for c in cands]
    step = (len(cands) - 1) / (N_SYSTEMS - 1)
    return [cands[round(k * step)][1] for k in range(N_SYSTEMS)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--controls", type=Path)
    args = ap.parse_args(argv)
    scores = json.loads((args.results / "scores.json").read_text())
    items = E.load_items(EVAL_SET)
    rng = random.Random(SEED)
    dev_both = sorted(i for i, r in items.items() if not r["in_blind_split"]
                      and all((r.get(f"{d}_reference") or "").strip() for d in E.DIALECTS))
    rows: list[dict] = []

    systems = choose_systems(scores)
    sentences = sorted(rng.sample(dev_both, N_SENTENCES))
    for label in systems:
        for d in E.DIALECTS:
            raw, _ = E.load_system_file(BASE / f"{label}_{d}.jsonl", d, sentences)
            for i in sentences:
                text = E.clean_hypothesis(raw[i], d)
                if text:
                    rows.append({"group": "system", "system": label, "requested": d, "sentence_id": i, "text": text})

    from lid_outputs import group, load_model, predict          # fasttext lives in the eval venv
    model = load_model()
    dev_refs = {d: sorted(i for i, r in items.items() if not r["in_blind_split"]
                          and (r.get(f"{d}_reference") or "").strip()) for d in E.DIALECTS}
    kua = dev_refs["oshikwanyama"]
    kua_lid = predict(model, [items[i]["oshikwanyama_reference"] for i in kua])
    flagged = [i for i, (lab, _) in zip(kua, kua_lid) if group(lab) == "Oshindonga"]
    refs = [("oshikwanyama", i, "glotlid_ndonga") for i in flagged][:MAX_REFS]
    rest = [(d, i) for d in E.DIALECTS for i in dev_refs[d] if not (d == "oshikwanyama" and i in flagged)]
    rng.shuffle(rest)
    refs += [(d, i, "random") for d, i in rest[: max(0, MAX_REFS - len(refs))]]
    for d, i, why in refs:
        rows.append({"group": "reference", "system": f"reference:{why}", "requested": d,
                     "sentence_id": i, "text": E.normalise(items[i][f"{d}_reference"])})

    tsn, _ = E.load_system_file(BASE / "nllb-200-3.3b_oshindonga.jsonl", "oshindonga", dev_both)
    for i in sorted(rng.sample([i for i in dev_both if i not in sentences], N_WRONG)):
        rows.append({"group": "wrong_language", "system": "nllb-200-3.3b", "requested": "tswana",
                     "sentence_id": i, "text": E.normalise(tsn[i])})

    if args.controls:
        for k, c in enumerate(json.loads(args.controls.read_text())):
            rows.append({"group": "control", "system": "known", "requested": c["dialect"],
                         "sentence_id": None, "text": E.normalise(c["text"]), "control_no": k})

    for n, r in enumerate(rows, start=1):
        r["row"] = f"R{n:04d}"
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "key.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for sheet, seed in (("A", 1), ("B", 2)):
        order = rows[:]
        random.Random(seed).shuffle(order)
        with (OUT / f"sheet_{sheet}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "text", f"label ({LABELS})", "comment"])
            for r in order:
                w.writerow([r["row"], r["text"], "", ""])
    counts = {g: sum(r["group"] == g for r in rows) for g in ("system", "reference", "wrong_language", "control")}
    plan = {"systems": systems, "sentences": sentences, "counts": counts,
            "kwanyama_refs_glotlid_ndonga": len(flagged), "seed": SEED}
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2))
    print(json.dumps({k: v for k, v in plan.items() if k != "sentences"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
