#!/usr/bin/env python3
"""One command for the Ongiini-Eval-OW automatic evaluation (docs/eval-protocol.md).

    ~/.venvs/ongiini-eval/bin/python scripts/run_evaluation.py

Steps: export the private eval set → run manifest → language ID (GlotLID)
→ learned metrics (SSA-COMET / COMET-22, in ~/.venvs/ongiini-comet) →
strict input checks → scores, paired bootstrap CIs, approximate-
randomisation significance clusters, failure rates, breakdowns, reference
points, system similarity, prompt robustness → results folder
data/private/results/<UTC date>/ with scores.json, scores.md and
items.jsonl (development split only). Then build_results_report.py.

When new references arrive (import_eval_translations.py), run it again:
the reference fingerprint in scores.json changes, the old folder stays.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_scoring as E  # noqa: E402
import system_similarity as SIM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/private/export/data/baselines"
EVAL_SET = ROOT / "data/private/export/data/eval_set.jsonl"
RESULTS = ROOT / "data/private/results"
ROBUST = BASE / "robustness"
EVAL_PY = Path.home() / ".venvs/ongiini-eval/bin/python"
COMET_PY = Path.home() / ".venvs/ongiini-comet/bin/python"
OSHIWAMBO = {"Oshindonga", "Oshikwanyama"}
SLICE_KEYS = ("phenomenon_tags", "length_bucket", "domain", "provenance")
ANCHOR_COPY, ANCHOR_WRONG = "anchor:english-copy", "anchor:wrong-sentence"


def run(cmd: list[str]) -> None:
    print("→", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def lid_group(label: str) -> str:
    return {"ndo_Latn": "Oshindonga", "kua_Latn": "Oshikwanyama", "her_Latn": "Herero",
            "kwn_Latn": "Kwangali", "tsn_Latn": "Tswana", "swh_Latn": "Swahili",
            "swa_Latn": "Swahili", "eng_Latn": "English", "empty": "empty"}.get(label, "other")


def score_split(systems: dict[str, dict], ids: list[int], items: dict, dialect: str,
                lid: dict, with_details: bool) -> dict:
    refs = [E.normalise(items[i][f"{dialect}_reference"]) for i in ids]
    english = [E.normalise(items[i]["english"]) for i in ids]
    boot = E.bootstrap_indices(len(ids))
    out: dict = {"n": len(ids), "systems": {}}
    stats: dict[str, E.SystemStats] = {}
    for label, s in systems.items():
        raw = [s["raw"][i] for i in ids]
        hyps = [E.clean_hypothesis(r, dialect) for r in raw]
        st = E.SystemStats.build(hyps, refs)
        stats[label] = st
        sc = E.corpus_scores(st)
        entry = {m: round(v, 1) for m, v in sc.items()}
        lo, hi = E.bootstrap_ci(st, boot)
        entry["chrf++_ci"] = [round(lo, 1), round(hi, 1)]
        if with_details:
            entry |= E.failure_rates(english, raw, hyps)
            labels = (lid.get(label) or {}).get(dialect) or {}
            groups = Counter(lid_group(labels.get(str(i), ["?"])[0]) for i in ids) if labels else Counter()
            if groups:
                entry["lid"] = {g: round(100 * c / len(ids), 1) for g, c in groups.most_common()}
                foreign = {k for k, i in enumerate(ids)
                           if lid_group(labels.get(str(i), ["?"])[0]) not in OSHIWAMBO | {"empty"}}
                zeroed = [("" if k in foreign else h) for k, h in enumerate(hyps)]
                entry["chrf++_foreign_zero"] = round(E.METRICS[E.PRIMARY].corpus_score(zeroed, [refs]).score, 1)
        out["systems"][label] = entry
    out["_stats"] = stats
    out["_hyps"] = {k: v.hyps for k, v in stats.items()}
    out["_refs"] = refs
    return out


def significance(stats: dict[str, E.SystemStats], ranked: list[str]) -> dict:
    pvals = {(a, b): E.approx_randomisation(stats[a], stats[b]) for a, b in combinations(ranked, 2)}
    sig = E.holm(pvals)
    return {"clusters": E.clusters(ranked, sig),
            "pairs": {f"{a}|{b}": {"p": round(p, 4), "significant": sig[(a, b)]} for (a, b), p in pvals.items()}}


def slices(stats: dict[str, E.SystemStats], ids: list[int], items: dict, labels: list[str]) -> dict:
    out: dict = {}
    for key in SLICE_KEYS:
        groups: dict[str, list[int]] = defaultdict(list)
        for k, i in enumerate(ids):
            vals = items[i].get(key) or []
            for v in (vals if isinstance(vals, list) else [vals]):
                if v:
                    groups[v].append(k)
        out[key] = {}
        for g, idx in sorted(groups.items()):
            idx_a = np.asarray(idx)
            boot = E.bootstrap_indices(len(idx_a), samples=500)
            cell = {"n": len(idx), "small": len(idx) < E.SMALL_N, "systems": {}}
            for label in labels:
                st = stats[label]
                sub = E.SystemStats(hyps=[], refs=[], stats={E.PRIMARY: st.stats[E.PRIMARY][idx_a]})
                lo, hi = E.bootstrap_ci(sub, boot)
                cell["systems"][label] = {"chrf++": round(E.score_from(E.PRIMARY, sub.stats[E.PRIMARY].sum(0)), 1),
                                          "ci": [round(lo, 1), round(hi, 1)]}
            out[key][g] = cell
    return out


def robustness(main_labels: list[str], dev_ids: dict[str, list[int]], items: dict) -> dict | None:
    if not ROBUST.exists():
        return None
    from scipy.stats import kendalltau
    out: dict = {}
    for d in E.DIALECTS:
        ids = dev_ids[d]
        refs = [E.normalise(items[i][f"{d}_reference"]) for i in ids]
        per_template: dict[str, dict[str, float]] = defaultdict(dict)
        for p in sorted(ROBUST.glob(f"*__*_{d}.jsonl")):
            label, template = p.name[: -len(f"_{d}.jsonl")].split("__")
            try:
                raw, _ = E.load_system_file(p, d, ids)
            except E.InputError as e:
                print(f"  robustness skipped: {e}")
                continue
            hyps = [E.clean_hypothesis(raw[i], d) for i in ids]
            per_template[template][label] = round(E.METRICS[E.PRIMARY].corpus_score(hyps, [refs]).score, 1)
        out[d] = {"chrf++": per_template}
        temps = sorted(per_template)
        taus = {}
        for a, b in combinations(temps, 2):
            common = sorted(set(per_template[a]) & set(per_template[b]))
            if len(common) >= 3:
                taus[f"{a}|{b}"] = round(float(kendalltau([per_template[a][x] for x in common],
                                                          [per_template[b][x] for x in common])[0]), 2)
        out[d]["kendall_tau"] = taus
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-lid", action="store_true")
    ap.add_argument("--skip-comet", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out_dir = (args.out or RESULTS / datetime.now(timezone.utc).strftime("%Y-%m-%d")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run([EVAL_PY, "scripts/export_eval_set.py"])
    run([EVAL_PY, "scripts/build_run_manifest.py"])
    if not args.skip_lid:
        run([EVAL_PY, "scripts/lid_outputs.py", "--out", out_dir / "lid.json"])
    if not args.skip_comet and COMET_PY.exists():
        run([COMET_PY, "scripts/score_comet.py", "--out", out_dir / "comet.json"])

    items = E.load_items(EVAL_SET)
    manifest = json.loads((BASE / "manifest.json").read_text())
    lid = json.loads((out_dir / "lid.json").read_text()) if (out_dir / "lid.json").exists() else {}
    comet = json.loads((out_dir / "comet.json").read_text()) if (out_dir / "comet.json").exists() else {}

    dev_ids = {d: sorted(i for i, it in items.items() if not it["in_blind_split"]
                         and (it.get(f"{d}_reference") or "").strip()) for d in E.DIALECTS}
    blind_ids = {d: sorted(i for i, it in items.items() if it["in_blind_split"]
                           and (it.get(f"{d}_reference") or "").strip()) for d in E.DIALECTS}

    result: dict = {
        "protocol": E.PROTOCOL_VERSION, "signatures": E.signatures(),
        "reference_fingerprint": E.reference_fingerprint(items),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_dev": {d: len(v) for d, v in dev_ids.items()}, "n_blind": {d: len(v) for d, v in blind_ids.items()},
        "manifest": manifest, "lid_calibration": lid.get("calibration"),
        "lid_dialect_level_ok": lid.get("dialect_level_ok"), "dialects": {},
    }
    item_rows = []
    problems = []
    for d in E.DIALECTS:
        dres: dict = {}
        for split, ids_by in (("dev", dev_ids), ("blind", blind_ids)):
            ids = ids_by[d]
            systems: dict[str, dict] = {}
            for s in manifest["systems"]:
                if d not in s["files"] or s["role"] == "diagnostic":
                    continue
                try:
                    raw, _ = E.load_system_file(ROOT / s["files"][d], d, ids)
                except E.InputError as e:
                    problems.append(str(e))
                    continue
                systems[s["label"]] = {"raw": raw, "role": s["role"]}
            mapping = E.derangement(ids)
            systems[ANCHOR_COPY] = {"raw": {i: items[i]["english"] for i in ids}, "role": "anchor"}
            systems[ANCHOR_WRONG] = {"raw": {i: items[mapping[i]][f"{d}_reference"] for i in ids}, "role": "anchor"}
            res = score_split(systems, ids, items, d, lid.get("systems", {}), with_details=(split == "dev"))
            mains = [k for k, v in systems.items() if v["role"] == "main"]
            ranked = sorted(mains, key=lambda k: -res["systems"][k]["chrf++"])
            res["ranking"] = ranked
            res["significance"] = significance(res["_stats"], ranked)
            if split == "dev":
                res["slices"] = slices(res["_stats"], ids, items, ranked)
                for label, hyps in res["_hyps"].items():
                    for i, h, sc in zip(ids, hyps, E.sentence_chrf(hyps, res["_refs"])):
                        item_rows.append({"dialect": d, "id": i, "system": label, "chrf++": sc})
                sim_names = [k for k, v in systems.items() if v["role"] in ("main", "reference-point")] + [ANCHOR_COPY]
                res["similarity"] = SIM.similarity({k: res["_hyps"][k] for k in sim_names}, res["_refs"])
                for label in comet.get("systems", {}):
                    if label in res["systems"] and d in comet["systems"][label]:
                        res["systems"][label]["comet"] = comet["systems"][label][d]
            else:
                res.pop("_hyps")             # blind: headline only, nothing per item
            for k in ("_stats", "_hyps", "_refs"):
                res.pop(k, None)
            dres[split] = res
        result["dialects"][d] = dres

    # diagnostic systems (few-shot OkaLM): own item set without their prompt examples
    diag: dict = {}
    for s in (s for s in manifest["systems"] if s["role"] == "diagnostic"):
        for d in s["files"]:
            ids = [i for i in dev_ids[d] if i not in s.get("exclude_ids", [])]
            try:
                raw, _ = E.load_system_file(ROOT / s["files"][d], d, ids)
            except E.InputError as e:
                problems.append(str(e))
                continue
            refs = [E.normalise(items[i][f"{d}_reference"]) for i in ids]
            st = E.SystemStats.build([E.clean_hypothesis(raw[i], d) for i in ids], refs)
            diag.setdefault(s["label"], {})[d] = {"n": len(ids), **{m: round(v, 1) for m, v in E.corpus_scores(st).items()}}
    result["diagnostic"] = diag
    # self-similarity of the two dialects per system (dev, common ids)
    selfsim = {}
    common = sorted(set(dev_ids["oshindonga"]) & set(dev_ids["oshikwanyama"]))
    for s in manifest["systems"]:
        if set(s["files"]) == set(E.DIALECTS) and s["role"] != "diagnostic":
            try:
                a, _ = E.load_system_file(ROOT / s["files"]["oshindonga"], "oshindonga", common)
                b, _ = E.load_system_file(ROOT / s["files"]["oshikwanyama"], "oshikwanyama", common)
            except E.InputError:
                continue
            selfsim[s["label"]] = SIM.dialect_self_similarity(
                [E.clean_hypothesis(a[i], "oshindonga") for i in common],
                [E.clean_hypothesis(b[i], "oshikwanyama") for i in common])
    result["dialect_self_similarity"] = selfsim
    refsim = SIM.dialect_self_similarity([E.normalise(items[i]["oshindonga_reference"]) for i in common],
                                         [E.normalise(items[i]["oshikwanyama_reference"]) for i in common])
    result["dialect_self_similarity_references"] = refsim
    mains = [s["label"] for s in manifest["systems"] if s["role"] == "main"]
    result["robustness"] = robustness(mains, dev_ids, items)
    result["problems"] = problems

    (out_dir / "scores.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    with (out_dir / "items.jsonl").open("w") as f:
        for r in item_rows:
            f.write(json.dumps(r) + "\n")
    (out_dir / "scores.md").write_text(markdown(result))
    print(f"wrote {out_dir.relative_to(ROOT)}/scores.json, scores.md, items.jsonl")
    if problems:
        print("INPUT PROBLEMS (systems skipped):\n  " + "\n  ".join(problems))
    return 0


def markdown(r: dict) -> str:
    names = {s["label"]: s["name"] for s in r["manifest"]["systems"]}
    names |= {ANCHOR_COPY: "English copied", ANCHOR_WRONG: "Wrong sentence"}
    L = [f"# Ongiini-Eval-OW automatic scores — {r['generated']}", "",
         f"Protocol {r['protocol']} · references {r['reference_fingerprint']} · "
         f"dev n = {r['n_dev']} · blind n = {r['n_blind']} (internal)", "",
         "Signatures: " + "; ".join(f"{k}: `{v}`" for k, v in r["signatures"].items()), ""]
    for d, dres in r["dialects"].items():
        dev = dres["dev"]
        cl = {x: k + 1 for k, c in enumerate(dev["significance"]["clusters"]) for x in c}
        L += [f"## {d.capitalize()} — development split (n = {dev['n']})", "",
              "| # | system | chrF++ (95 % CI) | spBLEU | chrF | empty % | derailed % | copy % | Oshiwambo % (LID) | SSA-COMET |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        order = dev["ranking"] + [k for k in dev["systems"] if k not in dev["ranking"]]
        for k in order:
            s = dev["systems"][k]
            osh = sum(v for g, v in (s.get("lid") or {}).items() if g in OSHIWAMBO)
            com = (s.get("comet") or {}).get("ssa-comet-mtl", {}).get("mean", "–")
            L.append(f"| {cl.get(k, '·')} | {names.get(k, k)} | {s['chrf++']} ({s['chrf++_ci'][0]}–{s['chrf++_ci'][1]}) | "
                     f"{s['spbleu']} | {s['chrf']} | {s.get('empty_pct', '–')} | {s.get('derailed_pct', '–')} | "
                     f"{s.get('english_copy_pct', '–')} | {round(osh, 1) if s.get('lid') else '–'} | {com} |")
        blind = dres["blind"]
        L += ["", f"Blind split (internal, n = {blind['n']}): " + ", ".join(
            f"{names.get(k, k)} {blind['systems'][k]['chrf++']}" for k in blind["ranking"][:5]) + " …", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
