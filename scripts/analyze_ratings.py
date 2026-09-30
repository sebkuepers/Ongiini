#!/usr/bin/env python3
"""Analyse the ongiini.ai/rate/ ratings (round 1: reference check).

Writes a private markdown report with, per dialect:
  1. reference acceptance — not-wrong (main) and good (secondary) with
     Wilson 95 % CIs; random sample, flagged top separately, plus a
     stratified whole-set estimate (the random sample excludes the
     flagged top, so it has to be re-weighted)
  2. paired reference vs Claude Opus 5 (same rater, same sentence):
     exact McNemar on wrong / not-wrong, difference with 95 % CI
  3. discrimination — reference vs Gemma vs planted errors; without a
     clear gap the acceptance numbers say nothing
  4. flagged sentences — confirmed (2× wrong), split (needs a third
     rater), cleared; with issue tags and suggestions
  5. rater quality — control hits, repeat consistency, time per task;
     raters with ≥2 planted errors marked ✓ are flagged, and (1) is
     reported with and without them

Only first ratings count for the estimates (repeats measure consistency);
"can't judge" is excluded and counted.

    python3 scripts/analyze_ratings.py --db data/private/ratings.sqlite [--out report.md]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT / "data/private/oshiwambo_eval_v3.tsv"
CONTROLS = ("control_error", "control_wrong")
FAST_MS = 3000


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pct(x: float) -> str:
    return "–" if x != x else f"{100 * x:.0f} %"


def ci(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    return f"{pct(p)} ({pct(lo)}–{pct(hi)}, n={n})" if n else "– (n=0)"


def dev_ref_counts() -> dict[str, int]:
    if not TSV.exists():
        return {}
    with TSV.open() as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["in_blind_split"] == "false"]
    return {d: sum(1 for r in rows if r.get(f"kaarina_{d}", "").strip()) for d in ("oshindonga", "oshikwanyama")}


def load(db: Path):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT r.*, i.round, i.sentence_id, i.dialect, i.english, i.text, i.grp, i.kind, i.source, "
        "i.expected, i.pair_of, i.severity, ra.label FROM ratings r JOIN items i USING (item_key) "
        "JOIN raters ra USING (rater_id) ORDER BY r.rater_id, r.seq")]
    items = [dict(r) for r in con.execute("SELECT * FROM items")]
    return rows, items


def rater_qc(rows: list[dict]) -> dict[int, dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["rater_id"]].append(r)
    out = {}
    for rid, rs in by.items():
        ctrl = [r for r in rs if r["grp"] in CONTROLS and not r["is_repeat"] and r["verdict"] != "cant_judge"]
        firsts = {r["item_key"]: r for r in rs if not r["is_repeat"]}
        reps = [(firsts[r["item_key"]], r) for r in rs if r["is_repeat"] and r["item_key"] in firsts]
        durs = sorted(r["duration_ms"] for r in rs if r["duration_ms"] is not None)
        out[rid] = {
            "label": rs[0]["label"], "n": len(rs),
            "controls": len(ctrl), "caught": sum(r["verdict"] == "wrong" for r in ctrl),
            "passed_as_good": sum(r["verdict"] == "good" for r in ctrl if r["grp"] == "control_error"),
            "repeats": len(reps), "same_verdict": sum(a["verdict"] == b["verdict"] for a, b in reps),
            "same_wrong": sum((a["verdict"] == "wrong") == (b["verdict"] == "wrong") for a, b in reps),
            "median_s": durs[len(durs) // 2] / 1000 if durs else float("nan"),
            "fast": sum(d < FAST_MS for d in durs),
            "cant": sum(r["verdict"] == "cant_judge" for r in rs),
        }
        out[rid]["flagged"] = out[rid]["passed_as_good"] >= 2
    return out


def acceptance(rs: list[dict]) -> tuple[int, int, int]:
    judged = [r for r in rs if r["verdict"] != "cant_judge"]
    return (sum(r["verdict"] != "wrong" for r in judged), sum(r["verdict"] == "good" for r in judged),
            len(judged))


def stratified(rand: list[dict], flag: list[dict], n_flag_items: int, n_total: int | None):
    """Whole-set not-wrong estimate: flagged top weighted by its share of the dev refs."""
    kr, _, nr = acceptance(rand)
    kf, _, nf = acceptance(flag)
    if not n_total or not nr or not nf:
        return None
    w = n_flag_items / n_total
    pr, pf = kr / nr, kf / nf
    p = (1 - w) * pr + w * pf
    se = math.sqrt((1 - w) ** 2 * pr * (1 - pr) / nr + w ** 2 * pf * (1 - pf) / nf)
    return p, max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def report(rows: list[dict], items: list[dict], exclude: set[int] = frozenset()) -> list[str]:
    qc = rater_qc(rows)
    flagged_raters = {rid for rid, q in qc.items() if q["flagged"]}
    totals = dev_ref_counts()
    L = ["# Rating report — reference check", ""]
    L.append(f"{len(rows)} answers from {len(qc)} raters · {sum(r['is_repeat'] for r in rows)} repeats · "
             f"{sum(r['verdict'] == 'cant_judge' for r in rows)} can't judge")
    L.append("")
    firsts = [r for r in rows if not r["is_repeat"] and r["rater_id"] not in exclude]
    for d in sorted({i["dialect"] for i in items}):
        F = [r for r in firsts if r["dialect"] == d]
        g = lambda name, rs=F: [r for r in rs if r["grp"] == name]
        L += [f"## {d.capitalize()}", "", "### 1. Reference acceptance", "",
              "| sample | not wrong (main) | good |", "|---|---|---|"]
        for name, label in (("random_ref", "random sample"), ("flagged_ref", "flagged top (back-translation)")):
            k, kg, n = acceptance(g(name))
            L.append(f"| {label} | {ci(k, n)} | {ci(kg, n)} |")
        clean = [r for r in g("random_ref") if r["rater_id"] not in flagged_raters]
        if flagged_raters:
            k, kg, n = acceptance(clean)
            L.append(f"| random, without flagged raters | {ci(k, n)} | {ci(kg, n)} |")
        n_flag = sum(1 for i in items if i["dialect"] == d and i["grp"] == "flagged_ref")
        st = stratified(g("random_ref"), g("flagged_ref"), n_flag, totals.get(d))
        if st:
            L.append(f"| whole dev set (stratified, approx.) | {pct(st[0])} ({pct(st[1])}–{pct(st[2])}) | |")
        L.append("")

        # 2. paired
        mine = {(r["rater_id"], r["item_key"]): r for r in F}
        pairs = [(mine.get((r["rater_id"], r["pair_of"])), r) for r in g("claude_pair")]
        pairs = [(a, b) for a, b in pairs if a and a["verdict"] != "cant_judge" and b["verdict"] != "cant_judge"]
        n = len(pairs)
        b_ = sum(a["verdict"] != "wrong" and c["verdict"] == "wrong" for a, c in pairs)
        c_ = sum(a["verdict"] == "wrong" and c["verdict"] != "wrong" for a, c in pairs)
        L += ["### 2. Reference vs Claude Opus 5 (paired, same rater)", ""]
        if n:
            ref_ok = sum(a["verdict"] != "wrong" for a, _ in pairs) / n
            cl_ok = sum(c["verdict"] != "wrong" for _, c in pairs) / n
            rp = [(a, c) for a, c in pairs if a["grp"] == "random_ref"]
            diff = (b_ - c_) / n
            se = math.sqrt(max(0.0, (b_ + c_) - (b_ - c_) ** 2 / n)) / n
            good_diff = (sum(a["verdict"] == "good" for a, _ in pairs)
                         - sum(c["verdict"] == "good" for _, c in pairs)) / n
            L += [f"{n} pairs · not wrong: reference {pct(ref_ok)}, Claude {pct(cl_ok)}",
                  f"- difference (reference − Claude): {100 * diff:+.0f} points, 95 % CI "
                  f"{100 * (diff - 1.96 * se):+.0f} to {100 * (diff + 1.96 * se):+.0f}",
                  f"- discordant: reference ok / Claude wrong {b_}, reference wrong / Claude ok {c_} · "
                  f"exact McNemar p = {mcnemar_exact(b_, c_):.2f}",
                  f"- good (secondary): {100 * good_diff:+.0f} points",
                  f"- random-sample pairs only ({len(rp)}): reference {pct(sum(a['verdict'] != 'wrong' for a, _ in rp) / len(rp)) if rp else '–'}, "
                  f"Claude {pct(sum(c['verdict'] != 'wrong' for _, c in rp) / len(rp)) if rp else '–'} "
                  "(the other pairs are flagged sentences, so the overall reference rate is lower by design)",
                  "- reading: \"not worse than Claude\" holds if the lower CI bound is above about −10 points; "
                  f"with {n} pairs only large gaps are detectable.", ""]
        else:
            L += ["no complete pairs yet", ""]

        # 3. discrimination
        L += ["### 3. Discrimination (not-wrong rate)", "", "| translation | not wrong |", "|---|---|"]
        for name, label in (("random_ref", "reference (random)"), ("claude_pair", "Claude Opus 5"),
                            ("gemma", "Gemma 4 26B"), ("control_error", "planted error"),
                            ("control_wrong", "other sentence")):
            k, _, n = acceptance(g(name))
            L.append(f"| {label} | {ci(k, n)} |")
        kr, _, nr = acceptance(g("random_ref"))
        ke, _, ne = acceptance(g("control_error"))
        if nr and ne:
            ok = wilson(ke, ne)[2] < wilson(kr, nr)[1]
            L += ["", "✓ planted errors clearly below the references — the scale discriminates." if ok else
                  "⚠ no clear gap between references and planted errors yet — do not draw conclusions "
                  "about reference quality."]
        L.append("")

        # 4. flagged
        L += ["### 4. Flagged sentences", ""]
        by_item = defaultdict(list)
        for r in g("flagged_ref"):
            by_item[r["item_key"]].append(r)
        conf, split, clear, open_ = [], [], [], []
        for i in sorted((i for i in items if i["dialect"] == d and i["grp"] == "flagged_ref"),
                        key=lambda i: (-i["severity"], i["sentence_id"])):
            vs = [r for r in by_item.get(i["item_key"], []) if r["verdict"] != "cant_judge"]
            wrong = sum(r["verdict"] == "wrong" for r in vs)
            bucket = (open_ if len(vs) < 2 else conf if wrong == len(vs) else
                      clear if wrong == 0 else split)
            bucket.append((i, vs))
        L.append(f"confirmed wrong {len(conf)} · split (needs a third rater) {len(split)} · "
                 f"cleared {len(clear)} · not yet 2 ratings {len(open_)}")
        for title, bucket in (("Confirmed", conf), ("Split", split)):
            if not bucket:
                continue
            L += ["", f"**{title}**", ""]
            for i, vs in bucket:
                L.append(f"- #{i['sentence_id']} (severity {i['severity']}) — {i['english']}")
                L.append(f"  - reference: {i['text']}")
                for r in vs:
                    tags = ", ".join(json.loads(r["issues"])) if r["issues"] else ""
                    L.append(f"  - {r['verdict']}{' · ' + tags if tags else ''}"
                             f"{' · suggests: ' + r['suggestion'] if r['suggestion'] else ''}")
        sugg = [r for r in g("random_ref") if r["suggestion"]]
        if sugg:
            L += ["", "**Suggestions on random-sample references**", ""]
            for r in sugg:
                L.append(f"- #{r['sentence_id']} {r['verdict']}: {r['english']}\n"
                         f"  - reference: {r['text']}\n  - suggests: {r['suggestion']}")
        L.append("")

    # 5. raters
    L += ["## 5. Raters", "", "| rater | answers | controls caught | planted ✓ | repeats same (wrong/not) "
          "| median s | < 3 s | can't judge |", "|---|---|---|---|---|---|---|---|"]
    for rid, q in sorted(qc.items()):
        L.append(f"| {q['label']}{' ⚑' if q['flagged'] else ''} | {q['n']} | {q['caught']}/{q['controls']} | "
                 f"{q['passed_as_good']} | {q['same_verdict']}/{q['repeats']} ({q['same_wrong']}) | "
                 f"{q['median_s']:.0f} | {q['fast']} | {q['cant']} |")
    if flagged_raters:
        L += ["", "⚑ = marked ✓ Good on two or more planted errors; section 1 also shows results without them."]
    reasons = Counter(r["cant_reason"] for r in rows if r["verdict"] == "cant_judge")
    if reasons:
        L += ["", "Can't judge: " + ", ".join(f"{k} {v}" for k, v in reasons.most_common())]
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "data/private/ratings.sqlite")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    rows, items = load(args.db)
    text = "\n".join(report(rows, items)) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
