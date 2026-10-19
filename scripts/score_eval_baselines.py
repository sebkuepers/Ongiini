#!/usr/bin/env python3
"""Score baseline system outputs against the eval-set references.

Reads the references from an export of the eval set (eval_set.jsonl —
by default the private preview written by export_eval_set.py) and one
or more system files in the baselines/*.jsonl format
({"id", "oshindonga_output", "oshikwanyama_output"}).

Reports corpus chrF++ and BLEU (sacrebleu) per dialect, plus the
derailment rate (runaway / meta-commentary outputs, see
retry_derailed_baselines.py), sliced by split, length bucket, domain and
phenomenon tag, next to two floor anchors (English copied; a reference
of another item). Only items that have references are scored.

Default is *sealed*: blind items are not scored at all and the headline
is the development split — for internal previews that may steer our own
system choices. --blind switches to the release / external-submission
view: blind-split headline with a 95% bootstrap CI on chrF++.

Output is aggregate only — no sentences — so the report is safe to
share. Everything is computed locally; references never leave the box.

    python3 scripts/score_eval_baselines.py \\
        --system claude=data/oshiwambo_eval/data/baselines/claude.jsonl \\
        --system gemma=data/private/export/data/baselines/gemma.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import sacrebleu

from retry_derailed_baselines import is_derailed

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFS = ROOT / "data/private/export/data/eval_set.jsonl"
DEFAULT_OUT = ROOT / "data/private/results"
DIALECTS = ("oshindonga", "oshikwanyama")
CHRF = sacrebleu.CHRF(word_order=2)
BLEU = sacrebleu.BLEU()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_system(paths: list[Path]) -> dict[int, dict]:
    """Accept both the baselines format (one line per item with
    <dialect>_output) and the public submission schema (one line per
    item and dialect with `dialect` + `translation`)."""
    outs: dict[int, dict] = {}
    for o in (o for path in paths for o in load_jsonl(path)):
        if "translation" in o:
            outs.setdefault(int(o["id"]), {"id": int(o["id"])})[
                f"{o['dialect']}_output"] = o["translation"]
        else:
            outs.setdefault(int(o["id"]), {"id": int(o["id"])}).update(o)
    return outs


def score(pairs: list[tuple[str, str, str]]) -> dict:
    """pairs: (english, hypothesis, reference)."""
    hyps = [h for _, h, _ in pairs]
    refs = [[r for _, _, r in pairs]]
    return {
        "n": len(pairs),
        "chrf++": round(CHRF.corpus_score(hyps, refs).score, 1),
        "bleu": round(BLEU.corpus_score(hyps, refs).score, 1),
        "derailed_pct": round(100 * sum(is_derailed(e, h) for e, h, _ in pairs) / len(pairs), 1),
    }


def bootstrap_ci(pairs: list[tuple[str, str, str]], n: int = 1000, seed: int = 42) -> list[float]:
    rng = random.Random(seed)
    scores = []
    for _ in range(n):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        scores.append(CHRF.corpus_score([h for _, h, _ in s], [[r for _, _, r in s]]).score)
    scores.sort()
    return [round(scores[int(0.025 * n)], 1), round(scores[int(0.975 * n)], 1)]


ANCHOR_COPY = "anchor: English copied"
ANCHOR_WRONG = "anchor: wrong sentence"


def anchors(items: dict[int, dict], seed: int = 42) -> dict[str, dict[int, dict]]:
    """Two floor references scored like systems: the English source copied
    unchanged, and a correct-language reference of a different item
    (seeded shuffle). A system near these is not translating."""
    rng = random.Random(seed)
    ids = sorted(items)
    copy = {i: {f"{d}_output": items[i]["english"] for d in DIALECTS} for i in ids}
    wrong: dict[int, dict] = {i: {} for i in ids}
    for d in DIALECTS:
        shuffled = ids[:]
        rng.shuffle(shuffled)
        for i, j in zip(ids, shuffled):
            wrong[i][f"{d}_output"] = items[j][f"{d}_reference"]
    return {ANCHOR_COPY: copy, ANCHOR_WRONG: wrong}


def main(argv: list[str] | None = None) -> int:
    global DIALECTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", type=Path, default=DEFAULT_REFS)
    ap.add_argument("--system", action="append", required=True,
                    help="name=file.jsonl[,file2.jsonl] — e.g. one file per "
                         "dialect (repeatable)")
    ap.add_argument("--blind", action="store_true",
                    help="Report the blind split as headline (release and "
                         "external submissions). Default: sealed — blind items "
                         "are not scored at all, headline and slices use the "
                         "development split, so the numbers can steer our own "
                         "system choices without touching the blind set.")
    ap.add_argument("--dialects", default=",".join(DIALECTS),
                    help="Comma-separated subset, e.g. oshikwanyama for a "
                         "single-dialect system")
    ap.add_argument("--exclude-ids", default="",
                    help="Comma-separated ids, or a JSON file of ids, to drop "
                         "for every system (e.g. few-shot prompt examples)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    DIALECTS = tuple(d for d in args.dialects.split(",") if d)
    exclude: set[int] = set()
    for part in filter(None, args.exclude_ids.split(",")):
        exclude |= (set(json.loads(Path(part).read_text())) if part.endswith(".json")
                    else {int(part)})

    all_items = {r["id"]: r for r in load_jsonl(args.refs)}
    items = {i: it for i, it in all_items.items()
             if all(it[f"{d}_reference"].strip() for d in DIALECTS)
             and (args.blind or not it["in_blind_split"])
             and i not in exclude}
    if not items:
        print("ERROR: no items with references in scope", file=sys.stderr)
        return 1
    systems = {}
    for spec in args.system:
        name, paths = spec.split("=", 1)
        outs = load_system([Path(p) for p in paths.split(",")])
        missing = [f"{i}/{d}" for i in items for d in DIALECTS
                   if f"{d}_output" not in outs.get(i, {})]
        if missing:
            print(f"ERROR: {name} lacks {len(missing)} outputs, e.g. {missing[:3]}",
                  file=sys.stderr)
            return 1
        systems[name] = outs
    systems.update(anchors(items))

    headline_split = "blind" if args.blind else "development"
    report: dict = {"headline_split": headline_split,
                    "items_scored": len(items),
                    "excluded_ids": sorted(exclude),
                    "dialects": list(DIALECTS),
                    "items_without_references": sum(
                        1 for it in all_items.values()
                        if not all(it[f"{d}_reference"].strip() for d in DIALECTS)),
                    "systems": {}}
    for name, outs in systems.items():
        sys_report = {}
        for d in DIALECTS:
            def pairs(pred) -> list[tuple[str, str, str]]:
                return [(it["english"], outs[i][f"{d}_output"].strip(),
                         it[f"{d}_reference"]) for i, it in items.items() if pred(it)]

            head = pairs(lambda it: it["in_blind_split"] == args.blind)
            slices = defaultdict(dict)
            for field in ("length_bucket", "domain"):
                for v in sorted({it[field] for it in items.values()}):
                    slices[field][v] = score(pairs(lambda it, f=field, v=v: it[f] == v))
            for tag in sorted({t for it in items.values() for t in it["phenomenon_tags"]}):
                slices["phenomenon"][tag] = score(pairs(lambda it, t=tag: t in it["phenomenon_tags"]))
            sys_report[d] = {
                "headline": {**score(head), "chrf++_ci95": bootstrap_ci(head)},
                "slices": slices,
            }
            if args.blind:
                sys_report[d]["development"] = score(pairs(lambda it: not it["in_blind_split"]))
        report["systems"][name] = sys_report

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scores.json").write_text(json.dumps(report, indent=2))
    (args.out / "scores.md").write_text(render_md(report))
    print(render_md(report))
    print(f"\nwrote {args.out}/scores.{{json,md}}", file=sys.stderr)
    return 0


def render_md(report: dict) -> str:
    names = list(report["systems"])
    split = report["headline_split"]
    scope = ("blind split" if split == "blind" else
             "development split (blind sealed)")
    lines = [f"## Headline — {scope}, chrF++ with 95% bootstrap CI", "",
             f"{report['items_scored']} items with references in scope; "
             f"{report['items_without_references']} items still without references"
             + (f"; excluded for all systems: {report['excluded_ids']}" if report["excluded_ids"] else "")
             + ".", "",
             "| System | Dialect | n | chrF++ | 95% CI | BLEU | derailed |",
             "|---|---|---|---|---|---|---|"]
    for n in names:
        for d in DIALECTS:
            b = report["systems"][n][d]["headline"]
            derailed = "–" if n.startswith("anchor:") else f"{b['derailed_pct']}%"
            lines.append(f"| {n} | {d} | {b['n']} | {b['chrf++']} | "
                         f"{b['chrf++_ci95'][0]}–{b['chrf++_ci95'][1]} | "
                         f"{b['bleu']} | {derailed} |")
    sliced = [n for n in names if not n.startswith("anchor:")]
    where = "all scored items" if split == "blind" else "development split"
    for field, title in (("length_bucket", "Length bucket"), ("domain", "Domain"),
                         ("phenomenon", "Phenomenon")):
        lines += ["", f"## {title} — {where}, diagnostic (chrF++)", "",
                  "| " + title + " | n | " + " | ".join(f"{n} {d[:5]}" for n in sliced for d in DIALECTS) + " |",
                  "|---|---|" + "---|" * (len(sliced) * len(DIALECTS))]
        first = report["systems"][sliced[0]][DIALECTS[0]]["slices"][field]
        for v, s in first.items():
            cells = [str(report["systems"][n][d]["slices"][field][v]["chrf++"])
                     for n in sliced for d in DIALECTS]
            lines.append(f"| {v} | {s['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
