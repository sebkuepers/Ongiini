#!/usr/bin/env python3
"""Score baseline system outputs against the eval-set references.

Reads the references from an export of the eval set (eval_set.jsonl —
by default the private preview written by export_eval_set.py) and one
or more system files in the baselines/*.jsonl format
({"id", "oshindonga_output", "oshikwanyama_output"}).

Reports corpus chrF++ and BLEU (sacrebleu) per dialect, plus the
derailment rate (runaway / meta-commentary outputs, see
retry_derailed_baselines.py), sliced by split, length bucket, domain and
phenomenon tag. Headline numbers come from the blind split with a 95%
bootstrap CI on chrF++; the slices use the full set and are diagnostic
only (the blind split is too small to slice).

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", type=Path, default=DEFAULT_REFS)
    ap.add_argument("--system", action="append", required=True,
                    help="name=path/to/baseline.jsonl (repeatable)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    items = {r["id"]: r for r in load_jsonl(args.refs)}
    systems = {}
    for spec in args.system:
        name, path = spec.split("=", 1)
        outs = {o["id"]: o for o in load_jsonl(Path(path))}
        if set(outs) != set(items):
            print(f"ERROR: {name} covers {len(outs)} ids, refs have {len(items)}",
                  file=sys.stderr)
            return 1
        systems[name] = outs

    report: dict = {"systems": {}}
    for name, outs in systems.items():
        sys_report = {}
        for d in DIALECTS:
            def pairs(pred) -> list[tuple[str, str, str]]:
                return [(it["english"], outs[i][f"{d}_output"].strip(),
                         it[f"{d}_reference"]) for i, it in items.items() if pred(it)]

            blind = pairs(lambda it: it["in_blind_split"])
            slices = defaultdict(dict)
            for field in ("length_bucket", "domain"):
                for v in sorted({it[field] for it in items.values()}):
                    slices[field][v] = score(pairs(lambda it, f=field, v=v: it[f] == v))
            for tag in sorted({t for it in items.values() for t in it["phenomenon_tags"]}):
                slices["phenomenon"][tag] = score(pairs(lambda it, t=tag: t in it["phenomenon_tags"]))
            sys_report[d] = {
                "blind": {**score(blind), "chrf++_ci95": bootstrap_ci(blind)},
                "development": score(pairs(lambda it: not it["in_blind_split"])),
                "full": score(pairs(lambda it: True)),
                "slices_full": slices,
            }
        report["systems"][name] = sys_report

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scores.json").write_text(json.dumps(report, indent=2))
    (args.out / "scores.md").write_text(render_md(report))
    print(render_md(report))
    print(f"\nwrote {args.out.relative_to(ROOT)}/scores.{{json,md}}", file=sys.stderr)
    return 0


def render_md(report: dict) -> str:
    names = list(report["systems"])
    lines = ["## Headline — blind split (chrF++ with 95% bootstrap CI)", "",
             "| System | Dialect | n | chrF++ | 95% CI | BLEU | derailed |",
             "|---|---|---|---|---|---|---|"]
    for n in names:
        for d in DIALECTS:
            b = report["systems"][n][d]["blind"]
            lines.append(f"| {n} | {d} | {b['n']} | {b['chrf++']} | "
                         f"{b['chrf++_ci95'][0]}–{b['chrf++_ci95'][1]} | "
                         f"{b['bleu']} | {b['derailed_pct']}% |")
    for field, title in (("length_bucket", "Length bucket"), ("domain", "Domain"),
                         ("phenomenon", "Phenomenon")):
        lines += ["", f"## {title} — full set, diagnostic (chrF++)", "",
                  "| " + title + " | n | " + " | ".join(f"{n} {d[:5]}" for n in names for d in DIALECTS) + " |",
                  "|---|---|" + "---|" * (len(names) * len(DIALECTS))]
        first = report["systems"][names[0]][DIALECTS[0]]["slices_full"][field]
        for v, s in first.items():
            cells = [str(report["systems"][n][d]["slices_full"][field][v]["chrf++"])
                     for n in names for d in DIALECTS]
            lines.append(f"| {v} | {s['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
