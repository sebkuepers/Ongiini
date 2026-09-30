#!/usr/bin/env python3
"""Import translator references from a returned .docx into the private TSV.

Reads the phone-edited Word document produced by
generate_oshiwambo_eval_doc.py (blocks of "Phrase N of 423 · id #X",
"English:", "Oshindonga:", "Oshikwanyama:", separator) and writes the
translations into <translator>_* columns of data/private/oshiwambo_eval_v2.tsv.

The references are unreleased — the blind split must never reach the
public repo — so everything this script reads or writes lives under the
gitignored data/private/. The tracked data/oshiwambo_eval_v2.tsv holds
the English sources only.

Translator comments are typed inline as paragraphs starting with "NB:"
after the Oshikwanyama answer. The doc has no per-dialect notes slot and
most notes cover both dialects, so each note goes into both notes columns.

Items where the translator broke a label while typing on the phone are
resolved by hand in a <docx>.fixes.json sidecar ({"<id>": {"odg"|"okw"|
"notes": "..."}}), kept next to the docx because it contains reference
text. Any other irregular block aborts the import so a new document never
gets silently mis-parsed.

    python3 scripts/import_eval_translations.py --translator kaarina \
        data/private/ongiini-eval-kaarina-2026-09.docx
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TSVS = (ROOT / "data/oshiwambo_eval_v2.tsv", ROOT / "data/oshiwambo_eval_v3.tsv")
PRIVATE_TSV = ROOT / "data/private/oshiwambo_eval_v3.tsv"

HEADER = re.compile(r"^Phrase \d+ of \d+\s*·\s*id #(\d+)")
NOTE = re.compile(r"^(NB|N\.B\.)\s*[:.]", re.I)

def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


NA = re.compile(r"^n\s*/?\s*a\b", re.I)


def parse_tables(doc) -> dict[int, dict]:
    """v3 layout: one 2-column table per phrase, rows labelled ID /
    English / Oshindonga / Oshikwanyama / Note."""
    blocks: dict[int, dict] = {}
    for table in doc.tables:
        cells = {row.cells[0].text.strip().lower(): row.cells[1].text.strip()
                 for row in table.rows if len(row.cells) >= 2}
        m = re.search(r"\d+", cells.get("id", ""))
        if not m:
            continue
        note = cells.get("note", "")
        b = {"english": cells.get("english", ""), "notes": [note] if note else [],
             "stray": [], "na": set()}
        for key, label in (("odg", "oshindonga"), ("okw", "oshikwanyama")):
            text = cells.get(label, "")
            if NA.match(text):
                b["notes"].append(f"{label}: {text}")
                b["na"].add(key)
                text = ""
            b[key] = [text] if text else []
        blocks[int(m[0])] = b
    return blocks


def parse(path: Path) -> dict[int, dict]:
    doc = Document(str(path))
    if doc.tables:
        return parse_tables(doc)
    blocks: dict[int, dict] = {}
    cur: dict | None = None
    field: str | None = None
    for raw in (p.text for p in doc.paragraphs):
        s = raw.strip()
        m = HEADER.match(s)
        if m:
            cur = {"odg": [], "okw": [], "notes": [], "stray": []}
            blocks[int(m[1])] = cur
            field = None
            continue
        if cur is None or not s:
            continue
        if s.startswith("That's it"):
            cur = None
        elif s.startswith("English:"):
            field = None
        elif s.startswith("Oshindonga:"):
            field = "odg"
            if rest := s[len("Oshindonga:"):].strip():
                cur["odg"].append(rest)
        elif s.startswith("Oshikwanyama:"):
            field = "okw"
            if rest := s[len("Oshikwanyama:"):].strip():
                cur["okw"].append(rest)
        elif set(s) <= set("─ "):
            field = "sep"
        elif NOTE.match(s):
            cur["notes"].append(s)
        elif field in ("odg", "okw"):
            cur[field].append(s)
        else:
            cur["stray"].append(s)
    return blocks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("docx", type=Path)
    ap.add_argument("--translator", required=True,
                    help="Column prefix, e.g. kaarina → kaarina_oshindonga")
    ap.add_argument("--tsv", type=Path, default=PRIVATE_TSV)
    ap.add_argument("--partial", action="store_true",
                    help="Import what is filled in and skip untranslated items "
                         "(for interim returns)")
    args = ap.parse_args(argv)

    if args.tsv.resolve() in {p.resolve() for p in PUBLIC_TSVS}:
        print("ERROR: refusing to write references into the tracked TSV",
              file=sys.stderr)
        return 1
    if not args.tsv.exists():
        print(f"ERROR: {args.tsv} missing — run build_eval_v3.py first", file=sys.stderr)
        return 1

    fixes_path = args.docx.with_suffix(".fixes.json")
    fixes = ({int(k): v for k, v in json.loads(fixes_path.read_text()).items()}
             if fixes_path.exists() else {})

    t = args.translator
    out_cols = [f"{t}_oshindonga", f"{t}_oshindonga_notes",
                f"{t}_oshikwanyama", f"{t}_oshikwanyama_notes"]
    with args.tsv.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = list(reader.fieldnames)
        rows = list(reader)
    split = cols.index("in_blind_split")
    cols[split:split] = [c for c in out_cols if c not in cols]

    blocks = parse(args.docx)
    ids = {int(r["id"]) for r in rows}
    if not set(blocks) <= ids:
        print(f"ERROR: ids not in the TSV: {sorted(set(blocks) - ids)}",
              file=sys.stderr)
        return 1

    errors = []
    n_done = n_skipped = 0
    for r in rows:
        i = int(r["id"])
        if i not in blocks:
            continue
        b = blocks[i]
        if "english" in b and clean(b["english"]) != clean(r["english"]):
            errors.append(f"id {i}: English in the doc differs from the TSV")
            continue
        fix = fixes.get(i, {})
        odg = fix.get("odg") or (b["odg"][0] if len(b["odg"]) == 1 else None)
        okw = fix.get("okw") or (b["okw"][0] if len(b["okw"]) == 1 else None)
        # Stray text is only tolerated when a fix covers the item, or when
        # it is the English source retyped by the translator.
        stray = [s for s in b["stray"] if clean(s) != clean(r["english"])]
        if args.partial and not b["odg"] and not b["okw"] and not fix:
            n_skipped += 1
            continue
        # A dialect marked n/a gets an empty reference; the note says why.
        if "odg" in b.get("na", ()):
            odg = ""
        if "okw" in b.get("na", ()):
            okw = ""
        if odg is None or okw is None or (stray and not fix):
            errors.append(f"id {i}: odg={b['odg']} okw={b['okw']} stray={stray}")
            continue
        notes = fix.get("notes") or " / ".join(b["notes"])
        r[f"{t}_oshindonga"] = clean(odg)
        r[f"{t}_oshikwanyama"] = clean(okw)
        r[f"{t}_oshindonga_notes"] = clean(notes)
        r[f"{t}_oshikwanyama_notes"] = clean(notes)
        n_done += 1

    if errors:
        print(f"ERROR: unresolved blocks (resolve them in {fixes_path.name}):", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        return 1

    with args.tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    n_notes = sum(1 for r in rows if int(r["id"]) in blocks and r[f"{t}_oshindonga_notes"])
    print(f"imported {n_done} items ({n_notes} with notes, {n_skipped} not yet "
          f"translated) into {args.tsv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
