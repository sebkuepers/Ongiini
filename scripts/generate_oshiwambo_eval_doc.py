#!/usr/bin/env python3
"""Generate a phone-friendly Word document for Elizabeth (the hired
Oshiwambo translator) to fill in ground-truth translations.

Reads data/oshiwambo_eval_v2.tsv (~400 items, 5 domains, phenomenon-
tagged). Produces a .docx with EN source only — Claude / Gemma machine
translations are deliberately hidden so Elizabeth's work isn't anchored
on them. Items are randomised so domain ordering doesn't bias her.

Layout decisions for mobile Word editing:
- No tables (mobile Word renders them painfully).
- Each phrase is its own block with predictable structure so her thumb
  knows where to land: bold label, then a blank line.
- The phrase ID is shown so she can refer back to specific items.
- The blind-split flag is NOT shown to her — that's our reporting
  bookkeeping, not her concern.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK
from docx.enum.table import WD_ROW_HEIGHT_RULE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[1] / "data/oshiwambo_eval_v2.tsv"
)
DEFAULT_OUT = (
    Path.home() / "Desktop/ongiini-eval-elizabeth-v2.docx"
)


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            en = (r.get("english") or "").strip()
            if not en:
                continue
            rows.append(r)
    return rows


def add_label(p, text: str, bold: bool = True, size: int = 12) -> None:
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)


def build_doc(rows: list[dict], shuffle_seed: int) -> Document:
    doc = Document()

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(12)

    # ── Title + intro ───────────────────────────────────────────
    doc.add_heading(
        "Ongiini AI — Oshindonga & Oshikwanyama eval (v2)", level=0
    )

    intro = doc.add_paragraph()
    intro.add_run(
        f"Tangi unene! Thank you for helping us. Below are "
        f"{len(rows)} short English sentences. For each one, please "
        "write the natural, everyday way a Namibian would say it in "
        "both Oshindonga AND Oshikwanyama."
    )

    intro2 = doc.add_paragraph()
    intro2.add_run(
        "You can edit this document on your phone. Tap the blank line "
        "below \"Oshindonga:\" or \"Oshikwanyama:\" and start typing. "
        "Save as you go — no need to finish in one sitting."
    )

    intro3 = doc.add_paragraph()
    intro3.add_run(
        "If a sentence doesn't translate naturally in one of the "
        "languages, just write \"n/a\" and a short note about why "
        "(e.g. \"only used in Oshindonga\", \"no direct equivalent\")."
    )

    intro4 = doc.add_paragraph()
    note_run = intro4.add_run(
        "Please write naturally — the way you would actually speak or "
        "text someone. Pick the most common, neutral register (not the "
        "most formal or textbook version, but also not the most casual "
        "slang). When two valid translations exist, pick the one a "
        "wider audience would understand."
    )
    note_run.italic = True

    intro5 = doc.add_paragraph()
    intro5.add_run(
        "Don't worry about getting things \"right\" — your job is to "
        "tell us how Namibians actually express each idea. Your "
        "translation IS the standard; we're measuring our AI against "
        "you, not the other way around."
    )

    # ── Phrases (shuffled so domain ordering doesn't bias) ───────
    rng = random.Random(shuffle_seed)
    ordered = list(rows)
    rng.shuffle(ordered)

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    doc.add_heading("The 425 phrases", level=1)

    total = len(ordered)
    for counter, r in enumerate(ordered, start=1):
        # Tiny grey counter at top of each block
        counter_p = doc.add_paragraph()
        cr = counter_p.add_run(f"Phrase {counter} of {total}  ·  id #{r['id']}")
        cr.bold = True
        cr.font.size = Pt(10)
        cr.font.color.rgb = RGBColor(0x6c, 0x6c, 0x6c)

        # English text
        en_p = doc.add_paragraph()
        en_label = en_p.add_run("English:  ")
        en_label.bold = True
        en_p.add_run(r["english"])

        # Oshindonga label + blank line
        odg_p = doc.add_paragraph()
        add_label(odg_p, "Oshindonga:")
        doc.add_paragraph()

        # Oshikwanyama label + blank line
        okw_p = doc.add_paragraph()
        add_label(okw_p, "Oshikwanyama:")
        doc.add_paragraph()

        # Visual separator
        sep = doc.add_paragraph("─" * 25)
        sep.runs[0].font.color.rgb = RGBColor(0xc0, 0xc0, 0xc0)

    # ── Closing ──────────────────────────────────────────────────
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    doc.add_heading("That's it — tangi unene!", level=1)
    doc.add_paragraph(
        "When you're done (or whenever you want to share what you "
        "have so far), please save the file and send it back. We'll "
        "review and let you know if anything needs another look. "
        "If a phrase had something strange about it or you weren't "
        "sure about it, you can also write a short note in the file."
    )

    return doc


ROW_LABELS = ("ID", "English", "Oshindonga", "Oshikwanyama", "Note")


def build_table_doc(rows: list[dict], shuffle_seed: int, name: str) -> Document:
    """One small 2-column table per phrase: ID / English / Oshindonga /
    Oshikwanyama / Note. Tables survive phone editing far better than
    free paragraphs (the v2 return had labels typed over and answers run
    together); import_eval_translations.py reads them back by row label.
    """
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(12)

    doc.add_heading("Ongiini AI — Oshindonga & Oshikwanyama eval (v3)", level=0)
    for text in (
        f"Tangi unene, {name}! These are {len(rows)} new English sentences "
        "for the benchmark. For each one, please write the natural, "
        "everyday way a Namibian would say it in Oshindonga AND "
        "Oshikwanyama — the same as last time.",
        "Each sentence has its own small table. Tap into the empty box "
        "next to \"Oshindonga\" or \"Oshikwanyama\" and type. Please do "
        "not delete or retype the labels or the ID.",
        "NEW this time: many of these English sentences were written by "
        "us, not by real users. If an English sentence sounds strange — "
        "something a Namibian would never say or write — please write "
        "\"unnatural\" in the Note box, and how people would say it "
        "instead. That helps us as much as the translation.",
        "Use the Note box too if a sentence doesn't translate naturally "
        "(write \"n/a\" in the translation box), or if you kept an English "
        "word like WhatsApp or NSFAF on purpose.",
    ):
        doc.add_paragraph(text)

    rng = random.Random(shuffle_seed)
    ordered = list(rows)
    rng.shuffle(ordered)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    for counter, r in enumerate(ordered, start=1):
        head = doc.add_paragraph()
        run = head.add_run(f"Phrase {counter} of {len(ordered)}")
        run.bold = True
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0x6c, 0x6c, 0x6c)
        head.paragraph_format.keep_with_next = True
        table = doc.add_table(rows=len(ROW_LABELS), cols=2)
        table.style = "Table Grid"
        table.autofit = False
        table.columns[0].width, table.columns[1].width = Cm(3.2), Cm(13.3)
        values = (f"#{r['id']}", r["english"], "", "", "")
        for row, label, value in zip(table.rows, ROW_LABELS, values):
            row.cells[0].width, row.cells[1].width = Cm(3.2), Cm(13.3)
            add_label(row.cells[0].paragraphs[0], label, size=11)
            row.cells[1].text = value
            # keep each phrase's table on one page
            no_split = OxmlElement("w:cantSplit")
            row._tr.get_or_add_trPr().append(no_split)
            for para in row.cells[0].paragraphs + row.cells[1].paragraphs:
                para.paragraph_format.keep_with_next = True
            if label in ("Oshindonga", "Oshikwanyama"):
                row.height, row.height_rule = Cm(1.4), WD_ROW_HEIGHT_RULE.AT_LEAST
        doc.add_paragraph()

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    doc.add_heading("That's it — tangi unene!", level=1)
    doc.add_paragraph(
        "Save the file and send it back whenever you are done, or earlier "
        "if you want us to check how it's going."
    )
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(DEFAULT_SOURCE),
                    help="Path to v2 TSV. Default: data/oshiwambo_eval_v2.tsv")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="Output .docx path. Default: ~/Desktop/...v2.docx")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for phrase shuffle (so re-runs are stable)")
    ap.add_argument("--format", choices=("blocks", "tables"), default="blocks",
                    help="blocks = v2 paragraph layout; tables = one table per phrase")
    ap.add_argument("--min-id", type=int, default=0,
                    help="Only include items with id >= this (e.g. 424 for v3)")
    ap.add_argument("--translator", default="",
                    help="Name used in the greeting (tables format)")
    args = ap.parse_args(argv)

    source = Path(args.source)
    out = Path(args.out)
    if not source.exists():
        print(f"ERROR: source not found at {source}", file=sys.stderr)
        return 1
    rows = [r for r in load_rows(source) if int(r["id"]) >= args.min_id]
    print(f"loaded {len(rows)} phrases from {source.name}", file=sys.stderr)
    doc = (build_table_doc(rows, args.seed, args.translator)
           if args.format == "tables" else build_doc(rows, args.seed))
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
