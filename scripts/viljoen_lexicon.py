#!/usr/bin/env python3
"""English → Oshindonga lookup built from the digitised Viljoen/Amakali/Namuandi dictionary.

Input: pages.jsonl from scripts/digitize_dictionary.py (data/private, never commit).
Both halves of the book feed one index keyed by lower-case English words/phrases:
- English→Oshindonga entries: headword ("qualification, ability") → equivalents
  as printed ("evulo (oma-)").
- Oshindonga→English entries: each short English sense ("parent", "hunter, one
  who shoots") → the full Oshindonga form with its plural class ("omuvali (aa-)").
Single-word "examples" (derived words such as "eiyuvo – conscience") are added as
entries too. Output: {english: [[oshindonga, note], ...]} and the example pairs.

    python3 scripts/viljoen_lexicon.py data/private/corpora/raw/nph-viljoen-1984/pages.jsonl \\
        data/private/corpora/dictionaries/viljoen-1984/
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def senses(text: str) -> list[str]:
    """Short English senses from a gloss: split on ; , and sentence ends, drop notes."""
    text = re.sub(r"\(.*?\)|Roberts'?\s*\d+\.?|Id\.", " ", text or "")
    out = []
    for part in re.split(r"[;,.]", text):
        p = re.sub(r"\s+", " ", part).strip().lower()
        p = re.sub(r"^(to|a|an|the) ", "", p)
        if p and len(p.split()) <= 3 and re.fullmatch(r"[a-z' -]+", p):
            out.append(p)
    return out


def main(argv=None) -> int:
    src, dst = (argv or sys.argv[1:])[:2]
    pages = [json.loads(l) for l in open(src) if l.strip()]
    lex: dict[str, list[list[str]]] = defaultdict(list)
    examples = []

    def add(key: str, ow: str, note: str = "") -> None:
        ow = re.sub(r"\s+", " ", ow).strip(" .;,")
        # sub-phrases ("grain of # ewe", "# ahead huma"): '#' stands for the headword
        if not ow or re.search(r"[#.!?:]", ow) or len(ow.split()) > 3:
            return
        if [ow, note] not in lex[key]:
            lex[key].append([ow, note])

    for p in pages:
        kind = p.get("page_kind")
        for e in p.get("entries") or []:
            if kind == "en_ow":
                for head in re.split(r"[,;]", e.get("stem") or ""):
                    head = re.sub(r"[#*]", "", head).strip().lower()
                    if head:
                        for eq in re.split(r"[,;]", e.get("gloss") or ""):
                            m = re.match(r"\s*(.+?)\s*(\(([^)]*)\))?\s*$", eq)
                            if m and m.group(1) and not m.group(1).lower().startswith("roberts"):
                                add(head, m.group(1), m.group(3) or "")
            elif kind == "ow_en":
                form = (e.get("full_form") or e.get("stem") or "").strip()
                note = e.get("plural") or e.get("marker") or ""
                for s in senses(e.get("gloss") or ""):
                    add(s, form, note)
                for x in e.get("examples") or []:
                    ow, en = (x.get("ow") or "").strip(), (x.get("en") or "").strip()
                    if len(ow.split()) == 1:          # a derived word, not a sentence
                        for s in senses(en):
                            add(s, ow)
                    elif ow and en:
                        examples.append({"ow": ow, "en": en, "headword": form, "page": p["page"],
                                         "source": "nph_viljoen_1984"})
    out = Path(dst)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lexicon_en_ow.json").write_text(json.dumps(lex, ensure_ascii=False, indent=0))
    with (out / "examples.jsonl").open("w") as f:
        for x in examples:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps({"english_keys": len(lex), "pairs": sum(len(v) for v in lex.values()),
                      "example_sentences": len(examples)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
