#!/usr/bin/env python3
"""Prepare the Oshiwambo training corpus from meyabase/osheng-articles (paper 3).

Internal experiments only; nothing written here is committed or published.
Every output row carries its source so a source can be dropped later
(e.g. ``namibian_scrape`` if The Namibian does not license its archive).

Inputs (data/private/meyabase/osheng-articles):
* scrape/namibian/articles/*.txt   — The Namibian Oshiwambo archive (scraped by Meyabase)
* mono_language/oshindonga/files/*.txt — further Oshindonga online articles
* bi_language/engndo|engkwa/<lang>/NNNN_*.txt — LAC / ASSAR booklets, line-aligned

Steps: NFC + standard quotes + whitespace; sentence split (line breaks, then
. ! ? before a capital); emails / phone numbers redacted with ongiini.pii and
such sentences dropped, as are URLs; 4–60 words; GlotLID v3 label per
sentence (English and non-Oshiwambo dropped from the monolingual part);
exact de-duplication; **eval-set guard**: drop any sentence equal to, or
sharing an 8-word sequence with, an eval-set reference or English source.

Outputs (data/private/corpus/osheng_v1/): mono.jsonl, parallel.jsonl, stats.json.

    ~/.venvs/ongiini-eval/bin/python scripts/prepare_osheng_corpus.py
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_scoring as E  # noqa: E402
from lid_outputs import group, load_model  # noqa: E402
from ongiini.pii import sanitize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/private/meyabase/osheng-articles"
OUT = ROOT / "data/private/corpus/osheng_v1"
EVAL_SET = ROOT / "data/private/export/data/eval_set.jsonl"
MIN_WORDS, MAX_WORDS, NGRAM = 4, 60, 8
OSHIWAMBO = {"Oshindonga", "Oshikwanyama"}

_QUOTES = str.maketrans({c: '"' for c in "“”„‟″ʺ˝ˮ˶«»"} | {c: "'" for c in "‘’‚‛′ʹ"})
_URL = re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|na|org|net|gov|edu)\b", re.I)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(]?[A-ZÀ-Ý0-9])")


def clean(text: str) -> str:
    return re.sub(r"[ \t ]+", " ", unicodedata.normalize("NFC", text).translate(_QUOTES)).strip()


def sentences(text: str) -> list[str]:
    out = []
    for line in clean(text).splitlines():
        line = line.strip()
        if line:
            out += [s.strip() for s in _SPLIT.split(line) if s.strip()]
    return out


def usable(s: str) -> bool:
    n = len(s.split())
    if not MIN_WORDS <= n <= MAX_WORDS or _URL.search(s):
        return False
    if sanitize(s) != s:                         # contained an email / phone number
        return False
    letters = sum(c.isalpha() for c in s)
    return letters >= 0.6 * len(s.replace(" ", ""))


def ngrams(s: str) -> set[tuple]:
    w = E.normalise(s).lower().split()
    return {tuple(w[i:i + NGRAM]) for i in range(len(w) - NGRAM + 1)}


def eval_guard():
    items = E.load_items(EVAL_SET)
    texts = [r["english"] for r in items.values()]
    texts += [r.get(f"{d}_reference") or "" for r in items.values() for d in E.DIALECTS]
    exact = {E.normalise(t).lower() for t in texts if t}
    grams = set().union(*(ngrams(t) for t in texts if t))
    return lambda s: E.normalise(s).lower() in exact or bool(ngrams(s) & grams)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model()
    hits_eval = eval_guard()
    stats: Counter = Counter()
    seen: set[str] = set()

    def lid(texts):
        labels, probs = model.predict([t.replace("\n", " ") for t in texts], k=1)
        return [(group(l[0].replace("__label__", "")), float(p[0])) for l, p in zip(labels, probs)]

    with (OUT / "mono.jsonl").open("w") as f:
        sources = [("namibian_scrape", sorted((SRC / "scrape/namibian/articles").glob("*.txt"))),
                   ("meyabase_mono_ndo", sorted((SRC / "mono_language/oshindonga/files").glob("*.txt")))]
        for source, files in sources:
            for path in files:
                stats[f"{source}:docs"] += 1
                sents = [s for s in sentences(path.read_text(errors="ignore")) if usable(s)]
                if not sents:
                    continue
                for k, (s, (lab, p)) in enumerate(zip(sents, lid(sents))):
                    key = E.normalise(s).lower()
                    if lab not in OSHIWAMBO:
                        stats[f"{source}:drop_lid_{lab}"] += 1
                        continue
                    if key in seen:
                        stats[f"{source}:drop_dup"] += 1
                        continue
                    if hits_eval(s):
                        stats[f"{source}:drop_eval_overlap"] += 1
                        continue
                    seen.add(key)
                    stats[f"{source}:kept"] += 1
                    stats[f"{source}:kept_{lab}"] += 1
                    f.write(json.dumps({"text": s, "lid": lab, "lid_prob": round(p, 3), "source": source,
                                        "doc": path.stem[:80], "i": k}, ensure_ascii=False) + "\n")

    with (OUT / "parallel.jsonl").open("w") as f:
        for pair, lang, dialect in (("engndo", "oshindonga", "oshindonga"), ("engkwa", "oshikwanyama", "oshikwanyama")):
            en_dir, ow_dir = SRC / "bi_language" / pair / "english", SRC / "bi_language" / pair / lang
            en_files = {p.name[:4]: p for p in en_dir.glob("[0-9][0-9][0-9][0-9]_*.txt")}
            for ow in sorted(ow_dir.glob("[0-9][0-9][0-9][0-9]_*.txt")):
                en = en_files.get(ow.name[:4])
                if not en:
                    stats[f"{pair}:no_english"] += 1
                    continue
                el = [clean(l) for l in en.read_text(errors="ignore").splitlines() if l.strip()]
                ol = [clean(l) for l in ow.read_text(errors="ignore").splitlines() if l.strip()]
                if len(el) != len(ol):
                    stats[f"{pair}:misaligned_docs"] += 1
                    continue
                stats[f"{pair}:docs"] += 1
                for k, (e, o) in enumerate(zip(el, ol)):
                    if not (usable(o) and len(e.split()) >= MIN_WORDS) or hits_eval(o) or hits_eval(e):
                        stats[f"{pair}:drop"] += 1
                        continue
                    if E.normalise(o).lower() in seen:
                        stats[f"{pair}:drop_dup"] += 1
                        continue
                    seen.add(E.normalise(o).lower())
                    lab, p = lid([o])[0]
                    stats[f"{pair}:kept"] += 1
                    f.write(json.dumps({"en": e, "ow": o, "dialect": dialect, "lid": lab, "lid_prob": round(p, 3),
                                        "source": "lac_assar", "doc": ow.stem[:80], "i": k}, ensure_ascii=False) + "\n")

    (OUT / "stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=2))
    print(json.dumps(dict(sorted(stats.items())), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
