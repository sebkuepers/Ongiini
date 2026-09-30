#!/usr/bin/env python3
"""Generate website/research/index.html from the public benchmark sources.

Every number and chart on the page is computed from
data/oshiwambo_eval_v3.tsv (English sources, tags, splits — no
references), so the page cannot drift from the data. Charts are static
inline SVG in the site's palette; each has a table twin. No scores are
shown: the leaderboard launches with v1.0.

    python3 scripts/build_research_page.py
    python3 scripts/build_product_knowledge.py   # page text feeds the bot
"""
from __future__ import annotations

import csv
import html
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data/oshiwambo_eval_v3.tsv"
TEMPLATE = Path(__file__).resolve().parent / "research_page_template.html"
SITE_CSS_FROM = ROOT / "website/contribute/index.html"   # shared ct-* design
OUT = ROOT / "website/research/index.html"

PHENOMENA = [  # tag, label, what it probes, example (English source from the set)
    ("negation", "Negation", "single and double negation, scope",
     "I never received the confirmation message you mentioned."),
    ("noun_class_agreement", "Noun-class agreement", "concord across noun, verb, adjective and numeral",
     "Those two old men sat under the tree near the river."),
    ("pronoun_coreference", "Pronoun coreference", "which person or thing a pronoun points to",
     "The nurse phoned the mother because she was worried."),
    ("tense_aspect", "Tense and aspect", "recent past, perfect, habitual, progressive",
     "It has been raining since morning."),
    ("numbers_dates", "Numbers and dates", "amounts in N$, dates, times",
     "Visiting hours are from 14:00 to 16:00 daily."),
    ("named_entities", "Named entities", "Namibian places, ministries, institutions",
     "Is the combi to Ondangwa full?"),
    ("code_switch", "Loanwords", "English words that stay English in everyday Oshiwambo",
     "My data bundle is finished."),
    ("politeness_register", "Politeness register", "Tate, Meme, Kuku — elder, peer and child address",
     "Kuku, did you sleep well?"),
    ("idiom_nonliteral", "Figurative language", "everyday non-literal expressions",
     "Money is tight this month, so we cannot buy new clothes."),
    ("polysemy", "Polysemy", "one English word, several Oshiwambo words",
     "Do you have a match to light the fire?"),
    ("multi_sentence", "Multi-sentence", "cohesion across two to four sentences",
     "The bus broke down halfway. We had to walk for almost two hours in the heat. "
     "By the time we arrived, the meeting was already over."),
]
PROVENANCE = [
    ("mined_paraphrased", "Paraphrased from real conversations",
     "Inspired by what people actually ask Ongiini AI — rewritten in full, no user text"),
    ("crafted", "Written for a phenomenon", "Each targets one of the eleven phenomena below"),
    ("v1_retained", "Everyday phrasebook", "Short and medium conversational items"),
    ("formal_drafted", "Formal and institutional", "Notices, letters and announcements"),
]
DOMAINS = [("chat", "Everyday chat"), ("challenge", "Phenomenon items"), ("formal", "Formal"),
           ("community", "Family and community"), ("religious", "Faith")]


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def hbar_chart(rows: list[tuple[str, int, str]], max_v: int, color: str, unit: str) -> str:
    """rows: (label, value, tooltip). Single-series horizontal bars."""
    label_w, row_h, w, pad = 250, 30, 640, 56
    h = len(rows) * row_h + 26
    x = lambda v: label_w + v / max_v * (w - label_w - pad)
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(unit)}">']
    step = 50 if max_v > 120 else 10
    for t in range(0, max_v + 1, step):
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="0" y2="{h - 26}" class="grid"/>'
                   f'<text x="{x(t):.1f}" y="{h - 8}" class="tick" text-anchor="middle">{t}</text>')
    for i, (label, v, tip) in enumerate(rows):
        y, bh = i * row_h + 8, 14
        bw = max(1.0, x(v) - x(0))
        r = min(4, bw / 2)
        path = (f"M{x(0):.1f},{y}H{x(0) + bw - r:.1f}Q{x(0) + bw:.1f},{y} {x(0) + bw:.1f},{y + r}"
                f"V{y + bh - r}Q{x(0) + bw:.1f},{y + bh} {x(0) + bw - r:.1f},{y + bh}H{x(0):.1f}Z")
        out.append(f'<g><title>{esc(tip)}</title>'
                   f'<text x="{label_w - 12}" y="{y + 11}" text-anchor="end" class="lbl">{esc(label)}</text>'
                   f'<path d="{path}" fill="{color}"/>'
                   f'<text x="{x(v) + 7:.1f}" y="{y + 11}" class="val">{v}</text></g>')
    out.append("</svg>")
    return "".join(out)


def stacked_chart(rows: list[tuple[str, int, int]], max_v: int) -> str:
    """rows: (label, development, blind). Two series with a 2px surface gap."""
    label_w, row_h, w, pad = 250, 30, 640, 70
    h = len(rows) * row_h + 26
    x = lambda v: label_w + v / max_v * (w - label_w - pad)
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Items per phenomenon">']
    for t in range(0, max_v + 1, 10):
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="0" y2="{h - 26}" class="grid"/>'
                   f'<text x="{x(t):.1f}" y="{h - 8}" class="tick" text-anchor="middle">{t}</text>')
    for i, (label, dev, blind) in enumerate(rows):
        y, bh = i * row_h + 8, 14
        x0, x1, x2 = x(0), x(dev), x(dev + blind)
        r = min(4, (x2 - x1 - 2) / 2)
        seg2 = (f"M{x1 + 2:.1f},{y}H{x2 - r:.1f}Q{x2:.1f},{y} {x2:.1f},{y + r}V{y + bh - r}"
                f"Q{x2:.1f},{y + bh} {x2 - r:.1f},{y + bh}H{x1 + 2:.1f}Z")
        out.append(f'<g><title>{esc(label)}: {dev + blind} items — {dev} development, {blind} blind</title>'
                   f'<text x="{label_w - 12}" y="{y + 11}" text-anchor="end" class="lbl">{esc(label)}</text>'
                   f'<rect x="{x0:.1f}" y="{y}" width="{x1 - x0:.1f}" height="{bh}" fill="var(--series-1)"/>'
                   f'<path d="{seg2}" fill="var(--series-2)"/>'
                   f'<text x="{x2 + 7:.1f}" y="{y + 11}" class="val">{dev + blind}</text></g>')
    out.append("</svg>")
    return "".join(out)


def table(head: list[str], rows: list[list]) -> str:
    th = "".join(f'<th class="{"num" if i else ""}">{esc(h)}</th>' for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td class="{"num" if i else ""}">{esc(str(c))}</td>'
                                  for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def main() -> int:
    with SOURCES.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    n = len(rows)
    blind = sum(r["in_blind_split"] == "true" for r in rows)
    prov = Counter(r["provenance"] for r in rows)
    dom = Counter(r["domain"] for r in rows)
    length = Counter(r["length_bucket"] for r in rows)
    tag_all = Counter(t for r in rows for t in r["phenomenon_tags"].split(";") if t)
    tag_blind = Counter(t for r in rows if r["in_blind_split"] == "true"
                        for t in r["phenomenon_tags"].split(";") if t)

    prov_rows = [(label, prov[k], f"{label}: {prov[k]} items — {sub}") for k, label, sub in PROVENANCE]
    phen_rows = [(label, tag_all[k] - tag_blind[k], tag_blind[k]) for k, label, _, _ in PHENOMENA]
    dom_rows = [(label, dom[k], f"{label}: {dom[k]} items") for k, label in DOMAINS]
    examples = "".join(
        f'<div class="ex"><div class="ex-tag">{esc(label)}</div><p class="ex-src">“{esc(ex)}”</p>'
        f'<p class="ex-why">{esc(why)}</p></div>' for _, label, why, ex in PHENOMENA)

    values = {
        "N_ITEMS": str(n), "N_BLIND": str(blind), "N_DEV": str(n - blind),
        "BLIND_PCT": f"{round(100 * blind / n)}", "N_PHEN": str(len(PHENOMENA)),
        "MIN_PHEN": str(min(tag_all[k] for k, *_ in PHENOMENA)),
        "LEN_S": f"{round(100 * length['S'] / n)}", "LEN_M": f"{round(100 * length['M'] / n)}",
        "LEN_L": f"{round(100 * length['L'] / n)}",
        "CHART_PROV": hbar_chart(prov_rows, 250, "var(--series-2)", "Items per source"),
        "TABLE_PROV": table(["Source", "Items"], [[l, v] for l, v, _ in prov_rows]),
        "CHART_PHEN": stacked_chart(phen_rows, 60),
        "TABLE_PHEN": table(["Phenomenon", "Development", "Blind", "Total"],
                            [[l, d, b, d + b] for l, d, b in phen_rows]),
        "CHART_DOM": hbar_chart(dom_rows, 250, "var(--series-1)", "Items per domain"),
        "TABLE_DOM": table(["Domain", "Items"], [[l, v] for l, v, _ in dom_rows]),
        "EXAMPLES": examples,
    }
    site = SITE_CSS_FROM.read_text()
    values["SITE_CSS"] = site[site.index("<style>") + len("<style>"): site.index("</style>")]
    page = TEMPLATE.read_text()
    for k, v in values.items():
        page = page.replace("{{" + k + "}}", v)
    assert "{{" not in page, "unfilled placeholder"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f"wrote {OUT.relative_to(ROOT)}  ({n} items, {blind} blind)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
