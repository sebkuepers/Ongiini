"""Extract product copy from website/*.html into a single canonical
markdown file used by the webhook's lookup_ongiini_docs tool.

Pure stdlib — no external deps. Runs anywhere with Python 3.10+.

Source pages walked:

  website/index.html          — hero, modes, privacy cards, free-tier,
                                contribute section, FAQ (JS array).
  website/privacy/index.html  — full GDPR Privacy Policy.
  website/terms/index.html    — full Terms of Service.
  website/imprint/index.html  — German imprint (§5 DDG, §18 (2) MStV).

Outputs:

  webhook/app/knowledge/product.md   — baked into the webhook image,
                                       consumed by lookup_ongiini_docs.
  website/product.md                 — also deployed by Cloudflare Pages
                                       so the same text is publicly
                                       inspectable at ongiini.ai/product.md.

Usage:
    python3 tools/build_product_knowledge.py [--check]

The --check flag exits non-zero (without writing) if the generated
content differs from what's currently committed — handy in CI to
catch website HTML changes that weren't accompanied by a regenerated
product.md.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEBSITE = ROOT / "website"
WEBHOOK_OUT = ROOT / "webhook" / "app" / "knowledge" / "product.md"
WEBSITE_OUT = WEBSITE / "product.md"


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Collapse runs of whitespace, including newlines, into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


# ──────────────────────────────────────────────────────────────────────
# Strategy 1 — walk index.html, harvest English text from data-en
# elements, grouped by the enclosing <section id>.
# ──────────────────────────────────────────────────────────────────────

class DataEnHarvester(HTMLParser):
    """Capture text inside elements that carry a `data-en` attribute.

    Tracks a section stack so each capture is tagged with the id of the
    nearest enclosing <section>. Skips <style> and <script> blocks so
    CSS/JS noise doesn't leak into the output.
    """

    # Tags that DataEn can decorate. We open a capture on any of these
    # and close on the matching end tag at the same depth.
    CAPTURE_TAGS = {
        "span", "p", "div", "li", "a", "strong", "em",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_style = False
        self.in_script = False
        self.section_stack: list[str | None] = []
        # When capturing: (outer_tag, depth, [text chunks], parent_class)
        self.capture: tuple[str, int, list[str], str] | None = None
        # (section_id, parent_class, text) tuples in document order.
        self.collected: list[tuple[str | None, str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = {k: (v or "") for k, v in attrs}
        if tag == "style":
            self.in_style = True
            return
        if tag == "script":
            self.in_script = True
            return
        if tag == "section":
            self.section_stack.append(d.get("id") or None)
        if self.capture:
            outer_tag, depth, chunks, parent_class = self.capture
            if tag == outer_tag:
                self.capture = (outer_tag, depth + 1, chunks, parent_class)
            return
        if "data-en" in d and tag in self.CAPTURE_TAGS:
            self.capture = (tag, 1, [], d.get("class", ""))

    def handle_endtag(self, tag: str) -> None:
        if tag == "style":
            self.in_style = False
            return
        if tag == "script":
            self.in_script = False
            return
        if tag == "section" and self.section_stack:
            self.section_stack.pop()
        if self.capture and tag == self.capture[0]:
            outer_tag, depth, chunks, parent_class = self.capture
            depth -= 1
            if depth <= 0:
                text = normalise("".join(chunks))
                if text:
                    section = self.section_stack[-1] if self.section_stack else None
                    self.collected.append((section, parent_class, text))
                self.capture = None
            else:
                self.capture = (outer_tag, depth, chunks, parent_class)

    def handle_data(self, data: str) -> None:
        if self.in_style or self.in_script:
            return
        if self.capture:
            self.capture[2].append(data)


# ──────────────────────────────────────────────────────────────────────
# Strategy 2 — pull the FAQ array out of index.html.
#
# The FAQ lives as a JS array literal in the page source:
#
#   var FAQ = [
#     { en: ["Question?", "Answer."], af: ["...", "..."] },
#     ...
#   ];
#
# Regex over the EN pair is reliable because the strings never contain
# unescaped double quotes — the source file uses straight quotes for JS
# string delimiters and curly quotes inside the content. We unescape
# the few backslash sequences (\", \\, \n) that show up.
# ──────────────────────────────────────────────────────────────────────

_FAQ_ENTRY_RE = re.compile(
    r"en:\s*\[\s*\"((?:[^\"\\]|\\.)*)\"\s*,\s*\"((?:[^\"\\]|\\.)*)\"\s*\]",
    re.DOTALL,
)


def _unescape_js(text: str) -> str:
    """Reverse the few JS escapes we expect: \\\", \\\\, \\n, \\t."""
    return (
        text.replace(r"\"", "\"")
            .replace(r"\\", "\\")
            .replace(r"\n", "\n")
            .replace(r"\t", "\t")
    )


def extract_faq(html: str) -> list[tuple[str, str]]:
    """Return [(question, answer)] tuples in source order."""
    entries = []
    for q_raw, a_raw in _FAQ_ENTRY_RE.findall(html):
        question = normalise(_unescape_js(q_raw))
        answer = normalise(_unescape_js(a_raw))
        if question and answer:
            entries.append((question, answer))
    return entries


# ──────────────────────────────────────────────────────────────────────
# Strategy 3 — semantic HTML → markdown for the legal pages.
#
# The three legal pages are clean: a single <main> wrapping <h1>, <h2>,
# <h3>, <p>, <ul>, <li> with no JS-driven copy. We linearise to markdown
# by tracking heading depth and converting block-level tags.
# ──────────────────────────────────────────────────────────────────────

class LegalPageRenderer(HTMLParser):
    """Render the <main> section of a legal HTML page to markdown.

    Conversions:
      <h1>..<h6>  → markdown headings, demoted by `demote` levels.
      <p>         → blank-line-separated paragraph.
      <ul>/<ol>   → markdown list with `- ` bullets.
      <strong>/<b>→ **bold**.
      <em>/<i>    → *italic*.
      <code>      → `code`.
      <a href>    → [text](href).
    Everything else flattens to plain text. Header bar / footer outside
    <main> is skipped.
    """

    BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li"}
    HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self, demote: int = 1, skip_h1: bool = True) -> None:
        """`demote=1` turns <h2> on the page into ### in the output. The
        page's <h1> is redundant with the section title we wrap it in,
        so it's skipped by default."""
        super().__init__(convert_charrefs=True)
        self.demote = demote
        self.skip_h1 = skip_h1
        self.in_main = False
        self.in_style = False
        self.in_script = False
        self.in_anchor: str | None = None
        self.list_stack: list[str] = []  # "ul" or "ol"
        self.ol_counters: list[int] = []
        self.current_block: list[str] = []
        self.current_block_tag: str | None = None
        self.blocks: list[str] = []  # finished markdown blocks

    # ── lifecycle ────────────────────────────────────────────────

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = {k: (v or "") for k, v in attrs}
        if tag == "main":
            self.in_main = True
            return
        if tag == "style":
            self.in_style = True
            return
        if tag == "script":
            self.in_script = True
            return
        if not self.in_main:
            return
        if tag in self.HEADING_TAGS or tag == "p":
            self._flush_block()
            self.current_block_tag = tag
        elif tag == "ul":
            self._flush_block()
            self.list_stack.append("ul")
        elif tag == "ol":
            self._flush_block()
            self.list_stack.append("ol")
            self.ol_counters.append(1)
        elif tag == "li":
            self._flush_block()
            self.current_block_tag = "li"
        elif tag in ("strong", "b"):
            self.current_block.append("**")
        elif tag in ("em", "i"):
            self.current_block.append("*")
        elif tag == "code":
            self.current_block.append("`")
        elif tag == "a" and d.get("href"):
            self.in_anchor = d["href"]
            self.current_block.append("[")
        elif tag == "br":
            self.current_block.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "main":
            self._flush_block()
            self.in_main = False
            return
        if tag == "style":
            self.in_style = False
            return
        if tag == "script":
            self.in_script = False
            return
        if not self.in_main:
            return
        if tag in self.HEADING_TAGS or tag == "p":
            self._flush_block()
        elif tag in ("ul", "ol"):
            self._flush_block()
            if self.list_stack:
                popped = self.list_stack.pop()
                if popped == "ol" and self.ol_counters:
                    self.ol_counters.pop()
            self.blocks.append("")  # blank line after list
        elif tag == "li":
            self._flush_block()
        elif tag in ("strong", "b"):
            self.current_block.append("**")
        elif tag in ("em", "i"):
            self.current_block.append("*")
        elif tag == "code":
            self.current_block.append("`")
        elif tag == "a" and self.in_anchor is not None:
            href = self.in_anchor
            self.current_block.append(f"]({href})")
            self.in_anchor = None

    def handle_data(self, data: str) -> None:
        if self.in_style or self.in_script or not self.in_main:
            return
        self.current_block.append(data)

    # ── flushing ─────────────────────────────────────────────────

    def _flush_block(self) -> None:
        if not self.current_block:
            self.current_block_tag = None
            return
        text = normalise("".join(self.current_block))
        self.current_block = []
        if not text:
            self.current_block_tag = None
            return
        tag = self.current_block_tag
        self.current_block_tag = None
        if tag in self.HEADING_TAGS:
            if tag == "h1" and self.skip_h1:
                return
            level = int(tag[1]) + self.demote
            level = min(level, 6)
            self.blocks.append(f"{'#' * level} {text}")
            self.blocks.append("")
        elif tag == "li":
            prefix = "- "
            if self.list_stack and self.list_stack[-1] == "ol":
                if self.ol_counters:
                    n = self.ol_counters[-1]
                    self.ol_counters[-1] = n + 1
                    prefix = f"{n}. "
            indent = "  " * max(0, len(self.list_stack) - 1)
            self.blocks.append(f"{indent}{prefix}{text}")
        elif tag == "p":
            self.blocks.append(text)
            self.blocks.append("")
        else:
            # Inline text outside a block — append to previous line.
            if self.blocks and self.blocks[-1]:
                self.blocks[-1] = self.blocks[-1] + " " + text
            else:
                self.blocks.append(text)

    # ── public ───────────────────────────────────────────────────

    def render(self) -> str:
        out_lines = self.blocks[:]
        # Collapse 3+ blank lines into 2.
        cleaned: list[str] = []
        blank = 0
        for line in out_lines:
            if line == "":
                blank += 1
                if blank <= 1:
                    cleaned.append("")
            else:
                blank = 0
                cleaned.append(line)
        # Strip trailing blanks.
        while cleaned and cleaned[-1] == "":
            cleaned.pop()
        return "\n".join(cleaned)


# ──────────────────────────────────────────────────────────────────────
# Driver — assemble the markdown.
# ──────────────────────────────────────────────────────────────────────

# Which sections from the home page carry product-knowledge value (i.e.
# what we want the LLM to be able to recite). Other sections (nav,
# footer, hero chat mockups, decorative numerals) are filtered out by
# section id.
HOME_SECTIONS_OF_INTEREST = [
    ("hero",       "What Ongiini is"),
    ("modes",      "How to use it"),
    ("languages",  "Languages supported"),
    ("privacy",    "Privacy summary (homepage cards)"),
    ("free",       "Pricing and the monthly limit"),
    ("open",       "Why it's built this way"),
    ("contribute", "How to support the project"),
    # Many sections don't exist as ids; that's fine — they're filtered
    # by the section_stack so the loop just skips them.
]


def render_home_section(
    collected: list[tuple[str | None, str, str]],
    section_id: str,
) -> list[str]:
    """Pull all data-en captures whose nearest section ancestor matches
    `section_id`. Dedupe runs of identical text (the page repeats some
    label lines across responsive variants). Return a list of plain
    paragraphs."""
    items = [text for sec, _cls, text in collected if sec == section_id]
    seen = set()
    out = []
    for text in items:
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def build_markdown(
    index_html: str,
    privacy_html: str,
    terms_html: str,
    imprint_html: str,
) -> str:
    parser = DataEnHarvester()
    parser.feed(index_html)

    faq_entries = extract_faq(index_html)

    today = date.today().isoformat()
    lines: list[str] = []
    lines.append("# Ongiini — Product Knowledge")
    lines.append("")
    lines.append(
        f"_Auto-generated from `website/*.html` on {today}. Do not edit by hand: "
        f"edit the source HTML and re-run `tools/build_product_knowledge.py`. "
        f"This file is consumed by the WhatsApp webhook's "
        f"`lookup_ongiini_docs` tool so the assistant always answers questions "
        f"about Ongiini itself from the same canonical copy that's on the website._"
    )
    lines.append("")

    # ── Home-page content, organised by section ──
    for section_id, heading in HOME_SECTIONS_OF_INTEREST:
        paragraphs = render_home_section(parser.collected, section_id)
        if not paragraphs:
            continue
        lines.append(f"## {heading}")
        lines.append("")
        for p in paragraphs:
            lines.append(p)
            lines.append("")

    # ── FAQ ──
    if faq_entries:
        lines.append("## Frequently asked questions")
        lines.append("")
        for q, a in faq_entries:
            lines.append(f"### {q}")
            lines.append("")
            lines.append(a)
            lines.append("")

    # ── Legal pages (each as its own H2 with the inside H1 demoted) ──
    for label, html in (
        ("Privacy policy (full text)",  privacy_html),
        ("Terms of service (full text)", terms_html),
        ("Imprint (German § 5 DDG)",     imprint_html),
    ):
        renderer = LegalPageRenderer(demote=1)
        renderer.feed(html)
        body = renderer.render().strip()
        if not body:
            continue
        lines.append(f"## {label}")
        lines.append("")
        lines.append(body)
        lines.append("")

    # Final newline + collapse triple-blanks.
    text = "\n".join(lines).rstrip() + "\n"
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero (without writing) if the generated content "
             "differs from what's currently committed. Use this in CI.",
    )
    args = ap.parse_args(argv)

    index_html = (WEBSITE / "index.html").read_text(encoding="utf-8")
    privacy_html = (WEBSITE / "privacy" / "index.html").read_text(encoding="utf-8")
    terms_html = (WEBSITE / "terms" / "index.html").read_text(encoding="utf-8")
    imprint_html = (WEBSITE / "imprint" / "index.html").read_text(encoding="utf-8")

    new_md = build_markdown(index_html, privacy_html, terms_html, imprint_html)

    if args.check:
        ok = True
        for path in (WEBHOOK_OUT, WEBSITE_OUT):
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            if existing != new_md:
                print(
                    f"product.md drift detected at {path.relative_to(ROOT)}: "
                    f"the website HTML has changed but product.md hasn't been "
                    f"regenerated. Run `python3 tools/build_product_knowledge.py` "
                    f"and commit the result.",
                    file=sys.stderr,
                )
                ok = False
        return 0 if ok else 1

    WEBHOOK_OUT.parent.mkdir(parents=True, exist_ok=True)
    WEBHOOK_OUT.write_text(new_md, encoding="utf-8")
    WEBSITE_OUT.write_text(new_md, encoding="utf-8")
    size_kb = len(new_md.encode("utf-8")) / 1024
    print(
        f"wrote {WEBHOOK_OUT.relative_to(ROOT)} and "
        f"{WEBSITE_OUT.relative_to(ROOT)} ({size_kb:.1f} KB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
