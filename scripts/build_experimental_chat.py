#!/usr/bin/env python3
"""Build the experimental.ongiini.ai chat page from the chat.ongiini.ai app.

Same interface as website/chat-app/index.html, with three changes:
- the API is same-origin (/v1/chat), served by the experimental backend behind
  the nginx of deploy/experimental/ (so no CORS and no link to production);
- a permanent banner says this is a test environment that always runs our latest
  Oshiwambo model, and points to chat.ongiini.ai for the regular assistant;
- not indexed by search engines.

    python3 scripts/build_experimental_chat.py --model-note "Gemma 4 12B + T2 (2026-10-08)"
Writes deploy/experimental/site/index.html.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

SRC = Path("website/chat-app/index.html")
OUT = Path("deploy/experimental/site/index.html")

BANNER_CSS = """
  .exp-banner{background:var(--terracotta-deep,#7a1f00);color:#fff;font:600 14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    padding:10px 16px;text-align:center}
  .exp-banner a{color:#fff;text-decoration:underline}
  .exp-banner small{display:block;font-weight:400;opacity:.9;margin-top:2px}
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-note", required=True, help="which model is live, shown in the banner")
    args = ap.parse_args(argv)
    page = SRC.read_text()
    replacements = [
        ("<title>Chat in your browser — Ongiini AI</title>", "<title>Experimental test environment — Ongiini AI</title>"),
        ('<meta name="robots" content="index, follow, max-image-preview:large" />',
         '<meta name="robots" content="noindex, nofollow" />'),
        ("""        : 'https://api.ongiini.ai/v1/chat';""", """        : '/v1/chat';  // experimental backend, same origin"""),
        ("</style>", BANNER_CSS + "</style>"),
    ]
    for old, new in replacements:
        if old not in page:
            raise SystemExit(f"chat-app changed, update build_experimental_chat.py: {old[:60]!r} not found")
        page = page.replace(old, new, 1)
    banner = (
        '<div class="exp-banner" role="note">'
        "Test environment: you are chatting with our latest experimental Oshiwambo model, not the regular Ongiini AI. "
        "It changes often and can make mistakes. "
        'For the regular AI assistant use <a href="https://chat.ongiini.ai">chat.ongiini.ai</a>.'
        f"<small>Model now: {html.escape(args.model_note)}</small></div>\n"
    )
    page = page.replace("<body>\n", "<body>\n" + banner, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
