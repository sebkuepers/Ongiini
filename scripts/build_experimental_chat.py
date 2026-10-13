#!/usr/bin/env python3
"""Build the experimental.ongiini.ai chat page from the chat.ongiini.ai app.

Same interface as website/chat-app/index.html, with these changes:
- the API is same-origin (/v1/chat), served by the experimental backend behind
  the nginx of deploy/experimental/ (so no CORS and no link to production);
- a permanent banner says this is a test environment that always runs our latest
  Oshiwambo model, and points to chat.ongiini.ai for the regular assistant;
- the start page names the model prominently and asks visitors to test it in
  Oshindonga and Oshikwanyama, with suggestions in both;
- every reply gets a one-click rating of its Oshiwambo (/v1/chat/feedback, see
  ongiini/api/chat.py), with a privacy note on the start page;
- not indexed by search engines.

    python3 scripts/build_experimental_chat.py --model-name "Ongiini Ombuto" \
        --model-note "Gemma 4 31B + Oshiwambo training T4b31 (2026-10-13)"
Writes deploy/experimental/site/index.html.
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

SRC = Path("website/chat-app/index.html")
OUT = Path("deploy/experimental/site/index.html")

BANNER_CSS = """
  .exp-banner{background:var(--terracotta-deep,#7a1f00);color:#fff;font:600 14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    padding:10px 16px;text-align:center}
  .exp-banner a{color:#fff;text-decoration:underline}
  .exp-banner small{display:block;font-weight:400;opacity:.9;margin-top:2px}
  .exp-kicker{margin:0 0 10px;font:600 13px/1.3 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;letter-spacing:.08em;
    text-transform:uppercase;color:var(--terracotta-deep)}
  .welcome-mark.exp-model{font-size:clamp(46px,12.5vw,120px)}
  .exp-model-name{color:var(--terracotta-deep)}
  .exp-model-meta{margin:12px 0 0;color:var(--ink-soft);font-size:14px}
  .welcome p.tagline.exp-ask{color:var(--ink);max-width:54ch}
  .exp-privacy{margin:18px auto 0;max-width:56ch;color:var(--ink-soft);font-size:13px;line-height:1.45}
  .turn--bot.exp-rated{flex-direction:column;align-items:flex-start;gap:6px}
  .exp-rate{display:flex;flex-wrap:wrap;align-items:center;gap:6px;font-size:13px;color:var(--ink-soft);padding-left:4px}
  .exp-rate button{padding:4px 10px;border:1px solid var(--rule);border-radius:999px;background:var(--bone);
    color:var(--ink);font-size:13px;cursor:pointer}
  .exp-rate button:hover{background:var(--bone-deep)}
  .exp-rate button.is-picked{background:var(--ink);color:var(--bone);border-color:var(--ink)}
  .exp-rate-status{min-width:6ch}
"""

SUGGESTIONS = [
    "Translate into Oshindonga: Where is the nearest clinic?",
    "Translate into Oshikwanyama: The rains came early this year.",
    "In Oshindonga, explain simply how vaccines work.",
    "Write a short SMS in Oshikwanyama inviting my aunt to a wedding.",
]

RATING_JS = """
      // ── experimental: rate the Oshiwambo of every reply ──
      var botTurnIndex = 0;
      var RATINGS = [['correct', '\u2713 Correct'], ['minor_errors', '~ Small mistakes'], ['wrong', '\u2717 Wrong']];
      function addLanguageRating(turnEl) {
        var idx = botTurnIndex++;
        turnEl.classList.add('exp-rated');
        var row = document.createElement('div');
        row.className = 'exp-rate';
        var label = document.createElement('span');
        label.textContent = 'Is the Oshiwambo correct?';
        row.appendChild(label);
        var status = document.createElement('span');
        status.className = 'exp-rate-status';
        RATINGS.forEach(function (opt) {
          var b = document.createElement('button');
          b.type = 'button';
          b.textContent = opt[1];
          b.addEventListener('click', async function () {
            row.querySelectorAll('button').forEach(function (x) { x.classList.remove('is-picked'); });
            b.classList.add('is-picked');
            status.textContent = 'Saving\u2026';
            try {
              var r = await fetch(API.replace('/v1/chat', '/v1/chat/feedback'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, turn: idx, rating: opt[0] }),
                credentials: 'omit',
              });
              status.textContent = r.ok ? 'Thank you!' : 'Could not save.';
            } catch (e) { status.textContent = 'Could not save.'; }
          });
          row.appendChild(b);
        });
        row.appendChild(status);
        turnEl.appendChild(row);
      }
"""


def welcome_html(name: str, note: str) -> str:
    first, _, rest = name.partition(" ")
    chips = "\n".join(f'        <button class="chip" data-suggest>{html.escape(t)}</button>' for t in SUGGESTIONS)
    return f"""<section class="welcome" id="welcome">
      <p class="exp-kicker">Test environment &middot; new Oshiwambo model</p>
      <h1 class="welcome-mark exp-model">{html.escape(first)} <span class="exp-model-name">{html.escape(rest)}</span><span class="dot">.</span></h1>
      <p class="exp-model-meta">{html.escape(note)}</p>
      <p class="tagline exp-ask"><strong>Please test it in Oshindonga and Oshikwanyama.</strong>
        Chat with it, ask questions or ask for translations, in either language.
        After every answer, tell us with one click whether its Oshiwambo is correct.</p>
      <div class="suggestions" id="suggestions">
{chips}
      </div>
      <p class="exp-privacy">When you rate an answer, we keep that question and answer, with personal
        details removed, to measure and improve the model.</p>
    </section>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Ongiini Ombuto", help="the model's name, shown big on the start page")
    ap.add_argument("--model-note", required=True, help="what is behind the name (base model, version, date)")
    args = ap.parse_args(argv)
    page = SRC.read_text()
    replacements = [
        ("<title>Chat in your browser — Ongiini AI</title>", "<title>Experimental test environment — Ongiini AI</title>"),
        ('<meta name="robots" content="index, follow, max-image-preview:large" />',
         '<meta name="robots" content="noindex, nofollow" />'),
        ("""        : 'https://api.ongiini.ai/v1/chat';""", """        : '/v1/chat';  // experimental backend, same origin"""),
        ("</style>", BANNER_CSS + "</style>"),
        ("      function appendTyping() {", RATING_JS + "      function appendTyping() {"),
        ("          appendBotTurn(data.reply || '');",
         "          appendBotTurn(data.reply || '');\n          addLanguageRating(turnsEl.lastElementChild);"),
        ("        turnsEl.innerHTML = '';", "        turnsEl.innerHTML = '';\n        botTurnIndex = 0;"),
    ]
    for old, new in replacements:
        if old not in page:
            raise SystemExit(f"chat-app changed, update build_experimental_chat.py: {old[:60]!r} not found")
        page = page.replace(old, new, 1)
    page, n = re.subn(r'<section class="welcome" id="welcome">.*?</section>',
                      lambda _: welcome_html(args.model_name, args.model_note), page, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("chat-app changed, update build_experimental_chat.py: welcome section not found")
    banner = (
        '<div class="exp-banner" role="note">'
        "Test environment: you are chatting with our latest experimental Oshiwambo model, not the regular Ongiini AI. "
        "It changes often and can make mistakes. "
        'For the regular AI assistant use <a href="https://chat.ongiini.ai">chat.ongiini.ai</a>.'
        f"<small>Model now: {html.escape(args.model_name)} \u2014 {html.escape(args.model_note)}</small></div>\n"
    )
    page = page.replace("<body>\n", "<body>\n" + banner, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
