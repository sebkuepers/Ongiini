"""Light-touch PII sanitisation for memory storage.

We redact a *conservative* set of patterns — only those with very low
false-positive rates. Better to miss redacting something than to mangle
the user's actual question.

Applied at WRITE time (in main.py before memory.save). The model still
sees the un-redacted user message in the current turn; redaction only
affects what lands on disk and what's replayed in future turns.

Patterns currently scrubbed:
- Email addresses
- Namibian mobile numbers (08x xxx xxxx, +264 8x xxx xxxx, 00264…,
  bare 2648x…) and any other "+"-prefixed international number
- Credit card numbers (13-19 digits, with optional spaces/dashes)
- IBAN-style international bank account numbers
- 11-digit standalone numbers that look like Namibian National IDs

What we deliberately DON'T try to scrub (too false-positive prone for
WhatsApp-style messages):
- Street addresses
- Person names
- Passport numbers (varied format)
- Namibian landlines, service and short numbers (061 374 400,
  +264 61 374 400, 10111) — those are how users are told to reach BIPA,
  clinics or the police, and are not personal. Mobile numbers are: they
  appeared verbatim in stored CVs and job applications.
- **URLs** — public web addresses, never user-shared PII. Facebook
  video IDs and similar long-digit path components were matching the
  credit-card regex and corrupting cited source links in replies. The
  sanitiser now splits text into URL / non-URL chunks and only scans
  the non-URL chunks.
"""

from __future__ import annotations

import re

# Order matters — broadest patterns last so earlier specific ones win.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Email — RFC-ish, conservative
    (re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), "[REDACTED:email]"),

    # IBAN — 2 country letters, 2 check digits, 11-30 alphanumerics
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "[REDACTED:iban]"),

    # Namibian mobile — 0 or (+|00)264 or bare 264, then 8x and seven
    # digits, spaces/dashes allowed. Landlines (06x) are left alone.
    # Must run before the card pattern: "+264 81 …" is 12 digits and
    # longer foreign numbers used to surface as "+[REDACTED:card]".
    (re.compile(r"(?<![\d+])(?:(?:\+|00)?264[ -]?|0)8\d(?:[ -]?\d){7}(?!\d)"),
     "[REDACTED:phone]"),

    # Any other "+"-prefixed international number (E.164: 8-15 digits).
    # +264 is excluded so Namibian landlines written with the country
    # code stay readable.
    (re.compile(r"(?<![\d+])\+(?!264)\d(?:[ -]?\d){7,14}(?!\d)"),
     "[REDACTED:phone]"),

    # Credit card — 13-19 digits, possibly broken by spaces or dashes,
    # bounded by non-digit. Catches "4111 1111 1111 1111" style.
    (re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"), "[REDACTED:card]"),

    # Namibian National ID — 11 digits standalone (i.e. not embedded in a
    # longer number). Some false-positive risk on year-of-birth-prefixed
    # 11-digit codes; the conservative bounded match is the price.
    (re.compile(r"(?<!\d)\d{11}(?!\d)"), "[REDACTED:id]"),
]


# Used to peel URLs out of the input before scanning. We use a permissive
# tail (\S+) because we want to preserve the URL exactly as the model
# emitted it — even trailing punctuation rides along. The transport layer
# already does deep-link cleanup separately.
_URL_RE = re.compile(r"https?://\S+")


def _scrub(chunk: str) -> str:
    for pat, replacement in _PATTERNS:
        chunk = pat.sub(replacement, chunk)
    return chunk


def sanitize(text: str) -> str:
    """Return `text` with known PII patterns replaced by tagged placeholders.

    URL substrings are preserved verbatim — long-digit path components
    (Facebook video IDs, news article slugs with timestamps, etc.) were
    triggering the credit-card / National-ID matchers and corrupting
    cited source links. URLs are public addresses, not user PII.

    Returns the input unchanged if no patterns match. Never raises on
    valid str input.
    """
    if not text:
        return text
    out: list[str] = []
    last_end = 0
    for m in _URL_RE.finditer(text):
        out.append(_scrub(text[last_end:m.start()]))
        out.append(m.group(0))    # URL kept verbatim
        last_end = m.end()
    out.append(_scrub(text[last_end:]))
    return "".join(out)


def sanitize_message(msg: dict) -> dict:
    """Return a NEW message dict with the `content` field sanitized."""
    if not isinstance(msg, dict):
        return msg
    content = msg.get("content")
    if isinstance(content, str):
        return {**msg, "content": sanitize(content)}
    return msg
