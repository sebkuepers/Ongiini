"""Anti-PII guardrails for the qualitative-analysis layer.

Two surfaces:

  (1) ANTI_PII_PROMPT — text prefix prepended to every analysis's
      extraction prompt. Tells the model in plain language what may
      and may not appear in its output. Includes a 'REDACTED'
      sentinel for messages the model judges impossible to generalise.

  (2) sanitise_label() — regex-level post-extraction filter. Even
      if the model misbehaves, this drops any label containing
      identifying patterns (4-digit numbers, capitalised possessives,
      known town names, anything our existing PII scrubber catches).
      Failed labels are dropped, not stored, never reach the page.

Both layers are conservative: the cost of dropping a useful label is
low (we'll see more from the same user later); the cost of leaking
a single identifying phrase is high (privacy harm + reputational risk
to the foundation).

Pure stdlib + the existing PII scrubber — no mem0, no openai. Importable
from anywhere without dragging the LLM dependency graph.
"""

from __future__ import annotations

import re

from .. import pii


ANTI_PII_PROMPT = (
    "PRIVACY RULES — THIS OUTPUT WILL BE PUBLISHED IN AGGREGATE STATISTICS, "
    "SO IT MUST BE GENERIC:\n"
    "- NEVER include a person's name (first name, surname, nickname).\n"
    "- NEVER include a specific place smaller than country: no town, no "
    "  village, no neighbourhood, no school name, no clinic name, no "
    "  employer name. 'Namibia' or 'rural Namibia' is OK; 'Oshakati' or "
    "  'Heinis' is NOT.\n"
    "- NEVER include specific numbers: no ages, no dates, no years, no "
    "  quantities ('three hectares', '47 years old', '2024' — all forbidden).\n"
    "- NEVER quote the user's exact wording or any verbatim phrasing that "
    "  could identify the source message.\n"
    "- Describe the TYPE or CATEGORY of what's being discussed, not the "
    "  specifics of one person's situation.\n"
    "- If you cannot produce a fully generic phrase without leaking any of "
    "  the above, output exactly 'REDACTED' — nothing else.\n\n"
    "Examples (BAD → GOOD):\n"
    "  'Joseph grade 11 chemistry homework' → 'grade 11 chemistry homework'\n"
    "  'Maize farming in Oshakati' → 'maize farming'\n"
    "  'Diabetes diagnosis at age 47' → 'diabetes information'\n"
    "  'Wife of farmer in Onengali' → 'spouse of a farmer'\n"
    "  'Three children at Augustineum High School' → 'parenting school-age children'\n\n"
)


# 4-digit numbers (years, IDs).
_RE_FOUR_DIGITS = re.compile(r"\b\d{4}\b")
# Capitalised possessive ("Joseph's"), strong proper-noun signal.
_RE_POSSESSIVE = re.compile(r"\b[A-Z][a-zA-Z]{2,}'s\b")

# Known Namibian town / settlement names. Last-resort backstop — the
# extraction prompt is the primary defence. Lower-cased, substring-
# matched. Picked the ~25 largest towns by population plus smaller
# settlements observed in past cache audits.
_NAMIBIAN_TOWNS: frozenset[str] = frozenset({
    "windhoek", "rundu", "walvis bay", "swakopmund", "oshakati",
    "katima mulilo", "grootfontein", "rehoboth", "katutura", "otjiwarongo",
    "okahandja", "ondangwa", "tsumeb", "keetmanshoop", "ongwediva",
    "henties bay", "luderitz", "lüderitz", "gobabis", "outapi",
    "mariental", "oshikango", "eenhana", "opuwo", "khorixas",
    "omuthiya", "okakarara", "usakos", "karasburg", "arandis",
    "okongo", "ruacana", "tsumkwe", "noordoewer",
    "heinis", "onengali", "augustineum",
})


_MAX_LABEL_LEN = 80


def sanitise_label(label: str | None) -> str | None:
    """Return the cleaned label, or None if it's not safe to publish.

    Drops:
      - empty / non-string input
      - the model's 'REDACTED' sentinel (case-insensitive)
      - labels containing any PII pattern the existing regex layer
        recognises (emails / phones / IDs / cards / IBANs)
      - labels containing 4-digit numbers (years / IDs)
      - capitalised possessives ('Joseph's')
      - labels containing a known Namibian town/village name
      - labels longer than 80 chars (model failed to produce a concise
        topic — usually means it echoed too much of the message)
    """
    if not label or not isinstance(label, str):
        return None
    text = label.strip()
    if not text or text.upper() == "REDACTED":
        return None
    if len(text) > _MAX_LABEL_LEN:
        return None
    if _RE_FOUR_DIGITS.search(text):
        return None
    if _RE_POSSESSIVE.search(text):
        return None
    cleaned = pii.sanitize(text)
    if "[REDACTED:" in cleaned:
        return None
    lowered = text.lower()
    for town in _NAMIBIAN_TOWNS:
        if town in lowered:
            return None
    return text
