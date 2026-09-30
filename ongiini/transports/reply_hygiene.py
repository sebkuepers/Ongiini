"""Reply hygiene shared by the WhatsApp and web-chat transports.

Deterministic, no network, no LLM:

  - **URL allowlist.** On turns where a citation tool ran, a URL in the
    reply must be one the tools actually returned this turn
    (``ReplyContext.allowed_urls``). Anything else was invented by the
    model — the most common confabulation we have seen — and is
    removed. URLs carrying HTML fragments (a Tavily snippet artefact)
    are always removed on those turns. This replaces the old per-URL
    HEAD check: the allowed URLs came back live from the search
    provider seconds earlier, and the check cost ~2 s per search turn.
  - **Boundary trimming.** A draft that stopped on its token limit is cut
    back to the last paragraph or sentence end, so the user never gets a
    half sentence, and an offer to continue is appended.
  - **Boundary capping.** Over-long bodies are cut at a boundary inside
    the transport's limit instead of mid-word.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

# Tools whose output the model cites as URLs. Transport-side knowledge
# (anti-trap #8 keeps tool names out of the framework).
CITATION_TOOLS = frozenset({"web_search", "fetch_url", "fetch_urls"})

URL_RE = re.compile(r"https?://[^\s)>\"\]]+")
_TRAILING_PUNCT = ".,;:!?)'\""
# What is left of a citation line once its URL is gone: bullets, dashes,
# a "source(s):" label, empty parens.
_EMPTY_CITATION_LINE_RE = re.compile(
    r"^[\s\-–—•*_>]*(?:(?:sources?|bron(?:ne)?|link)\s*:?)?[\s\-–—•*_()\[\]:.,]*$",
    re.IGNORECASE,
)
_SENTENCE_END_RE = re.compile(r"[.!?…](?:[\"')\]]*)(?=\s)")


def cites_tools(used_tools: tuple[str, ...]) -> bool:
    return bool(CITATION_TOOLS.intersection(used_tools))


def canonical_url(url: str) -> str:
    """Normalise for set membership: lowercase scheme + host, drop
    ``www.``, fragment and trailing slash; keep path and query."""
    url = url.strip().rstrip(_TRAILING_PUNCT)
    try:
        parts = urlsplit(url)
    except ValueError:
        return url.lower()
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/")
    return urlunsplit(("https", host, path, parts.query, ""))


def drop_unlisted_urls(text: str, allowed: tuple[str, ...]) -> tuple[str, int]:
    """Remove URLs that are not in ``allowed`` (or that carry HTML
    fragments). A line reduced to an empty citation stub is dropped.
    Returns ``(text, number_of_urls_removed)``.

    With an empty allowlist nothing is verified, so only malformed URLs
    are removed — a failed search must not also strip URLs the user
    gave us."""
    if not text:
        return text, 0
    allowed_set = {canonical_url(u) for u in allowed}
    removed = 0
    out: list[str] = []
    for line in text.split("\n"):
        new_line = line
        touched = False
        for raw in URL_RE.findall(line):
            url = raw.rstrip(_TRAILING_PUNCT)
            malformed = "<" in url or ">" in url
            unlisted = bool(allowed_set) and canonical_url(url) not in allowed_set
            if malformed or unlisted:
                new_line = new_line.replace(url, "")
                removed += 1
                touched = True
        if touched:
            new_line = re.sub(r"\(\s*\)", "", new_line).rstrip()
            if _EMPTY_CITATION_LINE_RE.match(new_line):
                continue
        out.append(new_line)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return cleaned, removed


def trim_to_boundary(text: str, *, min_keep_ratio: float = 0.5) -> str:
    """Cut ``text`` back to its last paragraph break or sentence end, if
    that keeps at least ``min_keep_ratio`` of it. Otherwise cut at the
    last whitespace. Used for drafts that hit the token limit."""
    text = text.rstrip()
    if not text:
        return text
    floor = int(len(text) * min_keep_ratio)
    para = text.rfind("\n\n")
    if para >= floor:
        return text[:para].rstrip()
    ends = [m.end() for m in _SENTENCE_END_RE.finditer(text + " ")]
    ends = [e for e in ends if e >= floor]
    if ends:
        return text[:ends[-1]].rstrip()
    space = text.rfind(" ")
    return (text[:space] if space >= floor else text).rstrip()


def cap_at_boundary(text: str, limit: int) -> str:
    """Fit ``text`` into ``limit`` chars, cutting at a boundary."""
    if len(text) <= limit:
        return text
    return trim_to_boundary(text[:limit], min_keep_ratio=0.6)
