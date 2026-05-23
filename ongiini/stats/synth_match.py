"""Synthesis label-matching helpers.

Pure stdlib so unit tests can import these without pulling in the mem0 /
qdrant / sentence-transformers chain that ``stats.analyses`` depends on.

These helpers exist because the LLM's cluster items don't always match
the cached extraction labels exactly:
  - case flips ('Yellowing Maize Leaves' vs 'yellowing maize leaves')
  - trailing punctuation drift ('foo.' vs 'foo')
  - the LLM copying the input payload's '(count: N)' suffix verbatim

Without normalisation, strict-equals matching returns zero hits and every
cluster's total collapses to 0 — the aggregator then dumps every cached
item into the long-tail "Other" bucket. Production-confirmed 2026-05-23.
"""

from __future__ import annotations

import re

# Trailing "(count: N)" the LLM may have copied verbatim from the input
# payload. Anchored at end-of-string so a 'count' word inside the label
# is not affected.
_COUNT_SUFFIX_RE = re.compile(r"\s*\(count\s*:\s*\d+\s*\)\s*$", re.IGNORECASE)


def norm_label(s: str) -> str:
    """Normalise a label for case-/punctuation-/count-suffix-insensitive
    matching between LLM cluster items and cached extraction labels."""
    s = s.strip()
    s = _COUNT_SUFFIX_RE.sub("", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(".,;:!?\"'`")
    return s
