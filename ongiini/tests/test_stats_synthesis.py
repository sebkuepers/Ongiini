"""Tests for ongiini/stats/synth_match.py — label normalisation.

Regression coverage for the 100% Other bug (2026-05-23): Gemma 4 copied
the input payload's "(count: N)" suffix verbatim into cluster items, and
the strict-equals matching against cached labels returned zero matches —
collapsing all 242 messages into the long-tail Other bucket. The fix
extends norm_label to strip that suffix.
"""

from __future__ import annotations

from ongiini.stats.synth_match import norm_label, _COUNT_SUFFIX_RE


# ---------- norm_label ----------

def test_norm_label_strips_count_suffix():
    """The PRIMARY regression: LLM-copied (count: N) suffix must be removed."""
    assert norm_label("small talk (count: 112)") == "small talk"
    assert norm_label("Current President of Namibia (count: 3)") == "current president of namibia"
    assert norm_label("hello (count:5)") == "hello"   # no space variant
    assert norm_label("hello (count : 5 )") == "hello"   # extra spaces


def test_norm_label_strips_count_suffix_case_insensitive():
    """Match COUNT, Count, count — all variants."""
    assert norm_label("foo (COUNT: 1)") == "foo"
    assert norm_label("foo (Count: 2)") == "foo"


def test_norm_label_preserves_inner_count_text():
    """A 'count' substring inside the label is NOT a trailing suffix
    and must be left alone."""
    assert norm_label("count my tokens (count: 4)") == "count my tokens"
    assert norm_label("I want to count things") == "i want to count things"


def test_norm_label_lowercases():
    assert norm_label("Yellowing Maize Leaves") == "yellowing maize leaves"


def test_norm_label_strips_trailing_punctuation():
    assert norm_label("yellowing maize leaves.") == "yellowing maize leaves"
    assert norm_label("question?") == "question"


def test_norm_label_collapses_whitespace():
    assert norm_label("  small   talk  ") == "small talk"


def test_norm_label_round_trip_input_to_cached():
    """End-to-end matching scenario: a cached label and the LLM's
    rendition of it (as written back from the input payload) must
    normalise to the same key."""
    cached = "current president of namibia"
    llm_returned = "Current President of Namibia (count: 3)"
    assert norm_label(cached) == norm_label(llm_returned)


def test_norm_label_handles_empty_string():
    assert norm_label("") == ""
    assert norm_label("   ") == ""
    assert norm_label("(count: 0)") == ""


# ---------- _COUNT_SUFFIX_RE direct ----------

def test_count_suffix_regex_only_matches_trailing():
    """The regex is anchored at end-of-string."""
    assert _COUNT_SUFFIX_RE.search("foo (count: 1)") is not None
    assert _COUNT_SUFFIX_RE.search("(count: 1) foo") is None  # leading, no match
    assert _COUNT_SUFFIX_RE.search("foo") is None
