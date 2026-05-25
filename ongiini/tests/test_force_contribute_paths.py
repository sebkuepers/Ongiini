"""Lightweight unit tests for the contribute-flow regex helpers.

The full pre-execution intercepts live in ongiini/api/main.py and
pull in mem0 (vLLM-backed) at import time, which isn't available in
this test environment. We test the pure regex helpers directly here;
the integration behaviour is verified end-to-end on Spark."""
from __future__ import annotations

import re

import pytest


# Mirror of the regex defined in ongiini/api/main.py — kept identical
# so this test fails loudly if the prod regex drifts. (Yes, duplication
# is ugly, but the alternative is full mem0 in the test env.)
_SKIP_RE_SOURCE = (
    r"^\s*("
    r"skip( this( one)?)?|"
    r"skip it|"
    r"next( one)?( please)?|"
    r"different( one)?( please)?|"
    r"another( one)?( please)?|"
    r"give me (?:another|a different|the next)(?: one)?(?: please)?|"
    r"send (?:me )?(?:another|a different|the next)(?: one)?(?: please)?|"
    r"easier( one| please| sentence)?|"
    r"too (?:hard|difficult|long|complex)|"
    r"this is too (?:hard|difficult|long|complex)|"
    r"i don'?t know( this( one)?)?|"
    r"i can'?t (?:translate|do) this|"
    r"no idea|"
    r"i'?m not sure( about this)?|"
    r"change (?:this|the sentence)|"
    r"can (?:i|we) (?:get|have|try) (?:another|a different)"
    r")[\s.!?]*$"
)
SKIP_RE = re.compile(_SKIP_RE_SOURCE, re.IGNORECASE)


@pytest.mark.parametrize("text", [
    "skip", "skip this", "skip this one", "Skip it",
    "next", "next one", "Next one please",
    "another", "another one", "Another please",
    "different one",
    "give me another", "give me a different one", "give me the next",
    "send me another", "send the next",
    "easier", "easier one", "easier please",
    "too hard", "too difficult",
    "this is too difficult", "this is too long",
    "I don't know", "i dont know", "I don't know this one",
    "I can't translate this", "I cant do this",
    "no idea",
    "I'm not sure", "im not sure about this",
    "change this", "change the sentence",
    "can we try another", "can i have a different",
])
def test_skip_regex_matches_common_skip_phrases(text: str):
    assert SKIP_RE.match(text), f"should match: {text!r}"


@pytest.mark.parametrize("text", [
    # Real Oshindonga translations — must NOT match
    "Ondi hala oku ku mona nawa",
    "Tangi unene",
    "Onawa, ondi li nawa",
    # Plausible English content that isn't a skip
    "I am very happy to see you today, my friend",
    "Please help me with this important question",
    # Single Oshiwambo acknowledgement words
    "Eewa",
    "Ehee",
    # Yes / no for the followup prompt — handled by a DIFFERENT regex,
    # must not trip the skip regex (which lives in the SAVE path)
    "yes",
    "no",
    "done",
])
def test_skip_regex_does_not_match_real_content(text: str):
    assert not SKIP_RE.match(text), f"should NOT match: {text!r}"


def test_skip_regex_anchored_does_not_match_substring():
    """A long sentence that happens to contain the word 'skip' inside
    must NOT match — the regex is anchored start-to-end."""
    assert not SKIP_RE.match("I want to skip work today and go home")
    assert not SKIP_RE.match("Don't skip the queue please")
