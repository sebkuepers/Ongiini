"""Validator tests for the 6 new card types (cloze, reorder,
multiple_choice, grammar, proverb, dialogue). Each test exercises
the shape contract — what's required + what gets rejected.

The validator runs against the LLM's emitted card payload BEFORE the
card is persisted. A bad shape here means the frontend would render
a broken card, so be strict at this boundary."""
from __future__ import annotations

import pytest

from ongiini.learning.cards import _validate_card
from ongiini.learning.llm import ModelOutputError


def _good(card_type: str, **extra) -> dict:
    """Minimum-good payload for a given card type, with caller-
    provided overrides merged in."""
    base = {
        "card_type": card_type,
        "prompt_text": "default prompt",
        "reference_answer": "x",
        "module_id": "m1",
    }
    base.update(extra)
    return base


# ──────────────────────────────────────────────────────────────────
# Shared contract — every exercise type needs reference_answer
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ct", [
    "vocab", "translation", "production",
    "cloze", "reorder", "multiple_choice", "grammar",
    "proverb", "dialogue",
])
def test_exercise_card_requires_reference_answer(ct):
    """Lessons are exempt (acknowledged not graded); every other card
    type needs something to grade against."""
    payload = _good(ct, reference_answer="")
    # Cloze needs ___; dialogue needs turns; reorder needs tokens; MC
    # needs options. Most will fail on the type-specific check before
    # reference_answer — we just want to confirm no path silently
    # accepts an empty reference_answer.
    with pytest.raises(ModelOutputError):
        _validate_card(payload)


def test_lesson_card_does_not_require_reference_answer():
    """Lessons are read-and-acknowledge."""
    _validate_card({
        "card_type": "lesson",
        "prompt_text": "Today's topic: greetings",
    })


# ──────────────────────────────────────────────────────────────────
# Cloze
# ──────────────────────────────────────────────────────────────────

def test_cloze_requires_blank_marker_in_prompt():
    """The frontend reads prompt_text to position the input slot —
    a cloze without ___ would render with no blank."""
    with pytest.raises(ModelOutputError, match="must contain '___'"):
        _validate_card(_good("cloze", prompt_text="No blank here"))


def test_cloze_accepts_triple_or_quadruple_underscore():
    _validate_card(_good(
        "cloze",
        prompt_text="Ich ___ Kaffee.",
        reference_answer="trinke",
    ))
    _validate_card(_good(
        "cloze",
        prompt_text="Ich ____ Kaffee.",
        reference_answer="trinke",
    ))


# ──────────────────────────────────────────────────────────────────
# Reorder
# ──────────────────────────────────────────────────────────────────

def test_reorder_requires_tokens_list_of_at_least_2():
    with pytest.raises(ModelOutputError, match="tokens"):
        _validate_card(_good(
            "reorder", prompt_text="Arrange:", tokens=["only-one"],
            reference_answer="only-one",
        ))


def test_reorder_rejects_non_string_tokens():
    with pytest.raises(ModelOutputError, match="non-empty"):
        _validate_card(_good(
            "reorder", prompt_text="Arrange:", tokens=["ich", 42, "Hause"],
            reference_answer="ich gehe Hause",
        ))


def test_reorder_happy_path():
    _validate_card(_good(
        "reorder", prompt_text="Arrange:",
        tokens=["nach", "ich", "Hause", "gehe", "jetzt"],
        reference_answer="ich gehe jetzt nach Hause",
    ))


# ──────────────────────────────────────────────────────────────────
# Multiple choice
# ──────────────────────────────────────────────────────────────────

def test_mc_requires_2_to_4_options():
    with pytest.raises(ModelOutputError, match="2-4 options"):
        _validate_card(_good("multiple_choice",
            options=[{"label": "A", "text": "x"}],
            reference_answer="A",
        ))
    with pytest.raises(ModelOutputError, match="2-4 options"):
        _validate_card(_good("multiple_choice",
            options=[{"label": chr(65+i), "text": "x"} for i in range(5)],
            reference_answer="A",
        ))


def test_mc_rejects_duplicate_option_labels():
    with pytest.raises(ModelOutputError, match="unique"):
        _validate_card(_good("multiple_choice",
            options=[
                {"label": "A", "text": "foo"},
                {"label": "A", "text": "bar"},
            ],
            reference_answer="A",
        ))


def test_mc_reference_answer_must_match_option_label():
    """The grader maps the learner's pick by label; if reference_answer
    doesn't match an option label there's no truth to score against."""
    with pytest.raises(ModelOutputError, match="match one of"):
        _validate_card(_good("multiple_choice",
            options=[
                {"label": "A", "text": "foo"},
                {"label": "B", "text": "bar"},
            ],
            reference_answer="C",     # not in {A, B}
        ))


def test_mc_explanation_field_is_optional_but_typed():
    # No explanations — fine.
    _validate_card(_good("multiple_choice",
        options=[
            {"label": "A", "text": "foo"},
            {"label": "B", "text": "bar"},
        ],
        reference_answer="A",
    ))
    # With explanations — fine.
    _validate_card(_good("multiple_choice",
        options=[
            {"label": "A", "text": "foo", "explanation": "why A"},
            {"label": "B", "text": "bar", "explanation": "why B"},
        ],
        reference_answer="A",
    ))
    # Non-string explanation — rejected.
    with pytest.raises(ModelOutputError, match="explanation"):
        _validate_card(_good("multiple_choice",
            options=[
                {"label": "A", "text": "foo", "explanation": 123},
                {"label": "B", "text": "bar"},
            ],
            reference_answer="A",
        ))


# ──────────────────────────────────────────────────────────────────
# Grammar
# ──────────────────────────────────────────────────────────────────

def test_grammar_requires_source_sentence():
    with pytest.raises(ModelOutputError, match="source_sentence"):
        _validate_card(_good(
            "grammar",
            prompt_text="Rewrite in perfect:",
            reference_answer="ich bin gegangen",
        ))


def test_grammar_happy_path():
    _validate_card(_good(
        "grammar",
        prompt_text="Rewrite in perfect:",
        source_sentence="ich gehe",
        reference_answer="ich bin gegangen",
    ))


# ──────────────────────────────────────────────────────────────────
# Dialogue
# ──────────────────────────────────────────────────────────────────

def test_dialogue_requires_turns_list():
    with pytest.raises(ModelOutputError, match="turns"):
        _validate_card(_good(
            "dialogue",
            prompt_text="Complete:",
            reference_answer="something",
        ))


def test_dialogue_each_turn_needs_speaker_and_text():
    with pytest.raises(ModelOutputError, match="speaker"):
        _validate_card(_good(
            "dialogue",
            prompt_text="Complete:",
            # 2 turns to satisfy the length check; second one missing speaker.
            turns=[
                {"speaker": "A", "text": "Hello"},
                {"text": "no speaker"},
            ],
            reference_answer="x",
        ))
    with pytest.raises(ModelOutputError, match="text"):
        _validate_card(_good(
            "dialogue",
            prompt_text="Complete:",
            turns=[
                {"speaker": "A", "text": "Hello"},
                {"speaker": "B"},   # missing text
            ],
            reference_answer="x",
        ))


def test_dialogue_happy_path():
    _validate_card(_good(
        "dialogue",
        prompt_text="Complete your line:",
        turns=[
            {"speaker": "Interviewer", "text": "Erzählen Sie etwas."},
            {"speaker": "You", "text": "___"},
        ],
        reference_answer="Mein Name ist Sebastian.",
    ))


# ──────────────────────────────────────────────────────────────────
# Proverb
# ──────────────────────────────────────────────────────────────────

def test_proverb_cultural_note_typed_when_present():
    with pytest.raises(ModelOutputError, match="cultural_note"):
        _validate_card(_good(
            "proverb",
            prompt_text="Complete the idiom:",
            reference_answer="Viele Köche verderben den Brei.",
            cultural_note=42,
        ))


def test_proverb_happy_path_with_cultural_note():
    _validate_card(_good(
        "proverb",
        prompt_text="'too many cooks spoil the broth' in German:",
        reference_answer="Viele Köche verderben den Brei.",
        cultural_note="Used about over-collaboration in workplaces.",
    ))


def test_proverb_cultural_note_field_optional():
    _validate_card(_good(
        "proverb",
        prompt_text="'too many cooks' in German:",
        reference_answer="Viele Köche verderben den Brei.",
    ))
