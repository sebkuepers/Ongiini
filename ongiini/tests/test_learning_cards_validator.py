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


def test_lesson_card_must_use_steps_shape():
    """Lessons MUST use the multi-step shape now — the coach is the
    only caller and asks the LLM specifically for steps[]. Legacy
    single-blob shape (prompt_text only) is rejected outright."""
    with pytest.raises(ModelOutputError, match="'steps' as a list"):
        _validate_card({
            "card_type": "lesson",
            "prompt_text": "Today's topic: greetings",
        })


# ──────────────────────────────────────────────────────────────────
# Multi-step lesson card (carousel shape)
# ──────────────────────────────────────────────────────────────────

def _good_steps(quick_check: bool = True) -> list[dict]:
    steps: list[dict] = [
        {"kind": "concept", "body": "Concept introduction sentence."},
        {"kind": "example", "body": "Concrete example sentence.",
         "examples": ["Hier ist ein Beispiel."]},
        {"kind": "contrast", "body": "Contrast with the opposite case."},
    ]
    if quick_check:
        steps.append({
            "kind": "quick_check",
            "prompt": "Which form fits a formal greeting?",
            "answer": "Guten Tag.",
            "hint": "Used in shops, hotels, offices.",
        })
    return steps


def test_stepped_lesson_validates_without_prompt_text():
    """The new shape carries content in steps[] — prompt_text is
    optional (synthesised on the persistence side)."""
    _validate_card({
        "card_type": "lesson",
        "title": "Formal vs Informal Greetings",
        "module_id": "m1",
        "topic_id": "t1",
        "steps": _good_steps(),
    })


def test_stepped_lesson_ignores_stray_prompt_text():
    """If the model includes both prompt_text and steps, the validator
    no longer fails — the new architecture asks ONLY for steps, so
    any prompt_text the model emits is just ignored downstream
    (the coach builds the lesson payload from steps + title)."""
    _validate_card({
        "card_type": "lesson",
        "prompt_text": "ignored leftover prose",
        "steps": _good_steps(),
    })


def test_stepped_lesson_rejects_under_min_steps():
    """One step isn't a carousel."""
    with pytest.raises(ModelOutputError, match="2-5 entries"):
        _validate_card({
            "card_type": "lesson",
            "topic_id": "t1",
            "steps": [{"kind": "concept", "body": "Just one."}],
        })


def test_stepped_lesson_rejects_over_max_steps():
    """Six steps is too many — carousel fatigue."""
    steps = [{"kind": "concept", "body": f"Step {i}."} for i in range(6)]
    with pytest.raises(ModelOutputError, match="2-5 entries"):
        _validate_card({
            "card_type": "lesson",
            "topic_id": "t1",
            "steps": steps,
        })


def test_stepped_lesson_rejects_unknown_step_kind():
    with pytest.raises(ModelOutputError, match="must be one of"):
        _validate_card({
            "card_type": "lesson",
            "topic_id": "t1",
            "steps": [
                {"kind": "concept", "body": "First."},
                {"kind": "bogus", "body": "Bad."},
            ],
        })


def test_stepped_lesson_rejects_quick_check_not_last():
    """The renderer pegs the reveal-answer interaction to the LAST
    step — a quick_check in the middle would break the UX."""
    with pytest.raises(ModelOutputError, match="LAST step"):
        _validate_card({
            "card_type": "lesson",
            "topic_id": "t1",
            "steps": [
                {"kind": "quick_check",
                 "prompt": "?", "answer": "Yes"},
                {"kind": "concept", "body": "Concept after quiz?"},
            ],
        })


# Multiple quick_check steps would also trip the "LAST step" check
# first (only the last index can be quick_check; everything else
# raises the LAST check earlier), so the "at most one" branch in the
# validator is defensive belt-and-braces. No test needed for an
# unreachable path.


def test_stepped_lesson_concept_requires_body():
    with pytest.raises(ModelOutputError, match="'concept' requires"):
        _validate_card({
            "card_type": "lesson",
            "topic_id": "t1",
            "steps": [
                {"kind": "concept"},  # no body
                {"kind": "example", "body": "Example."},
            ],
        })


def test_stepped_lesson_quick_check_requires_prompt_and_answer():
    with pytest.raises(ModelOutputError, match="'prompt'"):
        _validate_card({
            "card_type": "lesson",
            "topic_id": "t1",
            "steps": [
                {"kind": "concept", "body": "First."},
                {"kind": "quick_check", "answer": "Yes"},  # no prompt
            ],
        })
    with pytest.raises(ModelOutputError, match="'answer'"):
        _validate_card({
            "card_type": "lesson",
            "topic_id": "t1",
            "steps": [
                {"kind": "concept", "body": "First."},
                {"kind": "quick_check", "prompt": "?"},  # no answer
            ],
        })


def test_stepped_lesson_examples_must_be_strings():
    with pytest.raises(ModelOutputError, match="non-empty strings"):
        _validate_card({
            "card_type": "lesson",
            "topic_id": "t1",
            "steps": [
                {"kind": "concept", "body": "Concept.",
                 "examples": ["good", 42]},
                {"kind": "example", "body": "Example."},
            ],
        })


# Topic_id / module_id are no longer validated here — the selector
# picks them and the coach attaches them after generate_card_content
# returns. The model isn't asked to emit them, so we don't check.


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
        ))
    with pytest.raises(ModelOutputError, match="text"):
        _validate_card(_good(
            "dialogue",
            prompt_text="Complete:",
            turns=[
                {"speaker": "A", "text": "Hello"},
                {"speaker": "B"},   # missing text
            ],
        ))


def test_dialogue_requires_at_least_one_blank():
    """A dialogue with no '___' anywhere has nothing to drill — fatal."""
    with pytest.raises(ModelOutputError, match="___"):
        _validate_card(_good(
            "dialogue",
            prompt_text="Read the dialogue:",
            turns=[
                {"speaker": "A", "text": "Hello!"},
                {"speaker": "B", "text": "Hi!"},
            ],
        ))


def test_dialogue_blank_turn_requires_answer():
    """Every turn containing '___' MUST carry a non-empty 'answer'."""
    with pytest.raises(ModelOutputError, match="answer"):
        _validate_card(_good(
            "dialogue",
            prompt_text="Complete:",
            turns=[
                {"speaker": "A", "text": "Hallo! ___ Sie bereit?"},
                {"speaker": "B", "text": "Ja, ich bin bereit."},
            ],
        ))


def test_dialogue_multi_blank_turn_requires_pipe_separated_answer():
    """Two blanks in one turn need a pipe-separated answer with matching count."""
    with pytest.raises(ModelOutputError, match="blanks"):
        _validate_card(_good(
            "dialogue",
            prompt_text="Complete:",
            turns=[
                {"speaker": "A", "text": "Ich ___ einen Kaffee, ___ Sie auch?",
                 "answer": "trinke"},   # only 1 part for 2 blanks
                {"speaker": "B", "text": "Ja."},
            ],
        ))


def test_dialogue_happy_path():
    """New-shape happy path: per-turn 'answer' fields, synthesised
    composite reference_answer in pipe-separated form. Glosses must
    NOT contain '___' (the gloss is read-only context — the blank
    lives only in the target-language line so the renderer doesn't
    paint a second input that has no matching answer)."""
    payload = _good(
        "dialogue",
        prompt_text="Complete your line:",
        turns=[
            {"speaker": "Interviewer",
             "text": "Erzählen Sie etwas. (Tell us something.)"},
            {"speaker": "You",
             "text": "Mein Name ___ Sebastian. (My name is Sebastian.)",
             "answer": "ist"},
            {"speaker": "Interviewer",
             "text": "Und ___ alt sind Sie? (And how old are you?)",
             "answer": "wie"},
        ],
    )
    _validate_card(payload)
    assert payload["reference_answer"] == "ist | wie"


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
