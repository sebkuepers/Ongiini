"""Intake capture-spec tests — validators + missing-fields + completeness.

The LLM conducts the conversation; this module's deterministic job is
shape-validation of candidate values + a 'which fields are still null'
query. These tests lock the validators down and prove that the
missing/complete queries behave the way the API will need them to.
"""
from __future__ import annotations

import pytest

from ongiini.learning import intake


# ---------- validate: name ----------

def test_name_accepts_simple():
    r = intake.validate_field(intake.FIELD_NAME, "Sebastian")
    assert r.ok and r.value == "Sebastian"


def test_name_strips_whitespace():
    r = intake.validate_field(intake.FIELD_NAME, "  Sebastian  ")
    assert r.ok and r.value == "Sebastian"


def test_name_accepts_diacritics_apostrophe_hyphen_initials():
    """Namibian / diaspora names — diacritics, OW orthography,
    apostrophes, hyphens, initials. Lock these as accepted."""
    for name in [
        "Anna-Marie",
        "Naòmi",
        "O'Brien",
        "Hāneb",
        "M.K. Shilongo",
        "Köpfer",
    ]:
        r = intake.validate_field(intake.FIELD_NAME, name)
        assert r.ok, f"should accept {name!r}: {r.reason}"


def test_name_rejects_empty():
    assert not intake.validate_field(intake.FIELD_NAME, "").ok
    assert not intake.validate_field(intake.FIELD_NAME, "   ").ok


def test_name_rejects_too_long():
    assert not intake.validate_field(intake.FIELD_NAME, "x" * 41).ok


def test_name_rejects_digits():
    assert not intake.validate_field(intake.FIELD_NAME, "Player1").ok


def test_name_rejects_web_junk():
    for bad in ["alice@example.com", "<script>", "name|injection"]:
        assert not intake.validate_field(intake.FIELD_NAME, bad).ok, bad


def test_name_rejects_control_chars():
    assert not intake.validate_field(intake.FIELD_NAME, "alice\nbob").ok


def test_name_rejects_non_string():
    assert not intake.validate_field(intake.FIELD_NAME, 42).ok
    assert not intake.validate_field(intake.FIELD_NAME, None).ok


# ---------- validate: age ----------

def test_age_accepts_int():
    r = intake.validate_field(intake.FIELD_AGE, 25)
    assert r.ok and r.value == 25


def test_age_accepts_numeric_string():
    r = intake.validate_field(intake.FIELD_AGE, "33")
    assert r.ok and r.value == 33


def test_age_rejects_bool():
    """bool is a subclass of int — explicit reject so True can't pass."""
    assert not intake.validate_field(intake.FIELD_AGE, True).ok
    assert not intake.validate_field(intake.FIELD_AGE, False).ok


def test_age_rejects_word():
    assert not intake.validate_field(intake.FIELD_AGE, "twenty").ok


def test_age_rejects_out_of_range():
    assert not intake.validate_field(intake.FIELD_AGE, 5).ok
    assert not intake.validate_field(intake.FIELD_AGE, 200).ok


def test_age_rejects_other_types():
    assert not intake.validate_field(intake.FIELD_AGE, None).ok
    assert not intake.validate_field(intake.FIELD_AGE, [25]).ok


# ---------- validate: level ----------

@pytest.mark.parametrize("level", intake.VALID_LEVELS)
def test_level_accepts_canonical(level):
    r = intake.validate_field(intake.FIELD_LEVEL, level)
    assert r.ok and r.value == level


def test_level_accepts_capitalised():
    r = intake.validate_field(intake.FIELD_LEVEL, "Beginner")
    assert r.ok and r.value == "beginner"


def test_level_accepts_prefix_three_chars():
    r = intake.validate_field(intake.FIELD_LEVEL, "int")
    assert r.ok and r.value == "intermediate"


def test_level_rejects_too_short_prefix():
    assert not intake.validate_field(intake.FIELD_LEVEL, "b").ok


def test_level_rejects_unknown():
    assert not intake.validate_field(intake.FIELD_LEVEL, "fluent").ok


# ---------- validate: objective ----------

def test_objective_accepts_sentence():
    r = intake.validate_field(intake.FIELD_OBJECTIVE,
                              "I want to pass a job interview in Afrikaans.")
    assert r.ok


def test_objective_rejects_empty_and_too_short():
    assert not intake.validate_field(intake.FIELD_OBJECTIVE, "").ok
    assert not intake.validate_field(intake.FIELD_OBJECTIVE, "  ").ok
    assert not intake.validate_field(intake.FIELD_OBJECTIVE, "a").ok


def test_objective_rejects_too_long():
    assert not intake.validate_field(intake.FIELD_OBJECTIVE, "x" * 201).ok


# ---------- validate: unknown field ----------

def test_validate_unknown_field_returns_bad():
    r = intake.validate_field("favourite_colour", "blue")
    assert not r.ok


# ---------- missing_fields ----------

def test_missing_fields_none_profile_returns_all_fields():
    assert intake.missing_fields(None) == list(intake.INTAKE_FIELDS)


def test_missing_fields_empty_profile_returns_all_fields():
    assert intake.missing_fields({}) == list(intake.INTAKE_FIELDS)


def test_missing_fields_with_some_captured():
    profile = {"name": "Sebastian", "age": 35}
    missing = intake.missing_fields(profile)
    assert intake.FIELD_NAME not in missing
    assert intake.FIELD_AGE not in missing
    assert intake.FIELD_LEVEL in missing
    assert intake.FIELD_OBJECTIVE in missing


def test_missing_fields_treats_empty_string_as_missing():
    """A blank string is captured-but-empty — still missing for
    LLM-conducted intake purposes."""
    profile = {"name": "", "age": 35, "current_level": "beginner",
               "objective": "interview"}
    missing = intake.missing_fields(profile)
    assert missing == [intake.FIELD_NAME]


def test_missing_fields_treats_whitespace_only_as_missing():
    profile = {"name": "Sebastian", "age": 35,
               "current_level": "   ", "objective": "interview"}
    assert intake.missing_fields(profile) == [intake.FIELD_LEVEL]


def test_missing_fields_order_is_stable():
    """Stable iteration order so the LLM-side prompt sees consistent
    listing turn-to-turn."""
    profile = {"name": "x"}
    assert intake.missing_fields(profile) == [
        intake.FIELD_AGE, intake.FIELD_LEVEL, intake.FIELD_OBJECTIVE,
    ]


# ---------- is_complete ----------

def test_is_complete_false_when_nothing_captured():
    assert intake.is_complete(None) is False
    assert intake.is_complete({}) is False


def test_is_complete_false_when_one_field_missing():
    profile = {
        intake.FIELD_NAME: "Sebastian",
        intake.FIELD_AGE: 35,
        intake.FIELD_LEVEL: "beginner",
    }
    assert intake.is_complete(profile) is False


def test_is_complete_true_when_all_captured():
    profile = {
        intake.FIELD_NAME: "Sebastian",
        intake.FIELD_AGE: 35,
        intake.FIELD_LEVEL: "beginner",
        intake.FIELD_OBJECTIVE: "job interview",
    }
    assert intake.is_complete(profile) is True
