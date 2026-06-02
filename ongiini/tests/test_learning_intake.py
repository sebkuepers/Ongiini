"""Intake state machine — pure-Python tests, no DB."""
from __future__ import annotations

import pytest

from ongiini.learning import intake
from ongiini.learning.db import (
    INTAKE_START, INTAKE_NAME, INTAKE_AGE, INTAKE_LEVEL,
    INTAKE_OBJECTIVE, INTAKE_DONE,
)


# ---------- next_step ----------

def test_next_step_walks_in_order():
    assert intake.next_step(INTAKE_START) == INTAKE_NAME
    assert intake.next_step(INTAKE_NAME) == INTAKE_AGE
    assert intake.next_step(INTAKE_AGE) == INTAKE_LEVEL
    assert intake.next_step(INTAKE_LEVEL) == INTAKE_OBJECTIVE
    assert intake.next_step(INTAKE_OBJECTIVE) == INTAKE_DONE


def test_next_step_clamps_at_done():
    # Done is the terminal state; advancing further stays at done.
    assert intake.next_step(INTAKE_DONE) == INTAKE_DONE


def test_next_step_unknown_recovers_to_name():
    assert intake.next_step("garbage") == INTAKE_NAME


def test_is_done_only_true_for_done():
    assert intake.is_done(INTAKE_DONE) is True
    for s in (INTAKE_START, INTAKE_NAME, INTAKE_AGE,
              INTAKE_LEVEL, INTAKE_OBJECTIVE):
        assert intake.is_done(s) is False


# ---------- prompt_for ----------

def test_prompt_for_every_step_returns_text():
    for s in intake.STEP_SEQUENCE:
        out = intake.prompt_for(s)
        assert isinstance(out, str) and len(out) > 0


def test_prompt_for_unknown_step_recovers_to_name_prompt():
    assert intake.prompt_for("garbage") == intake.prompt_for(INTAKE_NAME)


# ---------- validate: name ----------

def test_name_accepts_simple():
    r = intake.validate(INTAKE_NAME, "Sebastian")
    assert r.ok and r.value == "Sebastian"


def test_name_strips_whitespace():
    r = intake.validate(INTAKE_NAME, "  Sebastian  ")
    assert r.ok and r.value == "Sebastian"


def test_name_accepts_diacritics_and_apostrophe_and_hyphen():
    """Reviewer flagged that an ASCII-only regex would reject
    legitimate Namibian + diaspora names. The relaxed validator now
    accepts these — common shapes worth locking down."""
    for name in [
        "Anna-Marie",
        "Naòmi",      # diacritic
        "O'Brien",
        "Hāneb",      # OW orthography
        "M.K. Shilongo",   # initials + period
        "Köpfer",     # diacritic + capital
    ]:
        r = intake.validate(INTAKE_NAME, name)
        assert r.ok, f"should accept {name!r}: {r.reason}"


def test_name_rejects_empty():
    r = intake.validate(INTAKE_NAME, "")
    assert not r.ok
    r2 = intake.validate(INTAKE_NAME, "   ")
    assert not r2.ok


def test_name_rejects_too_long():
    r = intake.validate(INTAKE_NAME, "x" * 41)
    assert not r.ok


def test_name_rejects_digits():
    r = intake.validate(INTAKE_NAME, "Player1")
    assert not r.ok


def test_name_rejects_web_junk():
    for bad in ["alice@example.com", "<script>", "name|injection"]:
        r = intake.validate(INTAKE_NAME, bad)
        assert not r.ok, f"should reject {bad!r}"


def test_name_rejects_control_chars():
    r = intake.validate(INTAKE_NAME, "alice\nbob")
    assert not r.ok


# ---------- validate: age ----------

def test_age_accepts_int():
    r = intake.validate(INTAKE_AGE, 25)
    assert r.ok and r.value == 25


def test_age_accepts_numeric_string():
    r = intake.validate(INTAKE_AGE, "33")
    assert r.ok and r.value == 33


def test_age_rejects_bool_subtly():
    """`bool` is a subclass of int — without explicit rejection True
    would pass as age=1. Lock this down."""
    r = intake.validate(INTAKE_AGE, True)
    assert not r.ok
    r2 = intake.validate(INTAKE_AGE, False)
    assert not r2.ok


def test_age_rejects_word():
    r = intake.validate(INTAKE_AGE, "twenty")
    assert not r.ok


def test_age_rejects_out_of_range():
    assert not intake.validate(INTAKE_AGE, 5).ok
    assert not intake.validate(INTAKE_AGE, 200).ok


def test_age_rejects_other_types():
    assert not intake.validate(INTAKE_AGE, None).ok
    assert not intake.validate(INTAKE_AGE, [25]).ok


# ---------- validate: level ----------

@pytest.mark.parametrize("level", intake.VALID_LEVELS)
def test_level_accepts_canonical(level):
    r = intake.validate(INTAKE_LEVEL, level)
    assert r.ok and r.value == level


def test_level_accepts_capitalised():
    r = intake.validate(INTAKE_LEVEL, "Beginner")
    assert r.ok and r.value == "beginner"


def test_level_accepts_prefix_three_chars():
    # Frontend button might submit a shortened label.
    r = intake.validate(INTAKE_LEVEL, "int")
    assert r.ok and r.value == "intermediate"


def test_level_rejects_too_short_prefix():
    # Just 'b' could be beginner — but it could also be a typo;
    # require at least 3 chars.
    r = intake.validate(INTAKE_LEVEL, "b")
    assert not r.ok


def test_level_rejects_unknown():
    r = intake.validate(INTAKE_LEVEL, "fluent")
    assert not r.ok


# ---------- validate: objective ----------

def test_objective_accepts_sentence():
    r = intake.validate(INTAKE_OBJECTIVE, "I want to pass a job interview in Afrikaans.")
    assert r.ok


def test_objective_rejects_empty():
    assert not intake.validate(INTAKE_OBJECTIVE, "").ok
    assert not intake.validate(INTAKE_OBJECTIVE, "  ").ok


def test_objective_rejects_too_short():
    assert not intake.validate(INTAKE_OBJECTIVE, "a").ok


def test_objective_rejects_too_long():
    r = intake.validate(INTAKE_OBJECTIVE, "x" * 201)
    assert not r.ok


# ---------- validate: terminal / pre-start steps ----------

def test_validate_start_returns_bad():
    r = intake.validate(INTAKE_START, "anything")
    assert not r.ok


def test_validate_done_returns_bad():
    r = intake.validate(INTAKE_DONE, "anything")
    assert not r.ok


def test_validate_unknown_step_returns_bad():
    r = intake.validate("garbage", "answer")
    assert not r.ok
