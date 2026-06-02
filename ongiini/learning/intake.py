"""Intake state machine for the learn.ongiini.ai onboarding flow.

Sebastian asked for explicit structured intake even when we already
have mem0 context — name, age, current Afrikaans level, and the
specific objective the learner is trying to reach. The reason: a
real conversation with a learner means knowing their actual goal, in
their own words, captured cleanly enough to drive card generation.

This module is pure-Python state-machine logic. No DB access. The
learner row's ``intake_step`` column is the persistent cursor; this
module answers two questions per turn:

  1. Given the current step, what should the AI ask next? — `prompt_for`
  2. Given an answer at the current step, is it valid? — `validate`

The DB layer applies the validated value to ``learner_profiles`` and
moves the cursor to the next step. The grading + card generator
modules read the completed profile from the DB; they don't talk to
this module.

State sequence:

    start  ─►  name  ─►  age  ─►  level  ─►  objective  ─►  done

``start`` is the marker before the first prompt — it lets the API
distinguish "we haven't asked anything yet" from "we just asked
name". Useful when a magic-link arrival pre-fills name/age — the
learner may land directly at ``level`` or even ``objective``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .db import (
    INTAKE_START, INTAKE_NAME, INTAKE_AGE, INTAKE_LEVEL,
    INTAKE_OBJECTIVE, INTAKE_DONE,
)


# Linear progression. ``next_step()`` advances by index lookup.
STEP_SEQUENCE = (
    INTAKE_START, INTAKE_NAME, INTAKE_AGE, INTAKE_LEVEL,
    INTAKE_OBJECTIVE, INTAKE_DONE,
)


# Valid level values. Anything else is rejected at validate time.
VALID_LEVELS = ("beginner", "elementary", "intermediate", "advanced")


# Bounds for free-text inputs. Length caps prevent obvious abuse and
# keep the prompts that include these values (card generation later)
# from blowing up the model's context.
MIN_NAME_LEN = 1
MAX_NAME_LEN = 40
MIN_AGE = 12         # 16+ is the ToS requirement, but the intake
                     # asks before the user is identified, so we leave
                     # a small buffer for younger users to be redirected
                     # by an upstream check rather than blocked here.
MAX_AGE = 120
MIN_OBJECTIVE_LEN = 3
MAX_OBJECTIVE_LEN = 200


@dataclass(frozen=True)
class ValidationResult:
    """The validated value plus, on failure, a human-readable reason."""
    ok: bool
    value: object | None = None
    reason: str | None = None

    @classmethod
    def good(cls, value: object) -> "ValidationResult":
        return cls(ok=True, value=value)

    @classmethod
    def bad(cls, reason: str) -> "ValidationResult":
        return cls(ok=False, reason=reason)


def next_step(current: str) -> str:
    """Return the next step name after ``current``.

    Clamps to ``done`` so callers can't walk past the end of the
    sequence. Unknown step strings (corrupt stored value) default to
    ``name`` — the start of the first real question — so a learner is
    never stuck.
    """
    if current not in STEP_SEQUENCE:
        return INTAKE_NAME
    idx = STEP_SEQUENCE.index(current)
    return STEP_SEQUENCE[min(idx + 1, len(STEP_SEQUENCE) - 1)]


def is_done(step: str) -> bool:
    return step == INTAKE_DONE


# ──────────────────────────────────────────────────────────────────
# What to ask next
# ──────────────────────────────────────────────────────────────────

# Each step gets one prompt the API surfaces to the frontend.
# Conversational, friendly — matches the rest of Ongiini's voice.
_PROMPTS = {
    INTAKE_START: "Welcome! Ready to put together a quick learning plan? Tap start.",
    INTAKE_NAME: "First — what should I call you?",
    INTAKE_AGE: "And how old are you? (Just to pitch examples at the right level.)",
    INTAKE_LEVEL: (
        "Where would you say your Afrikaans is today? "
        "Pick one: beginner, elementary, intermediate, advanced."
    ),
    INTAKE_OBJECTIVE: (
        "Last one — what do you actually want to be able to do in Afrikaans? "
        "A job interview, talking to in-laws, helping the kids with homework, "
        "something else? In one sentence."
    ),
    INTAKE_DONE: "All set — let's start.",
}


def prompt_for(step: str) -> str:
    """The conversational prompt the API surfaces for this step.

    Unknown step strings get the ``name`` prompt — same recovery rule
    as ``next_step``.
    """
    return _PROMPTS.get(step, _PROMPTS[INTAKE_NAME])


# ──────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────

# Reject only obvious junk: control chars, common shell / web punctuation,
# and digits (digits in a name are almost always a typo). Everything else
# is fair game — Namibian names use diacritics, apostrophes, hyphens,
# dots (initials like "M.K."), and characters from Khoekhoegowab and
# other languages that an ASCII-only regex would refuse. The length
# bounds below catch the abuse cases regex can't.
_NAME_REJECT_RE = re.compile(r"[\d<>{}\[\]@#$%^&*()=+|\\;:\"?!]")


def _validate_name(value: str) -> ValidationResult:
    v = (value or "").strip()
    if len(v) < MIN_NAME_LEN:
        return ValidationResult.bad("Please give me at least one character.")
    if len(v) > MAX_NAME_LEN:
        return ValidationResult.bad(f"That's pretty long — keep it under {MAX_NAME_LEN} characters.")
    # Reject control characters (covers \n, \t, null bytes, etc.).
    if any(ord(ch) < 0x20 for ch in v):
        return ValidationResult.bad("That looks like it has control characters — try again?")
    if _NAME_REJECT_RE.search(v):
        return ValidationResult.bad("Names can't contain digits or web symbols — try again?")
    return ValidationResult.good(v)


def _validate_age(value: object) -> ValidationResult:
    # Accept int or numeric string. Reject "I'm twenty" — keep MVP simple.
    if isinstance(value, bool):
        # bool is a subclass of int; explicitly reject so True/False can't
        # sneak through.
        return ValidationResult.bad("Send a number for your age.")
    if isinstance(value, int):
        age = value
    elif isinstance(value, str):
        v = value.strip()
        if not v.isdigit():
            return ValidationResult.bad("Send a number for your age (e.g. 24).")
        age = int(v)
    else:
        return ValidationResult.bad("Send a number for your age (e.g. 24).")

    if not (MIN_AGE <= age <= MAX_AGE):
        return ValidationResult.bad(
            f"That doesn't look right — give me an age between {MIN_AGE} and {MAX_AGE}."
        )
    return ValidationResult.good(age)


def _validate_level(value: str) -> ValidationResult:
    v = (value or "").strip().lower()
    if v in VALID_LEVELS:
        return ValidationResult.good(v)
    # Tolerate the frontend's button label coming back capitalised.
    for label in VALID_LEVELS:
        if label.startswith(v) and len(v) >= 3:
            return ValidationResult.good(label)
    return ValidationResult.bad(
        "Pick one of: " + ", ".join(VALID_LEVELS) + "."
    )


def _validate_objective(value: str) -> ValidationResult:
    v = (value or "").strip()
    if len(v) < MIN_OBJECTIVE_LEN:
        return ValidationResult.bad("Tell me a bit more — even one sentence works.")
    if len(v) > MAX_OBJECTIVE_LEN:
        return ValidationResult.bad(
            f"Got it — could you trim that to under {MAX_OBJECTIVE_LEN} characters?"
        )
    return ValidationResult.good(v)


_VALIDATORS = {
    INTAKE_NAME: _validate_name,
    INTAKE_AGE: _validate_age,
    INTAKE_LEVEL: _validate_level,
    INTAKE_OBJECTIVE: _validate_objective,
}


def validate(step: str, value: object) -> ValidationResult:
    """Validate the user's answer for the given step.

    ``start`` and ``done`` aren't user-answerable — calling validate
    on them is a programming error. We return a 'bad' result with a
    descriptive reason so the API surfaces something sane rather than
    crashing.
    """
    validator = _VALIDATORS.get(step)
    if validator is None:
        return ValidationResult.bad(
            f"Step '{step}' doesn't take a user answer."
        )
    return validator(value)
