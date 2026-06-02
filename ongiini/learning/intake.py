"""Intake capture spec for the learn.ongiini.ai onboarding.

**This module is deliberately minimal.** It does NOT script the
conversation. The LLM conducts the intake — it decides how to ask,
whether to ask a follow-up, how to react to whatever the learner
volunteers, whether to weave context from prior mem0 facts. This is
the central value proposition of the learning surface: extremely
smart, personalised conversation, not a fixed-question form.

What lives here:

  * The **target schema** — which fields the intake is trying to
    capture (name, age, level, objective).
  * **Field-level validators** — once the LLM extracts a value from a
    user's free-text reply, this module says "is the value's SHAPE
    OK" (e.g. age is an int between bounds, level is one of the
    canonical strings). Validators answer "is this storable", not
    "is this what the user really meant" — that's the LLM's job.
  * A ``missing_fields(profile)`` query — given the current profile
    state, which fields are still null. The API surfaces this list
    to the LLM each turn so it knows what still needs capturing.
  * ``is_complete(profile)`` — terminal check.

What does NOT live here:

  * Prompt text for the user (the LLM writes those).
  * A linear step sequence (the LLM may capture two fields in one
    answer, may double back to clarify level after objective, may
    skip a field a magic-link arrival already pre-filled).
  * Conversational follow-ups, encouragement, transitions.

The deterministic line: shape-validation + persistence; everything
else flows through the LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


# ──────────────────────────────────────────────────────────────────
# Target fields — the schema the intake is trying to fill.
# ──────────────────────────────────────────────────────────────────

FIELD_NAME = "name"
FIELD_AGE = "age"
FIELD_LEVEL = "current_level"
FIELD_OBJECTIVE = "objective"

# Listed in the order the LLM is encouraged (but not forced) to capture
# them in. Used as the default "where to start" hint when nothing is
# captured yet; the LLM is free to reorder based on what the learner
# volunteers.
INTAKE_FIELDS = (FIELD_NAME, FIELD_AGE, FIELD_LEVEL, FIELD_OBJECTIVE)


# Canonical level values. Lowercase. The LLM is told these are the
# allowed values; the validator below will accept reasonable variants
# but always normalises to canonical.
VALID_LEVELS = ("beginner", "elementary", "intermediate", "advanced")


# Bounds — defensive, not pedagogical. The validators reject obvious
# garbage (negative age, name with control chars). They do NOT enforce
# "right answer" — that's a content question the LLM owns.
MIN_NAME_LEN = 1
MAX_NAME_LEN = 40
MIN_AGE = 12
MAX_AGE = 120
MIN_OBJECTIVE_LEN = 2     # "CV" / "TV" / "AI" — short but valid
MAX_OBJECTIVE_LEN = 200


# ──────────────────────────────────────────────────────────────────
# Validation result type
# ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ValidationResult:
    """The validated, NORMALISED value (or a reason for rejection).

    On ok=True, ``value`` is the form to persist (e.g. "Beginner" →
    "beginner"). On ok=False, ``reason`` is plain-text the API can
    surface back to the LLM, which then explains the issue to the
    learner in its own voice.
    """
    ok: bool
    value: object | None = None
    reason: str | None = None

    @classmethod
    def good(cls, value: object) -> "ValidationResult":
        return cls(ok=True, value=value)

    @classmethod
    def bad(cls, reason: str) -> "ValidationResult":
        return cls(ok=False, reason=reason)


# ──────────────────────────────────────────────────────────────────
# Field validators (shape only — never semantic)
# ──────────────────────────────────────────────────────────────────

# Reject only obvious junk: digits, control chars, common web
# punctuation. Namibian names include diacritics, apostrophes,
# hyphens, dots ("M.K. Shilongo"), and characters from Khoekhoegowab
# and other languages, so an ASCII-only allowlist would reject real
# users. Length bounds catch abuse cases the regex can't.
_NAME_REJECT_RE = re.compile(r"[\d<>{}\[\]@#$%^&*()=+|\\;:\"?!]")


def _validate_name(value: object) -> ValidationResult:
    if not isinstance(value, str):
        return ValidationResult.bad("name must be a string")
    v = value.strip()
    if len(v) < MIN_NAME_LEN:
        return ValidationResult.bad("name is empty")
    if len(v) > MAX_NAME_LEN:
        return ValidationResult.bad(f"name longer than {MAX_NAME_LEN} characters")
    if any(ord(ch) < 0x20 for ch in v):
        return ValidationResult.bad("name contains control characters")
    if _NAME_REJECT_RE.search(v):
        return ValidationResult.bad("name contains digits or web punctuation")
    return ValidationResult.good(v)


def _validate_age(value: object) -> ValidationResult:
    # bool is a subclass of int — reject explicitly so True can't sneak through.
    if isinstance(value, bool):
        return ValidationResult.bad("age must be a number")
    if isinstance(value, int):
        age = value
    elif isinstance(value, str):
        v = value.strip()
        if not v.isdigit():
            return ValidationResult.bad("age must be a positive integer")
        age = int(v)
    else:
        return ValidationResult.bad("age must be a number")
    if not (MIN_AGE <= age <= MAX_AGE):
        return ValidationResult.bad(
            f"age must be between {MIN_AGE} and {MAX_AGE}"
        )
    return ValidationResult.good(age)


def _validate_level(value: object) -> ValidationResult:
    if not isinstance(value, str):
        return ValidationResult.bad("level must be a string")
    v = value.strip().lower()
    if v in VALID_LEVELS:
        return ValidationResult.good(v)
    # Accept prefix matches of >=3 chars so the LLM can pass back "beg"
    # or the user's free-text "I think I'm intermediate-ish" stripped
    # down to a recognisable token.
    for label in VALID_LEVELS:
        if label.startswith(v) and len(v) >= 3:
            return ValidationResult.good(label)
    return ValidationResult.bad(
        "level must be one of: " + ", ".join(VALID_LEVELS)
    )


def _validate_objective(value: object) -> ValidationResult:
    if not isinstance(value, str):
        return ValidationResult.bad("objective must be a string")
    v = value.strip()
    if len(v) < MIN_OBJECTIVE_LEN:
        return ValidationResult.bad("objective too short")
    if len(v) > MAX_OBJECTIVE_LEN:
        return ValidationResult.bad(f"objective longer than {MAX_OBJECTIVE_LEN} characters")
    return ValidationResult.good(v)


_VALIDATORS = {
    FIELD_NAME: _validate_name,
    FIELD_AGE: _validate_age,
    FIELD_LEVEL: _validate_level,
    FIELD_OBJECTIVE: _validate_objective,
}


def validate_field(field: str, value: object) -> ValidationResult:
    """Validate a candidate value for a named field.

    ``field`` must be one of ``INTAKE_FIELDS``. Returns a
    ValidationResult — ``ok=True`` with the canonical/normalised value
    to persist, or ``ok=False`` with a short machine-readable reason
    the LLM can paraphrase to the learner.

    This is the only deterministic step in the intake loop. Everything
    else — the question, the follow-up, the encouragement — is the
    LLM's.
    """
    validator = _VALIDATORS.get(field)
    if validator is None:
        return ValidationResult.bad(f"unknown field: {field}")
    return validator(value)


# ──────────────────────────────────────────────────────────────────
# Profile-state queries
# ──────────────────────────────────────────────────────────────────

def missing_fields(profile: Mapping[str, object] | None) -> list[str]:
    """Return the intake fields still null/missing on this profile.

    The LLM uses this every turn to decide what to talk about next —
    not as a forced sequence, but as an awareness of what's still
    needed. Order matches ``INTAKE_FIELDS`` so a caller iterating
    deterministically still gets a stable order.

    A profile of ``None`` (learner with no row yet) returns all fields.
    """
    if not profile:
        return list(INTAKE_FIELDS)
    missing = []
    for field in INTAKE_FIELDS:
        v = profile.get(field)
        # Treat empty string the same as null — captured-but-empty is
        # not captured.
        if v is None or (isinstance(v, str) and not v.strip()):
            missing.append(field)
    return missing


def is_complete(profile: Mapping[str, object] | None) -> bool:
    """True iff every intake field is captured.

    The API uses this to decide whether to surface the intake mode
    or the learning-card mode to the frontend. The LLM doesn't have
    to be consulted to answer "is the form done?".
    """
    return len(missing_fields(profile)) == 0
