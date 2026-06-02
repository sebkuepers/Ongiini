"""LLM-driven answer grading.

The learner types a free-text answer to a card. The model evaluates
against the card's reference_answer (or rubric for production cards),
the card's type, and the learner's level. Returns a rating
(``correct`` / ``partial`` / ``wrong``) plus a short feedback string.

The rating is the load-bearing field — store.record_attempt promotes /
demotes the Leitner box from it. Feedback is shown to the learner.

Same shape as curriculum.py and cards.py: single-shot LLM call,
structured JSON output, validation of the load-bearing fields.
"""
from __future__ import annotations

import logging
from typing import Any

from owela import Model

from .context import LearnerContext
from .db import RATINGS
from .llm import INJECTION_GUARD_LINE, ModelOutputError, ask_for_json, tag_learner_input

log = logging.getLogger("ongiini.learning.grading")


_REQUIRED_GRADING_KEYS = ("rating", "feedback")


def _validate_grading(payload: dict[str, Any]) -> None:
    if "error" in payload:
        raise ModelOutputError(f"model declined: {payload['error']}")
    for key in _REQUIRED_GRADING_KEYS:
        if key not in payload:
            raise ModelOutputError(f"grading missing required key: {key!r}")
    if payload["rating"] not in RATINGS:
        raise ModelOutputError(
            f"rating must be one of {RATINGS}; got {payload['rating']!r}"
        )
    if not isinstance(payload["feedback"], str) or not payload["feedback"].strip():
        raise ModelOutputError("grading feedback must be a non-empty string")


def _build_system_prompt(skill_content: str) -> str:
    return (
        "You are grading a learner's free-text answer to one card. "
        "The skill reference below names their target + source language "
        "pair and lays out the rubric — be generous but honest. "
        f"{INJECTION_GUARD_LINE} "
        "Emit ONLY the JSON object — no prose, no Markdown fences.\n\n"
        f"{skill_content}"
    )


def _build_user_prompt(
    ctx: LearnerContext,
    *,
    card: dict[str, Any],
    user_answer: str,
    hint_used: bool,
) -> str:
    p = ctx.profile or {}
    return (
        "LEARNER:\n"
        f"  level: {p.get('current_level') or 'beginner'}\n"
        f"  objective: {tag_learner_input(ctx.goal_context or p.get('objective'))}\n"
        "\nCARD:\n"
        f"  card_type: {card.get('card_type')}\n"
        f"  prompt_text: {card.get('prompt_text')}\n"
        f"  reference_answer: {card.get('reference_answer') or '(none)'}\n"
        f"  hint_used: {hint_used}\n"
        "\nLEARNER'S ANSWER:\n"
        f"  {tag_learner_input(user_answer)}\n"
        "\nTASK: Grade the answer. Output JSON only — { rating, feedback }. "
        "Feedback must be 1–3 sentences and directly usable. For 'correct', "
        "confirm and show the canonical form if their spelling drifted. For "
        "'partial', name the specific gap. For 'wrong', give the right "
        "answer in one breath without shaming. Don't lecture; the learner "
        "will see many more cards on this pattern. A blank or 'I don't know' "
        "answer is 'wrong' — give the right answer and a one-line nudge."
    )


async def grade_answer(
    ctx: LearnerContext,
    *,
    card: dict[str, Any],
    user_answer: str,
    hint_used: bool,
    model: Model,
    skill_content: str,
) -> dict[str, Any]:
    """Ask the model to grade the learner's answer. Returns dict with
    ``rating`` (one of RATINGS) and ``feedback`` (str).

    Raises ``ModelOutputError`` on malformed output.

    Note: an empty / whitespace-only answer is sent to the model just
    like any other — the SKILL.md rubric covers it as "wrong" with a
    specific feedback shape. Bypassing the model with a hard-coded
    English nudge would break the "LLM owns grading" contract and would
    surface English text to a learner whose UI we may later localise.
    """
    payload = await ask_for_json(
        system_prompt=_build_system_prompt(skill_content),
        user_prompt=_build_user_prompt(
            ctx, card=card, user_answer=user_answer or "",
            hint_used=hint_used,
        ),
        model=model,
    )
    _validate_grading(payload)
    return payload
