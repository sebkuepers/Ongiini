"""LLM-driven card generation.

When the SRS queue is empty (no card is due for re-review), the API
asks the model to author the NEXT card for this learner. The model
sees the full LearnerContext — profile, curriculum outline, progress
distribution — and picks the card_type (vocab / translation /
production), the prompt, the reference answer, and an optional hint.

Same shape as ``curriculum.py``: single-shot LLM call returning
structured JSON; we validate the load-bearing fields and let the
LLM own everything else.

The model chooses the direction for vocab cards (EN→AF or AF→EN)
per card based on the learner's level and goal — Sebastian's
explicit instruction. We don't constrain it from the backend.
"""
from __future__ import annotations

import logging
from typing import Any

from owela import Model

from .context import LearnerContext
from .db import CARD_TYPES
from .llm import INJECTION_GUARD_LINE, ModelOutputError, ask_for_json, tag_learner_input

log = logging.getLogger("ongiini.learning.cards")


_REQUIRED_CARD_KEYS = ("card_type", "prompt_text")


def _validate_card(payload: dict[str, Any]) -> None:
    if "error" in payload:
        raise ModelOutputError(f"model declined: {payload['error']}")
    for key in _REQUIRED_CARD_KEYS:
        if key not in payload:
            raise ModelOutputError(f"card missing required key: {key!r}")
    ct = payload["card_type"]
    # Type-check before the membership test — None / list / int would
    # all pass `not in tuple` and produce a confusing error downstream.
    if not isinstance(ct, str):
        raise ModelOutputError(f"card_type must be a string; got {type(ct).__name__}")
    if ct not in CARD_TYPES:
        raise ModelOutputError(f"card_type must be one of {CARD_TYPES}; got {ct!r}")
    if not isinstance(payload["prompt_text"], str) or not payload["prompt_text"].strip():
        raise ModelOutputError("card prompt_text must be a non-empty string")


def _build_system_prompt(skill_content: str) -> str:
    # MVP: re-embed the full SKILL.md per call. See curriculum.py for
    # the same comment — simplicity over per-call token cost.
    return (
        "You are authoring ONE learning card for a specific Afrikaans "
        "learner. Use the skill reference below to pick the right card "
        f"type and the right difficulty. {INJECTION_GUARD_LINE} "
        "Emit ONLY the JSON object — no prose, no Markdown fences.\n\n"
        f"{skill_content}"
    )


def _build_user_prompt(ctx: LearnerContext) -> str:
    p = ctx.profile or {}
    import json as _json
    outline_json = "(none yet)"
    if ctx.curriculum_outline:
        outline_json = _json.dumps(ctx.curriculum_outline, indent=2, ensure_ascii=False)
    progress_summary = "(no progress yet)"
    if ctx.progress:
        progress_summary = (
            f"total_seen={ctx.progress.get('total_seen', 0)}, "
            f"total_correct={ctx.progress.get('total_correct', 0)}, "
            f"by_box={ctx.progress.get('by_box', {})}"
        )
    return (
        "LEARNER:\n"
        f"  name: {tag_learner_input(p.get('name'))}\n"
        f"  level: {p.get('current_level') or 'beginner'}\n"
        f"  objective: {tag_learner_input(ctx.goal_context or p.get('objective'))}\n"
        f"\nCURRICULUM OUTLINE:\n{outline_json}\n"
        f"\nPROGRESS: {progress_summary}\n"
        "\nTASK: Author the next card. Pick the card_type that fits the "
        "in-progress module and the learner's current ability. If lots "
        "of cards are stuck in Leitner box 1, prefer consolidating cards "
        "(easier, in-module) over introducing new themes. Output JSON only."
    )


async def generate_card(
    ctx: LearnerContext,
    *,
    model: Model,
    skill_content: str,
) -> dict[str, Any]:
    """Ask the model to author the next card. Returns a dict with at
    least ``card_type`` and ``prompt_text``; ``reference_answer`` and
    ``hint_text`` and ``difficulty`` may be present.

    Raises ``ModelOutputError`` on bad JSON / missing required fields /
    unknown card_type.
    """
    payload = await ask_for_json(
        system_prompt=_build_system_prompt(skill_content),
        user_prompt=_build_user_prompt(ctx),
        model=model,
    )
    _validate_card(payload)
    return payload
