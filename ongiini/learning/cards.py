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
    # module_id is soft-required: if missing the card still saves (the
    # store accepts None for back-compat), but the per-module progress
    # bar can't count it. Log via downstream so we can spot LLM
    # regressions without breaking the turn. Type-check when present.
    if "module_id" in payload and payload["module_id"] is not None:
        if not isinstance(payload["module_id"], str):
            raise ModelOutputError(
                f"module_id must be a string; got "
                f"{type(payload['module_id']).__name__}"
            )


def _build_system_prompt(skill_content: str) -> str:
    # MVP: re-embed the full SKILL.md per call. See curriculum.py for
    # the same comment — simplicity over per-call token cost.
    return (
        "You are authoring ONE learning card for a specific learner. "
        "The skill reference below names their target + source "
        "language pair and gives the card-type guidance + JSON shape. "
        f"{INJECTION_GUARD_LINE} "
        "Emit ONLY the JSON object — no prose, no Markdown fences.\n\n"
        f"{skill_content}"
    )


def _render_module_digest(ctx: LearnerContext) -> str:
    """Per-module rollup the recent_cards window can't carry once the
    learner is deep in a module. Without this the model loses track of
    'lessons already given for module mod-1' after ~12 turns and
    re-emits the original lesson — the exact bug Sebastian hit at the
    13-card mark."""
    if not ctx.module_digest:
        return "(no per-module data yet)"
    lines: list[str] = []
    for mod_id, d in ctx.module_digest.items():
        lessons = d.get("lessons_given", 0)
        ex_emit = d.get("exercises_emitted", 0)
        ex_seen = d.get("exercises_attempted", 0)
        ex_correct = d.get("exercises_correct", 0)
        lines.append(
            f"  - {mod_id}: lessons_given={lessons}, "
            f"exercises_emitted={ex_emit}, "
            f"exercises_attempted={ex_seen}, "
            f"exercises_correct={ex_correct}"
        )
    return "\n".join(lines)


def _render_recent_cards(ctx: LearnerContext) -> str:
    """Compact bullet list of the most recent cards the learner has
    seen on this goal's thread. Without this the model has no way to
    tell that it just emitted a lesson — so on every "Got it →" tap
    it re-decides "module just started → lesson" and emits the same
    thing again. Showing it the recent cards lets it pick the right
    next move (drill the lesson it just gave, or start a new sub-topic).
    """
    if not ctx.recent_cards:
        return "(none yet — this is the first card on this goal)"
    lines: list[str] = []
    for c in ctx.recent_cards:
        kind = c.get("kind") or "?"
        ct = c.get("card_type") or kind
        ans = "answered" if c.get("answered") else "active"
        title = c.get("title")
        prompt = (c.get("prompt_text") or "")
        # Compact the prompt — first 80 chars is enough to recognise
        # the topic; we don't need to echo the whole lesson body.
        snippet = (title or prompt).strip().splitlines()[0][:80] if (title or prompt) else "(empty)"
        lines.append(f"  - [{kind}/{ct}] [{ans}] {snippet}")
    return "\n".join(lines)


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
        f"  focus: {tag_learner_input(ctx.goal_title or ctx.goal_context or p.get('objective'))}\n"
        f"\nCURRICULUM OUTLINE:\n{outline_json}\n"
        f"\nPROGRESS: {progress_summary}\n"
        f"\nMODULE DIGEST (per-module rollup — load-bearing for "
        "lesson-vs-drill decisions; refer here BEFORE recent cards):\n"
        f"{_render_module_digest(ctx)}\n"
        f"\nRECENT CARDS ON THIS GOAL (oldest first, last 12):\n"
        f"{_render_recent_cards(ctx)}\n"
        "\nTASK: Author the next card. "
        "INCLUDE the JSON key `module_id` matching one of the module "
        "ids from the outline above — required so per-module progress "
        "can be tracked.\n"
        "Rules:\n"
        " - If MODULE DIGEST shows lessons_given >= 1 for the "
        "in-progress module, EMIT AN EXERCISE, not another lesson — "
        "the lesson has already been taught. The recent-cards window "
        "may have rolled past it; the digest is authoritative.\n"
        " - If MODULE DIGEST shows lessons_given == 0 for a freshly-"
        "started module, emit a lesson first.\n"
        " - If lots of cards are stuck in Leitner box 1, prefer "
        "consolidating cards (easier, in-module) over introducing new "
        "themes.\n"
        " - Pick the card_type that fits the in-progress module and "
        "the learner's level.\n"
        "Output JSON only."
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
