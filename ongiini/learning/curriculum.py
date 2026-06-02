"""LLM-driven curriculum outline design + revision.

The model authors the curriculum outline for one specific learner,
given their profile + goal context. We persist the JSON it returns;
the SKILL.md document teaches it the shape.

Two entry points:

  * ``design_outline(ctx, model, skill_content)`` — called when the
    learner has just completed intake and there's no outline yet.
    Returns the parsed outline dict; caller is responsible for
    ``store.save_curriculum_outline()``.
  * ``revise_outline(ctx, model, skill_content, change_reason)`` —
    called when the learner says something that should change the plan
    ("my interview moved up to tomorrow", "I want to switch focus to
    asking-questions"). Returns the revised outline.

The LLM owns the inner JSON structure; we only validate that the
output is a dict and has the load-bearing keys (``summary``,
``modules``). Anything beyond that is the LLM's design space.
"""
from __future__ import annotations

import logging
from typing import Any

from owela import Model

from .context import LearnerContext
from .llm import INJECTION_GUARD_LINE, ModelOutputError, ask_for_json, tag_learner_input

log = logging.getLogger("ongiini.learning.curriculum")


# ──────────────────────────────────────────────────────────────────
# Validation — just enough to fail fast on a confused model output.
# Inner module shape is the LLM's space, not ours.
# ──────────────────────────────────────────────────────────────────

_REQUIRED_OUTLINE_KEYS = ("summary", "modules")


def _validate_outline(payload: dict[str, Any]) -> None:
    """Raise ModelOutputError if the outline shape is fundamentally
    wrong. We do NOT validate inner module fields — the LLM chooses
    how to structure each module."""
    if "error" in payload:
        raise ModelOutputError(f"model declined: {payload['error']}")
    for key in _REQUIRED_OUTLINE_KEYS:
        if key not in payload:
            raise ModelOutputError(f"outline missing required key: {key!r}")
    if not isinstance(payload["modules"], list):
        raise ModelOutputError("outline.modules must be a list")
    if not payload["modules"]:
        raise ModelOutputError("outline.modules must not be empty")
    # The list elements must be dicts — a list of strings or ints would
    # crash the persistence layer downstream with a harder-to-diagnose
    # error.
    if not all(isinstance(m, dict) for m in payload["modules"]):
        raise ModelOutputError("outline.modules entries must be objects")


# ──────────────────────────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────────────────────────

def _build_system_prompt(skill_content: str) -> str:
    """The system prompt the model sees on curriculum-design turns.

    Loads the learning-afrikaans skill verbatim so the model has the
    JSON shape, the rubric, the anchor vocabulary, and the
    personalisation guidance in front of it. The skill content does
    most of the work — this preface just frames the task.

    Note on cost: re-embedding SKILL.md on every call is intentional
    for the MVP. The total prompt is a few thousand tokens; the
    simplicity of one source of truth for the LLM's behaviour matters
    more than the per-call savings. Phase 2 can split SKILL.md into
    addressable sections if production load demands it."""
    return (
        "You are designing a personalised Afrikaans learning curriculum "
        "for one specific learner. Use the skill reference below to "
        "decide JSON shape, module composition, and how to personalise. "
        f"{INJECTION_GUARD_LINE} "
        "Emit ONLY the JSON object — no prose, no Markdown fences.\n\n"
        f"{skill_content}"
    )


def _render_context_for_prompt(ctx: LearnerContext) -> str:
    """Render the LearnerContext as a readable snippet inside the user
    message. Free-text fields are wrapped in <learner_input> tags so
    the model knows what's data vs instruction (system prompt teaches
    it to never treat tag content as instructions). Numeric / enum
    fields don't need tagging — they're closed-vocabulary."""
    p = ctx.profile or {}
    lines = ["LEARNER CONTEXT:"]
    lines.append(f"  name: {tag_learner_input(p.get('name'))}")
    lines.append(f"  age: {p.get('age') if p.get('age') is not None else '(not given)'}")
    lines.append(f"  current_level: {p.get('current_level') or '(not given)'}")
    lines.append(f"  objective (from intake): {tag_learner_input(p.get('objective'))}")
    lines.append(f"  goal_context (override): {tag_learner_input(ctx.goal_context)}")
    if ctx.mem0_facts:
        lines.append("  prior facts from long-term memory:")
        for fact in ctx.mem0_facts[:6]:
            lines.append(f"    - {tag_learner_input(fact)}")
    if ctx.recent_excerpts:
        lines.append("  recent conversation excerpts:")
        for excerpt in ctx.recent_excerpts[:3]:
            lines.append(f"    > {tag_learner_input(excerpt)}")
    return "\n".join(lines)


def _build_design_user_prompt(ctx: LearnerContext) -> str:
    return (
        _render_context_for_prompt(ctx)
        + "\n\nTASK: Author the initial curriculum outline JSON. "
        "Pick a sensible module count for this learner's goal (tighter "
        "for time-pressured objectives, broader for open-ended ones). "
        "The first module should have status 'in_progress'; the rest "
        "'not_started'."
    )


def _build_revise_user_prompt(ctx: LearnerContext, change_reason: str) -> str:
    outline_json = "(none yet — this is effectively a fresh design)"
    if ctx.curriculum_outline:
        import json as _json
        outline_json = _json.dumps(ctx.curriculum_outline, indent=2, ensure_ascii=False)
    progress_summary = "(no progress yet)"
    if ctx.progress:
        progress_summary = (
            f"total_seen={ctx.progress.get('total_seen', 0)}, "
            f"total_correct={ctx.progress.get('total_correct', 0)}, "
            f"by_box={ctx.progress.get('by_box', {})}"
        )
    return (
        _render_context_for_prompt(ctx)
        + f"\n\nCURRENT OUTLINE:\n{outline_json}"
        + f"\n\nPROGRESS SO FAR: {progress_summary}"
        + f"\n\nREASON TO REVISE: {tag_learner_input(change_reason)}"
        + "\n\nTASK: Emit a REVISED curriculum outline JSON. Preserve "
        "module ids where the underlying topic is the same. Update "
        "module statuses to reflect actual progress. Drop or compress "
        "modules that no longer matter; insert new ones if the reason "
        "demands it."
    )


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

async def design_outline(
    ctx: LearnerContext,
    *,
    model: Model,
    skill_content: str,
) -> dict[str, Any]:
    """Author a fresh curriculum outline for this learner.

    Raises ``ModelOutputError`` if the LLM returns malformed JSON or
    an outline that's missing the required keys. Callers should treat
    that as a transient failure and either retry or surface a 503.
    """
    payload = await ask_for_json(
        system_prompt=_build_system_prompt(skill_content),
        user_prompt=_build_design_user_prompt(ctx),
        model=model,
    )
    _validate_outline(payload)
    return payload


async def revise_outline(
    ctx: LearnerContext,
    *,
    model: Model,
    skill_content: str,
    change_reason: str,
) -> dict[str, Any]:
    """Revise the existing curriculum outline. ``change_reason`` is a
    short phrase describing why the revision is needed — usually the
    learner said something that should reshape the plan ('my interview
    moved up'). Returns the revised outline; caller persists."""
    if not change_reason or not change_reason.strip():
        raise ValueError("change_reason is required")
    payload = await ask_for_json(
        system_prompt=_build_system_prompt(skill_content),
        user_prompt=_build_revise_user_prompt(ctx, change_reason),
        model=model,
    )
    _validate_outline(payload)
    return payload
