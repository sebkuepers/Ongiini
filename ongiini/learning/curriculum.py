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
        "You are designing a personalised language-learning curriculum "
        "for one specific learner. The skill reference below names "
        "their target + source language and gives the JSON shape, "
        "module composition rules, and personalisation guidance. "
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
    lines.append("")
    lines.append("FOCUS FOR THIS CURRICULUM (authoritative — the learner picked")
    lines.append("this specifically for THIS plan; design around it, NOT around")
    lines.append("an older intake objective if they differ):")
    lines.append(f"  goal title: {tag_learner_input(ctx.goal_title)}")
    lines.append(f"  goal context (optional notes): {tag_learner_input(ctx.goal_context)}")
    if p.get('objective') and ctx.goal_title:
        lines.append(
            f"  (older intake objective — use ONLY if the goal title above is "
            f"empty or generic): {tag_learner_input(p.get('objective'))}"
        )
    elif p.get('objective'):
        lines.append(f"  intake objective (used as fallback): {tag_learner_input(p.get('objective'))}")
    if ctx.mem0_facts:
        lines.append("  prior facts from long-term memory:")
        for fact in ctx.mem0_facts[:6]:
            lines.append(f"    - {tag_learner_input(fact)}")
    if ctx.recent_excerpts:
        lines.append("  recent conversation excerpts:")
        for excerpt in ctx.recent_excerpts[:3]:
            lines.append(f"    > {tag_learner_input(excerpt)}")
    # Track D — surface the learner's recent error categories so the
    # designer can pad a remedial module if any one category dominates
    # (e.g. 12 gender_error attempts in the recent window → add a
    # module on noun gender). Empty for fresh learners — no signal.
    if ctx.error_patterns:
        parts = [
            f"{e.get('tag')}×{e.get('count')}"
            for e in ctx.error_patterns
            if (
                isinstance(e, dict)
                and isinstance(e.get("tag"), str)
                and e.get("tag")
                # count==0 wouldn't be useful but is technically valid;
                # use `is not None` so a zero-count never silently
                # disappears from the surfaced block.
                and e.get("count") is not None
            )
        ]
        if parts:
            lines.append("  recent error categories (top 5):")
            lines.append(f"    {', '.join(parts)}")
            lines.append(
                "    If one category dominates (count ≥ 8), include a "
                "remedial module that targets it."
            )
    return "\n".join(lines)


def _build_design_user_prompt(ctx: LearnerContext) -> str:
    return (
        _render_context_for_prompt(ctx)
        + "\n\nTASK: Author the initial curriculum outline JSON. "
        "Pick a sensible module count for this learner's goal (tighter "
        "for time-pressured objectives, broader for open-ended ones). "
        "The first module should have status 'in_progress'; the rest "
        "'not_started'. "
        "Each module SHOULD include a `topics` list — mix `kind: \"lesson\"` "
        "topics with `kind: \"practice\"` topics, starting each module "
        "with a lesson topic (teach first, then drill)."
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


# ──────────────────────────────────────────────────────────────────
# Design-review loop — design → critique → (revise if not ready) ×N
# ──────────────────────────────────────────────────────────────────

_DEFAULT_MAX_ITERATIONS = 3


async def design_outline_with_review(
    ctx: LearnerContext,
    *,
    model: Model,
    skill_content: str,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> dict[str, Any]:
    """Author a curriculum outline that's been QA'd by an LLM critic.

    Loop:
      1. Designer LLM writes the outline.
      2. Critic LLM scores it against the checklist in SKILL.md.
      3. If critic.ready → return outline.
      4. Else: pass critic.issues as change_reason to revise_outline.
      5. Cap at ``max_iterations`` (default 3) — after that, return
         the last outline with a WARNING log. We do NOT block the
         learner on a stubborn critic.

    Soft-fail on revise: if a revision fails JSON-validation, we
    keep the most recent valid outline and stop iterating rather than
    rolling back to nothing. The critic's degraded-mode (`ready=True`
    on model failure) means a flaky critic falls through to "ship as
    designed."

    Total LLM calls: 1 (designer) + 1–N×2 (critic + maybe revise
    per iteration). For a typical median-1-iteration goal that's 2
    calls; worst case at N=3 it's 7."""
    from . import curriculum_critic as critic_mod   # local — break cycle

    # Clamp non-positive iteration counts. The function name promises
    # "with_review"; a caller asking for 0 (or negative) iterations
    # would silently get an un-reviewed outline. Treat that as a
    # caller bug, log it at debug, and ship the designed outline.
    if max_iterations <= 0:
        log.debug(
            "curriculum: design_outline_with_review called with "
            "max_iterations=%d; skipping critic loop",
            max_iterations,
        )
        return await design_outline(
            ctx, model=model, skill_content=skill_content,
        )

    # Initial design call. The model occasionally returns malformed
    # JSON on the very first turn (extra prose around the object, etc.)
    # and the learner sees "I had trouble putting your plan together".
    # One transparent retry costs one LLM call in the rare flaky case
    # and saves the user from having to hit Send again. If both
    # attempts fail we let ModelOutputError bubble — the coach has its
    # own user-facing error path.
    try:
        outline = await design_outline(
            ctx, model=model, skill_content=skill_content,
        )
    except ModelOutputError as exc:
        log.warning(
            "curriculum: design_outline failed on first attempt (%s); "
            "retrying once before surfacing the error",
            exc,
        )
        outline = await design_outline(
            ctx, model=model, skill_content=skill_content,
        )

    for iteration in range(1, max_iterations + 1):
        critique = await critic_mod.critique_outline(
            ctx, outline, model=model, skill_content=skill_content,
        )
        log.info(
            "curriculum: critic iter=%d score=%d ready=%s issues=%d",
            iteration, critique.score, critique.ready, len(critique.issues),
        )
        if critique.ready:
            return outline
        if iteration == max_iterations:
            # Out of iterations — ship the last version. Warning level
            # so ops can spot stubborn critics or weak designer prompts.
            log.warning(
                "curriculum: max iterations (%d) hit without critic "
                "approval; shipping last outline. final_score=%d "
                "open_issues=%d",
                max_iterations, critique.score, len(critique.issues),
            )
            return outline
        # Build the change reason for the next revise. If the critic
        # said "not ready" but didn't actually list issues (degenerate
        # but possible — e.g. ready=false with score=3 and no specific
        # complaints), synthesize a meaningful prompt from the score
        # so the revise model still has something to act on. Without
        # this the revise turn is shaped by the boilerplate header
        # alone, which is barely better than a re-roll.
        if critique.issues:
            change_reason = "Critic feedback to address:\n" + "\n".join(
                f"- {item}" for item in critique.issues
            )
        else:
            change_reason = (
                f"The critic scored this {critique.score}/10 but did "
                "not list specific issues. Strengthen the plan overall: "
                "tighten goal alignment, ensure each module has a lesson "
                "topic before practice topics, and improve recycling "
                "across modules."
            )
        # Revise based on the critic's specific issues. revise_outline
        # has its own _validate_outline call; on failure we surface
        # ModelOutputError to the caller (coach has its own retry
        # path) so we don't silently ship a broken revision.
        try:
            outline = await revise_outline(
                ctx, model=model, skill_content=skill_content,
                change_reason=change_reason,
            )
        except ModelOutputError as exc:
            log.warning(
                "curriculum: revise_outline failed on iter=%d, "
                "shipping prior version: %s", iteration, exc,
            )
            return outline

    return outline
