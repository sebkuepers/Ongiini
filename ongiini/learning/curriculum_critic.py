"""LLM critic for the curriculum design loop.

A second LLM call that reviews a freshly-designed curriculum outline
against a checklist (goal alignment, level appropriateness,
scaffolding, recycling, module count, topic completeness, cultural
anchoring) and returns a structured verdict. The orchestrator in
``curriculum.design_outline_with_review`` uses the verdict to decide
whether to ship the outline or pass it to ``revise_outline`` with the
critic's issues as the change reason.

Design principles:

  * Soft-fail. A flaky or bad-JSON critic must NEVER block the learner
    from getting their plan. On any model error or shape error we
    degrade to ``ready=True, score=0, issues=["critic_failed"]`` —
    the orchestrator logs it and ships the un-reviewed outline. The
    learner sees a working curriculum either way.
  * The checklist lives in the rendered skill content (CURRICULUM
    REVIEW CHECKLIST section of ``_core.md.tmpl``), not duplicated
    here. Single source of truth — when we tune the checklist we tune
    it in one place.
  * Independent of the designer prompt. The critic sees the outline +
    learner context fresh, not the design prompt or any of the
    designer's reasoning. That's the point of LLM-as-judge — a
    different read on the same artifact.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from owela import Model

from .context import LearnerContext
from .llm import INJECTION_GUARD_LINE, ModelOutputError, ask_for_json, tag_learner_input

log = logging.getLogger("ongiini.learning.curriculum_critic")


# Max length we'll trust from the model for an issue / strength string.
# A list of 12-word actionables is much more useful than a paragraph;
# this caps each entry so a runaway model doesn't bloat the prompt for
# the next revise turn.
_MAX_ITEM_LEN = 240


@dataclass(frozen=True)
class CritiqueResult:
    """Structured verdict from the curriculum critic.

    ``ready`` is the gate the orchestrator reads. ``score`` and the
    text lists are kept for logging + future analytics (we want to
    learn whether iteration 1 outlines typically score 7 or 4)."""
    ready: bool
    score: int = 0
    issues: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)

    @classmethod
    def degraded(cls, reason: str) -> "CritiqueResult":
        """Soft-fail factory — used when the model errors or returns
        garbage. ``ready=True`` because we don't want the loop to
        keep retrying a broken critic forever; we ship the outline
        un-reviewed and let the warning log raise the alarm."""
        return cls(
            ready=True, score=0,
            issues=[f"critic_failed: {reason}"][:1],
        )


# ──────────────────────────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────────────────────────

def _build_system_prompt(skill_content: str) -> str:
    """The critic's system prompt. Embeds the rendered skill so the
    critic sees the curriculum design rules + checklist (same source
    of truth the designer used)."""
    return (
        "You are reviewing a learning curriculum someone else just "
        "designed. You are NOT designing it yourself — your job is "
        "to score it against the CURRICULUM REVIEW CHECKLIST in the "
        "skill reference below and either approve it (ready=true) or "
        "return a short list of specific, actionable issues the "
        "designer should fix. Be strict but fair: a 3-module plan "
        "with the right shape and no glaring gaps is GOOD ENOUGH — "
        "you don't have to find issues to justify your existence. "
        f"{INJECTION_GUARD_LINE} "
        "Emit ONLY the JSON object — no prose, no Markdown fences.\n\n"
        f"{skill_content}"
    )


def _build_user_prompt(ctx: LearnerContext, outline: dict[str, Any]) -> str:
    """Render the outline + the learner context the designer saw, so
    the critic can score against the same evidence."""
    import json as _json
    p = ctx.profile or {}
    outline_json = _json.dumps(outline, indent=2, ensure_ascii=False)
    parts = [
        "LEARNER CONTEXT:",
        f"  name: {tag_learner_input(p.get('name'))}",
        f"  age: {p.get('age') if p.get('age') is not None else '(not given)'}",
        f"  current_level: {tag_learner_input(p.get('current_level')) if p.get('current_level') else '(not given)'}",
        "",
        "FOCUS FOR THIS CURRICULUM (this is what the plan must serve):",
        f"  goal title: {tag_learner_input(ctx.goal_title)}",
        f"  goal context: {tag_learner_input(ctx.goal_context)}",
        f"  intake objective fallback: {tag_learner_input((p or {}).get('objective'))}",
        "",
        "OUTLINE UNDER REVIEW:",
        outline_json,
        "",
        "TASK: Review the outline against the CURRICULUM REVIEW "
        "CHECKLIST in the skill reference above. Return JSON:",
        '  {"ready": bool, "score": 1-10,',
        '   "issues": ["specific actionable item", ...],',
        '   "strengths": ["what works", ...]}',
        "",
        "Rules:",
        " - ready=true ONLY if the outline meets ALL checklist items "
        "well enough to ship — no fatal gaps in goal alignment, "
        "scaffolding, or skill coverage.",
        " - issues MUST be specific + actionable. \"Module 1 needs "
        "a lesson topic\" is good. \"Could be better\" is useless.",
        " - At most 5 issues, at most 3 strengths.",
        " - Each entry under 30 words.",
        " - score is a sanity check (1=unusable, 10=excellent); the "
        "orchestrator reads ready first.",
    ]
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────
# Validation + soft-fail
# ──────────────────────────────────────────────────────────────────

def _trim_list(value: Any, cap: int) -> list[str]:
    """Coerce ``value`` into a list of trimmed strings. Tolerant of
    a single string, a list with mixed types, or a missing field."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s:
            continue
        out.append(s[:_MAX_ITEM_LEN])
        if len(out) >= cap:
            break
    return out


def _validate_critique(payload: dict[str, Any]) -> CritiqueResult:
    """Parse the LLM's JSON into a CritiqueResult. Tolerant of missing
    fields (default to safe values) but strict on the ``ready`` gate —
    if absent or non-bool, default to True so we don't block the
    learner. Returns a degraded result on completely unusable shapes."""
    if not isinstance(payload, dict):
        return CritiqueResult.degraded("payload was not a JSON object")
    if "error" in payload:
        return CritiqueResult.degraded(f"model error: {payload['error']}")

    ready_raw = payload.get("ready")
    ready = bool(ready_raw) if isinstance(ready_raw, bool) else True

    # Score parsing: distinguish "model said something" (0–10 valid)
    # from "model said nothing" (default 0, surfaces as no-signal in
    # logs). Lower clamp is 0, NOT 1 — otherwise a legitimate 0 from
    # the model and a missing-field default both collapse to 1 and the
    # downstream signal is muddied. Also reject bool explicitly: it
    # passes isinstance(_, int) in Python but `score: true` from the
    # model is a parse mistake we should treat as missing.
    score_raw = payload.get("score")
    if isinstance(score_raw, bool):
        score = 0
    elif isinstance(score_raw, (int, float)):
        score = max(0, min(10, int(score_raw)))
    else:
        score = 0

    return CritiqueResult(
        ready=ready,
        score=score,
        issues=_trim_list(payload.get("issues"), cap=5),
        strengths=_trim_list(payload.get("strengths"), cap=3),
    )


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

async def critique_outline(
    ctx: LearnerContext,
    outline: dict[str, Any],
    *,
    model: Model,
    skill_content: str,
) -> CritiqueResult:
    """Score an outline against the checklist. Never raises — model
    errors degrade to ``ready=True, score=0`` so the orchestrator
    keeps moving (with a logged warning)."""
    try:
        payload = await ask_for_json(
            system_prompt=_build_system_prompt(skill_content),
            user_prompt=_build_user_prompt(ctx, outline),
            model=model,
        )
    except ModelOutputError as exc:
        log.warning("curriculum_critic: model output error: %s", exc)
        return CritiqueResult.degraded(str(exc)[:120])
    except Exception as exc:                                # noqa: BLE001
        log.warning("curriculum_critic: model crashed: %s", exc)
        return CritiqueResult.degraded(str(exc)[:120])
    return _validate_critique(payload)
