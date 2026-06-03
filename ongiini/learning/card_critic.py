"""LLM critic for the card-content authoring loop.

A second LLM call that reviews a freshly-generated card payload
against the **Card review checklist** in SKILL.md and returns a
structured verdict. The orchestrator in
``cards.generate_card_content_with_review`` uses the verdict to
decide whether to ship the card or pass it back to
``generate_card_content`` with the critic's issues as a steering
note (the card-level analog of ``revise_outline``).

The checklist (level appropriateness / gloss completeness /
type-specific shape / cultural anchoring) lives in
``_core.md.tmpl`` — same single-source-of-truth posture as the
curriculum critic.

Design principles — same as ``curriculum_critic.py``:

  * Soft-fail. A flaky or bad-JSON critic must NEVER block the
    learner. On any model error we degrade to ``ready=True,
    score=0, issues=["critic_failed: ..."]`` and ship the
    un-reviewed card.
  * The checklist lives in the rendered skill content, not
    duplicated here.
  * Independent of the author prompt. The critic sees the card +
    learner context fresh.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from owela import Model

from .context import LearnerContext
from .llm import INJECTION_GUARD_LINE, ModelOutputError, ask_for_json, tag_learner_input

log = logging.getLogger("ongiini.learning.card_critic")


# Cap for individual issue/strength strings. Same posture as the
# curriculum critic — a list of 12-word actionables is more useful
# than a paragraph.
_MAX_ITEM_LEN = 240


@dataclass(frozen=True)
class CardCritiqueResult:
    """Structured verdict from the card critic.

    Same shape as the curriculum-side ``CritiqueResult`` — ``ready``
    is the gate the orchestrator reads; ``score`` and the lists are
    kept for logging."""
    ready: bool
    score: int = 0
    issues: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)

    @classmethod
    def degraded(cls, reason: str) -> "CardCritiqueResult":
        """Soft-fail factory — ``ready=True`` so the orchestrator
        ships the un-reviewed card rather than retrying a broken
        critic forever. The orchestrator logs the reason."""
        return cls(
            ready=True, score=0,
            issues=[f"critic_failed: {reason}"],
        )


# ──────────────────────────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────────────────────────

def _build_system_prompt(skill_content: str) -> str:
    """The critic's system prompt. Embeds the rendered skill so the
    critic sees the Card review checklist + gloss + cloze + CEFR
    guidance (same source of truth the author used)."""
    return (
        "You are reviewing a single learning card someone else just "
        "wrote. You are NOT writing it yourself — your job is to "
        "score it against the CARD REVIEW CHECKLIST in the skill "
        "reference below and either approve it (ready=true) or return "
        "a short list of specific, actionable issues the author "
        "should fix. Be strict but fair: a correct, glossed, "
        "level-appropriate card with the right shape is GOOD ENOUGH "
        "— you don't have to find issues to justify your existence. "
        f"{INJECTION_GUARD_LINE} "
        "Emit ONLY the JSON object — no prose, no Markdown fences.\n\n"
        f"{skill_content}"
    )


def _build_user_prompt(
    ctx: LearnerContext,
    payload: dict[str, Any],
    *,
    card_type: str,
    module_title: str,
    topic_title: str,
) -> str:
    """Render the card + the learner context + the selector's pick so
    the critic can score against the same evidence the author saw."""
    import json as _json
    p = ctx.profile or {}
    card_json = _json.dumps(payload, indent=2, ensure_ascii=False)
    parts = [
        "LEARNER CONTEXT:",
        f"  name: {tag_learner_input(p.get('name'))}",
        f"  current_level: {tag_learner_input(p.get('current_level')) if p.get('current_level') else '(not given)'}",
        f"  focus: {tag_learner_input(ctx.goal_title or ctx.goal_context or p.get('objective'))}",
        "",
        "CARD UNDER REVIEW:",
        f"  card_type (selector's pick): {card_type}",
        f"  module: {tag_learner_input(module_title)}",
        f"  topic:  {tag_learner_input(topic_title)}",
        "",
        "  payload:",
        card_json,
        "",
        "TASK: Review the card against the CARD REVIEW CHECKLIST in "
        "the skill reference above. Return JSON:",
        '  {"ready": bool, "score": 1-10,',
        '   "issues": ["specific actionable item", ...],',
        '   "strengths": ["what works", ...]}',
        "",
        "Rules:",
        " - ready=true ONLY if the card meets ALL checklist items "
        "well enough to ship — no fatal gaps in level, gloss "
        "completeness, or type-specific shape.",
        " - issues MUST be specific + actionable. \"Step 2 body "
        "has no gloss — add '(Good day…)' after the German sentence\" "
        "is good. \"Could be better\" is useless.",
        " - At most 4 issues, at most 2 strengths.",
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


def _validate_critique(payload: dict[str, Any]) -> CardCritiqueResult:
    """Parse the LLM's JSON into a CardCritiqueResult. Same posture
    as ``curriculum_critic._validate_critique`` — tolerant on missing
    fields, defaults ``ready=True`` if the field is absent or
    non-bool so a soft-failed critic doesn't block the learner."""
    if not isinstance(payload, dict):
        return CardCritiqueResult.degraded("payload was not a JSON object")
    if "error" in payload:
        return CardCritiqueResult.degraded(f"model error: {payload['error']}")

    ready_raw = payload.get("ready")
    ready = bool(ready_raw) if isinstance(ready_raw, bool) else True

    # Score: bool first (Python bool isinstance int), then int/float,
    # else 0. Clamp 0–10.
    score_raw = payload.get("score")
    if isinstance(score_raw, bool):
        score = 0
    elif isinstance(score_raw, (int, float)):
        score = max(0, min(10, int(score_raw)))
    else:
        score = 0

    return CardCritiqueResult(
        ready=ready,
        score=score,
        issues=_trim_list(payload.get("issues"), cap=4),
        strengths=_trim_list(payload.get("strengths"), cap=2),
    )


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

async def critique_card(
    ctx: LearnerContext,
    payload: dict[str, Any],
    *,
    model: Model,
    skill_content: str,
    card_type: str,
    module_title: str,
    topic_title: str,
) -> CardCritiqueResult:
    """Score a card payload against the Card review checklist. Never
    raises — model errors degrade to ``ready=True, score=0`` so the
    orchestrator keeps moving (with a logged warning)."""
    try:
        result = await ask_for_json(
            system_prompt=_build_system_prompt(skill_content),
            user_prompt=_build_user_prompt(
                ctx, payload,
                card_type=card_type,
                module_title=module_title,
                topic_title=topic_title,
            ),
            model=model,
        )
    except ModelOutputError as exc:
        log.warning("card_critic: model output error: %s", exc)
        return CardCritiqueResult.degraded(str(exc)[:120])
    except Exception as exc:                                # noqa: BLE001
        log.warning("card_critic: model crashed: %s", exc)
        return CardCritiqueResult.degraded(str(exc)[:120])
    return _validate_critique(result)
