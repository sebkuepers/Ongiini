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
from .db import (
    CARD_CLOZE, CARD_DIALOGUE, CARD_GRAMMAR, CARD_LESSON,
    CARD_MULTIPLE_CHOICE, CARD_PROVERB, CARD_REORDER, CARD_TYPES,
    EXERCISE_CARD_TYPES,
)
from .llm import INJECTION_GUARD_LINE, ModelOutputError, ask_for_json, tag_learner_input

log = logging.getLogger("ongiini.learning.cards")


# Always required, regardless of card_type or shape variant.
_REQUIRED_CARD_KEYS = ("card_type",)

# Allowed kinds for entries in a lesson card's ``steps`` array. Order
# is the recommended ordering inside a single lesson; the validator
# only enforces ordering constraints on ``quick_check`` (must be last).
LESSON_STEP_KINDS = ("concept", "example", "contrast", "quick_check")
LESSON_STEPS_MIN = 2
LESSON_STEPS_MAX = 5


def _validate_lesson_steps(steps: Any) -> None:
    """Validate the ``steps`` array of a multi-step lesson card.

    The carousel renderer relies on these invariants — the model is
    free to vary the content per step, but the shape MUST hold or the
    frontend would render an empty card."""
    if not isinstance(steps, list):
        raise ModelOutputError(
            "lesson 'steps' must be a list"
        )
    if not (LESSON_STEPS_MIN <= len(steps) <= LESSON_STEPS_MAX):
        raise ModelOutputError(
            f"lesson 'steps' must have {LESSON_STEPS_MIN}-{LESSON_STEPS_MAX} "
            f"entries; got {len(steps)}"
        )
    quick_check_seen = False
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ModelOutputError(
                f"lesson step at index {idx} must be an object"
            )
        kind = step.get("kind")
        if kind not in LESSON_STEP_KINDS:
            raise ModelOutputError(
                f"lesson step 'kind' must be one of {LESSON_STEP_KINDS}; "
                f"got {kind!r}"
            )
        if kind == "quick_check":
            # quick_check is at most one and MUST be the final step —
            # the renderer pegs the "Reveal answer" reveal to the last
            # step, so a quick_check in the middle would break the UX.
            if quick_check_seen:
                raise ModelOutputError(
                    "lesson 'steps' may contain at most one 'quick_check'"
                )
            if idx != len(steps) - 1:
                raise ModelOutputError(
                    "lesson 'quick_check' step must be the LAST step"
                )
            quick_check_seen = True
            prompt = step.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ModelOutputError(
                    "lesson 'quick_check' step requires a non-empty 'prompt'"
                )
            answer = step.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ModelOutputError(
                    "lesson 'quick_check' step requires a non-empty 'answer'"
                )
            if "hint" in step and not isinstance(step["hint"], str):
                raise ModelOutputError(
                    "lesson 'quick_check' 'hint' must be a string when present"
                )
        else:
            # concept / example / contrast — body is the carousel slide
            # content; examples is optional.
            body = step.get("body")
            if not isinstance(body, str) or not body.strip():
                raise ModelOutputError(
                    f"lesson step '{kind}' requires a non-empty 'body'"
                )
            if "examples" in step:
                ex = step["examples"]
                if not isinstance(ex, list):
                    raise ModelOutputError(
                        f"lesson step '{kind}' 'examples' must be a list"
                    )
                if not all(isinstance(e, str) and e.strip() for e in ex):
                    raise ModelOutputError(
                        f"lesson step '{kind}' 'examples' must all be "
                        "non-empty strings"
                    )


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

    # Shape rules per card_type:
    #   * Lesson cards MUST use the multi-step shape (steps[] with
    #     2-5 entries). The legacy single-blob ``prompt_text`` shape
    #     is gone — the model is asked specifically for steps[] under
    #     the new content brief, so accepting both opens room for
    #     model confusion + dropped content.
    #   * Exercise cards still require a non-empty ``prompt_text``.
    # ``module_id`` and ``topic_id`` are NOT validated here. The
    # selector picks them deterministically and the coach attaches
    # them after calling the model — the model is not asked to emit
    # them, so we don't check for them.
    if ct == CARD_LESSON:
        steps = payload.get("steps")
        if not isinstance(steps, list):
            raise ModelOutputError(
                "lesson card must include 'steps' as a list (2-5 "
                "entries — concept / example / contrast / quick_check)"
            )
        _validate_lesson_steps(steps)
    else:
        if "prompt_text" not in payload:
            raise ModelOutputError(
                "card missing required key: 'prompt_text'"
            )
        if not isinstance(payload["prompt_text"], str) or not payload["prompt_text"].strip():
            raise ModelOutputError("card prompt_text must be a non-empty string")

    # All exercise types require a non-empty reference_answer the
    # grader can score against. Lesson cards are exempt (they're
    # acknowledged, not graded). Note: this tightened in Phase 2 —
    # vocab/translation/production used to accept null reference_answer
    # but the grader always needed *something* to score against, so
    # the laxer contract was silently producing worse grading on
    # malformed cards. For production cards the reference_answer is
    # a free-form rubric, not a canonical string.
    if ct in EXERCISE_CARD_TYPES:
        ref = payload.get("reference_answer")
        if not isinstance(ref, str) or not ref.strip():
            raise ModelOutputError(
                f"{ct} card requires a non-empty reference_answer"
            )

    # Per-type structural extras. These shape-checks live HERE rather
    # than in a generic "extra payload" pass because the frontend
    # renderer relies on the field being present and the right type;
    # a missing or wrong-typed extra would render an empty card.
    if ct == CARD_REORDER:
        tokens = payload.get("tokens")
        if not isinstance(tokens, list) or len(tokens) < 2:
            raise ModelOutputError(
                "reorder card requires a 'tokens' list of at least 2 strings"
            )
        if not all(isinstance(t, str) and t.strip() for t in tokens):
            raise ModelOutputError(
                "reorder 'tokens' must all be non-empty strings"
            )
    elif ct == CARD_MULTIPLE_CHOICE:
        options = payload.get("options")
        if not isinstance(options, list) or not (2 <= len(options) <= 4):
            raise ModelOutputError(
                "multiple_choice requires 2-4 options"
            )
        labels: set[str] = set()
        for opt in options:
            if not isinstance(opt, dict):
                raise ModelOutputError(
                    "multiple_choice option must be an object"
                )
            label = opt.get("label")
            text = opt.get("text")
            if not isinstance(label, str) or not label.strip():
                raise ModelOutputError(
                    "multiple_choice option needs a non-empty 'label'"
                )
            if not isinstance(text, str) or not text.strip():
                raise ModelOutputError(
                    "multiple_choice option needs non-empty 'text'"
                )
            if label in labels:
                raise ModelOutputError(
                    f"multiple_choice option labels must be unique; "
                    f"saw {label!r} twice"
                )
            labels.add(label)
            # explanation is optional but if present must be a string —
            # frontend renders it post-grading.
            if "explanation" in opt and not isinstance(opt["explanation"], str):
                raise ModelOutputError(
                    "multiple_choice 'explanation' must be a string"
                )
        # reference_answer must match one of the option labels so the
        # grader and renderer can map "the right one" deterministically.
        ref = payload.get("reference_answer")
        if ref not in labels:
            raise ModelOutputError(
                "multiple_choice reference_answer must match one of "
                "the option labels"
            )
    elif ct == CARD_GRAMMAR:
        src = payload.get("source_sentence")
        if not isinstance(src, str) or not src.strip():
            raise ModelOutputError(
                "grammar card requires non-empty 'source_sentence'"
            )
    elif ct == CARD_DIALOGUE:
        turns = payload.get("turns")
        if not isinstance(turns, list) or len(turns) < 2:
            raise ModelOutputError(
                "dialogue card requires 'turns' list of at least 2 entries"
            )
        for turn in turns:
            if not isinstance(turn, dict):
                raise ModelOutputError("dialogue turn must be an object")
            if not isinstance(turn.get("speaker"), str) or not turn["speaker"].strip():
                raise ModelOutputError(
                    "dialogue turn requires non-empty 'speaker'"
                )
            if not isinstance(turn.get("text"), str):
                raise ModelOutputError(
                    "dialogue turn requires 'text' string (may be '___')"
                )
    elif ct == CARD_PROVERB:
        # cultural_note is optional, but if present must be a string —
        # frontend renders it after grading.
        if "cultural_note" in payload and not isinstance(payload["cultural_note"], str):
            raise ModelOutputError(
                "proverb 'cultural_note' must be a string"
            )
    elif ct == CARD_CLOZE:
        # The prompt_text MUST carry the blank placeholder so the
        # frontend can render the input slot. Accept 3 or more
        # underscores in a row — `___` is the canonical marker;
        # any longer run still matches the substring check.
        if "___" not in payload["prompt_text"]:
            raise ModelOutputError(
                "cloze prompt_text must contain '___' as the blank marker"
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


def _build_content_brief(
    ctx: LearnerContext,
    *,
    card_type: str,
    module_id: str,
    module_title: str,
    topic_id: str,
    topic_title: str,
) -> str:
    """Tight prompt: the SELECTOR has already decided card_type +
    module + topic. The model's only job is to write the content.

    No outline JSON, no module digest, no pacing rules, no recent
    cards — none of those decisions are the model's anymore."""
    p = ctx.profile or {}
    if card_type == CARD_LESSON:
        content_brief = (
            "Produce a LESSON card teaching this topic. Use the lesson "
            "card shape from the skill reference: a `steps` array with "
            "2-5 entries — kinds: concept / example / contrast / "
            "quick_check (last only). Output JSON: "
            "{ title, steps }. "
            "DO NOT include card_type, module_id, topic_id, or "
            "prompt_text — the coach attaches scaffolding."
        )
    else:
        content_brief = (
            f"Produce a {card_type} EXERCISE card drilling this topic. "
            "Use the shape from the skill reference for this card_type "
            "(prompt_text + reference_answer + any per-type extras like "
            "options / tokens / turns / source_sentence). "
            "Output JSON. DO NOT include card_type, module_id, or "
            "topic_id — the coach attaches scaffolding."
        )
    return (
        "LEARNER:\n"
        f"  name: {tag_learner_input(p.get('name'))}\n"
        f"  level: {p.get('current_level') or 'beginner'}\n"
        f"  focus: {tag_learner_input(ctx.goal_title or ctx.goal_context or p.get('objective'))}\n"
        "\nCARD TO AUTHOR (selected by the coach — these are FIXED, "
        "you don't pick them, you produce content for them):\n"
        f"  card_type: {card_type}\n"
        f"  module: {tag_learner_input(module_title)} (id: {module_id})\n"
        f"  topic:  {tag_learner_input(topic_title)} (id: {topic_id})\n"
        f"\n{content_brief}"
    )


async def generate_card_content(
    ctx: LearnerContext,
    *,
    model: Model,
    skill_content: str,
    card_type: str,
    module_id: str,
    module_title: str,
    topic_id: str,
    topic_title: str,
    steering_note: str | None = None,
) -> dict[str, Any]:
    """Ask the model to author the CONTENT of one card, with the
    card_type / module / topic already chosen by the selector.

    Returns the validated content payload. The coach attaches
    ``card_type``, ``module_id``, ``topic_id`` (the model is told NOT
    to emit them) and persists.

    ``steering_note`` is appended to the user prompt when the caller
    is asking for a corrective re-roll (e.g. after the card critic
    rejected the first attempt). Mirrors the steering_note pattern
    on ``curriculum.revise_outline``.

    Raises ``ModelOutputError`` on malformed shape."""
    # Inject the just-decided card_type into the payload BEFORE the
    # validator runs. The model is told not to emit card_type (so the
    # brief stays single-purpose), but the per-type validators below
    # need to see it to pick the right shape checks.
    user_prompt = _build_content_brief(
        ctx,
        card_type=card_type, module_id=module_id, module_title=module_title,
        topic_id=topic_id, topic_title=topic_title,
    )
    if steering_note:
        user_prompt = (
            user_prompt
            + "\n\nSTEERING NOTE (the previous attempt was reviewed and "
            "rejected — address this before re-emitting):\n"
            + steering_note
        )
    payload = await ask_for_json(
        system_prompt=_build_system_prompt(skill_content),
        user_prompt=user_prompt,
        model=model,
    )
    payload["card_type"] = card_type
    _validate_card(payload)
    return payload


# Iteration cap for the card-content review loop. Same posture as the
# curriculum design loop: 1 author + 1 critic + optional 1 revise.
# A second critic call after revise would double cost again with
# diminishing returns; we ship after one revise regardless.
_CARD_REVIEW_MAX_ITERATIONS = 1


async def generate_card_content_with_review(
    ctx: LearnerContext,
    *,
    model: Model,
    skill_content: str,
    card_type: str,
    module_id: str,
    module_title: str,
    topic_id: str,
    topic_title: str,
) -> dict[str, Any]:
    """Author + critic + maybe revise. Mirror of
    ``curriculum.design_outline_with_review``.

    Flow:
      1. ``generate_card_content`` — first draft.
      2. ``card_critic.critique_card`` — score against the Card
         review checklist in SKILL.md.
      3. If ``critique.ready`` → ship.
      4. Else → ``generate_card_content`` again with the critic's
         issues as ``steering_note``; ship whatever comes back.

    Soft-fails: critic crash → ship original (critic returns
    ``ready=True`` on degraded). Revise crash → ship original.

    Worst case: 3 LLM calls per card (author + critic + revise).
    Best case: 2 (author + critic-approves)."""
    from . import card_critic as critic_mod   # local — break cycle

    payload = await generate_card_content(
        ctx,
        model=model, skill_content=skill_content,
        card_type=card_type, module_id=module_id, module_title=module_title,
        topic_id=topic_id, topic_title=topic_title,
    )

    # Belt-and-braces: critique_card already soft-fails on
    # ModelOutputError / Exception inside its ask_for_json call, but
    # anything that raises BEFORE that (e.g. a non-serialisable nested
    # object slipping past the validator into json.dumps) would still
    # propagate. Catch here too so the orchestrator's stated contract
    # — "critic crash → ship the original card" — is absolute.
    try:
        critique = await critic_mod.critique_card(
            ctx, payload,
            model=model, skill_content=skill_content,
            card_type=card_type,
            module_title=module_title,
            topic_title=topic_title,
        )
    except Exception as exc:                                # noqa: BLE001
        log.warning(
            "card_critic: critic crashed pre-call on card_type=%s "
            "topic=%s; shipping the original. error=%s",
            card_type, topic_id, exc,
        )
        return payload
    log.info(
        "card_critic: card_type=%s topic=%s score=%d ready=%s issues=%d",
        card_type, topic_id, critique.score, critique.ready,
        len(critique.issues),
    )
    if critique.ready:
        return payload

    if _CARD_REVIEW_MAX_ITERATIONS <= 0:
        return payload

    # Revise pass — feed the critic's issues back as the steering
    # note. Build a meaningful steering string even if the critic
    # didn't list issues (degenerate but possible).
    if critique.issues:
        steering = "Critic feedback to address:\n" + "\n".join(
            f"- {item}" for item in critique.issues
        )
    else:
        steering = (
            f"The critic scored this card {critique.score}/10 but did "
            "not list specific issues. Tighten the card overall: "
            "ensure every target-language sentence has an inline "
            "source-language gloss, the level matches the learner, "
            "and the shape is right for the card_type."
        )
    try:
        revised = await generate_card_content(
            ctx,
            model=model, skill_content=skill_content,
            card_type=card_type, module_id=module_id, module_title=module_title,
            topic_id=topic_id, topic_title=topic_title,
            steering_note=steering,
        )
    except ModelOutputError as exc:
        log.warning(
            "card_critic: revise failed on card_type=%s topic=%s; "
            "shipping the original. error=%s",
            card_type, topic_id, exc,
        )
        return payload
    return revised
