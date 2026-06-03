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

    # Lesson cards have TWO valid shapes:
    #   1. New multi-step shape: ``steps`` array with 2-5 entries.
    #      ``prompt_text`` is optional (the per-step ``body`` carries
    #      content); the persistence layer synthesises a prompt_text
    #      from title/first-step for the DB NOT NULL constraint.
    #   2. Legacy single-blob shape: ``prompt_text`` is required, no
    #      ``steps``. Kept for backward-compat with cards authored
    #      before the carousel landed.
    # Exercise cards still require ``prompt_text`` unconditionally.
    if ct == CARD_LESSON and "steps" in payload:
        _validate_lesson_steps(payload["steps"])
        # The two shapes are mutually exclusive: the model must pick
        # ONE. Allowing both lets the model hedge by emitting prose in
        # prompt_text alongside a steps[] array — the prose then gets
        # silently dropped by the coach (which forwards steps to the
        # frontend). Reject loudly so the LLM's mixed-shape confusion
        # surfaces rather than hides.
        pt = payload.get("prompt_text")
        if isinstance(pt, str) and pt.strip():
            raise ModelOutputError(
                "lesson cards must use EITHER 'steps' OR 'prompt_text', "
                "not both — the prose in prompt_text would be silently "
                "discarded. Pick the multi-step carousel (steps[]) or "
                "the legacy single-blob shape (prompt_text only)."
            )
        if pt is not None and not isinstance(pt, str):
            raise ModelOutputError("lesson prompt_text must be a string when present")
    else:
        if "prompt_text" not in payload:
            raise ModelOutputError(
                "card missing required key: 'prompt_text'"
            )
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
    # topic_id is also soft-required — drives topic-aware pacing
    # (no exercises on untaught topics). Same back-compat policy as
    # module_id: optional, type-checked when present.
    if "topic_id" in payload and payload["topic_id"] is not None:
        if not isinstance(payload["topic_id"], str):
            raise ModelOutputError(
                f"topic_id must be a string; got "
                f"{type(payload['topic_id']).__name__}"
            )

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


def _render_module_digest(ctx: LearnerContext) -> str:
    """Per-module rollup the recent_cards window can't carry once the
    learner is deep in a module. Without this the model loses track of
    'lessons already given for module mod-1' after ~12 turns and
    re-emits the original lesson — the exact bug Sebastian hit at the
    13-card mark.

    The per-topic breakdown (``topics_taught`` / ``topics_drilled``)
    drives the teach-then-test pacing rule below — the model can read
    "topic t2 has been drilled but never taught" and steer back to a
    lesson card. We filter the per-topic keys against the outline's
    declared `topics[].id` set so a stray / typoed topic_id from a
    past LLM hiccup doesn't get echoed back as a "real" taught topic."""
    if not ctx.module_digest:
        return "(no per-module data yet)"
    # Build the per-module known-topic whitelist from the outline so
    # bogus topic_ids (mismatched / typoed by a prior LLM call) don't
    # leak back into the prompt.
    known_topics_by_module: dict[str, set[str]] = {}
    outline = ctx.curriculum_outline or {}
    raw_modules = outline.get("modules") if isinstance(outline, dict) else None
    if isinstance(raw_modules, list):
        for m in raw_modules:
            if not isinstance(m, dict):
                continue
            mod_id = m.get("id")
            topics = m.get("topics")
            if not isinstance(mod_id, str) or not isinstance(topics, list):
                continue
            ids = {
                t.get("id") for t in topics
                if isinstance(t, dict) and isinstance(t.get("id"), str)
            }
            known_topics_by_module[mod_id] = ids
    lines: list[str] = []
    for mod_id, d in ctx.module_digest.items():
        lessons = d.get("lessons_given", 0)
        ex_emit = d.get("exercises_emitted", 0)
        ex_seen = d.get("exercises_attempted", 0)
        ex_correct = d.get("exercises_correct", 0)
        taught_raw = d.get("topics_taught") or {}
        drilled_raw = d.get("topics_drilled") or {}
        # Filter to ids declared in the outline if we have one;
        # otherwise (legacy outlines without topics lists) keep the
        # raw set — the runtime rule already skips pacing checks for
        # those.
        known = known_topics_by_module.get(mod_id)
        if known is not None:
            taught_keys = sorted(k for k in taught_raw.keys() if k in known)
            drilled_keys = sorted(k for k in drilled_raw.keys() if k in known)
        else:
            taught_keys = sorted(taught_raw.keys())
            drilled_keys = sorted(drilled_raw.keys())
        taught_str = ", ".join(taught_keys) if taught_keys else "(none)"
        drilled_str = ", ".join(drilled_keys) if drilled_keys else "(none)"
        lines.append(
            f"  - {mod_id}: lessons_given={lessons}, "
            f"exercises_emitted={ex_emit}, "
            f"exercises_attempted={ex_seen}, "
            f"exercises_correct={ex_correct}, "
            f"topics_taught=[{taught_str}], "
            f"topics_drilled=[{drilled_str}]"
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
        "INCLUDE the JSON keys `module_id` AND `topic_id` matching the "
        "in-progress module and one of its `topics[].id` entries from "
        "the outline above — both required so per-module / per-topic "
        "progress can be tracked.\n"
        "Topic-aware pacing rules (the single most important section):\n"
        " 1. Walk the in-progress module's `topics` list. Find every "
        "topic whose `kind` is \"lesson\" — these MUST be taught "
        "before any exercise is drilled in this module.\n"
        " 2. Cross-reference with MODULE DIGEST -> `topics_taught` for "
        "that module. The set of TAUGHT lesson topics is "
        "`topics_taught.keys()`.\n"
        " 3. If there are LESSON topics not yet in `topics_taught` "
        "(\"untaught lesson topics\"), EMIT A LESSON card targeting the "
        "earliest one in outline order. Use the multi-step shape with "
        "a `steps` array (see SKILL.md for the lesson shape).\n"
        " 4. If every lesson topic in the in-progress module has been "
        "taught, emit an EXERCISE card. The exercise's `topic_id` MUST "
        "be either:\n"
        "   (a) a `practice` topic from the in-progress module's "
        "topics list, OR\n"
        "   (b) one of the already-taught lesson topics, drilled with "
        "a DIFFERENT card_type (vocab → cloze → dialogue, etc.) for "
        "spaced recycling.\n"
        " 5. NEVER emit an exercise targeting a topic whose id is not "
        "in `topics_taught` AND not declared as a `practice` topic in "
        "the outline. Cards on untaught material are the bug we are "
        "trying to prevent.\n"
        " 6. If lots of cards are stuck in Leitner box 1, prefer "
        "consolidating cards (easier, on already-taught topics) over "
        "introducing new themes.\n"
        " 7. Pick the card_type that fits the topic and the learner's "
        "level.\n"
        "Output JSON only."
    )


async def generate_card(
    ctx: LearnerContext,
    *,
    model: Model,
    skill_content: str,
    steering_note: str | None = None,
) -> dict[str, Any]:
    """Ask the model to author the next card. Returns a dict with at
    least ``card_type`` and ``prompt_text``; ``reference_answer`` and
    ``hint_text`` and ``difficulty`` may be present.

    ``steering_note`` is appended to the user prompt when the coach is
    asking for a corrective re-roll (e.g. after a topic-pacing
    violation). It lets us nudge the LLM without rebuilding the whole
    prompt structure.

    Raises ``ModelOutputError`` on bad JSON / missing required fields /
    unknown card_type.
    """
    user_prompt = _build_user_prompt(ctx)
    if steering_note:
        user_prompt = (
            user_prompt
            + "\n\nSTEERING NOTE (the previous attempt was rejected — "
            "address this before re-emitting):\n"
            + steering_note
        )
    payload = await ask_for_json(
        system_prompt=_build_system_prompt(skill_content),
        user_prompt=user_prompt,
        model=model,
    )
    _validate_card(payload)
    return payload
