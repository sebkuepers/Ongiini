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
import re
from typing import Any

from owela import Model

from . import store
from .context import LearnerContext
from .db import (
    CARD_CLOZE, CARD_DIALOGUE, CARD_GRAMMAR, CARD_LESSON,
    CARD_MULTIPLE_CHOICE, CARD_PROVERB, CARD_REORDER, CARD_STORY,
    CARD_TYPES, EXERCISE_CARD_TYPES,
)
from .llm import INJECTION_GUARD_LINE, ModelOutputError, ask_for_json, tag_learner_input

log = logging.getLogger("ongiini.learning.cards")


# Matches any parenthesised substring on a single line. Used to enforce
# the "no `___` inside a gloss" rule across all exercise types — the
# inline source-language gloss is read-only context, and a blank inside
# it creates a phantom input slot with no corresponding answer.
_GLOSS_PARENS_RE = re.compile(r"\(([^)]*)\)")


def _gloss_contains_blank(text: str) -> bool:
    """Return True if any ``(...)`` substring inside ``text`` contains
    a ``___`` blank marker. The inline gloss must show the missing
    piece spelled out in source-language so the learner reads it as
    read-only context; a blank in the gloss is a fatal authoring
    error and we reject the card before it reaches the frontend
    renderer (which would either ignore it or paint a phantom input)."""
    if not isinstance(text, str) or "___" not in text or "(" not in text:
        return False
    for match in _GLOSS_PARENS_RE.finditer(text):
        if "___" in match.group(1):
            return True
    return False


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


# Story card shape constants. A story is 4-8 short paragraphs of
# comprehensible input followed by 1-3 lenient comprehension
# questions. The paragraph count keeps the read short enough to feel
# bite-sized but long enough to give the target structure 4-6
# repetitions (per Krashen i+1 + repetition-for-acquisition).
STORY_PARAGRAPHS_MIN = 4
STORY_PARAGRAPHS_MAX = 8
STORY_QUESTIONS_MIN = 1
STORY_QUESTIONS_MAX = 3


def _validate_story_payload(payload: dict[str, Any]) -> None:
    """Validate the story card shape and synthesise ``prompt_text`` +
    ``reference_answer`` so the rest of the persistence + grading
    pipeline (which expects single strings) keeps working.

    Story shape:

    ```json
    {
      "card_type": "story",
      "title": "At the bakery (Beim Bäcker)",
      "paragraphs": [
        {"target": "Sebastian geht in die Bäckerei.",
         "gloss":  "(Sebastian goes into the bakery.)"},
        ...
      ],
      "comprehension_questions": [
        {"prompt": "Where does Sebastian go?",
         "answer": "to the bakery"},
        ...
      ]
    }
    ```

    Each paragraph carries a target-language sentence + the mandatory
    inline source-language gloss; the gloss-blank-leak rule applies
    here too — a `___` inside the gloss is rejected.
    """
    paragraphs = payload.get("paragraphs")
    if not isinstance(paragraphs, list):
        raise ModelOutputError(
            "story card must include 'paragraphs' as a list"
        )
    if not (STORY_PARAGRAPHS_MIN <= len(paragraphs) <= STORY_PARAGRAPHS_MAX):
        raise ModelOutputError(
            f"story 'paragraphs' must have {STORY_PARAGRAPHS_MIN}-"
            f"{STORY_PARAGRAPHS_MAX} entries; got {len(paragraphs)}"
        )
    target_lines: list[str] = []
    for idx, para in enumerate(paragraphs):
        if not isinstance(para, dict):
            raise ModelOutputError(
                f"story paragraph {idx} must be an object"
            )
        target = para.get("target")
        gloss = para.get("gloss")
        if not isinstance(target, str) or not target.strip():
            raise ModelOutputError(
                f"story paragraph {idx} requires non-empty 'target' "
                "string in <<TARGET_LANGUAGE>>"
            )
        if not isinstance(gloss, str) or not gloss.strip():
            raise ModelOutputError(
                f"story paragraph {idx} requires non-empty 'gloss' "
                "string in <<SOURCE_LANGUAGE>>"
            )
        if _gloss_contains_blank(gloss):
            raise ModelOutputError(
                f"story paragraph {idx} gloss contains '___' — the "
                "gloss must be the full source-language translation, "
                "not a fill-in-the-blank"
            )
        target_lines.append(target.strip())

    questions = payload.get("comprehension_questions")
    if not isinstance(questions, list):
        raise ModelOutputError(
            "story card must include 'comprehension_questions' as a list"
        )
    if not (STORY_QUESTIONS_MIN <= len(questions) <= STORY_QUESTIONS_MAX):
        raise ModelOutputError(
            f"story 'comprehension_questions' must have "
            f"{STORY_QUESTIONS_MIN}-{STORY_QUESTIONS_MAX} entries; "
            f"got {len(questions)}"
        )
    answers: list[str] = []
    for idx, q in enumerate(questions):
        if not isinstance(q, dict):
            raise ModelOutputError(
                f"story comprehension question {idx} must be an object"
            )
        prompt = q.get("prompt")
        answer = q.get("answer")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ModelOutputError(
                f"story comprehension question {idx} requires "
                "non-empty 'prompt'"
            )
        if not isinstance(answer, str) or not answer.strip():
            raise ModelOutputError(
                f"story comprehension question {idx} requires "
                "non-empty 'answer'"
            )
        answers.append(answer.strip())

    # Title is optional but recommended; the frontend uses it as the
    # story header.
    if "title" in payload and not isinstance(payload["title"], str):
        raise ModelOutputError(
            "story 'title' must be a string when present"
        )

    # Synthesise prompt_text from the first paragraph (used by the SRS
    # picker's prompt_text lookups + by message rendering as a thumb
    # summary) and reference_answer from the pipe-joined comprehension
    # answers (used by the grader and the message-shape contract that
    # exercise cards carry a single reference_answer string).
    payload["prompt_text"] = (
        payload.get("title") or target_lines[0][:160]
    )
    payload["reference_answer"] = " | ".join(answers)


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
    elif ct == CARD_STORY:
        # Stories carry content in paragraphs[] + comprehension_questions[],
        # not prompt_text/reference_answer. Validated separately;
        # prompt_text + reference_answer are synthesised so the rest of
        # the persistence + grading pipeline (which expects strings) keeps
        # working transparently.
        _validate_story_payload(payload)
    else:
        if "prompt_text" not in payload:
            raise ModelOutputError(
                "card missing required key: 'prompt_text'"
            )
        if not isinstance(payload["prompt_text"], str) or not payload["prompt_text"].strip():
            raise ModelOutputError("card prompt_text must be a non-empty string")
        # Reject any blank marker leaked into the inline gloss. The
        # gloss is read-only context — a `___` inside `(...)` either
        # paints a phantom input slot on the frontend or confuses the
        # learner about what they're being asked to fill in (Sebastian
        # saw `"Excuse me, where is ___ station?"` and typed `ist`
        # because the English implied a verb-shaped gap).
        if _gloss_contains_blank(payload["prompt_text"]):
            raise ModelOutputError(
                "card prompt_text has a '___' blank inside a "
                "parenthesised gloss — the gloss must spell out the "
                "missing piece in source-language so the learner "
                "reads it as read-only context"
            )

    # All exercise types require a non-empty reference_answer the
    # grader can score against. Lesson cards are exempt (they're
    # acknowledged, not graded). Note: this tightened in Phase 2 —
    # vocab/translation/production used to accept null reference_answer
    # but the grader always needed *something* to score against, so
    # the laxer contract was silently producing worse grading on
    # malformed cards. For production cards the reference_answer is
    # a free-form rubric, not a canonical string.
    # Dialogue cards synthesise ``reference_answer`` from per-turn
    # ``answer`` fields further down — they're allowed to omit a
    # top-level reference_answer because each blank carries its own
    # canonical fill on the turn it belongs to. Stories synthesise
    # ``reference_answer`` from their comprehension_questions during
    # _validate_story_payload (one canonical answer per question,
    # pipe-joined). Every other exercise type still requires the
    # single-string reference_answer here.
    if (ct in EXERCISE_CARD_TYPES
            and ct != CARD_DIALOGUE and ct != CARD_STORY):
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
        if _gloss_contains_blank(src):
            raise ModelOutputError(
                "grammar card source_sentence has '___' inside a "
                "parenthesised gloss — the gloss must spell out the "
                "source-language equivalent of the full target sentence"
            )
    elif ct == CARD_DIALOGUE:
        turns = payload.get("turns")
        if not isinstance(turns, list) or len(turns) < 2:
            raise ModelOutputError(
                "dialogue card requires 'turns' list of at least 2 entries"
            )
        # Multi-slot answers: each turn that contains '___' MUST carry
        # an 'answer' string with the canonical fill. The frontend
        # renders one input per blank, the grader compares each. The
        # composite reference_answer is synthesised from these so older
        # storage paths (SRS replay, history) keep working.
        slot_answers: list[str] = []
        any_blank = False
        for turn in turns:
            if not isinstance(turn, dict):
                raise ModelOutputError("dialogue turn must be an object")
            if not isinstance(turn.get("speaker"), str) or not turn["speaker"].strip():
                raise ModelOutputError(
                    "dialogue turn requires non-empty 'speaker'"
                )
            text = turn.get("text")
            if not isinstance(text, str):
                raise ModelOutputError(
                    "dialogue turn requires 'text' string (may be '___')"
                )
            # Reject blank markers leaked into the gloss. The
            # frontend's split-on-`___` renderer would paint a
            # phantom input inside the parenthesised source-language
            # gloss with no matching answer slot. Catch here before
            # it ever reaches the renderer.
            if _gloss_contains_blank(text):
                raise ModelOutputError(
                    "dialogue turn text has '___' inside a "
                    "parenthesised gloss — the gloss must show the "
                    "missing piece spelled out in source-language"
                )
            if "___" in text:
                any_blank = True
                ans = turn.get("answer")
                if not isinstance(ans, str) or not ans.strip():
                    raise ModelOutputError(
                        "dialogue turn containing '___' requires a "
                        "non-empty 'answer' string with the canonical "
                        "fill for that blank"
                    )
                # Number of blanks per turn must equal what the renderer
                # will draw. Multi-blank single-turn is allowed if the
                # model emits a pipe-joined 'answer' with the same count.
                blanks = text.count("___")
                ans_parts = [p.strip() for p in ans.split("|")]
                if blanks > 1 and len(ans_parts) != blanks:
                    raise ModelOutputError(
                        f"dialogue turn has {blanks} blanks but 'answer' "
                        f"has {len(ans_parts)} pipe-separated parts — "
                        f"emit one per blank in order"
                    )
                slot_answers.extend(ans_parts if blanks > 1 else [ans.strip()])
        if not any_blank:
            raise ModelOutputError(
                "dialogue card requires at least one turn with '___' as "
                "the slot the learner fills"
            )
        # Synthesise the composite reference_answer so the rest of the
        # storage / grading pipeline (which expects a single string)
        # keeps working transparently. The pipe is the slot delimiter
        # the grader expects.
        payload["reference_answer"] = " | ".join(slot_answers)
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


def _format_recent_drills(
    recent: list[dict[str, Any]],
    *,
    header: str,
) -> str:
    """Render a list of recent drills with the given header. Used for
    both the per-topic and per-module variation visibility — same
    shape, different scope, same "don't repeat the example sentence"
    instruction. Empty string when there's nothing to surface."""
    if not recent:
        return ""
    lines = [header]
    for r in recent:
        ct = r.get("card_type") or "?"
        pt = (r.get("prompt_text") or "").strip().replace("\n", " ")
        ra = (r.get("reference_answer") or "").strip().replace("\n", " ")
        if len(pt) > 180:
            pt = pt[:177] + "…"
        if len(ra) > 80:
            ra = ra[:77] + "…"
        lines.append(f"  - {ct}: \"{pt}\" → \"{ra}\"")
    return "\n".join(lines) + "\n"


def _format_recent_topic_drills(recent: list[dict[str, Any]]) -> str:
    """Per-topic header — the strictest variation gate: same topic
    means the author was just here on the previous card_type slot."""
    return _format_recent_drills(
        recent,
        header=("RECENT DRILLS ON THIS TOPIC (do NOT reuse the same "
                "example sentence — pick a different one):"),
    )


def _format_recent_module_drills(recent: list[dict[str, Any]]) -> str:
    """Per-module header — the wider lens. Catches duplicates that
    cross topics within one module ("Thank you very much → Vielen
    Dank" showing up twice in module 1 even though the topics
    differed)."""
    return _format_recent_drills(
        recent,
        header=("RECENT DRILLS IN THIS MODULE (across topics — do NOT "
                "duplicate any of these example sentences in your "
                "new card):"),
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

    For exercise cards we also surface the last few drills already
    emitted on the SAME topic so the author picks a different example
    sentence instead of re-using the canonical one across every
    card_type in the rotation (the "Ich trinke einen Kaffee" bug).
    Lessons skip this — their content is the topic itself, not an
    example sentence."""
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
        recent_block = ""
    else:
        content_brief = (
            f"Produce a {card_type} EXERCISE card drilling this topic. "
            "Use the shape from the skill reference for this card_type "
            "(prompt_text + reference_answer + any per-type extras like "
            "options / tokens / turns / source_sentence). "
            "Output JSON. DO NOT include card_type, module_id, or "
            "topic_id — the coach attaches scaffolding."
        )
        recent_block = ""
        if ctx.goal_id:
            try:
                topic_recent = store.recent_topic_prompts(
                    ctx.goal_id, topic_id, limit=4,
                )
                module_recent = store.recent_module_prompts(
                    ctx.goal_id, module_id, limit=8,
                )
                # Dedup the module list against the topic list — the
                # same drills will appear in both queries since the
                # topic is inside the module. Keep only module-level
                # drills the topic block doesn't already surface.
                topic_keys = {
                    (r.get("card_type"), r.get("prompt_text"))
                    for r in topic_recent
                }
                module_only = [
                    r for r in module_recent
                    if (r.get("card_type"), r.get("prompt_text")) not in topic_keys
                ]
                topic_block = _format_recent_topic_drills(topic_recent)
                module_block = _format_recent_module_drills(module_only)
                recent_block = topic_block + module_block
            except Exception as exc:                            # noqa: BLE001
                # Variation guidance is best-effort — a failure to read
                # recent prompts must NOT block authoring.
                log.warning(
                    "cards: recent prompt lookup failed; continuing "
                    "without variation context. error=%s", exc,
                )
    # Surface the learner's recent error pattern so a noun-rich vocab
    # card prefers nouns the learner keeps getting wrong, a grammar
    # card targets the conjugation they keep slipping on, etc. Top 5
    # tags only; empty when the learner is fresh.
    err_block = ""
    if ctx.error_patterns:
        parts = [
            f"{e.get('tag')}×{e.get('count')}"
            for e in ctx.error_patterns
            if (
                isinstance(e, dict)
                and isinstance(e.get("tag"), str)
                and e.get("tag")
                and e.get("count") is not None
            )
        ]
        if parts:
            err_block = (
                "\nLEARNER'S RECENT ERROR PROFILE (target these "
                "weaknesses where the card_type allows):\n"
                f"  {', '.join(parts)}\n"
            )
    return (
        "LEARNER:\n"
        f"  name: {tag_learner_input(p.get('name'))}\n"
        f"  level: {p.get('current_level') or 'beginner'}\n"
        f"  focus: {tag_learner_input(ctx.goal_title or ctx.goal_context or p.get('objective'))}\n"
        + err_block
        + "\nCARD TO AUTHOR (selected by the coach — these are FIXED, "
        "you don't pick them, you produce content for them):\n"
        f"  card_type: {card_type}\n"
        f"  module: {tag_learner_input(module_title)} (id: {module_id})\n"
        f"  topic:  {tag_learner_input(topic_title)} (id: {topic_id})\n"
        + (f"\n{recent_block}" if recent_block else "")
        + f"\n{content_brief}"
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
            module_id=module_id,
            module_title=module_title,
            topic_id=topic_id,
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
