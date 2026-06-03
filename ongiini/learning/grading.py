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
from .db import (
    CARD_CLOZE,
    CARD_DIALOGUE,
    CARD_GRAMMAR,
    CARD_MULTIPLE_CHOICE,
    CARD_PRODUCTION,
    CARD_PROVERB,
    CARD_REORDER,
    CARD_STORY,
    CARD_TRANSLATION,
    CARD_VOCAB,
    EXERCISE_CARD_TYPES,
    RATINGS,
)
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


_TYPE_RUBRICS: dict[str, str] = {
    CARD_VOCAB: (
        "Vocab: meaning-equivalent answer is 'correct'. Minor spelling "
        "drift on a clear meaning is 'correct' (note the canonical "
        "form). Wrong word entirely is 'wrong'."
    ),
    CARD_TRANSLATION: (
        "Translation: meaning preservation is what matters. Word order "
        "and minor article slips are 'correct' if intelligible. A "
        "wrong tense that changes the meaning is 'partial'. "
        "Untranslated / wrong-language is 'wrong'."
    ),
    CARD_PRODUCTION: (
        "Production: graded against the rubric in reference_answer. "
        "Free-form output that meets the rubric's intent is 'correct'. "
        "Misses one element but otherwise solid is 'partial'."
    ),
    CARD_CLOZE: (
        "Cloze: the answer is the missing word(s). Exact match "
        "(case-insensitive) is 'correct'. A clear typo on the right "
        "word is 'correct' (note the canonical form). Different "
        "word that doesn't fit the slot is 'wrong'."
    ),
    CARD_REORDER: (
        "Reorder: the answer should be the same tokens in the "
        "reference_answer order (case-insensitive, tolerant of "
        "spacing + punctuation). One-token transposition is "
        "'partial'. Major reordering or missing tokens is 'wrong'."
    ),
    CARD_MULTIPLE_CHOICE: (
        "Multiple choice: the learner submits the LABEL of their "
        "pick (A / B / C / D). Exact label match → 'correct'. Any "
        "other label → 'wrong' (no 'partial' for MC). Feedback must "
        "explain why the chosen distractor was wrong AND why the "
        "right one is right, using the option's 'explanation' field."
    ),
    CARD_GRAMMAR: (
        "Grammar: the transformation has to be correct. The right "
        "verb form with one minor word-order issue → 'partial'. "
        "Wrong tense / mood / person → 'wrong'. Be strict on the "
        "exact morphology being drilled; that's the whole point. "
        "IMPORTANT: if the learner's answer is MORE register-coherent "
        "or MORE idiomatic than the reference_answer (e.g. replacing "
        "'Hallo, wie geht es Ihnen?' — informal greeting with formal "
        "pronoun — with 'Guten Tag, wie geht es Ihnen?' when "
        "transforming to formal), score 'correct' and acknowledge the "
        "better choice in feedback. Punishing a learner for natural "
        "register coherence breaks trust. Token-level fidelity to the "
        "reference is NOT the goal — the morphological transformation "
        "being drilled is."
    ),
    CARD_PROVERB: (
        "Proverb: the canonical idiom is the reference. Exact match "
        "(modulo capitalisation / punctuation) → 'correct'. A close "
        "variant that means the same thing → 'partial'. Unrelated "
        "saying → 'wrong'."
    ),
    CARD_STORY: (
        "Story: graded comprehension of a short reading. Reference "
        "answers are pipe-separated ('|') in question order. The "
        "learner's submission arrives in the same shape. Score "
        "**LENIENTLY** — this is comprehensible-input, not retrieval "
        "practice. ANY answer in either <<TARGET_LANGUAGE>> or "
        "<<SOURCE_LANGUAGE>> that captures the gist of the reference "
        "is 'correct'. Synonyms, paraphrases, partial answers that "
        "show the learner understood the gist are 'correct'. Only "
        "score 'partial' if the learner clearly misread one of the "
        "questions; 'wrong' if the response is unrelated to the "
        "story. Feedback must be encouraging — point out what the "
        "learner caught and only briefly note any miss. The point of "
        "stories is to build confidence with new input, not to gate "
        "progress on perfect recall."
    ),
    CARD_DIALOGUE: (
        "Dialogue: a multi-slot fill-in across the conversation. The "
        "reference_answer is the canonical answers for each blank, "
        "pipe-separated ('|') in turn-blank order. The learner's "
        "answer arrives in the same pipe-separated shape, one entry "
        "per blank. Compare slot-by-slot (case-insensitive, tolerant "
        "of minor typos): all slots correct → 'correct'; one or two "
        "slots wrong → 'partial'; majority wrong / wrong-language → "
        "'wrong'. Feedback names which slot was wrong and what the "
        "right form was. If the learner submitted fewer slots than "
        "expected, score what they sent and note the missing ones."
    ),
}

# Safety net: any exercise card type MUST have a dedicated rubric. The
# previous shape used a silent vocab fallback on missing-key — exactly
# the silent-quality-loss pattern we want to avoid. Run at import so a
# typo or a new card_type without a matching rubric is caught at
# startup rather than at grade-time.
_MISSING = set(EXERCISE_CARD_TYPES) - set(_TYPE_RUBRICS)
assert not _MISSING, (
    f"_TYPE_RUBRICS is missing entries for exercise card types: {_MISSING}"
)
del _MISSING


def _build_user_prompt(
    ctx: LearnerContext,
    *,
    card: dict[str, Any],
    user_answer: str,
    hint_used: bool,
) -> str:
    p = ctx.profile or {}
    ct = card.get("card_type") or CARD_VOCAB
    rubric_line = _TYPE_RUBRICS.get(ct)
    if rubric_line is None:
        # Unknown card_type at grade time — log loudly rather than
        # silently scoring with the wrong rubric, then fall back to
        # vocab. The startup assert means this can only fire if the
        # card_type comes from outside CARD_TYPES (e.g. a bug in the
        # validator path or a stored card from a future version).
        log.warning("grading: no rubric for card_type=%r, falling back to vocab", ct)
        rubric_line = _TYPE_RUBRICS[CARD_VOCAB]
    # Surface the per-type structural extras the rubric depends on:
    # multiple_choice options + their explanations, dialogue turns,
    # grammar source sentence, reorder tokens.
    extras: list[str] = []
    if ct == CARD_MULTIPLE_CHOICE:
        opts = card.get("options") or []
        if isinstance(opts, list):
            for o in opts:
                if not isinstance(o, dict):
                    continue
                expl = o.get("explanation") or "(no explanation)"
                extras.append(
                    f"  option {o.get('label')}: {o.get('text')!r} — {expl}"
                )
    elif ct == CARD_DIALOGUE:
        # Surface each turn with its expected answer when present so
        # the grader can name which slot was wrong in the feedback —
        # the rubric above grades pipe-separated multi-slot answers,
        # but without the per-turn map the LLM can't tell the learner
        # WHICH blank they got wrong.
        for t in card.get("turns") or []:
            if not isinstance(t, dict):
                continue
            ans = t.get("answer")
            ans_tail = f"  [answer: {ans!r}]" if isinstance(ans, str) and ans.strip() else ""
            extras.append(f"  {t.get('speaker')}: {t.get('text')}{ans_tail}")
    elif ct == CARD_STORY:
        # Show the grader the actual story so it can judge
        # comprehension leniently — the rubric assumes the grader has
        # read what the learner read. Then surface each question + its
        # canonical answer so the slot-by-slot match works.
        title = card.get("title")
        if isinstance(title, str) and title.strip():
            extras.append(f"  story title: {title}")
        for i, p in enumerate(card.get("paragraphs") or []):
            if not isinstance(p, dict):
                continue
            extras.append(
                f"  paragraph {i+1}: {p.get('target')} {p.get('gloss')}"
            )
        for i, q in enumerate(card.get("comprehension_questions") or []):
            if not isinstance(q, dict):
                continue
            extras.append(
                f"  question {i+1}: {q.get('prompt')!r} "
                f"[canonical answer: {q.get('answer')!r}]"
            )
    elif ct == CARD_GRAMMAR:
        extras.append(f"  source_sentence: {card.get('source_sentence')}")
    elif ct == CARD_REORDER:
        tokens = card.get("tokens") or []
        extras.append(f"  tokens: {tokens}")
    extras_block = ("\nCARD EXTRAS:\n" + "\n".join(extras)) if extras else ""
    return (
        "LEARNER:\n"
        f"  level: {p.get('current_level') or 'beginner'}\n"
        f"  focus: {tag_learner_input(ctx.goal_title or ctx.goal_context or p.get('objective'))}\n"
        "\nCARD:\n"
        f"  card_type: {ct}\n"
        f"  prompt_text: {card.get('prompt_text')}\n"
        f"  reference_answer: {card.get('reference_answer') or '(none)'}\n"
        f"  hint_used: {hint_used}"
        f"{extras_block}\n"
        f"\nRUBRIC FOR THIS CARD TYPE:\n  {rubric_line}\n"
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
