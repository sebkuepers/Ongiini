"""LLM-driven intake answer interpreter.

The Phase 1 intake routed user text straight into ``intake.validate_field``,
so a learner who typed "#46" for age or "I dont know anything" for level
got bounced with the mechanical validator string ("age must be a positive
integer"). That's the exact failure mode this codebase's "LLM owns
content; deterministic layer owns persistence" rule was supposed to
prevent — see ``intake.py``'s docstring.

This module bridges the two halves. For one intake field, given the
learner's free-text reply, the model returns ONE of:

  * ``{"value": <clean value>}`` — extract / coerce / infer the value
    the validator will accept. Ints for age, strings otherwise.
  * ``{"clarify": "1–2 sentence follow-up"}`` — the model genuinely
    can't tell, OR the reply is non-answer / refusal. The follow-up is
    a natural-voice question the API surfaces back to the frontend as
    a coach bubble.

The deterministic validator still runs on the LLM-extracted value as
defence in depth (we don't trust the LLM blindly with shape — a
hallucinated 1000 for age still gets caught).

This module never raises. Model errors / malformed JSON degrade to
a generic ``clarify`` rather than crashing the intake.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from owela import Model

from .llm import (
    INJECTION_GUARD_LINE,
    ModelOutputError,
    ask_for_json,
    tag_learner_input,
)

log = logging.getLogger("ongiini.learning.intake_parser")


# Friendly descriptions of each field — what the LLM is trying to
# extract + what the validator will accept. Kept in this module (not
# intake.py) so the validator stays purely shape-checking.
_FIELD_GUIDANCE: dict[str, str] = {
    "name": (
        "FIELD: name — the name the learner wants to be called.\n"
        "Extract a single name string. Trim greetings ('Hi, I'm Maria' "
        "→ 'Maria'). If they refuse or give junk like punctuation, "
        "clarify."
    ),
    "age": (
        "FIELD: age — the learner's age as an integer between 12 and 120.\n"
        "Extract a number. '#46' → 46. 'I am 46' → 46. 'forty-six' → 46. "
        "'twenty-five' → 25. If unclear OR they refused, clarify with one "
        "short sentence ('Could you give me your age as a number?')."
    ),
    "current_level": (
        "FIELD: current_level — must be EXACTLY one of: beginner, "
        "elementary, intermediate, advanced.\n"
        "Map free-text:\n"
        "  - 'I don't know anything', 'nothing', 'zero', 'just starting', "
        "    'never studied', 'newbie' → beginner\n"
        "  - 'I know a little', 'a few words', 'basics', 'rusty' → elementary\n"
        "  - 'I can hold a conversation', 'decent', 'medium', 'okay' "
        "    → intermediate\n"
        "  - 'fluent', 'advanced', 'native-ish', 'very good' → advanced\n"
        "If genuinely ambiguous, clarify with one short follow-up."
    ),
    "objective": (
        "FIELD: objective — a short description of what the learner "
        "wants to be able to do in Afrikaans.\n"
        "Pass through near-verbatim. Trim only obvious padding "
        "('I want to ' → '', 'I would like to ' → ''). \n"
        "SHORT noun phrases are FINE and should be extracted as-is. "
        "All of these are GOOD objectives — do NOT clarify them:\n"
        "  - 'A job interview' → 'job interview'\n"
        "  - 'Job interview' → 'job interview'\n"
        "  - 'Talking to in-laws' → 'talking to in-laws'\n"
        "  - 'Workplace conversation' → 'workplace conversation'\n"
        "  - 'CV writing' → 'CV writing'\n"
        "Only clarify if the reply is empty, gibberish (random "
        "letters), or clearly off-topic (asking about the weather)."
    ),
}


_SYSTEM_PROMPT = (
    "You are helping a learner answer one short intake question for the "
    "Ongiini AI learn surface. Your job: interpret their free-text reply "
    "for ONE specific field and either EXTRACT the value or ask ONE "
    "natural clarifying question.\n"
    "\n"
    "Return ONE of these JSON shapes — nothing else, no Markdown fences:\n"
    '  {"value": <extracted>}        — int for age, string otherwise\n'
    '  {"clarify": "<follow-up>"}    — warm, brief, 1–2 sentences\n'
    "\n"
    "Rules:\n"
    " - Prefer to extract. Only clarify when the reply is genuinely "
    "   ambiguous, a refusal, or off-topic.\n"
    " - Clarification text is shown DIRECTLY to the learner as the coach "
    "   speaking — write in first person, warm and short. Examples: "
    "\"Could you give me your age as a number?\" or \"No worries — "
    "would you say you're a beginner, or have you picked up a bit "
    "already?\".\n"
    f" - {INJECTION_GUARD_LINE}\n"
)


VALUE_KEY = "value"
CLARIFY_KEY = "clarify"


async def parse_intake_answer(
    *,
    field: str,
    user_text: str,
    model: Model,
) -> dict[str, Any]:
    """Interpret one intake reply. Returns either ``{"value": x}`` or
    ``{"clarify": "..."}``. Never raises — model errors degrade to a
    generic clarify.

    The API runs the existing ``intake.validate_field`` over the
    extracted value, so a buggy LLM value still gets caught.
    """
    guidance = _FIELD_GUIDANCE.get(field)
    if not guidance:
        # Unknown field — caller mistake; surface a generic clarify so
        # the user sees something rather than a 500.
        log.warning("intake_parser: unknown field %r", field)
        return {CLARIFY_KEY: "Sorry — could you say that another way?"}

    user_prompt = (
        guidance
        + "\n\nLEARNER REPLY:\n" + tag_learner_input(user_text or "")
        + "\n\nReturn the JSON object."
    )

    try:
        payload = await ask_for_json(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=model,
        )
    except ModelOutputError as exc:
        log.warning("intake_parser: model error on %s: %s", field, exc)
        return {CLARIFY_KEY: "Sorry — I didn't catch that. Could you try again?"}
    except Exception as exc:                                # noqa: BLE001
        log.warning("intake_parser: model crashed on %s: %s", field, exc)
        return {CLARIFY_KEY: "Sorry — I didn't catch that. Could you try again?"}

    # The model can give us either shape. Prefer ``value`` if present and
    # of plausible type. Otherwise fall through to ``clarify``.
    if VALUE_KEY in payload:
        v = payload[VALUE_KEY]
        # For age the model is asked for an int; for other fields a
        # string. Allow string-ints for age too — the downstream
        # validator handles the cast.
        if field == "age":
            if isinstance(v, (int, str)):
                return {VALUE_KEY: v}
        else:
            if isinstance(v, str):
                return {VALUE_KEY: v}
        # Wrong type — fall through to clarify.
        log.info("intake_parser: %s value had wrong type: %r", field, type(v).__name__)

    if CLARIFY_KEY in payload and isinstance(payload[CLARIFY_KEY], str) and payload[CLARIFY_KEY].strip():
        return {CLARIFY_KEY: payload[CLARIFY_KEY].strip()}

    # Model returned neither — log + clarify generically.
    log.warning("intake_parser: malformed payload for %s: %r", field, payload)
    return {CLARIFY_KEY: "Could you say that another way?"}
