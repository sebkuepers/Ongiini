"""Conversation mode — the free-chat side of the learn surface.

Track C of the quality roadmap: a learner can switch from "Cards" to
"Chat" mode and talk to the coach in the TARGET LANGUAGE at their
CEFR level. The coach replies in target language, calibrates to the
learner's level + stated focus + recent error profile, and surfaces
a small Notes block (1-3 corrections + 1-3 new high-frequency words)
after each turn.

This is the one thing AI can do that Duolingo fundamentally can't —
genuine open-ended target-language conversation calibrated to the
individual learner. Compounds with stories (Track A): the learner
reads a bakery story, then can practice ordering at the bakery in
chat mode.

Architecture mirrors the per-card-critic flow in
``card_critic.py``:
  * dedicated module with a system prompt, response shape, and
    structured-output parser
  * the API endpoint wires it; the coach.run_turn flow is NOT
    involved (chat is a separate surface)
  * persistence uses ``messages.append`` with the new MSG_CHAT_*
    kinds so the chat rehydrates from the goal on reload
"""
from __future__ import annotations

import json as _json
import logging
from dataclasses import dataclass, field
from typing import Any

from owela import Model

from .context import LearnerContext
from .llm import INJECTION_GUARD_LINE, ask_for_json, tag_learner_input

log = logging.getLogger("ongiini.learning.conversation")


# Cap per-entry length on corrections + new words so a runaway model
# can't blow the next turn's context window. The Notes block is
# advisory, not load-bearing — being terse is fine.
_MAX_NOTE_ITEM_LEN = 240
_MAX_CORRECTIONS = 3
_MAX_NEW_WORDS = 3
# How many recent chat turns to feed back so the conversation has
# context. Chat is the surface where context-carry is the AI's
# differentiator vs Duolingo, so we lean larger here than a critic
# loop would. 20 entries ≈ 10 exchanges — enough to keep a "ok let's
# rehearse checking into a hotel" scenario coherent across the kind
# of mid-length conversation an A0-B1 learner can sustain. A
# character-budget guard caps the cumulative size at ~6000 chars so
# a paste-bomb in one turn can't blow the prompt window even within
# the entry cap.
_HISTORY_WINDOW = 20
_HISTORY_CHAR_BUDGET = 6000
# Cap the learner's free-form turn so an over-eager paste doesn't
# blow the prompt budget. 800 chars ≈ 150 words, well above any
# A0-B1 single-turn output.
_MAX_LEARNER_TURN_CHARS = 800


@dataclass(frozen=True)
class ConversationTurn:
    """One coach reply: the bubble the learner reads, plus the
    structured Notes block the frontend renders below it."""
    reply: str
    corrections: list[dict[str, str]] = field(default_factory=list)
    new_words: list[dict[str, str]] = field(default_factory=list)


def _build_system_prompt(
    ctx: LearnerContext,
    *,
    skill_content: str,
) -> str:
    """The coach's system prompt for conversation mode. The full
    skill reference is embedded so the coach has the same CEFR
    calibration + native-speaker authenticity gate the cards use,
    plus a conversation-specific persona on top."""
    p = ctx.profile or {}
    level = p.get("current_level") or "beginner"
    name = p.get("name") or ""
    focus = (
        ctx.goal_title or ctx.goal_context or p.get("objective")
        or "general practice"
    )
    return (
        "You are a friendly language tutor having a free-form "
        "conversation with a single learner. You will reply ONLY in "
        f"<<TARGET_LANGUAGE>> at the learner's level ({level}) using "
        "the CEFR calibration in the skill reference below. The "
        "conversation is bound to a single learner with a specific "
        "focus — anchor your replies in that focus:\n"
        f"  - learner name (if any): {tag_learner_input(name) or '(not given)'}\n"
        f"  - learner level: {level}\n"
        f"  - focus: {tag_learner_input(focus)}\n\n"
        "CONVERSATIONAL POSTURE:\n"
        " - Reply in <<TARGET_LANGUAGE>> ONLY. Do not write any\n"
        "   <<SOURCE_LANGUAGE>> in the `reply` field. Code-switching\n"
        "   breaks the immersion.\n"
        " - Match the learner's level. Use only vocabulary + grammar\n"
        "   the learner can decode (~90% known, per Krashen i+1).\n"
        "   At A0/A1 use simple present tense, basic vocab, short\n"
        "   sentences. At intermediate, paragraph-level scaffolding\n"
        "   is fine.\n"
        " - 1-3 sentences per reply for A0-A1, up to 5 for B1+. Don't\n"
        "   lecture — invite the next turn.\n"
        " - Lead with content. If the learner asked a question or\n"
        "   shared something, respond to it before pivoting. Don't\n"
        "   robotically ask 'how was your day?' every turn.\n"
        " - Anchor to the focus when it's natural. A learner whose\n"
        "   focus is 'travel to Germany' can practice ordering at\n"
        "   the bakery, asking for train tickets, checking into a\n"
        "   hotel. Don't force it though.\n"
        " - Use natural <<TARGET_LANGUAGE>>. Same native-speaker\n"
        "   authenticity gate as cards: no register-incoherent or\n"
        "   grammatically-possible-but-unsaid constructions.\n\n"
        "OUTPUT SHAPE:\n"
        "You output ONLY the JSON object — no prose, no fences:\n"
        '{ "reply":      "<your <<TARGET_LANGUAGE>> reply>",\n'
        '  "corrections":[\n'
        '    {"learner": "<exact phrase from learner>",\n'
        '     "correct": "<the canonical version>",\n'
        '     "note":    "<one-sentence <<SOURCE_LANGUAGE>> "\n'
        '                "explanation of what was off>"}\n'
        "  ],\n"
        '  "new_words":  [\n'
        '    {"word":    "<<TARGET_LANGUAGE>> word the learner saw",\n'
        '     "meaning": "<<SOURCE_LANGUAGE>> meaning"}\n'
        "  ]\n"
        "}\n"
        "RULES FOR THE NOTES BLOCKS:\n"
        f" - corrections: 0-{_MAX_CORRECTIONS} entries. Only flag\n"
        "   things that genuinely affect intelligibility or are core\n"
        "   to the level (sein/haben at A0, perfekt at A2, case\n"
        "   marking at B1). Don't pile on stylistic nitpicks.\n"
        f" - new_words: 0-{_MAX_NEW_WORDS} entries. Pick high-\n"
        "   frequency words from your reply the learner probably\n"
        "   didn't know. Skip if your reply only used words the\n"
        "   learner already has (early in the curriculum). Each\n"
        "   word's meaning is a short <<SOURCE_LANGUAGE>> gloss.\n"
        " - corrections and new_words are advisory — empty is fine\n"
        "   when the learner's input was clean and your reply only\n"
        "   used known vocabulary.\n"
        f"{INJECTION_GUARD_LINE}\n"
        "\n"
        f"{skill_content}"
    )


def _build_user_prompt(
    *,
    user_text: str,
    history: list[dict[str, Any]],
) -> str:
    """Render the conversation so far + the learner's new turn. The
    history is a flat list of ``{role, text}`` dicts oldest-first;
    only the last _HISTORY_WINDOW entries are surfaced so the
    prompt budget stays predictable as the conversation grows."""
    parts: list[str] = []
    # Take the last _HISTORY_WINDOW entries, then walk backwards and
    # trim until the cumulative char count fits the budget. We drop
    # from the OLDEST end first so the most recent context (which is
    # most relevant) is always preserved.
    candidate = history[-_HISTORY_WINDOW:] if history else []
    recent: list[dict[str, Any]] = []
    char_budget = _HISTORY_CHAR_BUDGET
    for entry in reversed(candidate):
        text_len = len(str(entry.get("text") or ""))
        if text_len > char_budget:
            break
        recent.insert(0, entry)
        char_budget -= text_len
    if recent:
        parts.append("CONVERSATION SO FAR (oldest first):")
        for entry in recent:
            role = entry.get("role")
            text = entry.get("text") or ""
            if not text:
                continue
            label = "coach" if role == "coach" else "learner"
            parts.append(f"  {label}: {tag_learner_input(text)}")
        parts.append("")
    parts.append("LEARNER'S NEW TURN:")
    parts.append(f"  {tag_learner_input(user_text)}")
    parts.append("")
    parts.append(
        "Reply in <<TARGET_LANGUAGE>> at the learner's level. Return "
        "the JSON shape (reply / corrections / new_words) from the "
        "system instructions."
    )
    return "\n".join(parts)


def _coerce_note_entry(entry: Any, required_keys: tuple[str, ...]) -> dict[str, str] | None:
    """Validate one notes entry. Returns the trimmed dict or None
    when the shape is malformed (caller drops Nones)."""
    if not isinstance(entry, dict):
        return None
    out: dict[str, str] = {}
    for key in required_keys:
        value = entry.get(key)
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        out[key] = text[:_MAX_NOTE_ITEM_LEN]
    return out


def _validate_turn(payload: dict[str, Any]) -> ConversationTurn:
    """Parse the LLM's JSON into a ConversationTurn. Tolerant of
    missing notes blocks (they're advisory); strict on the reply
    being a non-empty string (without that there's nothing to show)."""
    if "error" in payload:
        # Soft-fail upstream — return an empty reply marker rather
        # than raising; the API can surface a friendly retry text.
        return ConversationTurn(reply="")
    reply = payload.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        return ConversationTurn(reply="")
    corrections: list[dict[str, str]] = []
    raw_corr = payload.get("corrections")
    if isinstance(raw_corr, list):
        for entry in raw_corr[:_MAX_CORRECTIONS]:
            coerced = _coerce_note_entry(entry, ("learner", "correct", "note"))
            if coerced is not None:
                corrections.append(coerced)
    new_words: list[dict[str, str]] = []
    raw_nw = payload.get("new_words")
    if isinstance(raw_nw, list):
        for entry in raw_nw[:_MAX_NEW_WORDS]:
            coerced = _coerce_note_entry(entry, ("word", "meaning"))
            if coerced is not None:
                new_words.append(coerced)
    return ConversationTurn(
        reply=reply.strip(),
        corrections=corrections,
        new_words=new_words,
    )


async def chat_turn(
    ctx: LearnerContext,
    *,
    user_text: str,
    history: list[dict[str, Any]],
    model: Model,
    skill_content: str,
) -> ConversationTurn:
    """Drive one conversation turn. Returns the coach's reply +
    structured Notes block.

    Soft-fails: a ModelOutputError or empty reply yields a
    ConversationTurn(reply='') — the API layer surfaces a friendly
    "couldn't reply, try again" text. No exception escapes — chat
    mode must not crash the surface."""
    text = (user_text or "").strip()[:_MAX_LEARNER_TURN_CHARS]
    if not text:
        return ConversationTurn(reply="")
    try:
        payload = await ask_for_json(
            system_prompt=_build_system_prompt(ctx, skill_content=skill_content),
            user_prompt=_build_user_prompt(user_text=text, history=history),
            model=model,
        )
    except Exception as exc:                                # noqa: BLE001
        log.warning("conversation: model call failed: %s", exc)
        return ConversationTurn(reply="")
    return _validate_turn(payload)


def serialise_turn(turn: ConversationTurn) -> dict[str, Any]:
    """Render a ConversationTurn for persistence + transport. Used
    by the API to build the chat_coach + chat_notes message payloads."""
    return {
        "reply": turn.reply,
        "corrections": turn.corrections,
        "new_words": turn.new_words,
    }


def build_history_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the persisted MSG_CHAT_* rows into the [{role, text}]
    history shape the system prompt expects. Filters out the notes
    block (advisory; not part of the conversational history) and
    keeps only the chat-mode kinds (so a goal that's mostly cards
    doesn't bleed lesson + exercise context into the chat memory)."""
    from .db import MSG_CHAT_COACH, MSG_CHAT_LEARNER
    out: list[dict[str, Any]] = []
    for row in messages:
        kind = row.get("kind")
        payload = row.get("payload") or {}
        if kind == MSG_CHAT_LEARNER:
            text = payload.get("text") or ""
            if text:
                out.append({"role": "learner", "text": text})
        elif kind == MSG_CHAT_COACH:
            text = payload.get("reply") or payload.get("text") or ""
            if text:
                out.append({"role": "coach", "text": text})
    return out
