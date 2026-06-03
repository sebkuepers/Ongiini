"""The LLM-driven coach orchestrator — Phase 2's central brain.

When the frontend POSTs to ``/v1/learn/turn`` with the learner's text
(or nothing, meaning "give me what's next"), this module decides:

  * Was it an answer to the active card?      → grade + advance
  * Was it a question to the coach?            → respond with text
  * Was it off-topic?                          → politely redirect
  * Was it a "what's next" with no active card?→ design outline if
                                                 needed + author the
                                                 next card

Every decision is persisted to ``learner_messages`` so the chat thread
is the system of record. The frontend just renders the messages this
returns.

Cardinal design rule: the LLM owns content; this module owns routing
and persistence. The classifier (``turn_classifier``) is itself an
LLM call but it's bounded to a 3-way enum — no free-text output that
would be hard to reason about.
"""
from __future__ import annotations

import logging
from typing import Any

from owela import Model

from . import cards as cards_mod
from . import context as ctx_mod
from . import curriculum
from . import grading
from . import messages
from . import selector
from . import store
from . import turn_classifier
from .db import (
    EXERCISE_CARD_TYPES,
    MSG_COACH_TEXT,
    MSG_EXERCISE,
    MSG_FEEDBACK,
    MSG_LEARNER_TEXT,
    MSG_LESSON,
    MSG_PROGRESS,
)
from .llm import INJECTION_GUARD_LINE, ModelOutputError, ask_for_json, tag_learner_input

log = logging.getLogger("ongiini.learning.coach")


_RECENT_TEXT_PAIRS = 4
# Soft cap on coach answer length. 600 chars ≈ 3-4 sentences with an
# example, which is what the system prompt asks for. The earlier
# 280-char cap silently clipped useful explanations — a violation of
# the "no silent quality loss" rule. Trims on a sentence boundary
# when possible.
_QUESTION_MAX_OUTPUT = 600
# Cap on how many modules from the outline we surface to the
# question-handler. Most curriculums have 3-6; 10 leaves headroom.
_QUESTION_MAX_MODULES = 10


def _error_coach_payload(text: str, error_code: str) -> dict[str, Any]:
    """Build a ``coach_text`` payload that carries an error code in a
    meta field. The frontend renders the text inline like any other
    coach message but can also surface a retry affordance or count
    error rate from the meta. Backwards-compatible: clients that
    don't read meta just see the text."""
    return {"text": text, "meta": {"error": error_code}}


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

async def run_turn(
    *,
    learner_id: str,
    goal_id: str,
    user_text: str | None,
    model: Model,
    skill_content: str,
) -> list[dict[str, Any]]:
    """Process one learner turn. Returns the NEW messages appended to
    the thread (the API hands these straight back to the frontend).

    ``user_text=None`` (or empty) means "give me what's next" — used
    on the very first turn after a goal is created and after each
    feedback so the next card surfaces without an extra click.
    """
    if not learner_id or not goal_id:
        raise ValueError("learner_id and goal_id are required")

    ctx = ctx_mod.build_learner_context(learner_id, goal_id=goal_id)
    active_exercise = messages.latest_unanswered_exercise(
        learner_id=learner_id, goal_id=goal_id,
    )

    if user_text and user_text.strip():
        return await _handle_learner_text(
            learner_id=learner_id, goal_id=goal_id,
            user_text=user_text,
            active_exercise=active_exercise,
            ctx=ctx, model=model, skill_content=skill_content,
        )

    # No text — "give me what's next". If there's an unanswered
    # exercise we don't surface another card (the learner is mid-card),
    # we just return nothing and the frontend keeps its current state.
    if active_exercise:
        return []

    return await _produce_next_thing(
        learner_id=learner_id, goal_id=goal_id,
        ctx=ctx, model=model, skill_content=skill_content,
        prefix_messages=[],
    )


# ──────────────────────────────────────────────────────────────────
# Routing: learner sent text
# ──────────────────────────────────────────────────────────────────

async def _handle_learner_text(
    *,
    learner_id: str,
    goal_id: str,
    user_text: str,
    active_exercise: dict[str, Any] | None,
    ctx: ctx_mod.LearnerContext,
    model: Model,
    skill_content: str,
) -> list[dict[str, Any]]:
    new_messages: list[dict[str, Any]] = []

    # Snapshot the conversation BEFORE persisting the learner message,
    # so the classifier doesn't see the same text twice (once in
    # LEARNER'S MESSAGE, once as the latest "learner:" line in
    # RECENT CONVERSATION). The just-appended message gets surfaced
    # via the user_text input instead.
    active_card_payload = (
        active_exercise.get("payload") if active_exercise else None
    )
    recent_pairs = messages.recent_text_pairs(
        learner_id=learner_id, goal_id=goal_id,
        max_pairs=_RECENT_TEXT_PAIRS,
    )

    # Persist the learner's message so it shows up in the thread
    # regardless of how the rest of the routing goes.
    learner_msg = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=MSG_LEARNER_TEXT, payload={"text": user_text},
    )
    new_messages.append(learner_msg)

    verdict = await turn_classifier.classify_turn(
        user_text=user_text,
        active_card=active_card_payload,
        recent_text_pairs=recent_pairs,
        model=model,
    )
    log.info(
        "coach: verdict=%s learner=%s goal=%s has_card=%s",
        verdict, learner_id[:8], goal_id[:8], bool(active_exercise),
    )

    if verdict == turn_classifier.VERDICT_ANSWER and active_exercise:
        new_messages.extend(await _handle_answer(
            learner_id=learner_id, goal_id=goal_id,
            user_text=user_text, active_exercise=active_exercise,
            ctx=ctx, model=model, skill_content=skill_content,
        ))
    elif verdict == turn_classifier.VERDICT_QUESTION:
        new_messages.extend(await _handle_question(
            learner_id=learner_id, goal_id=goal_id,
            user_text=user_text, ctx=ctx,
            recent_pairs=recent_pairs,
            model=model, skill_content=skill_content,
        ))
    elif verdict == turn_classifier.VERDICT_OFF_TOPIC:
        new_messages.append(_emit_off_topic_redirect(
            learner_id=learner_id, goal_id=goal_id, ctx=ctx,
        ))
    else:
        # answer-verdict with NO active exercise — defensive fallback.
        # The classifier shouldn't emit ANSWER without an active card,
        # but if it does we treat it as a question.
        new_messages.extend(await _handle_question(
            learner_id=learner_id, goal_id=goal_id,
            user_text=user_text, ctx=ctx,
            recent_pairs=recent_pairs,
            model=model, skill_content=skill_content,
        ))

    return new_messages


# ──────────────────────────────────────────────────────────────────
# Verdict: ANSWER — grade + advance
# ──────────────────────────────────────────────────────────────────

async def _handle_answer(
    *,
    learner_id: str,
    goal_id: str,
    user_text: str,
    active_exercise: dict[str, Any],
    ctx: ctx_mod.LearnerContext,
    model: Model,
    skill_content: str,
) -> list[dict[str, Any]]:
    new_messages: list[dict[str, Any]] = []

    # Atomic claim — only ONE concurrent caller wins this; everyone else
    # bails out before the model call so we don't double-grade, double-
    # advance Leitner, or duplicate attempt rows. The claim flips
    # answered 0→1 in the same SQL statement that checks it.
    if not messages.claim_exercise(active_exercise["message_id"]):
        log.info(
            "coach: exercise already claimed by another caller learner=%s card=%s",
            learner_id[:8], active_exercise["card_id"],
        )
        # Don't surface anything — the other caller will write the
        # feedback + progress. The frontend's optimistic learner_msg
        # is already in place.
        return new_messages

    card = store.get_card(active_exercise["card_id"])
    if not card:
        # Card was deleted between us claiming the exercise and now
        # (rare). Surface a coach text + move on.
        log.warning(
            "coach: active exercise references missing card %s",
            active_exercise["card_id"],
        )
        msg = messages.append(
            learner_id=learner_id, goal_id=goal_id,
            kind=MSG_COACH_TEXT,
            payload=_error_coach_payload(
                "Hmm, lost track of that card — let's keep going.",
                error_code="card_missing",
            ),
        )
        new_messages.append(msg)
        new_messages.extend(await _produce_next_thing(
            learner_id=learner_id, goal_id=goal_id,
            ctx=ctx, model=model, skill_content=skill_content,
            prefix_messages=[],
        ))
        return new_messages

    try:
        grade = await grading.grade_answer(
            ctx, card=card,
            user_answer=user_text,
            hint_used=False,    # frontend tracks this; Phase 2 doesn't expose yet
            model=model, skill_content=skill_content,
        )
    except ModelOutputError as exc:
        log.warning("coach: grade_answer failed: %s", exc)
        msg = messages.append(
            learner_id=learner_id, goal_id=goal_id,
            kind=MSG_COACH_TEXT,
            payload=_error_coach_payload(
                "I had trouble grading that just now — could you try again?",
                error_code="grading_failed",
            ),
        )
        new_messages.append(msg)
        return new_messages

    # Persist the attempt + advance Leitner state.
    attempt = store.record_attempt(
        learner_id=learner_id, card_id=card["card_id"],
        user_answer=user_text, ai_feedback=grade["feedback"],
        rating=grade["rating"], hint_used=False,
    )

    # Feedback message — the coloured callout in the thread.
    feedback_msg = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=MSG_FEEDBACK,
        payload={"rating": grade["rating"], "feedback": grade["feedback"]},
        card_id=card["card_id"],
    )
    new_messages.append(feedback_msg)

    # Progress chip — small "now in box N · X cards · Y% right" stamp.
    progress_payload = store.progress_for(learner_id)
    progress_payload["new_box"] = attempt["new_box"]
    progress_msg = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=MSG_PROGRESS, payload=progress_payload,
        card_id=card["card_id"],
    )
    new_messages.append(progress_msg)

    # And the next thing — lesson or exercise — so the learner has
    # something to do without an extra click. ``exclude_card_id`` is
    # the card we just graded; the SRS replay path uses this so the
    # learner doesn't see the exact card they just answered come back
    # back-to-back (Anki-style "Again" lets it surface after one new
    # card).
    new_messages.extend(await _produce_next_thing(
        learner_id=learner_id, goal_id=goal_id,
        ctx=ctx_mod.build_learner_context(learner_id, goal_id=goal_id),
        model=model, skill_content=skill_content,
        prefix_messages=[],
        exclude_card_id=card["card_id"],
    ))
    return new_messages


# ──────────────────────────────────────────────────────────────────
# Verdict: QUESTION — on-topic coach response
# ──────────────────────────────────────────────────────────────────

async def _handle_question(
    *,
    learner_id: str,
    goal_id: str,
    user_text: str,
    ctx: ctx_mod.LearnerContext,
    recent_pairs: list[dict[str, Any]],
    model: Model,
    skill_content: str,
) -> list[dict[str, Any]]:
    answer_text, error_code = await _coach_respond_to_question(
        ctx=ctx, user_text=user_text,
        recent_pairs=recent_pairs,
        model=model, skill_content=skill_content,
    )
    payload: dict[str, Any] = {"text": answer_text}
    if error_code:
        payload["meta"] = {"error": error_code}
    msg = messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=MSG_COACH_TEXT, payload=payload,
    )
    return [msg]


async def _coach_respond_to_question(
    *,
    ctx: ctx_mod.LearnerContext,
    user_text: str,
    recent_pairs: list[dict[str, Any]],
    model: Model,
    skill_content: str,
) -> tuple[str, str | None]:
    """Ask the model to answer the learner's on-topic question in 1-3
    sentences. Returns ``(text, error_code)`` — ``error_code`` is
    ``None`` on success and a short string on fallback so the API can
    surface it via the message ``meta``."""
    from .skill_renderer import LANGUAGE_DISPLAY
    target_name = LANGUAGE_DISPLAY.get(
        ctx.target_language, ctx.target_language.title(),
    )
    system_prompt = (
        f"You are the learner's {target_name} coach. Answer their question "
        "directly and plainly. 1-3 sentences. If a quick example helps, "
        "include one. Don't lecture; the learner will see more cards on "
        "this pattern. Stay focused on the curriculum and the language. "
        f"{INJECTION_GUARD_LINE} "
        'Reply with ONLY a JSON object: {"text": "your reply"}'
    )
    p = ctx.profile or {}
    parts = [
        "LEARNER:",
        f"  level: {p.get('current_level') or 'beginner'}",
        f"  objective: {tag_learner_input(ctx.goal_context or p.get('objective'))}",
    ]
    # Surface the curriculum modules to the question handler. Summary
    # alone is too thin for 'where am I in the plan?' style questions,
    # which are common. Cap at _QUESTION_MAX_MODULES so a sprawling
    # outline doesn't blow up the prompt.
    if ctx.curriculum_outline:
        parts.append("\nCURRICULUM SUMMARY:")
        parts.append(f"  {ctx.curriculum_outline.get('summary', '')}")
        modules = ctx.curriculum_outline.get("modules") or []
        if modules:
            parts.append("\nMODULES:")
            for m in modules[:_QUESTION_MAX_MODULES]:
                if not isinstance(m, dict):
                    continue
                status = m.get("status") or "not_started"
                title = m.get("title") or "(untitled)"
                parts.append(f"  - [{status}] {title}")
    if recent_pairs:
        parts.append("\nRECENT CONVERSATION:")
        for m in recent_pairs[-4:]:
            role = "coach" if m.get("kind") == MSG_COACH_TEXT else "learner"
            txt = (m.get("payload") or {}).get("text", "")
            parts.append(f"  {role}: {tag_learner_input(txt)}")
    parts.append(f"\nLEARNER'S QUESTION:\n  {tag_learner_input(user_text)}")
    user_prompt = "\n".join(parts)

    try:
        payload = await ask_for_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
        )
    except ModelOutputError as exc:
        log.warning("coach: question response failed: %s", exc)
        return (
            "Let me try to come back to that — for now, "
            "want to keep going with the next card?",
            "question_failed",
        )

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return (
            "Let me try to come back to that — for now, "
            "want to keep going with the next card?",
            "question_empty",
        )
    # Cap length so a long-winded model response doesn't dominate the
    # thread. We trim on a sentence boundary if possible — never a hard
    # mid-word cut.
    text = text.strip()
    if len(text) > _QUESTION_MAX_OUTPUT:
        cutoff = text.rfind(". ", 0, _QUESTION_MAX_OUTPUT)
        text = text[:cutoff + 1] if cutoff > 0 else text[:_QUESTION_MAX_OUTPUT] + "…"
    return text, None


# ──────────────────────────────────────────────────────────────────
# Verdict: OFF_TOPIC — polite redirect (no LLM call needed)
# ──────────────────────────────────────────────────────────────────

def _emit_off_topic_redirect(
    *,
    learner_id: str,
    goal_id: str,
    ctx: ctx_mod.LearnerContext,
) -> dict[str, Any]:
    """Persist a templated polite redirect. No LLM call — the message
    is the same regardless of what the learner asked, so spending a
    model call to vary it is waste. The text points the learner at the
    chat / WhatsApp surfaces for general-purpose questions.

    Target language comes from the goal — interpolated via the
    skill_renderer's display map so the text reads naturally for any
    of the supported languages (Afrikaans / English / German)."""
    from .skill_renderer import LANGUAGE_DISPLAY
    target_name = LANGUAGE_DISPLAY.get(
        ctx.target_language, ctx.target_language.title(),
    )
    text = (
        f"I'm focused on your {target_name} learning right now and that's "
        "outside the curriculum. For general questions, the chat at "
        "chat.ongiini.ai or the WhatsApp assistant is set up for that. "
        "Want to keep going with the next card?"
    )
    return messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=MSG_COACH_TEXT, payload={"text": text},
    )


# ──────────────────────────────────────────────────────────────────
# Auto-advance modules when their target is hit
# ──────────────────────────────────────────────────────────────────

# When exercises_emitted overshoots the estimate by this factor we
# advance even if attempted is still short — the model has clearly
# moved past plan and continuing to drill the same module is worse
# than acknowledging it's done.
_OVERSHOOT_THRESHOLD = 1.5


def _advance_module_if_complete(
    *,
    goal_id: str,
    outline: dict[str, Any] | None,
    digest: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """If the current in_progress module has hit its
    ``estimated_cards`` target (or overshot it materially), mark it
    ``completed`` and promote the next ``not_started`` module to
    ``in_progress``. Persists + returns the new outline. Returns None
    when no change is needed.

    This is the fix for "10 / 6" — the model previously kept emitting
    cards in the same module forever because the outline's
    `modules[].status` was static after design time. Now the
    progression is data-driven from the per-module digest.
    """
    if not outline:
        return None
    raw_modules = outline.get("modules")
    if not isinstance(raw_modules, list) or not raw_modules:
        return None
    modules = [dict(m) if isinstance(m, dict) else m for m in raw_modules]

    changed = False
    for i, m in enumerate(modules):
        if not isinstance(m, dict):
            continue
        if m.get("status") != "in_progress":
            continue
        mod_id = m.get("id")
        if not isinstance(mod_id, str):
            continue
        est = m.get("estimated_cards")
        if not isinstance(est, int) or est <= 0:
            continue
        d = digest.get(mod_id, {})
        emitted = int(d.get("exercises_emitted", 0)) + int(d.get("lessons_given", 0))
        attempted = int(d.get("exercises_attempted", 0))
        if attempted < est and emitted < int(est * _OVERSHOOT_THRESHOLD):
            continue
        m["status"] = "completed"
        changed = True
        # Promote the next not_started module to in_progress.
        for j in range(i + 1, len(modules)):
            if isinstance(modules[j], dict) and modules[j].get("status") == "not_started":
                modules[j]["status"] = "in_progress"
                break
        break

    if not changed:
        return None
    new_outline = dict(outline)
    new_outline["modules"] = modules
    store.save_curriculum_outline(goal_id, new_outline)
    log.info("coach: advanced module on goal %s", goal_id)
    return new_outline


# ──────────────────────────────────────────────────────────────────
# Produce the next lesson or exercise card
#
# Orchestrator only. ALL pacing / sequencing decisions live in
# ``selector.select_next_card``. ALL content authoring lives in
# ``cards.generate_card_content``. This function just glues them
# together, persists the result, and turns it into a message.
# ──────────────────────────────────────────────────────────────────


def _persist_and_emit_lesson(
    *,
    learner_id: str,
    goal_id: str,
    sel: "selector.CardSelection",
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Save the lesson card row + emit the matching learner_message.
    Synthesises ``prompt_text`` for the DB NOT NULL column from the
    model's title or first step's body."""
    steps = payload.get("steps") or []
    first_body = ""
    if isinstance(steps, list) and steps and isinstance(steps[0], dict):
        first_body = str(steps[0].get("body") or "")
    persist_prompt_text = (
        (payload.get("title") or first_body or "(lesson)")
    ).strip()[:240] or "(lesson)"
    card_id = store.save_card(
        goal_id,
        sel.card_type,
        persist_prompt_text,
        module_id=sel.module_id,
        topic_id=sel.topic_id,
    )
    return messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=MSG_LESSON,
        payload={
            "title": payload.get("title") or persist_prompt_text[:60],
            "steps": steps,
        },
        card_id=card_id,
    )


_EXERCISE_EXTRA_KEYS = (
    "options", "tokens", "turns", "source_sentence", "cultural_note",
)


def _persist_and_emit_exercise(
    *,
    learner_id: str,
    goal_id: str,
    sel: "selector.CardSelection",
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Save the exercise card row + emit the matching learner_message.
    Preserves per-type structural extras (options / tokens / turns /
    source_sentence / cultural_note) by storing them BOTH in the
    card row (so SRS replay can rebuild the renderer payload) and
    in the message payload (so the frontend renders the first
    emission correctly)."""
    extras = {k: payload[k] for k in _EXERCISE_EXTRA_KEYS if k in payload}
    card_id = store.save_card(
        goal_id,
        sel.card_type,
        payload["prompt_text"],
        reference_answer=payload.get("reference_answer"),
        hint_text=payload.get("hint_text"),
        difficulty=payload.get("difficulty"),
        module_id=sel.module_id,
        topic_id=sel.topic_id,
        extras=extras or None,
    )
    msg_payload: dict[str, Any] = {
        "card_type": sel.card_type,
        "prompt_text": payload["prompt_text"],
        "hint_text": payload.get("hint_text"),
        "difficulty": payload.get("difficulty"),
    }
    msg_payload.update(extras)
    return messages.append(
        learner_id=learner_id, goal_id=goal_id,
        kind=MSG_EXERCISE,
        payload=msg_payload,
        card_id=card_id,
    )


async def _produce_next_thing(
    *,
    learner_id: str,
    goal_id: str,
    ctx: ctx_mod.LearnerContext,
    model: Model,
    skill_content: str,
    prefix_messages: list[dict[str, Any]],
    exclude_card_id: str | None = None,
) -> list[dict[str, Any]]:
    """Author the next card and emit it as a message.

    Flow (every step is deterministic except the two LLM calls
    explicitly noted):

      1. Ensure a curriculum outline exists.                  [LLM if missing]
      2. _advance_module_if_complete.                         [deterministic]
      3. SRS replay — return any due card without an LLM call.[deterministic]
      4. selector.select_next_card.                           [deterministic]
         If the selector returns ``advance_first``, run step 2
         again and reselect (one extra pass; never loops).
         If the selector returns ``graduation``, emit a coach
         text and return.
      5. cards.generate_card_content under a tight brief.     [LLM]
      6. Persist + emit the appropriate message kind.         [deterministic]

    The model is told: "Produce content for card_type=X on topic Y in
    module Z." It can't pick the wrong topic or card_type because
    we don't ask it to pick — the selector did.
    """
    new_messages: list[dict[str, Any]] = list(prefix_messages)

    # Step 1 — outline exists?
    if not ctx.curriculum_outline:
        try:
            outline = await curriculum.design_outline_with_review(
                ctx, model=model, skill_content=skill_content,
            )
        except ModelOutputError as exc:
            log.warning("coach: design_outline failed: %s", exc)
            msg = messages.append(
                learner_id=learner_id, goal_id=goal_id,
                kind=MSG_COACH_TEXT,
                payload=_error_coach_payload(
                    "I had trouble putting your plan together just now — "
                    "could you try again in a moment?",
                    error_code="design_outline_failed",
                ),
            )
            new_messages.append(msg)
            return new_messages
        store.save_curriculum_outline(goal_id, outline)
        ctx = ctx_mod.build_learner_context(learner_id, goal_id=goal_id)

    # Step 2 — advance modules if the current one is finished.
    advanced = _advance_module_if_complete(
        goal_id=goal_id,
        outline=ctx.curriculum_outline,
        digest=ctx.module_digest,
    )
    if advanced is not None:
        ctx = ctx_mod.build_learner_context(learner_id, goal_id=goal_id)

    # Step 3 — SRS replay. A previously-failed card whose Leitner
    # next_due_at <= now wins over a brand-new card so the learner
    # actually re-encounters the things they got wrong.
    due_cards = store.next_due_cards(
        learner_id,
        goal_id=goal_id,
        exclude_card_id=exclude_card_id,
        limit=1,
    )
    if due_cards:
        d = due_cards[0]
        replay_payload: dict[str, Any] = {
            "card_type": d["card_type"],
            "prompt_text": d["prompt_text"],
            "hint_text": d.get("hint_text"),
            "difficulty": d.get("difficulty"),
            "review_box": int(d.get("box") or 1),
        }
        # Re-attach per-type structural extras so reorder/MC/dialogue/
        # grammar/proverb cards render correctly on re-review. Without
        # this a failed reorder card would resurface as just a prompt
        # with no token chips — Sebastian's "I couldn't see the words"
        # bug for repeated cards.
        extras = d.get("extras")
        if isinstance(extras, dict):
            for k in _EXERCISE_EXTRA_KEYS:
                if k in extras:
                    replay_payload[k] = extras[k]
        msg = messages.append(
            learner_id=learner_id, goal_id=goal_id,
            kind=MSG_EXERCISE,
            payload=replay_payload,
            card_id=d["card_id"],
        )
        new_messages.append(msg)
        return new_messages

    # Step 4 — selector picks card_type + module + topic.
    sel = selector.select_next_card(
        outline=ctx.curriculum_outline,
        module_digest=ctx.module_digest,
    )
    if sel.advance_first:
        # Selector says nothing to do in this module — run advance
        # once more and reselect. This is the rare path that fires
        # when a module has hit its drill quotas mid-run; the count
        # threshold in _advance_module_if_complete didn't catch it.
        advanced2 = _advance_module_if_complete(
            goal_id=goal_id,
            outline=ctx.curriculum_outline,
            digest=ctx.module_digest,
        )
        if advanced2 is not None:
            ctx = ctx_mod.build_learner_context(learner_id, goal_id=goal_id)
        sel = selector.select_next_card(
            outline=ctx.curriculum_outline,
            module_digest=ctx.module_digest,
        )
        # If even after advance the selector STILL says advance_first
        # (the outline is malformed — no topics list, or every module
        # is empty), treat it as graduation. Better to land on a
        # friendly "tell me what's next" than to loop or crash.
        if sel.advance_first:
            log.warning(
                "coach: selector returned advance_first twice in a row — "
                "treating as graduation. outline likely malformed."
            )
            sel = selector.CardSelection(graduation=True, phase="graduation_loop")
    if sel.graduation:
        msg = messages.append(
            learner_id=learner_id, goal_id=goal_id,
            kind=MSG_COACH_TEXT,
            payload={
                "text": (
                    "You've finished every module in this plan — well done. "
                    "Tell me what you'd like next: a new focus, a different "
                    "language, or more practice on the same material."
                ),
                "meta": {"event": "graduation"},
            },
        )
        new_messages.append(msg)
        return new_messages

    log.info(
        "coach: selector → phase=%s card_type=%s module=%s topic=%s",
        sel.phase, sel.card_type, sel.module_id, sel.topic_id,
    )

    # Step 5 — author the content. Tight brief; model just writes.
    # One transparent retry on transient JSON-shape issues (Gemma
    # occasionally drops a fence or a key).
    payload: dict[str, Any] | None = None
    last_exc: ModelOutputError | None = None
    for attempt in range(2):
        try:
            payload = await cards_mod.generate_card_content(
                ctx,
                model=model,
                skill_content=skill_content,
                card_type=sel.card_type,
                module_id=sel.module_id,
                module_title=sel.module_title,
                topic_id=sel.topic_id,
                topic_title=sel.topic_title,
            )
            last_exc = None
            break
        except ModelOutputError as exc:
            last_exc = exc
            if attempt == 0:
                log.info("coach: generate_card_content transient fail, retrying: %s", exc)
            else:
                log.warning("coach: generate_card_content failed after retry: %s", exc)
    if last_exc is not None or payload is None:
        msg = messages.append(
            learner_id=learner_id, goal_id=goal_id,
            kind=MSG_COACH_TEXT,
            payload=_error_coach_payload(
                "I had trouble generating your next card — could you try "
                "again?",
                error_code="generate_card_failed",
            ),
        )
        new_messages.append(msg)
        return new_messages

    # Step 6 — persist + emit the right message kind.
    if sel.card_type not in EXERCISE_CARD_TYPES:
        msg = _persist_and_emit_lesson(
            learner_id=learner_id, goal_id=goal_id, sel=sel, payload=payload,
        )
    else:
        msg = _persist_and_emit_exercise(
            learner_id=learner_id, goal_id=goal_id, sel=sel, payload=payload,
        )
    new_messages.append(msg)
    return new_messages

