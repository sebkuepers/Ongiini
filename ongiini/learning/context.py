"""LearnerContext — what the LLM sees when designing or revising a
curriculum, generating a card, or grading an answer.

The whole reason this learning surface is interesting is that the
model sees the *specific learner* — their profile, their stated
goal, anything we know from prior conversation. This module is the
single place that assembles that context so prompts elsewhere
(curriculum design, card generation, grading) all see the same shape.

What the LLM gets:

  * ``profile``        — the four intake fields once captured (name,
                         age, current_level, objective). Always
                         available after intake completes.
  * ``goal_context``   — the original learning objective text the
                         magic-link carried (if any), e.g. "job
                         interview at SPAR". For cold visitors with
                         no magic link this is None and the LLM
                         relies on the intake objective field.
  * ``recent_excerpts``— short-term conversation excerpts from the
                         chat / WhatsApp thread that produced the
                         magic link. Empty for cold visitors. Phase-2
                         wiring; for MVP we surface the shape but do
                         not yet populate it from short-term memory.
  * ``mem0_facts``    — long-term mem0 facts for this user, queried
                         by the objective text. Same Phase-2 story —
                         shape is here so prompts can reference it
                         from day one, populated in Phase 2 once we
                         carry the raw msisdn through the magic-link
                         token.

The context provider is deliberately READ-ONLY. It pulls from the
store and (later) from the chat memory tiers; it never writes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from . import messages as _messages
from . import store
from .db import MSG_EXERCISE, MSG_LESSON

log = logging.getLogger("ongiini.learning.context")


@dataclass(frozen=True)
class LearnerContext:
    """Snapshot of everything the LLM should see for one turn.

    Frozen so prompts can hash / cache by context identity and so a
    caller can't accidentally mutate the assembled view back into the
    store."""

    learner_id: str
    profile: dict[str, object] | None
    goal_id: str | None = None
    goal_context: str | None = None
    recent_excerpts: list[str] = field(default_factory=list)
    mem0_facts: list[str] = field(default_factory=list)

    # The current curriculum outline (LLM-authored, JSON-parsed). None
    # before the LLM has written one — the curriculum-design prompt
    # uses this signal to decide "are we drafting an outline, or
    # revising the existing one?"
    curriculum_outline: dict[str, object] | None = None

    # Progress snapshot — total cards seen, total correct, per-box
    # distribution. The curriculum-design prompt wants this to decide
    # whether the learner is stuck or moving; the card-generation
    # prompt wants it to pick difficulty.
    progress: dict[str, object] | None = None

    # The most recent lesson / exercise cards on this goal's thread,
    # oldest first. The card-generation prompt uses this so the model
    # can see "I just emitted a lesson on greetings — switch to a
    # drill" rather than re-emitting another lesson on every "Got it
    # →" tap (the Phase 2 bug: same lesson three times in a row
    # because the model had nothing in context except the outline).
    # Capped at 6 — enough to anchor decisions, small enough to keep
    # prompt cost flat.
    recent_cards: list[dict[str, object]] = field(default_factory=list)


def build_learner_context(
    learner_id: str,
    *,
    goal_context: str | None = None,
    goal_id: str | None = None,
) -> LearnerContext:
    """Assemble the LearnerContext for the given learner.

    ``goal_context`` overrides whatever's on the profile — used at
    magic-link landing where the magic link carries a fresh objective
    that may be more current than what was stored. If absent, the
    learner's profile objective fills in.

    ``goal_id`` picks up the existing curriculum outline from that
    goal. Optional: callers without a goal_id (e.g. early intake
    turns before a goal exists) just don't get an outline back, which
    is correct.
    """
    profile = store.get_profile(learner_id)
    outline: dict[str, object] | None = None
    if goal_id:
        outline = store.get_curriculum_outline(goal_id)
    # Fall back to the intake objective when the caller didn't supply
    # one. Keeps a sensible context for cold-visitor flows where the
    # only "why" we have is what intake captured.
    fallback_objective = None
    if profile and isinstance(profile.get("objective"), str):
        fallback_objective = str(profile.get("objective")) or None

    progress = (
        store.progress_for(learner_id, goal_id=goal_id) if learner_id and goal_id
        else (store.progress_for(learner_id) if learner_id else None)
    )

    recent_cards: list[dict[str, object]] = []
    if learner_id and goal_id:
        thread = _messages.list_for_goal(
            learner_id=learner_id, goal_id=goal_id, limit=40,
        )
        for row in thread:
            if row.get("kind") in (MSG_LESSON, MSG_EXERCISE):
                p = row.get("payload") or {}
                recent_cards.append({
                    "kind": row["kind"],
                    "card_type": p.get("card_type"),
                    "title": p.get("title"),
                    "prompt_text": p.get("prompt_text") or p.get("body"),
                    "answered": bool(row.get("answered")),
                })
        # Keep only the tail so the prompt budget stays predictable.
        recent_cards = recent_cards[-6:]

    return LearnerContext(
        learner_id=learner_id,
        profile=profile,
        goal_id=goal_id,
        goal_context=goal_context or fallback_objective,
        recent_excerpts=[],     # Phase 2 — wire to chat short-term memory
        mem0_facts=[],          # Phase 2 — wire to mem0 search by objective
        curriculum_outline=outline,
        progress=progress,
        recent_cards=recent_cards,
    )
