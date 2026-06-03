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
    # The user's most recent stated focus for THIS curriculum. The
    # primary anchor for the curriculum-design prompt — when set, it
    # overrides the older profile.objective so a learner who created
    # a German goal titled "Job Interview at Serviceplan" doesn't get
    # a curriculum based on the restaurant-themed objective they
    # entered at intake last week.
    goal_title: str | None = None
    # Optional longer-form context the user supplied when creating
    # the goal (the modal's "Context" textarea).
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
    # drill" rather than re-emitting another lesson.
    # Capped at 12 — enough to keep an early-module lesson in the
    # window even after a long drill streak.
    recent_cards: list[dict[str, object]] = field(default_factory=list)

    # Language pair for this goal, mirrored from the row so prompt
    # builders (curriculum / cards / grading / coach.question_handler)
    # and the off-topic redirect text can read them without re-querying
    # the store. ``source_language`` defaults to 'english' to match
    # back-fill on legacy rows; ``target_language`` defaults to
    # 'afrikaans' to match the prior single-language behaviour.
    source_language: str = "english"
    target_language: str = "afrikaans"

    # Per-module rollup: how many lessons + exercises this learner has
    # already seen for each module_id in the outline. This is the
    # SURVIVAL signal that the recent_cards window can't carry once
    # the learner is deep in a module — without it the model loses
    # track of "yep, I gave the greetings lesson 14 turns ago" and
    # re-emits it.
    # Shape: {module_id: {"lessons_given": int, "exercises_emitted": int,
    #                     "exercises_attempted": int, "exercises_correct": int,
    #                     "cards_in_module": int}}
    module_digest: dict[str, dict[str, object]] = field(default_factory=dict)

    # Top error categories from the learner's recent grader-tagged
    # attempts (Track D). Drives adaptive curriculum design: the
    # designer's next outline can target a learner's actual weak
    # spots (e.g. "12 gender_error attempts → next module covers
    # noun gender") rather than running the same plan for every
    # learner with the same stated goal. Shape:
    # ``[{tag: 'gender_error', count: 12}, ...]`` capped at top 5.
    # Empty list when the learner is fresh or all attempts were
    # correct.
    error_patterns: list[dict[str, object]] = field(default_factory=list)


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
    # Pull source/target/level + title + context from the goal row.
    # Defaults match the back-fill (source=english, target=afrikaans)
    # so callers without a goal still get sensible behaviour for the
    # legacy single-language flow.
    source_language = "english"
    target_language = "afrikaans"
    goal_level: str | None = None
    goal_title: str | None = None
    goal_row_context: str | None = None
    if goal_id:
        outline = store.get_curriculum_outline(goal_id)
        goal_row = next(
            (g for g in store.list_goals(learner_id, include_archived=True)
             if g["goal_id"] == goal_id),
            None,
        )
        if goal_row:
            if isinstance(goal_row.get("source_language"), str):
                source_language = goal_row["source_language"]
            if isinstance(goal_row.get("language"), str):
                target_language = goal_row["language"]
            if isinstance(goal_row.get("current_level"), str) and goal_row["current_level"].strip():
                goal_level = goal_row["current_level"]
            if isinstance(goal_row.get("title"), str) and goal_row["title"].strip():
                goal_title = goal_row["title"].strip()
            if isinstance(goal_row.get("context"), str) and goal_row["context"].strip():
                goal_row_context = goal_row["context"].strip()
    # The goal_context parameter is an explicit caller override (e.g.
    # the magic-link path supplying a fresh objective for a new goal).
    # Persisted goal data is preferred when the caller didn't supply
    # one — we DELIBERATELY do not fall back to profile.objective here
    # because doing so caused the prior bug: a learner with stale
    # intake objective + a freshly-named new goal got a curriculum
    # designed for the old objective.

    progress = (
        store.progress_for(learner_id, goal_id=goal_id) if learner_id and goal_id
        else (store.progress_for(learner_id) if learner_id else None)
    )

    recent_cards: list[dict[str, object]] = []
    module_digest: dict[str, dict[str, object]] = {}
    if learner_id and goal_id:
        thread = _messages.list_for_goal(
            learner_id=learner_id, goal_id=goal_id, limit=200,
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
        # Keep only the tail of the recent window so the prompt budget
        # stays predictable — the deeper history is summarised below
        # in module_digest, which doesn't grow with thread length.
        recent_cards = recent_cards[-12:]

        # Build the module digest via the store helper — same query
        # the API uses to serve /turn so context + UI see the same
        # numbers.
        module_digest = store.progress_for_modules(learner_id, goal_id)

    error_patterns: list[dict[str, object]] = []
    if learner_id:
        try:
            error_patterns = store.error_pattern_summary(
                learner_id, goal_id=goal_id,
            )
        except Exception as exc:                                # noqa: BLE001
            # Best-effort context decoration. A failure here MUST NOT
            # block curriculum design or card authoring.
            log.warning(
                "context: error_pattern_summary lookup failed; "
                "continuing without it. error=%s", exc,
            )

    # Augment the profile dict with the resolved goal-level so prompts
    # that read ctx.profile['current_level'] get the per-goal value
    # when one is set. Profile-level stays as the fallback.
    if profile is not None and goal_level:
        profile = dict(profile)
        profile["current_level"] = goal_level

    return LearnerContext(
        learner_id=learner_id,
        profile=profile,
        goal_id=goal_id,
        goal_title=goal_title,
        goal_context=goal_context or goal_row_context,
        recent_excerpts=[],     # Phase 2 — wire to chat short-term memory
        mem0_facts=[],          # Phase 2 — wire to mem0 search by objective
        curriculum_outline=outline,
        progress=progress,
        recent_cards=recent_cards,
        module_digest=module_digest,
        error_patterns=error_patterns,
        source_language=source_language,
        target_language=target_language,
    )


