"""Deterministic card selector — the planner that decides WHAT to
emit next, leaving the LLM to write the content.

Why this exists: earlier iterations let the LLM decide each turn
whether to emit a lesson or an exercise, which topic to target, and
which card_type to drill — and we patched the resulting failures
with prompt retries, force-notes, and topic_id remappings. The LLM
is unreliable as a planner over state; deterministic code is good
at this. This module concentrates ALL pacing/sequencing decisions
in one ~100-line pure function.

The selector reads two pieces of state:
  * the curriculum outline (modules → topics; each topic has a
    kind in {"lesson", "practice"})
  * the per-module digest from ``store.progress_for_modules`` —
    ``topics_taught`` and ``topics_drilled`` keyed by topic_id

…and returns a ``CardSelection`` describing the next thing to
author. The coach takes that selection, calls the LLM for content
under a tight brief, attaches the scaffolding fields (the LLM is
not trusted to set them), and persists.

This module has NO side effects. No DB writes, no LLM calls. Easy
to test, easy to reason about, easy to tune.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .db import (
    CARD_CLOZE,
    CARD_DIALOGUE,
    CARD_GRAMMAR,
    CARD_LESSON,
    CARD_MULTIPLE_CHOICE,
    CARD_TRANSLATION,
    CARD_VOCAB,
)


# ──────────────────────────────────────────────────────────────────
# Tunable constants — the architecture stays the same when you
# change these; they're knobs, not invariants.
# ──────────────────────────────────────────────────────────────────

# How many lesson cards we author per lesson topic before moving on
# to the practice topics. One well-built multi-step lesson (3-5
# swipeable steps) typically covers a topic; bump this if a topic
# wants splitting across multiple cards.
TARGET_LESSONS_PER_TOPIC = 1

# How many exercise cards we author per practice topic before the
# module is "drilled enough". Each drill picks a different card_type
# from the rotation so the learner sees the topic in multiple forms.
TARGET_DRILLS_PER_PRACTICE_TOPIC = 2

# Round-robin rotation for exercise card_type variety. The n-th drill
# on a topic picks rotation[n % len(rotation)]. The order is designed
# so the first drill is the easiest (vocab) and the last is the hardest
# (dialogue) — readable on inspection.
EXERCISE_TYPE_ROTATION: tuple[str, ...] = (
    CARD_VOCAB,
    CARD_CLOZE,
    CARD_TRANSLATION,
    CARD_MULTIPLE_CHOICE,
    CARD_GRAMMAR,
    CARD_DIALOGUE,
)


# ──────────────────────────────────────────────────────────────────
# Selection result
# ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CardSelection:
    """The selector's verdict — fields the coach needs to author and
    persist the next card.

    Exactly one of (``card_type``, ``graduation``) is meaningful per
    selection. When ``graduation`` is True there's no card to author;
    the caller emits a "you've finished the plan" coach message.

    When ``advance_first`` is True, the caller must call
    ``coach._advance_module_if_complete`` BEFORE acting on the rest
    of this selection — the selector detected that the in-progress
    module is fully drilled and the next module should take over
    before we re-select.
    """
    card_type: str | None = None
    module_id: str | None = None
    module_title: str | None = None
    topic_id: str | None = None
    topic_title: str | None = None
    graduation: bool = False
    advance_first: bool = False
    # Diagnostic — which phase of the decision table produced this
    # selection (teach / drill / recycle). Useful in logs and tests.
    phase: str = ""


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _in_progress_module(outline: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the first module with ``status == 'in_progress'`` from
    the outline, or None if the outline is missing/malformed or
    every module is completed (the graduation case)."""
    if not outline:
        return None
    modules = outline.get("modules")
    if not isinstance(modules, list):
        return None
    for m in modules:
        if isinstance(m, dict) and m.get("status") == "in_progress":
            return m
    return None


def _topics_by_kind(
    module: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (lesson_topics, practice_topics) for one module, each
    entry a dict with at least ``id`` and ``title`` (filtered to
    well-shaped entries)."""
    topics = module.get("topics")
    if not isinstance(topics, list):
        return [], []
    lessons: list[dict[str, Any]] = []
    practices: list[dict[str, Any]] = []
    for t in topics:
        if not isinstance(t, dict):
            continue
        tid = t.get("id")
        if not isinstance(tid, str) or not tid:
            continue
        kind = t.get("kind")
        if kind == "lesson":
            lessons.append(t)
        elif kind == "practice":
            practices.append(t)
    return lessons, practices


def pick_exercise_type(module_drilled_total: int) -> str:
    """Round-robin pick from ``EXERCISE_TYPE_ROTATION`` indexed by the
    total number of exercises already emitted in the in-progress
    MODULE — not per-topic.

    Why per-module: keying per-topic meant every new topic restarted
    at slot 0 (vocab), which made entire modules feel vocab-heavy
    because the second slot (cloze) was the only other type that
    fired within the default drill quota. Per-module rotation cycles
    through translation / multiple_choice / grammar / dialogue across
    the practice topics within one module, even if each topic only
    gets 2 drills.

    Pure function so the same state always gives the same answer."""
    rot = EXERCISE_TYPE_ROTATION
    return rot[max(0, module_drilled_total) % len(rot)]


# ──────────────────────────────────────────────────────────────────
# The decision table
# ──────────────────────────────────────────────────────────────────

def select_next_card(
    *,
    outline: dict[str, Any] | None,
    module_digest: dict[str, dict[str, Any]],
) -> CardSelection:
    """Decide what the next card should be — pure function over
    outline + digest.

    Walks the phase table in order; the first hit wins:
      1. **graduation** — no in-progress module left
      2. **teach** — first lesson topic with fewer than
         TARGET_LESSONS_PER_TOPIC taught lessons
      3. **drill** — first practice topic with fewer than
         TARGET_DRILLS_PER_PRACTICE_TOPIC drills
      4. **recycle** — drill a taught lesson topic with the fewest
         drills (spaced recycling)
      5. **advance_first** — nothing to do in this module, ask the
         caller to advance and re-select

    SRS replay is NOT in this function — the coach handles it
    upstream before calling the selector. Keeps this function pure
    over the outline+digest pair.
    """
    in_progress = _in_progress_module(outline)
    if in_progress is None:
        return CardSelection(graduation=True, phase="graduation")

    mod_id = in_progress.get("id")
    if not isinstance(mod_id, str):
        # Malformed outline — coach should handle this defensively
        # upstream (logs + advance). Treat as graduation here.
        return CardSelection(graduation=True, phase="graduation_malformed")
    mod_title = in_progress.get("title") or mod_id

    lessons, practices = _topics_by_kind(in_progress)
    digest = module_digest.get(mod_id, {}) if module_digest else {}
    taught: dict[str, int] = digest.get("topics_taught") or {}
    drilled: dict[str, int] = digest.get("topics_drilled") or {}
    # Module-level rotation index: total exercises emitted in this
    # module so far. Drives card_type variety so the user doesn't see
    # vocab on every new practice topic in a row.
    module_drilled_total = sum(int(v or 0) for v in drilled.values())

    # Phase 2 — teach: first lesson topic that hasn't hit its lesson quota.
    for t in lessons:
        tid = t["id"]
        if int(taught.get(tid, 0)) < TARGET_LESSONS_PER_TOPIC:
            return CardSelection(
                card_type=CARD_LESSON,
                module_id=mod_id,
                module_title=mod_title,
                topic_id=tid,
                topic_title=t.get("title") or tid,
                phase="teach",
            )

    # Phase 3 — drill: first practice topic that hasn't hit its drill quota.
    for t in practices:
        tid = t["id"]
        if int(drilled.get(tid, 0)) < TARGET_DRILLS_PER_PRACTICE_TOPIC:
            return CardSelection(
                card_type=pick_exercise_type(module_drilled_total),
                module_id=mod_id,
                module_title=mod_title,
                topic_id=tid,
                topic_title=t.get("title") or tid,
                phase="drill",
            )

    # Phase 4 — recycle: drill an already-taught lesson topic with
    # the FEWEST drills (spaced recycling). Tie-break by outline
    # order via the list traversal.
    if lessons:
        # Build (topic, drilled_count) and pick the smallest.
        ranked = sorted(
            ((t, int(drilled.get(t["id"], 0))) for t in lessons),
            key=lambda pair: pair[1],
        )
        t, count = ranked[0]
        # Only recycle if there's HEADROOM — if every taught topic has
        # already been drilled to the same target as practice topics,
        # advance instead of piling on more. Earlier this cap was
        # 2× TARGET_DRILLS_PER_PRACTICE_TOPIC which made modules feel
        # over-drilled (10 cards before advancing in the German smoke).
        if count < TARGET_DRILLS_PER_PRACTICE_TOPIC:
            return CardSelection(
                card_type=pick_exercise_type(module_drilled_total),
                module_id=mod_id,
                module_title=mod_title,
                topic_id=t["id"],
                topic_title=t.get("title") or t["id"],
                phase="recycle",
            )

    # Phase 5 — nothing left to do in this module; ask the coach to
    # advance and re-select on the next turn. The advance helper is
    # data-driven from per-module emit counts; it'll promote the
    # next ``not_started`` module to ``in_progress``.
    return CardSelection(advance_first=True, phase="advance_first")
