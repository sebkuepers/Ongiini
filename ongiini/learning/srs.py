"""Leitner-style spaced-repetition scheduler.

Pure functions only — no I/O, no DB access. The state (which box a
card is in for a given learner) lives in card_review_state; this
module just answers two questions:

  1. Given a learner's answer (correct or not), what box does the card
     move to? — ``promote(box, correct)``
  2. Given a box number, when should the card next appear? —
     ``next_due_at(box, now)``

The MVP uses five boxes with classic Leitner-ish intervals: a wrong
answer always knocks the card back to box 1 (review again now), a
correct answer promotes one box, and the boxes have geometric-ish
intervals so mastered cards return less often:

  box 1 → next due NOW                    (re-review in the same session)
  box 2 → next due in ~1 day
  box 3 → next due in ~3 days
  box 4 → next due in ~7 days  (one week)
  box 5 → next due in ~14 days (two weeks; "mastered" tier)

This is deliberately not SM-2 / FSRS — MVP wants something the user
can understand at a glance from a 5-bar mini-chart in the stats
sidebar, not an opaque algorithm. We can swap in a smarter scheduler
once we have enough data to see what the algorithm is actually doing.

A note on the ``next_due_at`` return type: it returns an ISO-8601 UTC
**string**, NOT a ``datetime`` object. The schema stores the column
as TEXT, and the SRS query uses string range comparisons
(``WHERE next_due_at <= ?``). Returning a string here matches the
storage shape and removes the "did I remember to .isoformat()?" foot-
gun at every caller. Tests can compare strings directly or re-parse
when they need timestamp arithmetic.
"""
from __future__ import annotations

from datetime import datetime, timedelta

# Box → hours until next-due. Box 1 returns immediately so the same
# session re-surfaces things the learner just got wrong; later boxes
# space out geometrically-ish.
BOX_INTERVALS_HOURS: dict[int, int] = {
    1: 0,
    2: 24,
    3: 72,
    4: 168,    # 7 days
    5: 336,    # 14 days — "mastered"
}

MIN_BOX = 1
MAX_BOX = max(BOX_INTERVALS_HOURS)


def promote(box: int, correct: bool) -> int:
    """Return the new box for a card after an answer.

    Wrong answer → always demoted to box 1, regardless of the prior box.
    Correct answer → promoted by one, capped at ``MAX_BOX``.

    A 'partial' rating from the grader counts as ``correct=True``
    for promotion purposes — the goal in the MVP is forward motion, not
    perfect-only mastery. The grading layer is responsible for the
    correct/partial/wrong split; this function only sees the bool.

    Defensive on the input box: anything below 1 is clamped to 1,
    anything above MAX_BOX is clamped to MAX_BOX. Means callers can
    safely pass in stored values without pre-validating.
    """
    box = max(MIN_BOX, min(MAX_BOX, box))
    if not correct:
        return MIN_BOX
    return min(MAX_BOX, box + 1)


def next_due_at(box: int, now: datetime) -> str:
    """Return the ISO-8601 UTC timestamp string the card is next due at.

    ``box`` is the box AFTER promotion (i.e. the new state). ``now`` is
    the timestamp of the answer being recorded — pass an explicit value
    rather than relying on ``datetime.utcnow()`` so tests can pin the
    clock without monkey-patching.

    Returns a string (not a ``datetime``) because the schema column
    is TEXT and the SRS query relies on lexicographic comparison —
    see the module docstring.

    Clamps box to the supported range so the caller never has to worry
    about a KeyError from a corrupt stored value.
    """
    box = max(MIN_BOX, min(MAX_BOX, box))
    due = now + timedelta(hours=BOX_INTERVALS_HOURS[box])
    return due.isoformat(timespec="seconds")


def box_label(box: int) -> str:
    """Human-readable label for a box. Used by the stats panel."""
    box = max(MIN_BOX, min(MAX_BOX, box))
    return {
        1: "new / shaky",
        2: "warming up",
        3: "getting there",
        4: "solid",
        5: "mastered",
    }[box]
