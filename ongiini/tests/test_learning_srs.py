"""Pure-function tests for the Leitner SRS scheduler.

No DB access. These lock the box-promotion and next-due semantics so
when we swap in a smarter scheduler later it's a deliberate change
rather than a silent drift.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ongiini.learning import srs


UTC = timezone.utc


# ---------- promote ----------

def test_promote_correct_advances_one_box():
    assert srs.promote(1, correct=True) == 2
    assert srs.promote(2, correct=True) == 3
    assert srs.promote(3, correct=True) == 4
    assert srs.promote(4, correct=True) == 5


def test_promote_caps_at_max_box():
    assert srs.promote(srs.MAX_BOX, correct=True) == srs.MAX_BOX


def test_promote_wrong_always_demotes_to_box_one():
    # No matter how mastered, a wrong answer drops to box 1.
    for box in range(srs.MIN_BOX, srs.MAX_BOX + 1):
        assert srs.promote(box, correct=False) == srs.MIN_BOX


def test_promote_clamps_below_min():
    # Defensive: a stored value of 0 (or negative) shouldn't crash.
    assert srs.promote(0, correct=True) == 2     # treated as box 1, then +1
    assert srs.promote(-3, correct=False) == 1


def test_promote_clamps_above_max():
    # Defensive: a stored value of 99 from a future schema shouldn't crash.
    assert srs.promote(99, correct=True) == srs.MAX_BOX
    assert srs.promote(99, correct=False) == 1


# ---------- next_due_at ----------

def test_next_due_box1_is_immediately():
    # Box 1 = re-review in the same session.
    now = datetime(2026, 6, 2, 14, 0, 0, tzinfo=UTC)
    due = srs.next_due_at(1, now)
    assert due == "2026-06-02T14:00:00+00:00"


def test_next_due_box2_is_one_day_out():
    now = datetime(2026, 6, 2, 14, 0, 0, tzinfo=UTC)
    due = srs.next_due_at(2, now)
    assert due == "2026-06-03T14:00:00+00:00"


def test_next_due_intervals_increase_monotonically():
    # The whole point of Leitner — later boxes return less often.
    now = datetime(2026, 6, 2, 14, 0, 0, tzinfo=UTC)
    timestamps = [srs.next_due_at(b, now) for b in range(1, srs.MAX_BOX + 1)]
    # Strings are ISO-8601 → lexicographic compare matches chronological.
    assert timestamps == sorted(timestamps)


def test_next_due_box5_is_two_weeks_out():
    now = datetime(2026, 6, 2, 14, 0, 0, tzinfo=UTC)
    due = srs.next_due_at(5, now)
    assert due == "2026-06-16T14:00:00+00:00"


def test_next_due_returns_string():
    # Locked behaviour: the storage column is TEXT and the SRS query
    # uses lexicographic comparison. Returning a string keeps callers
    # from forgetting to .isoformat().
    now = datetime(2026, 6, 2, 14, 0, 0, tzinfo=UTC)
    assert isinstance(srs.next_due_at(3, now), str)


def test_next_due_clamps_corrupt_box():
    now = datetime(2026, 6, 2, 14, 0, 0, tzinfo=UTC)
    # Box 99 from a future schema migration shouldn't crash — treat as MAX.
    assert srs.next_due_at(99, now) == srs.next_due_at(srs.MAX_BOX, now)
    # Negative box → treat as MIN.
    assert srs.next_due_at(-1, now) == srs.next_due_at(srs.MIN_BOX, now)


# ---------- box_label ----------

def test_box_labels_cover_all_boxes():
    # Stats panel needs a label for every box. Don't ship a KeyError.
    for box in range(srs.MIN_BOX, srs.MAX_BOX + 1):
        assert isinstance(srs.box_label(box), str)
        assert len(srs.box_label(box)) > 0


def test_box_label_clamps_corrupt_input():
    assert srs.box_label(99) == srs.box_label(srs.MAX_BOX)
    assert srs.box_label(-1) == srs.box_label(srs.MIN_BOX)
