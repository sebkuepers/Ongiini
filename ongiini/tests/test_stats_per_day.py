"""Tests for the per-day timeseries computations in stats.aggregator.

Specifically the _count_conversations_per_day helper which buckets
conversations by their start-day so /statistics can show daily session
volume alongside daily message volume.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ongiini.stats.aggregator import (
    CONVERSATION_GAP,
    _count_conversations,
    _count_conversations_per_day,
)


UTC = timezone.utc


def _ts(*args) -> datetime:
    """Make a UTC timestamp from (year, month, day, hour, minute) args."""
    return datetime(*args, tzinfo=UTC)


# ---------- _count_conversations_per_day ----------

def test_single_user_single_day_single_conversation():
    per_user = {
        "u1": [_ts(2026, 5, 28, 10, 0),
               _ts(2026, 5, 28, 10, 5),
               _ts(2026, 5, 28, 10, 20)],
    }
    out = _count_conversations_per_day(per_user)
    assert out == {"2026-05-28": 1}


def test_gap_starts_new_conversation_same_day():
    """Two bursts on the same day, separated by > 30min gap, count as
    two conversations both bucketed to that day."""
    per_user = {
        "u1": [
            _ts(2026, 5, 28, 9, 0),
            _ts(2026, 5, 28, 9, 10),
            # 31-min gap → new conversation
            _ts(2026, 5, 28, 9, 41),
            _ts(2026, 5, 28, 9, 50),
        ],
    }
    out = _count_conversations_per_day(per_user)
    assert out == {"2026-05-28": 2}


def test_gap_of_exactly_30_min_starts_new_conversation():
    """Boundary check — CONVERSATION_GAP is 30min; >= triggers split.
    Locks the gap-of-exactly-30-min behavior so we'd notice if it
    accidentally changed to strict-greater-than."""
    assert CONVERSATION_GAP == timedelta(minutes=30)
    per_user = {
        "u1": [_ts(2026, 5, 28, 10, 0), _ts(2026, 5, 28, 10, 30)],
    }
    out = _count_conversations_per_day(per_user)
    assert out == {"2026-05-28": 2}


def test_conversation_spans_midnight_counts_on_start_day_only():
    """If a conversation has a sub-30min gap that crosses midnight,
    it counts ONCE on its start day — not on both days."""
    per_user = {
        "u1": [
            _ts(2026, 5, 28, 23, 50),
            _ts(2026, 5, 28, 23, 55),
            _ts(2026, 5, 29, 0, 10),  # 15min after last msg → same conversation
        ],
    }
    out = _count_conversations_per_day(per_user)
    assert out == {"2026-05-28": 1}
    assert "2026-05-29" not in out


def test_two_conversations_one_per_day():
    """User pings on Mon, sleeps, pings again Tue. Two conversations
    bucketed one to each day."""
    per_user = {
        "u1": [
            _ts(2026, 5, 28, 14, 0),
            _ts(2026, 5, 29, 9, 0),
        ],
    }
    out = _count_conversations_per_day(per_user)
    assert out == {"2026-05-28": 1, "2026-05-29": 1}


def test_multiple_users_aggregate():
    per_user = {
        "u1": [_ts(2026, 5, 28, 10, 0)],
        "u2": [_ts(2026, 5, 28, 10, 0)],
        "u3": [_ts(2026, 5, 28, 14, 0), _ts(2026, 5, 29, 14, 0)],
    }
    out = _count_conversations_per_day(per_user)
    assert out["2026-05-28"] == 3   # u1 + u2 + u3-day-1
    assert out["2026-05-29"] == 1   # u3-day-2


def test_empty_input_returns_empty_dict():
    assert _count_conversations_per_day({}) == {}
    assert _count_conversations_per_day({"u1": []}) == {}


def test_per_day_sum_matches_total_conversation_count():
    """Invariant: summing per-day buckets MUST equal the total count
    from _count_conversations. If they ever diverge, one of the two
    has a bug."""
    per_user = {
        "u1": [_ts(2026, 5, 28, 10, 0), _ts(2026, 5, 28, 10, 5),
               _ts(2026, 5, 28, 11, 0)],     # 2 conversations on 5-28
        "u2": [_ts(2026, 5, 28, 23, 50), _ts(2026, 5, 29, 0, 10)],   # 1 spanning midnight
        "u3": [_ts(2026, 5, 29, 12, 0)],     # 1 on 5-29
    }
    by_day = _count_conversations_per_day(per_user)
    assert sum(by_day.values()) == _count_conversations(per_user)
