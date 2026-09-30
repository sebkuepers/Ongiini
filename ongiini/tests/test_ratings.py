"""Tests for the translation-rating store and task assignment."""
from __future__ import annotations

import pytest

from ongiini import ratings


@pytest.fixture
def con(tmp_path):
    c = ratings.connect(tmp_path / "r.sqlite")
    yield c
    c.close()


def task(tid, item, dialect="oshikwanyama", kind="ref", priority=2, target=1, **kw):
    return {"task_id": tid, "round": "r1", "item_id": item, "dialect": dialect,
            "english": f"en {item}", "text": f"{kind} {item}", "kind": kind,
            "source": kw.get("source", kind), "priority": priority, "target": target,
            "expected": kw.get("expected")}


def rater(con, dialects=("oshikwanyama",)):
    token = ratings.add_rater(con, "R", list(dialects))
    return token, ratings.rater_for(con, token)


def test_token_is_stored_hashed_and_resolves(con):
    token, r = rater(con)
    assert r is not None and r["token_hash"] == ratings.hash_token(token)
    assert token not in str(dict(r))
    assert ratings.rater_for(con, "wrong-token") is None


def test_priority_order_and_target(con):
    ratings.load_tasks(con, [task("a", 1, priority=3), task("b", 2, priority=1),
                             task("c", 3, priority=2)])
    _, r = rater(con)
    order = []
    while (t := ratings.next_task(con, r)):
        order.append(t["task_id"])
        ratings.record(con, r, t["task_id"], "correct", None, None)
    assert order == ["b", "c", "a"]


def test_task_leaves_queue_once_target_reached(con):
    ratings.load_tasks(con, [task("a", 1, target=1), task("b", 2, target=2)])
    _, r1 = rater(con)
    _, r2 = rater(con)
    for r in (r1,):
        while (t := ratings.next_task(con, r)):
            ratings.record(con, r, t["task_id"], "correct", None, None)
    t = ratings.next_task(con, r2)
    assert t["task_id"] == "b"           # "a" already has its one rating


def test_only_rater_dialects_and_no_double_rating(con):
    ratings.load_tasks(con, [task("a", 1, dialect="oshindonga"), task("b", 2)])
    _, r = rater(con, ("oshikwanyama",))
    t = ratings.next_task(con, r)
    assert t["task_id"] == "b"
    ratings.record(con, r, "b", "minor", None, 1200)
    assert ratings.next_task(con, r) is None
    with pytest.raises(ValueError):
        ratings.record(con, r, "a", "correct", None, None)   # other dialect


def test_control_served_every_tenth(con):
    ratings.load_tasks(con, [task(f"t{i}", i, target=5) for i in range(20)]
                       + [task("ctl", 99, kind="control", expected="wrong", priority=9)])
    _, r = rater(con)
    served = []
    for _ in range(12):
        t = ratings.next_task(con, r)
        served.append(t["task_id"])
        ratings.record(con, r, t["task_id"], "wrong" if t["task_id"] == "ctl" else "correct", None, None)
    assert served.index("ctl") == ratings.CONTROL_EVERY - 1


def test_rater_never_sees_kind_or_source(con):
    ratings.load_tasks(con, [task("a", 1, source="kaarina")])
    _, r = rater(con)
    t = ratings.next_task(con, r)
    assert set(t) == {"task_id", "dialect", "english", "text"}


def test_suggestion_is_pii_scrubbed(con):
    ratings.load_tasks(con, [task("a", 1)])
    _, r = rater(con)
    ratings.record(con, r, "a", "wrong", "Better: ... call me on 081 234 5678", None)
    s = con.execute("SELECT suggestion FROM ratings").fetchone()[0]
    assert "[REDACTED:phone]" in s and "081" not in s


def test_unknown_verdict_rejected(con):
    ratings.load_tasks(con, [task("a", 1)])
    _, r = rater(con)
    with pytest.raises(ValueError):
        ratings.record(con, r, "a", "great", None, None)


def test_report_counts_controls_per_rater(con):
    ratings.load_tasks(con, [task("a", 1, source="kaarina"),
                             task("ctl", 2, kind="control", expected="wrong")])
    _, r = rater(con)
    ratings.record(con, r, "a", "correct", None, None)
    ratings.record(con, r, "ctl", "correct", None, None)   # missed the control
    rep = ratings.report(con)
    assert rep["by_source"]["kaarina/oshikwanyama"]["correct"] == 1
    assert rep["raters"][0]["controls"] == 1 and rep["raters"][0]["controls_ok"] == 0
