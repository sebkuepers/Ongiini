"""Tests for the side-by-side translation-rating store."""
from __future__ import annotations

import pytest

from ongiini import ratings


@pytest.fixture
def con(tmp_path):
    c = ratings.connect(tmp_path / "r.sqlite")
    yield c
    c.close()


def screen(sid, item, dialect="oshikwanyama", priority=2, target=1, control=False):
    cands = [{"cand_id": f"{sid}-k", "text": f"ref {item}", "kind": "ref", "source": "kaarina"},
             {"cand_id": f"{sid}-c", "text": f"claude {item}", "kind": "model", "source": "claude-opus-5"}]
    cands.append({"cand_id": f"{sid}-x", "text": "other sentence", "kind": "control",
                  "source": "kaarina-ref-of-9", "expected": "no"} if control else
                 {"cand_id": f"{sid}-g", "text": f"gemma {item}", "kind": "model", "source": "gemma-4-26b"})
    return {"screen_id": sid, "round": "r1", "item_id": item, "dialect": dialect,
            "english": f"en {item}", "priority": priority, "target": target, "candidates": cands}


def rater(con, dialects=("oshikwanyama",), label="R"):
    token = ratings.add_rater(con, label, list(dialects))
    return token, ratings.rater_for(con, token)


def answer_all(con, r, s, verdict="send", choice=None):
    ids = [c["cid"] for c in s["candidates"]]
    ratings.record(con, r, s["screen_id"], verdicts={i: verdict for i in ids}, wrong_dialect=[],
                   choice=choice or ids[0])


def test_token_stored_hashed(con):
    token, r = rater(con)
    assert r["token_hash"] == ratings.hash_token(token) and token not in str(dict(r))
    assert ratings.rater_for(con, "nope") is None


def test_priority_order_and_targets(con):
    ratings.load_screens(con, [screen("a", 1, priority=3), screen("b", 2, priority=1),
                               screen("c", 3, priority=2, target=2)])
    _, r1 = rater(con)
    order = []
    while (s := ratings.next_screen(con, r1)):
        order.append(s["screen_id"])
        answer_all(con, r1, s)
    assert order == ["b", "c", "a"]
    _, r2 = rater(con)
    assert ratings.next_screen(con, r2)["screen_id"] == "c"   # only c wants a 2nd rater


def test_screen_never_reveals_kind_or_source(con):
    ratings.load_screens(con, [screen("a", 1, control=True)])
    _, r = rater(con)
    s = ratings.next_screen(con, r)
    assert set(s) == {"screen_id", "dialect", "english", "candidates"}
    assert all(set(c) == {"cid", "text"} for c in s["candidates"])
    assert "kaarina" not in str(s) and "control" not in str(s)


def test_candidate_order_differs_between_raters(con):
    ratings.load_screens(con, [screen(f"s{i}", i, target=2) for i in range(12)])
    _, r1 = rater(con, label="A")
    _, r2 = rater(con, label="B")
    orders = set()
    for _ in range(12):
        for r in (r1, r2):
            s = ratings.next_screen(con, r)
            orders.add(tuple(c["cid"][-1] for c in s["candidates"]))
            answer_all(con, r, s)
    assert len(orders) > 1


def test_every_candidate_needs_a_verdict_and_a_valid_choice(con):
    ratings.load_screens(con, [screen("a", 1)])
    _, r = rater(con)
    s = ratings.next_screen(con, r)
    ids = [c["cid"] for c in s["candidates"]]
    with pytest.raises(ValueError):
        ratings.record(con, r, "a", verdicts={ids[0]: "send"}, wrong_dialect=[], choice=ids[0])
    with pytest.raises(ValueError):
        ratings.record(con, r, "a", verdicts={i: "great" for i in ids}, wrong_dialect=[], choice=ids[0])
    with pytest.raises(ValueError):
        ratings.record(con, r, "a", verdicts={i: "send" for i in ids}, wrong_dialect=[], choice="bogus")
    ratings.record(con, r, "a", verdicts={i: "no" for i in ids}, wrong_dialect=[ids[1]],
                   choice="none", suggestion="Better one, call 081 234 5678")
    row = con.execute("SELECT choice, suggestion FROM screen_ratings").fetchone()
    assert row["choice"] == "none" and "[REDACTED:phone]" in row["suggestion"]
    assert con.execute("SELECT SUM(wrong_dialect) FROM candidate_ratings").fetchone()[0] == 1


def test_skip_needs_no_verdicts_and_counts_as_done(con):
    ratings.load_screens(con, [screen("a", 1), screen("b", 2)])
    _, r = rater(con)
    ratings.record(con, r, "a", verdicts=None, wrong_dialect=None, choice=None, skipped=True)
    assert ratings.progress(con, r["rater_id"])["done"] == 1
    assert ratings.next_screen(con, r)["screen_id"] == "b"


def test_other_dialect_rejected(con):
    ratings.load_screens(con, [screen("a", 1, dialect="oshindonga")])
    _, r = rater(con, ("oshikwanyama",))
    assert ratings.next_screen(con, r) is None
    with pytest.raises(ValueError):
        ratings.record(con, r, "a", verdicts={}, wrong_dialect=[], choice="none")


def test_report_absolute_relative_and_controls(con):
    ratings.load_screens(con, [screen("a", 1), screen("b", 2, control=True)])
    _, r = rater(con)
    ratings.record(con, r, "a", verdicts={"a-k": "send", "a-c": "fix", "a-g": "no"},
                   wrong_dialect=[], choice="a-k")
    ratings.record(con, r, "b", verdicts={"b-k": "send", "b-c": "send", "b-x": "send"},
                   wrong_dialect=[], choice="several")                      # missed the control
    rep = ratings.report(con)
    k = rep["by_source"]["kaarina/oshikwanyama"]
    assert k["send"] == 2 and k["chosen"] == 1 and k["shown"] == 2 and k["send_pct"] == 100.0
    assert rep["raters"][0]["controls"] == 1 and rep["raters"][0]["controls_ok"] == 0


def test_delete_round_and_rater(con):
    ratings.load_screens(con, [screen("a", 1)])
    _, r = rater(con)
    answer_all(con, r, ratings.next_screen(con, r))
    assert ratings.delete_round(con, "r1") == 1
    ratings.delete_rater(con, r["rater_id"])
    for t in ("screens", "candidates", "screen_ratings", "candidate_ratings", "raters"):
        assert con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0
