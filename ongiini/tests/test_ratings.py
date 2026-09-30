"""Tests for the v3 rating store: one translation per task, grouped items,
pairing and control rules, repeats, group mixing."""
from __future__ import annotations

import json
import random

import pytest

from ongiini import ratings


@pytest.fixture
def con(tmp_path):
    c = ratings.connect(tmp_path / "r.sqlite")
    yield c
    c.close()


def item(key, sid, grp, kind="ref", target=1, pair_of=None, severity=0, dialect="oshikwanyama",
         expected=None, source=None):
    return {"item_key": key, "round": "r1", "sentence_id": sid, "dialect": dialect,
            "english": f"en {sid}", "text": f"{grp} {sid}", "grp": grp, "kind": kind,
            "source": source or grp, "expected": expected, "target": target, "pair_of": pair_of,
            "severity": severity}


def full_set(n=40):
    items = []
    for s in range(n):
        items.append(item(f"ref{s}", s, "random_ref"))
        items.append(item(f"cl{s}", s, "claude_pair", kind="model", pair_of=f"ref{s}"))
    for s in range(100, 110):
        items.append(item(f"flag{s}", s, "flagged_ref", target=2, severity=s % 3))
        items.append(item(f"gem{s}", s, "gemma", kind="model"))
    for s in range(200, 215):
        items.append(item(f"err{s}", s, "control_error", kind="control", expected="wrong"))
        items.append(item(f"ref{s}", s, "random_ref"))          # same sentence as a control
    items.append(item("wrong1", 300, "control_wrong", kind="control", expected="wrong"))
    items.append({**item("prac1", 400, "practice", kind="practice", expected="wrong"),
                  "explanation": "The number does not match the English."})
    return items


def rater(con, dialects=("oshikwanyama",), label="R"):
    token = ratings.add_rater(con, label, list(dialects))
    return token, ratings.rater_for(con, token)


def run(con, r, n, verdict="good"):
    served = []
    for _ in range(n):
        it = ratings.next_item(con, r)
        if it is None:
            break
        served.append(it["item_key"])
        ratings.record(con, r, it["item_key"], verdict)
    return served


def test_token_stored_hashed(con):
    token, r = rater(con)
    assert r["token_hash"] == ratings.hash_token(token) and token not in str(dict(r))
    assert ratings.rater_for(con, "nope") is None


def test_items_never_reveal_origin_but_practice_has_feedback(con):
    ratings.load_items(con, full_set())
    _, r = rater(con)
    it = ratings.next_item(con, r)
    assert set(it) == {"item_key", "dialect", "english", "text"}
    prac = ratings.practice_items(con, r)
    assert prac and prac[0]["expected"] == "wrong" and prac[0]["explanation"]
    with pytest.raises(ValueError):
        ratings.record(con, r, "prac1", "wrong")       # practice is never scored


def test_claude_pair_only_to_same_rater_after_gap(con):
    ratings.load_items(con, full_set())
    _, a = rater(con, label="A")
    _, b = rater(con, label="B")
    served_a = run(con, a, 40)
    served_b = run(con, b, 40)
    pos_a = {}
    for i, k in enumerate(served_a):
        pos_a.setdefault(k, i)                         # first sighting, not the repeat
    for k in served_a:
        if k.startswith("cl"):
            ref = "ref" + k[2:]
            assert ref in pos_a and pos_a[k] - pos_a[ref] >= ratings.PAIR_GAP
    for k in served_b:
        if k.startswith("cl"):
            assert "ref" + k[2:] in served_b                # never someone else's pair


def test_gemma_never_to_rater_who_saw_that_reference(con):
    items = [item("flag1", 1, "flagged_ref", target=2), item("gem1", 1, "gemma", kind="model")]
    ratings.load_items(con, items)
    _, a = rater(con)
    ratings.record(con, a, "flag1", "good")
    assert ratings.next_item(con, a) is None
    _, b = rater(con, label="B")
    served = run(con, b, 5)
    assert served[0] in ("flag1", "gem1") and not ({"flag1", "gem1"} <= set(served))


def test_control_error_and_reference_of_same_sentence_exclude_each_other(con):
    ratings.load_items(con, [item("err1", 1, "control_error", kind="control", expected="wrong"),
                             item("ref1", 1, "random_ref")])
    _, r = rater(con)
    served = run(con, r, 5)
    assert len(served) == 1


def test_repeats_after_gap_marked_and_capped(con):
    ratings.load_items(con, full_set(60))
    _, r = rater(con)
    run(con, r, 80)
    rows = con.execute("SELECT item_key, seq, is_repeat FROM ratings WHERE rater_id = ?",
                       (r["rater_id"],)).fetchall()
    firsts = {x["item_key"]: x["seq"] for x in rows if not x["is_repeat"]}
    reps = [x for x in rows if x["is_repeat"]]
    assert 0 < len(reps) <= ratings.MAX_REPEATS
    for x in reps:
        assert x["seq"] - firsts[x["item_key"]] >= ratings.REPEAT_GAP
        assert not x["item_key"].startswith(("err", "wrong"))


def test_sessions_mix_groups_from_the_start(con):
    ratings.load_items(con, full_set())
    _, r = rater(con)
    first20 = run(con, r, 20)
    groups = {k.rstrip("0123456789") for k in first20}
    assert {"ref", "flag"} <= groups and ({"err", "wrong"} & groups)


def test_flagged_items_get_two_raters(con):
    ratings.load_items(con, [item("flag1", 1, "flagged_ref", target=2)])
    _, a = rater(con, label="A")
    _, b = rater(con, label="B")
    _, c = rater(con, label="C")
    assert run(con, a, 3) == ["flag1"] and run(con, b, 3) == ["flag1"] and run(con, c, 3) == []


def test_record_validation_and_scrubbing(con):
    ratings.load_items(con, [item("ref1", 1, "random_ref"), item("ref2", 2, "random_ref"),
                             item("ref3", 3, "random_ref")])
    _, r = rater(con)
    with pytest.raises(ValueError):
        ratings.record(con, r, "ref1", "great")
    with pytest.raises(ValueError):
        ratings.record(con, r, "ref1", "wrong", issues=["meaning"])
    with pytest.raises(ValueError):
        ratings.record(con, r, "ref1", "cant_judge", cant_reason="tired")
    ratings.record(con, r, "ref1", "wrong", issues=["word_choice", "unnatural"],
                   suggestion="better, call 081 234 5678")
    ratings.record(con, r, "ref2", "good", issues=["word_choice"])      # issues ignored on good
    ratings.record(con, r, "ref3", "cant_judge", cant_reason="english")
    rows = {x["item_key"]: x for x in con.execute("SELECT * FROM ratings")}
    assert json.loads(rows["ref1"]["issues"]) == ["unnatural", "word_choice"]
    assert "[REDACTED:phone]" in rows["ref1"]["suggestion"]
    assert rows["ref2"]["issues"] is None and rows["ref3"]["cant_reason"] == "english"


def test_other_dialect_rejected(con):
    ratings.load_items(con, [item("ref1", 1, "random_ref", dialect="oshindonga")])
    _, r = rater(con, ("oshikwanyama",))
    assert ratings.next_item(con, r) is None
    with pytest.raises(ValueError):
        ratings.record(con, r, "ref1", "good")


def test_simulation_three_raters_rules_hold(con):
    ratings.load_items(con, full_set(50))
    rs = [rater(con, label=f"R{i}")[1] for i in range(3)]
    rng = random.Random(1)
    active = list(rs)
    while active:
        r = rng.choice(active)
        it = ratings.next_item(con, r)
        if it is None:
            active.remove(r)
            continue
        ratings.record(con, r, it["item_key"], rng.choice(["good", "almost", "wrong"]))
    rows = con.execute("SELECT r.rater_id, r.item_key, r.seq, r.is_repeat, i.grp, i.sentence_id, "
                       "i.pair_of FROM ratings r JOIN items i USING (item_key)").fetchall()
    by_rater: dict[int, dict] = {}
    for x in rows:
        by_rater.setdefault(x["rater_id"], {})[(x["item_key"], x["is_repeat"])] = x
    for rid, mine in by_rater.items():
        firsts = {k[0]: v for k, v in mine.items() if not k[1]}
        ref_like = [v["sentence_id"] for v in firsts.values() if v["grp"] in ratings.REF_LIKE]
        assert len(ref_like) == len(set(ref_like))            # one reference-like per sentence
        for v in firsts.values():
            if v["grp"] == "claude_pair":
                assert firsts[v["pair_of"]]["seq"] + ratings.PAIR_GAP <= v["seq"]
    n_per = dict(con.execute("SELECT item_key, COUNT(DISTINCT rater_id) FROM ratings GROUP BY 1").fetchall())
    for key, n in n_per.items():
        if key.startswith("flag"):
            assert n <= 2
        elif key.startswith(("ref", "cl", "gem")):
            assert n == 1


def test_delete_round_and_rater(con):
    ratings.load_items(con, full_set(5))
    _, r = rater(con)
    run(con, r, 3)
    assert ratings.delete_round(con, "r1") > 0
    ratings.delete_rater(con, r["rater_id"])
    assert con.execute("SELECT COUNT(*) FROM ratings").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM raters").fetchone()[0] == 0
