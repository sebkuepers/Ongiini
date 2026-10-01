"""Tests for the Ongiini-Eval-OW scoring core (docs/eval-protocol.md).

Synthetic data only — no references, no system outputs. Run with the
eval venv (sacrebleu 2.4.3, numpy):

    ~/.venvs/ongiini-eval/bin/python -m pytest scripts/test_eval_scoring.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
E = pytest.importorskip("eval_scoring")
S = pytest.importorskip("system_similarity")

HYPS = ["omukwaniilwa okwa ya kegumbo", "ondi li nawa", "", "okwa li a hala okulya", "eendja odi li"]
REFS = ["omukwaniilwa okwa ya keumbo", "ondi li nawa unene", "otwa ya", "okwa hala okulya", "eendja oda pya"]


# ── normalisation ────────────────────────────────────────────────────

def test_normalise_nfc_and_whitespace():
    assert E.normalise("  á\n\tb  ") == "á b"
    assert E.normalise(None) == ""


def test_clean_hypothesis_label_and_quotes_once():
    assert E.clean_hypothesis('Oshindonga: "Ondi li nawa"', "oshindonga") == "Ondi li nawa"
    assert E.clean_hypothesis("oshikwanyama:  “Nawa”", "oshikwanyama") == "Nawa"
    assert E.clean_hypothesis('""x""', "oshindonga") == '"x"'           # one pair only
    assert E.clean_hypothesis("Oshindonga: x", "oshikwanyama") == "Oshindonga: x"


# ── input checks ─────────────────────────────────────────────────────

def _write(tmp_path, rows):
    p = tmp_path / "sys_oshindonga.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def _row(i, t="x", template="t1", dialect="oshindonga"):
    return {"id": i, "dialect": dialect, "translation": t, "prompt_template_id": template}


def test_load_system_file_strict(tmp_path):
    raw, templates = E.load_system_file(_write(tmp_path, [_row(1), _row(2, "y")]), "oshindonga", [2, 1])
    assert raw == {2: "y", 1: "x"} and templates == {"t1"}
    with pytest.raises(E.InputError, match="missing"):
        E.load_system_file(_write(tmp_path, [_row(1)]), "oshindonga", [1, 2])
    with pytest.raises(E.InputError, match="duplicate"):
        E.load_system_file(_write(tmp_path, [_row(1), _row(1)]), "oshindonga", [1])
    with pytest.raises(E.InputError, match="mixed"):
        E.load_system_file(_write(tmp_path, [_row(1), _row(2, template="t2")]), "oshindonga", [1, 2])
    with pytest.raises(E.InputError, match="dialect"):
        E.load_system_file(_write(tmp_path, [_row(1, dialect="oshikwanyama")]), "oshindonga", [1])


# ── reference points ─────────────────────────────────────────────────

def test_derangement_has_no_fixed_points_and_is_reproducible():
    ids = list(range(1, 51))
    d = E.derangement(ids)
    assert sorted(d) == ids and sorted(d.values()) == ids
    assert all(k != v for k, v in d.items())
    assert d == E.derangement(ids)


# ── metrics from sufficient statistics ───────────────────────────────

def test_summed_statistics_equal_corpus_score():
    s = E.SystemStats.build(HYPS, REFS)
    got = E.corpus_scores(s)
    for name, metric in E.METRICS.items():
        assert got[name] == pytest.approx(metric.corpus_score(HYPS, [REFS]).score, abs=1e-9)


def test_signatures_pin_the_protocol():
    sig = E.signatures()
    assert "nc:6|nw:2" in sig["chrf++"] and "version:2.4.3" in sig["chrf++"]
    assert "nw:0" in sig["chrf"]
    assert "tok:flores200" in sig["spbleu"]


def test_bootstrap_paired_and_reproducible():
    a = E.bootstrap_indices(5, samples=200)
    assert a.shape == (200, 5) and np.array_equal(a, E.bootstrap_indices(5, samples=200))
    s = E.SystemStats.build(HYPS, REFS)
    lo, hi = E.bootstrap_ci(s, a)
    assert lo <= E.corpus_scores(s)["chrf++"] <= hi


def test_approx_randomisation():
    s = E.SystemStats.build(HYPS, REFS)
    assert E.approx_randomisation(s, s, trials=500) == pytest.approx(1.0)   # identical → p = 1
    good = E.SystemStats.build(REFS * 8, REFS * 8)
    bad = E.SystemStats.build(["zzz"] * 40, REFS * 8)
    p = E.approx_randomisation(good, bad, trials=500)
    assert p < 0.01 and p == E.approx_randomisation(good, bad, trials=500)


def test_holm():
    pv = {("a", "b"): 0.001, ("a", "c"): 0.02, ("b", "c"): 0.04}
    # thresholds 0.05/3, 0.05/2, 0.05/1 → first two pass, third passes too
    assert E.holm(pv) == {("a", "b"): True, ("a", "c"): True, ("b", "c"): True}
    pv[("a", "c")] = 0.03                                                      # > 0.025 → stop
    assert E.holm(pv) == {("a", "b"): True, ("a", "c"): False, ("b", "c"): False}


def test_clusters():
    sig = {("a", "c"): True, ("b", "c"): True, ("a", "b"): False}
    assert E.clusters(["a", "b", "c"], sig) == [["a", "b"], ["c"]]
    assert E.clusters(["a", "b", "c"], {}) == [["a", "b", "c"]]


def test_failure_rates():
    en = ["The king went home", "I am fine", "We left"]
    fr = E.failure_rates(en, ["x", "I am fine", ""], ["x", "I am fine", ""])   # item 2 copies its source
    assert fr["empty_pct"] == pytest.approx(33.3) and fr["english_copy_pct"] == pytest.approx(33.3)


# ── similarity ───────────────────────────────────────────────────────

def test_similarity_symmetric_and_ordered():
    out = S.similarity({"a": HYPS, "b": HYPS, "c": ["zzz"] * 5}, REFS)
    assert out["pairs"]["a|b"]["raw"] == pytest.approx(100.0)
    assert out["pairs"]["a|c"]["raw"] < 10
    assert sorted(out["order"]) == ["a", "b", "c"]
    assert abs(out["order"].index("a") - out["order"].index("b")) == 1   # identical systems adjacent


# ── language ID grouping ─────────────────────────────────────────────

def test_lid_groups():
    L = pytest.importorskip("lid_outputs")          # needs fasttext (eval venv)
    assert L.group("ndo_Latn") == "Oshindonga" and L.group("kua_Latn") == "Oshikwanyama"
    assert L.group("swh_Latn") == "Swahili" and L.group("xyz_Latn") == "other"
