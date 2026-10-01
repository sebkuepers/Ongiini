"""System-to-system similarity for Ongiini-Eval-OW (docs/eval-protocol.md §10).

Exploratory. For each dialect, on the development items with a reference:

* ``raw``      symmetric corpus chrF++ between two systems' outputs
                (mean of A-as-hypothesis-vs-B and B-vs-A);
* ``shared_errors``  the same, restricted to items **both** systems got
                wrong (sentence chrF++ vs the reference < WRONG); shared
                wrong outputs are the signal for shared sources;
* ``excess``   shared_errors minus what their closeness to the reference
                alone predicts (mean over items of min(chrF_A, chrF_B) vs
                the reference on those items — two systems that are both
                near the reference are near each other for that reason);
* average-linkage clustering on (100 - raw) for the dendrogram order.

Read as a hint about shared training sources or strategies, never as
evidence about training data.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

import eval_scoring as E

WRONG = 30.0
MIN_SHARED = 20


def _chrf(h: list[str], r: list[str]) -> float:
    return E.METRICS[E.PRIMARY].corpus_score(h, [r]).score


def similarity(outputs: dict[str, list[str]], refs: list[str]) -> dict:
    names = sorted(outputs)
    sent = {n: E.sentence_chrf(outputs[n], refs) for n in names}
    pairs = {}
    for a, b in combinations(names, 2):
        ha, hb = outputs[a], outputs[b]
        raw = (_chrf(ha, hb) + _chrf(hb, ha)) / 2
        both = [i for i in range(len(refs)) if sent[a][i] < WRONG and sent[b][i] < WRONG
                and ha[i] and hb[i]]
        entry = {"raw": round(raw, 1), "n_both_wrong": len(both)}
        if len(both) >= MIN_SHARED:
            sa, sb = [ha[i] for i in both], [hb[i] for i in both]
            shared = (_chrf(sa, sb) + _chrf(sb, sa)) / 2
            expected = float(np.mean([min(sent[a][i], sent[b][i]) for i in both]))
            entry |= {"shared_errors": round(shared, 1), "expected": round(expected, 1),
                      "excess": round(shared - expected, 1)}
        pairs[f"{a}|{b}"] = entry
    return {"names": names, "pairs": pairs, "order": _cluster_order(names, pairs)}


def _cluster_order(names: list[str], pairs: dict) -> list[str]:
    """Average-linkage agglomeration on distance 100 - raw; returns leaf order."""
    def dist(a, b):
        k = f"{a}|{b}" if f"{a}|{b}" in pairs else f"{b}|{a}"
        return 100 - pairs[k]["raw"]
    clusters = [[n] for n in names]
    while len(clusters) > 1:
        best = None
        for i, j in combinations(range(len(clusters)), 2):
            d = np.mean([dist(a, b) for a in clusters[i] for b in clusters[j]])
            if best is None or d < best[0]:
                best = (d, i, j)
        _, i, j = best
        merged = clusters[i] + clusters[j]
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]
    return clusters[0]


def dialect_self_similarity(odg: list[str], okw: list[str]) -> float:
    """How alike a system's Oshindonga and Oshikwanyama outputs are."""
    return round((_chrf(odg, okw) + _chrf(okw, odg)) / 2, 1)
