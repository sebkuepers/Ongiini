"""Scoring core for Ongiini-Eval-OW — implements docs/eval-protocol.md.

Pure functions, no I/O policy: run_evaluation.py decides what to score and
where results go. Corpus metrics are computed from sacrebleu's per-segment
sufficient statistics, so bootstrap resampling and approximate
randomisation re-use one extraction per system and stay exact
(the summed statistics give the same score as ``corpus_score``).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

from retry_derailed_baselines import is_derailed

PROTOCOL_VERSION = "eval-protocol-v1 (2026-10-01)"
DIALECTS = ("oshindonga", "oshikwanyama")
METRICS = {
    "chrf++": CHRF(word_order=2),          # primary
    "spbleu": BLEU(tokenize="flores200"),  # secondary
    "chrf": CHRF(word_order=0),            # robustness
}
PRIMARY = "chrf++"
BOOTSTRAP_N, AR_TRIALS, SEED, ALPHA = 1000, 10_000, 42, 0.05
SMALL_N = 15
COPY_THRESHOLD = 60.0                      # chrF++ of output vs English source

_QUOTES = [('"', '"'), ("“", "”"), ("„", "“"), ("‘", "’"), ("'", "'"), ("«", "»")]


def normalise(text: str | None) -> str:
    """NFC, every whitespace run (incl. line breaks) → one space, trim."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def clean_hypothesis(text: str | None, dialect: str) -> str:
    """Same for every system: normalise, drop one leading '<Dialect>:' label
    and one pair of surrounding quotation marks."""
    t = normalise(text)
    label = dialect.capitalize() + ":"
    if t.lower().startswith(label.lower()):
        t = t[len(label):].strip()
    for a, b in _QUOTES:
        if len(t) >= 2 and t.startswith(a) and t.endswith(b):
            t = t[len(a):-len(b)].strip()
            break
    return t


# ── data ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def load_items(eval_set: Path) -> dict[int, dict]:
    return {int(r["id"]): r for r in load_jsonl(eval_set)}


def reference_fingerprint(items: dict[int, dict]) -> str:
    h = hashlib.sha256()
    for i in sorted(items):
        for d in DIALECTS:
            h.update(f"{i}\t{d}\t{normalise(items[i].get(f'{d}_reference'))}\n".encode())
    return h.hexdigest()[:16]


class InputError(ValueError):
    pass


def load_system_file(path: Path, dialect: str, ids: list[int]) -> tuple[dict[int, str], set[str]]:
    """Strict: every id in ``ids`` exactly once, dialect field matches, returns
    {id: raw translation} and the set of prompt_template_ids found."""
    rows = load_jsonl(path)
    out: dict[int, str] = {}
    templates: set[str] = set()
    for r in rows:
        i = int(r["id"])
        if r.get("dialect", dialect) != dialect:
            raise InputError(f"{path.name}: id {i} has dialect {r.get('dialect')!r}, expected {dialect}")
        if i in out:
            raise InputError(f"{path.name}: duplicate id {i}")
        out[i] = r.get("translation", r.get(f"{dialect}_output", "")) or ""
        templates.add(r.get("prompt_template_id", "legacy"))
    missing = [i for i in ids if i not in out]
    if missing:
        raise InputError(f"{path.name}: {len(missing)} scored ids missing, e.g. {missing[:5]}")
    if len(templates) > 1:
        raise InputError(f"{path.name}: mixed prompt templates {sorted(templates)}")
    return {i: out[i] for i in ids}, templates


def derangement(ids: list[int], seed: int = SEED) -> dict[int, int]:
    """Map each id to a different id (no fixed points), reproducibly."""
    rng = random.Random(seed)
    while True:
        perm = ids[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(ids, perm)):
            return dict(zip(ids, perm))


# ── metrics from sufficient statistics ───────────────────────────────

@dataclass
class SystemStats:
    hyps: list[str]
    refs: list[str]
    stats: dict[str, np.ndarray] = field(default_factory=dict)   # metric → (n, k)

    @classmethod
    def build(cls, hyps: list[str], refs: list[str]) -> "SystemStats":
        s = cls(hyps, refs)
        for name, m in METRICS.items():
            s.stats[name] = np.asarray(m._extract_corpus_statistics(hyps, [refs]), dtype=float)
        return s


def score_from(metric: str, summed: np.ndarray) -> float:
    return METRICS[metric]._compute_score_from_stats(summed.tolist()).score


def corpus_scores(s: SystemStats, idx: np.ndarray | None = None) -> dict[str, float]:
    return {m: score_from(m, (st if idx is None else st[idx]).sum(0)) for m, st in s.stats.items()}


def signatures() -> dict[str, str]:
    for metric in METRICS.values():            # sacrebleu needs one scored call (nrefs)
        metric.corpus_score(["a"], [["a"]])
    return {m: str(metric.get_signature()) for m, metric in METRICS.items()}


def bootstrap_indices(n: int, samples: int = BOOTSTRAP_N, seed: int = SEED) -> np.ndarray:
    """One set of resampled item indices shared by all systems (paired)."""
    return np.random.default_rng(seed).integers(0, n, size=(samples, n))


def bootstrap_ci(s: SystemStats, boot: np.ndarray, metric: str = PRIMARY) -> tuple[float, float]:
    st = s.stats[metric]
    sums = st[boot].sum(axis=1)                                    # (samples, k)
    vals = np.sort([score_from(metric, row) for row in sums])
    return float(vals[int(0.025 * len(vals))]), float(vals[int(0.975 * len(vals)) - 1])


def approx_randomisation(a: SystemStats, b: SystemStats, metric: str = PRIMARY,
                         trials: int = AR_TRIALS, seed: int = SEED) -> float:
    """Paired approximate randomisation (two-sided): swap each item's
    statistics between the two systems with p = 0.5."""
    sa, sb = a.stats[metric], b.stats[metric]
    observed = abs(score_from(metric, sa.sum(0)) - score_from(metric, sb.sum(0)))
    rng = np.random.default_rng(seed)
    swap = rng.random((trials, sa.shape[0])) < 0.5
    diff = sb - sa
    sum_a = sa.sum(0) + swap.astype(float) @ diff                  # (trials, k)
    sum_b = sb.sum(0) - swap.astype(float) @ diff
    hits = sum(abs(score_from(metric, x) - score_from(metric, y)) >= observed - 1e-12
               for x, y in zip(sum_a, sum_b))
    return (hits + 1) / (trials + 1)


def holm(pvals: dict[tuple[str, str], float], alpha: float = ALPHA) -> dict[tuple[str, str], bool]:
    """Holm–Bonferroni: True where the pair differs significantly."""
    order = sorted(pvals, key=pvals.get)
    m, out, stop = len(order), {}, False
    for k, pair in enumerate(order):
        if not stop and pvals[pair] <= alpha / (m - k):
            out[pair] = True
        else:
            stop = True
            out[pair] = False
    return out


def clusters(ranked: list[str], significant: dict[tuple[str, str], bool]) -> list[list[str]]:
    """WMT-style: sorted best→worst; a new cluster starts at the first system
    that is significantly worse than every member of the current cluster."""
    def sig(a, b):
        return significant.get((a, b), significant.get((b, a), False))
    out: list[list[str]] = []
    for name in ranked:
        if out and not all(sig(name, m) for m in out[-1]):
            out[-1].append(name)
        else:
            out.append([name])
    return out


def failure_rates(english: list[str], raw: list[str], cleaned: list[str]) -> dict[str, float]:
    n = len(cleaned)
    copy_metric = METRICS[PRIMARY]
    copies = sum(copy_metric.sentence_score(h, [e]).score > COPY_THRESHOLD
                 for e, h in zip(english, cleaned) if h)
    return {"empty_pct": round(100 * sum(not h for h in cleaned) / n, 1),
            "derailed_pct": round(100 * sum(is_derailed(e, r) for e, r in zip(english, raw)) / n, 1),
            "english_copy_pct": round(100 * copies / n, 1)}


def sentence_chrf(hyps: list[str], refs: list[str]) -> list[float]:
    m = METRICS[PRIMARY]
    return [round(m.sentence_score(h, [r]).score, 2) for h, r in zip(hyps, refs)]


__all__ = [n for n in dir() if not n.startswith("_")]
_ = sacrebleu  # version is part of the signatures
