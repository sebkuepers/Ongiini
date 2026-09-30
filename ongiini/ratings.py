"""Human rating of translations — store and assignment (v3).

Backs ongiini.ai/rate/, where invited native speakers judge translations
one at a time: "Is this a good translation?" good | almost | wrong, with
optional issue tags and a better translation after almost/wrong, or
"can't judge" with a reason. Round 1 validates the translator's
references (see the plan in docs / scripts/build_rating_tasks.py).

Items are one translation of one sentence, in a group:
  random_ref     unbiased sample of the references
  flagged_ref    references flagged by back-translation (2 raters each)
  claude_pair    Claude Opus 5 for a sentence whose reference THIS rater
                 already judged, at least PAIR_GAP tasks earlier (paired
                 comparison within rater)
  gemma          Gemma 4 26B, only to raters who did NOT see that
                 sentence's reference (discrimination check)
  control_error  the reference with a planted error (number, place,
                 missing clause) — expected "wrong"; never shown to a
                 rater who saw the real reference of that sentence
  control_wrong  the reference of a different sentence — attention check
  practice       onboarding examples with feedback (sent with the session,
                 never assigned or scored)
A rater also gets a few REPEATS of items they rated at least REPEAT_GAP
tasks earlier, to measure consistency.

Every session mixes the groups in fixed proportions (weighted round
robin), so an early drop-out still leaves a usable random sample.

Raters are invited by personal link; only sha256(token) is stored, plus a
pseudonymous label and dialects. No phone numbers. Suggestions are
PII-scrubbed. SQLite at <data_dir>/ratings.sqlite.
"""
from __future__ import annotations

import hashlib
import json
import random
import secrets
import sqlite3
import time
from pathlib import Path

from . import pii
from .config import settings

VERDICTS = ("good", "almost", "wrong", "cant_judge")
ISSUES = ("word_choice", "spelling_grammar", "unnatural", "other_dialect")
CANT_REASONS = ("english", "unfamiliar_word", "other")
PAIR_GAP = 5
REPEAT_GAP = 15
MAX_REPEATS = 5
# Share of a rater's tasks per group (≈ the per-dialect budget in the plan).
WEIGHTS = {"random_ref": 45, "flagged_ref": 25, "claude_pair": 30, "gemma": 10,
           "control_error": 12, "control_wrong": 3, "repeat": 5}
REF_LIKE = ("random_ref", "flagged_ref", "control_error")

SCHEMA = """
CREATE TABLE IF NOT EXISTS raters (
  rater_id   INTEGER PRIMARY KEY,
  token_hash TEXT UNIQUE NOT NULL,
  label      TEXT NOT NULL,
  dialects   TEXT NOT NULL,
  created_at REAL NOT NULL,
  active     INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS items (
  item_key    TEXT PRIMARY KEY,           -- opaque to raters
  round       TEXT NOT NULL,
  sentence_id INTEGER NOT NULL,
  dialect     TEXT NOT NULL,
  english     TEXT NOT NULL,
  text        TEXT NOT NULL,
  grp         TEXT NOT NULL,
  kind        TEXT NOT NULL,              -- ref | model | control | practice (never sent)
  source      TEXT NOT NULL,              -- never sent
  expected    TEXT,                       -- controls / practice
  explanation TEXT,                       -- practice feedback
  target      INTEGER NOT NULL DEFAULT 1, -- raters wanted
  pair_of     TEXT,                       -- claude_pair: the reference item
  severity    INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ratings (
  rating_id   INTEGER PRIMARY KEY,
  item_key    TEXT NOT NULL REFERENCES items(item_key),
  rater_id    INTEGER NOT NULL REFERENCES raters(rater_id),
  verdict     TEXT NOT NULL,
  cant_reason TEXT,
  issues      TEXT,                       -- JSON list
  suggestion  TEXT,
  duration_ms INTEGER,
  is_repeat   INTEGER NOT NULL DEFAULT 0,
  seq         INTEGER NOT NULL,           -- the rater's n-th answer
  created_at  REAL NOT NULL,
  UNIQUE (item_key, rater_id, is_repeat)
);
"""


def db_path() -> Path:
    return settings.data_dir / "ratings.sqlite"


def connect(path: Path | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(str(path or db_path()))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ── admin ─────────────────────────────────────────────────────────────

def add_rater(con: sqlite3.Connection, label: str, dialects: list[str]) -> str:
    """Create a rater and return the raw invite token (shown once)."""
    token = secrets.token_urlsafe(24)
    with con:
        con.execute("INSERT INTO raters (token_hash, label, dialects, created_at) VALUES (?,?,?,?)",
                    (hash_token(token), label, ",".join(dialects), time.time()))
    return token


def load_items(con: sqlite3.Connection, items: list[dict]) -> int:
    cols = ("item_key", "round", "sentence_id", "dialect", "english", "text", "grp", "kind",
            "source", "expected", "explanation", "target", "pair_of", "severity")
    defaults = {"expected": None, "explanation": None, "target": 1, "pair_of": None, "severity": 0}
    with con:
        con.executemany(f"INSERT OR IGNORE INTO items ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                        [tuple({**defaults, **it}[c] for c in cols) for it in items])
    return con.execute("SELECT COUNT(*) FROM items").fetchone()[0]


def delete_round(con: sqlite3.Connection, round_: str) -> int:
    with con:
        con.execute("DELETE FROM ratings WHERE item_key IN (SELECT item_key FROM items WHERE round = ?)",
                    (round_,))
        return con.execute("DELETE FROM items WHERE round = ?", (round_,)).rowcount


def delete_rater(con: sqlite3.Connection, rater_id: int) -> None:
    with con:
        con.execute("DELETE FROM ratings WHERE rater_id = ?", (rater_id,))
        con.execute("DELETE FROM raters WHERE rater_id = ?", (rater_id,))


# ── rater-facing ──────────────────────────────────────────────────────

def rater_for(con: sqlite3.Connection, token: str) -> sqlite3.Row | None:
    return con.execute("SELECT * FROM raters WHERE token_hash = ? AND active = 1",
                       (hash_token(token),)).fetchone()


def progress(con: sqlite3.Connection, rater_id: int) -> dict:
    mine = con.execute("SELECT COUNT(*) FROM ratings WHERE rater_id = ?", (rater_id,)).fetchone()[0]
    everyone = con.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
    return {"done": mine, "everyone": everyone}


def practice_items(con: sqlite3.Connection, rater: sqlite3.Row) -> list[dict]:
    dialects = rater["dialects"].split(",")
    rows = con.execute(f"SELECT dialect, english, text, expected, explanation FROM items "
                       f"WHERE grp = 'practice' AND dialect IN ({','.join('?' * len(dialects))}) "
                       f"ORDER BY item_key", dialects).fetchall()
    return [dict(r) for r in rows]


def _public(row: sqlite3.Row) -> dict:
    return {"item_key": row["item_key"], "dialect": row["dialect"],
            "english": row["english"], "text": row["text"]}


def next_item(con: sqlite3.Connection, rater: sqlite3.Row) -> dict | None:
    """Pick the rater's next item: the group furthest behind its share,
    then within the group the least-rated / most severe item that the
    pairing rules allow."""
    rid = rater["rater_id"]
    dialects = rater["dialects"].split(",")
    marks = ",".join("?" * len(dialects))
    mine = con.execute(
        "SELECT r.item_key, r.seq, r.is_repeat, i.grp, i.sentence_id, i.dialect FROM ratings r "
        "JOIN items i USING (item_key) WHERE r.rater_id = ?", (rid,)).fetchall()
    seq = max((r["seq"] for r in mine), default=0)
    firsts = {r["item_key"]: r for r in mine if not r["is_repeat"]}
    repeats = {r["item_key"] for r in mine if r["is_repeat"]}
    seen_ref_like = {(r["sentence_id"], r["dialect"]) for r in firsts.values() if r["grp"] in REF_LIKE}
    seen_ref = {(r["sentence_id"], r["dialect"]) for r in firsts.values()
                if r["grp"] in ("random_ref", "flagged_ref")}
    counts = {g: 0 for g in WEIGHTS}
    for r in mine:
        g = "repeat" if r["is_repeat"] else r["grp"]
        counts[g] = counts.get(g, 0) + 1

    rows = con.execute(f"""
      SELECT i.*, (SELECT COUNT(DISTINCT rater_id) FROM ratings x WHERE x.item_key = i.item_key) AS n
      FROM items i WHERE i.dialect IN ({marks}) AND i.grp != 'practice'""", dialects).fetchall()

    def eligible(it) -> bool:
        key = (it["sentence_id"], it["dialect"])
        if it["item_key"] in firsts:
            return False
        g = it["grp"]
        if g in ("random_ref", "flagged_ref"):
            return it["n"] < it["target"] and key not in seen_ref_like
        if g == "claude_pair":
            ref = firsts.get(it["pair_of"])
            return it["n"] < it["target"] and ref is not None and seq - ref["seq"] >= PAIR_GAP
        if g == "gemma":
            return it["n"] < it["target"] and key not in seen_ref
        if g == "control_error":
            return key not in seen_ref_like
        if g == "control_wrong":
            return True
        return False

    pools: dict[str, list] = {g: [] for g in WEIGHTS}
    for it in rows:
        if eligible(it):
            pools[it["grp"]].append(it)
    if counts.get("repeat", 0) < MAX_REPEATS:
        pools["repeat"] = [it for it in rows if it["item_key"] in firsts
                           and it["item_key"] not in repeats
                           and it["grp"] not in ("control_error", "control_wrong")
                           and seq - firsts[it["item_key"]]["seq"] >= REPEAT_GAP]
    total_w = sum(WEIGHTS.values())
    order = sorted((g for g in WEIGHTS if pools.get(g)),
                   key=lambda g: -(WEIGHTS[g] / total_w * (seq + 1) - counts.get(g, 0)))
    if not order:
        return None
    rng = random.Random(f"{rid}:{seq}")
    pool = pools[order[0]]
    best = min(pool, key=lambda it: (it["n"], -it["severity"], rng.random()))
    return _public(best)


def record(con: sqlite3.Connection, rater: sqlite3.Row, item_key: str, verdict: str, *,
           cant_reason: str | None = None, issues: list[str] | None = None,
           suggestion: str | None = None, duration_ms: int | None = None) -> None:
    item = con.execute("SELECT dialect, grp FROM items WHERE item_key = ?", (item_key,)).fetchone()
    if not item or item["dialect"] not in rater["dialects"].split(",") or item["grp"] == "practice":
        raise ValueError("unknown item")
    if verdict not in VERDICTS:
        raise ValueError("unknown verdict")
    if cant_reason is not None and cant_reason not in CANT_REASONS:
        raise ValueError("unknown reason")
    tags = sorted(set(issues or []))
    if any(t not in ISSUES for t in tags):
        raise ValueError("unknown issue")
    rid = rater["rater_id"]
    prior = con.execute("SELECT is_repeat FROM ratings WHERE item_key = ? AND rater_id = ?",
                        (item_key, rid)).fetchall()
    if len(prior) >= 2:
        raise ValueError("already rated")
    is_repeat = 1 if prior else 0
    seq = con.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM ratings WHERE rater_id = ?",
                      (rid,)).fetchone()[0]
    clean = pii.sanitize(suggestion.strip())[:1000] if suggestion and suggestion.strip() else None
    with con:
        con.execute("INSERT INTO ratings (item_key, rater_id, verdict, cant_reason, issues, suggestion, "
                    "duration_ms, is_repeat, seq, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (item_key, rid, verdict, cant_reason if verdict == "cant_judge" else None,
                     json.dumps(tags) if tags and verdict in ("almost", "wrong") else None,
                     clean, duration_ms, is_repeat, seq, time.time()))


def summary(con: sqlite3.Connection) -> dict:
    """Quick counts for the admin; the real analysis is scripts/analyze_ratings.py."""
    out = {"items": dict(con.execute("SELECT grp, COUNT(*) FROM items GROUP BY grp").fetchall()),
           "ratings": dict(con.execute("SELECT i.grp, COUNT(*) FROM ratings r JOIN items i USING (item_key) "
                                       "GROUP BY i.grp").fetchall()),
           "raters": [dict(r) for r in con.execute(
               "SELECT ra.rater_id, ra.label, ra.dialects, ra.active, COUNT(r.rating_id) n "
               "FROM raters ra LEFT JOIN ratings r USING (rater_id) GROUP BY ra.rater_id")]}
    return out
