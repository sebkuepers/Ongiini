"""Human rating of translations — store and task assignment.

Backs the ongiini.ai/rate/ surface where invited native speakers judge
translations one at a time. Round 1 ("Kaarina check") shows, blind and
mixed, the translator's reference next to two machine translations, so
the reference can be validated by people who read the language.

Design points:
  * Raters are invited by a personal link; only sha256(token) is stored,
    plus a pseudonymous label and the dialects they rate. No phone numbers.
  * Tasks carry a priority and a target number of ratings. The next task
    for a rater is the highest-priority one in their dialects that they
    have not rated and that still needs ratings — so whatever gets
    collected first is the most useful (flagged references first).
  * Every CONTROL_EVERY-th task is a control: a deliberately broken
    translation whose right answer is "wrong". Raters who miss controls
    can be excluded at analysis time.
  * Suggestions are PII-scrubbed before storage, like every other
    free-text write in the service.

SQLite at <data_dir>/ratings.sqlite.
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

VERDICTS = ("correct", "minor", "wrong", "cant_judge")
CONTROL_EVERY = 10

SCHEMA = """
CREATE TABLE IF NOT EXISTS raters (
  rater_id   INTEGER PRIMARY KEY,
  token_hash TEXT UNIQUE NOT NULL,
  label      TEXT NOT NULL,
  dialects   TEXT NOT NULL,          -- comma-separated
  created_at REAL NOT NULL,
  active     INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS tasks (
  task_id   TEXT PRIMARY KEY,        -- opaque to raters
  round     TEXT NOT NULL,
  item_id   INTEGER NOT NULL,
  dialect   TEXT NOT NULL,
  english   TEXT NOT NULL,
  text      TEXT NOT NULL,
  kind      TEXT NOT NULL,           -- ref | model | control (never sent to raters)
  source    TEXT NOT NULL,           -- e.g. kaarina, claude-opus-5 (never sent)
  priority  INTEGER NOT NULL,        -- 1 = highest
  target    INTEGER NOT NULL,        -- ratings wanted
  expected  TEXT                     -- controls: the right verdict
);
CREATE TABLE IF NOT EXISTS ratings (
  rating_id   INTEGER PRIMARY KEY,
  task_id     TEXT NOT NULL REFERENCES tasks(task_id),
  rater_id    INTEGER NOT NULL REFERENCES raters(rater_id),
  verdict     TEXT NOT NULL,
  suggestion  TEXT,
  duration_ms INTEGER,
  created_at  REAL NOT NULL,
  UNIQUE (task_id, rater_id)
);
"""


def db_path() -> Path:
    return settings.data_dir / "ratings.sqlite"


def connect(path: Path | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(str(path or db_path()))
    con.row_factory = sqlite3.Row
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


def load_tasks(con: sqlite3.Connection, tasks: list[dict]) -> int:
    with con:
        con.executemany(
            "INSERT OR IGNORE INTO tasks (task_id, round, item_id, dialect, english, text, kind, "
            "source, priority, target, expected) VALUES (:task_id, :round, :item_id, :dialect, "
            ":english, :text, :kind, :source, :priority, :target, :expected)",
            [{"expected": None, **t} for t in tasks])
    return con.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]


# ── rater-facing ──────────────────────────────────────────────────────

def rater_for(con: sqlite3.Connection, token: str) -> sqlite3.Row | None:
    return con.execute("SELECT * FROM raters WHERE token_hash = ? AND active = 1",
                       (hash_token(token),)).fetchone()


def progress(con: sqlite3.Connection, rater_id: int) -> dict:
    done = con.execute("SELECT COUNT(*) FROM ratings WHERE rater_id = ?", (rater_id,)).fetchone()[0]
    return {"done": done}


def next_task(con: sqlite3.Connection, rater: sqlite3.Row) -> dict | None:
    """Highest-priority open task in the rater's dialects; a control every
    CONTROL_EVERY-th task. Returns only what the rater may see."""
    dialects = rater["dialects"].split(",")
    marks = ",".join("?" * len(dialects))
    done = progress(con, rater["rater_id"])["done"]
    want_control = done % CONTROL_EVERY == CONTROL_EVERY - 1
    base = f"""
      SELECT t.task_id, t.dialect, t.english, t.text, t.kind, t.priority,
             (SELECT COUNT(*) FROM ratings r WHERE r.task_id = t.task_id) AS n
      FROM tasks t
      WHERE t.dialect IN ({marks})
        AND NOT EXISTS (SELECT 1 FROM ratings r WHERE r.task_id = t.task_id AND r.rater_id = ?)
    """
    args = [*dialects, rater["rater_id"]]
    rows = []
    if want_control:
        rows = con.execute(base + " AND t.kind = 'control'", args).fetchall()
    if not rows:
        rows = con.execute(base + " AND t.kind != 'control' AND n < t.target", args).fetchall()
    if not rows:
        return None
    # Priority first, then the least-rated, then a per-rater shuffle so two
    # raters don't walk the queue in the same order.
    rng = random.Random(f"{rater['rater_id']}:{done}")
    best = min(rows, key=lambda r: (r["priority"], r["n"], rng.random()))
    return {"task_id": best["task_id"], "dialect": best["dialect"],
            "english": best["english"], "text": best["text"]}


def record(con: sqlite3.Connection, rater: sqlite3.Row, task_id: str, verdict: str,
           suggestion: str | None, duration_ms: int | None) -> None:
    if verdict not in VERDICTS:
        raise ValueError("unknown verdict")
    task = con.execute("SELECT dialect FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    if not task or task["dialect"] not in rater["dialects"].split(","):
        raise ValueError("unknown task")
    clean = pii.sanitize(suggestion.strip())[:1000] if suggestion and suggestion.strip() else None
    with con:
        con.execute("INSERT OR REPLACE INTO ratings (task_id, rater_id, verdict, suggestion, "
                    "duration_ms, created_at) VALUES (?,?,?,?,?,?)",
                    (task_id, rater["rater_id"], verdict, clean, duration_ms, time.time()))


# ── analysis ──────────────────────────────────────────────────────────

def report(con: sqlite3.Connection) -> dict:
    """Aggregates for the admin: per source, per rater control accuracy."""
    score = {"correct": 2, "minor": 1, "wrong": 0}
    out: dict = {"by_source": {}, "raters": [], "totals": {}}
    for r in con.execute("""SELECT t.source, t.dialect, r.verdict, COUNT(*) n FROM ratings r
                            JOIN tasks t USING (task_id) WHERE t.kind != 'control'
                            GROUP BY 1,2,3"""):
        s = out["by_source"].setdefault(f"{r['source']}/{r['dialect']}", {v: 0 for v in VERDICTS})
        s[r["verdict"]] = r["n"]
    for k, s in out["by_source"].items():
        judged = sum(s[v] for v in score)
        s["mean_0_2"] = round(sum(score[v] * s[v] for v in score) / judged, 2) if judged else None
    for r in con.execute("""SELECT ra.label, COUNT(ri.rating_id) n,
                              SUM(t.kind = 'control') controls,
                              SUM(t.kind = 'control' AND ri.verdict = t.expected) controls_ok
                            FROM raters ra LEFT JOIN ratings ri USING (rater_id)
                            LEFT JOIN tasks t USING (task_id) GROUP BY ra.rater_id"""):
        out["raters"].append(dict(r))
    out["totals"] = {"tasks": con.execute("SELECT COUNT(*) FROM tasks").fetchone()[0],
                     "ratings": con.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]}
    return out


if __name__ == "__main__":  # python -m ongiini.ratings report
    import sys
    print(json.dumps(report(connect()), indent=2) if sys.argv[1:] == ["report"] else __doc__)
