"""Human rating of translations — store and screen assignment.

Backs ongiini.ai/rate/, where invited native speakers judge translations.
One screen = one English sentence with 2–3 candidate translations, shown
side by side in a per-rater random order, labelled A/B/C. For each screen
the rater gives:

  * per candidate, an absolute verdict anchored in an action —
    "Would you send this as it is?"  send | fix | no
    (+ an optional "wrong dialect" flag)
  * a relative choice — which one would you send?
    <candidate> | several | none  ("none" invites a better translation)
  * optionally: "the English sentence itself is strange", a suggestion

Round 1 validates the translator's references: her reference sits blind
among Claude Opus 5 and Gemma 4 26B. The absolute verdict says whether a
translation is good enough; the choice separates candidates that are
all "send". Controls live inside screens: on some screens one candidate
is a fluent reference of a *different* sentence (right answer: no).

Raters are invited by personal link; only sha256(token) is stored, plus
a pseudonymous label and dialects. No phone numbers. Suggestions are
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

VERDICTS = ("send", "fix", "no")
CHOICE_SPECIAL = ("several", "none")

SCHEMA = """
CREATE TABLE IF NOT EXISTS raters (
  rater_id   INTEGER PRIMARY KEY,
  token_hash TEXT UNIQUE NOT NULL,
  label      TEXT NOT NULL,
  dialects   TEXT NOT NULL,
  created_at REAL NOT NULL,
  active     INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS screens (
  screen_id TEXT PRIMARY KEY,
  round     TEXT NOT NULL,
  item_id   INTEGER NOT NULL,
  dialect   TEXT NOT NULL,
  english   TEXT NOT NULL,
  priority  INTEGER NOT NULL,           -- 1 = highest
  target    INTEGER NOT NULL            -- raters wanted
);
CREATE TABLE IF NOT EXISTS candidates (
  cand_id   TEXT PRIMARY KEY,           -- opaque to raters
  screen_id TEXT NOT NULL REFERENCES screens(screen_id),
  text      TEXT NOT NULL,
  kind      TEXT NOT NULL,              -- ref | model | control (never sent)
  source    TEXT NOT NULL,              -- e.g. kaarina, claude-opus-5 (never sent)
  expected  TEXT                        -- controls: the right verdict ("no")
);
CREATE TABLE IF NOT EXISTS screen_ratings (
  screen_id      TEXT NOT NULL REFERENCES screens(screen_id),
  rater_id       INTEGER NOT NULL REFERENCES raters(rater_id),
  choice         TEXT,                  -- cand_id | several | none | NULL when skipped
  skipped        INTEGER NOT NULL DEFAULT 0,
  english_strange INTEGER NOT NULL DEFAULT 0,
  suggestion     TEXT,
  duration_ms    INTEGER,
  created_at     REAL NOT NULL,
  PRIMARY KEY (screen_id, rater_id)
);
CREATE TABLE IF NOT EXISTS candidate_ratings (
  cand_id       TEXT NOT NULL REFERENCES candidates(cand_id),
  rater_id      INTEGER NOT NULL REFERENCES raters(rater_id),
  verdict       TEXT NOT NULL,
  wrong_dialect INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (cand_id, rater_id)
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


def load_screens(con: sqlite3.Connection, screens: list[dict]) -> int:
    """screens: [{screen_id, round, item_id, dialect, english, priority, target,
    candidates: [{cand_id, text, kind, source, expected?}]}]"""
    with con:
        for s in screens:
            con.execute("INSERT OR IGNORE INTO screens VALUES (?,?,?,?,?,?,?)",
                        (s["screen_id"], s["round"], s["item_id"], s["dialect"], s["english"],
                         s["priority"], s["target"]))
            for c in s["candidates"]:
                con.execute("INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?)",
                            (c["cand_id"], s["screen_id"], c["text"], c["kind"], c["source"],
                             c.get("expected")))
    return con.execute("SELECT COUNT(*) FROM screens").fetchone()[0]


def delete_round(con: sqlite3.Connection, round_: str) -> int:
    ids = [r[0] for r in con.execute("SELECT screen_id FROM screens WHERE round = ?", (round_,))]
    with con:
        for sid in ids:
            con.execute("DELETE FROM candidate_ratings WHERE cand_id IN "
                        "(SELECT cand_id FROM candidates WHERE screen_id = ?)", (sid,))
            con.execute("DELETE FROM screen_ratings WHERE screen_id = ?", (sid,))
            con.execute("DELETE FROM candidates WHERE screen_id = ?", (sid,))
            con.execute("DELETE FROM screens WHERE screen_id = ?", (sid,))
    return len(ids)


def delete_rater(con: sqlite3.Connection, rater_id: int) -> None:
    with con:
        con.execute("DELETE FROM candidate_ratings WHERE rater_id = ?", (rater_id,))
        con.execute("DELETE FROM screen_ratings WHERE rater_id = ?", (rater_id,))
        con.execute("DELETE FROM raters WHERE rater_id = ?", (rater_id,))


# ── rater-facing ──────────────────────────────────────────────────────

def rater_for(con: sqlite3.Connection, token: str) -> sqlite3.Row | None:
    return con.execute("SELECT * FROM raters WHERE token_hash = ? AND active = 1",
                       (hash_token(token),)).fetchone()


def progress(con: sqlite3.Connection, rater_id: int) -> dict:
    done = con.execute("SELECT COUNT(*) FROM screen_ratings WHERE rater_id = ?", (rater_id,)).fetchone()[0]
    return {"done": done}


def next_screen(con: sqlite3.Connection, rater: sqlite3.Row) -> dict | None:
    """Highest-priority open screen in the rater's dialects, candidates in a
    per-rater random order. Returns only what the rater may see."""
    dialects = rater["dialects"].split(",")
    marks = ",".join("?" * len(dialects))
    rows = con.execute(f"""
      SELECT s.screen_id, s.dialect, s.english, s.priority, s.target,
             (SELECT COUNT(*) FROM screen_ratings r WHERE r.screen_id = s.screen_id) AS n
      FROM screens s
      WHERE s.dialect IN ({marks})
        AND NOT EXISTS (SELECT 1 FROM screen_ratings r
                        WHERE r.screen_id = s.screen_id AND r.rater_id = ?)
    """, [*dialects, rater["rater_id"]]).fetchall()
    rows = [r for r in rows if r["n"] < r["target"]]
    if not rows:
        return None
    done = progress(con, rater["rater_id"])["done"]
    rng = random.Random(f"{rater['rater_id']}:{done}")
    best = min(rows, key=lambda r: (r["priority"], r["n"], rng.random()))
    cands = [{"cid": c["cand_id"], "text": c["text"]} for c in
             con.execute("SELECT cand_id, text FROM candidates WHERE screen_id = ? ORDER BY cand_id",
                         (best["screen_id"],))]
    random.Random(f"{rater['rater_id']}:{best['screen_id']}").shuffle(cands)
    return {"screen_id": best["screen_id"], "dialect": best["dialect"],
            "english": best["english"], "candidates": cands}


def record(con: sqlite3.Connection, rater: sqlite3.Row, screen_id: str, *,
           verdicts: dict[str, str] | None, wrong_dialect: list[str] | None,
           choice: str | None, english_strange: bool = False, suggestion: str | None = None,
           skipped: bool = False, duration_ms: int | None = None) -> None:
    screen = con.execute("SELECT dialect FROM screens WHERE screen_id = ?", (screen_id,)).fetchone()
    if not screen or screen["dialect"] not in rater["dialects"].split(","):
        raise ValueError("unknown screen")
    cand_ids = {r[0] for r in con.execute("SELECT cand_id FROM candidates WHERE screen_id = ?",
                                          (screen_id,))}
    if not skipped:
        verdicts = verdicts or {}
        if set(verdicts) != cand_ids or any(v not in VERDICTS for v in verdicts.values()):
            raise ValueError("a verdict is needed for every translation")
        if choice not in cand_ids and choice not in CHOICE_SPECIAL:
            raise ValueError("choose one translation, several or none")
    wd = set(wrong_dialect or []) & cand_ids
    clean = pii.sanitize(suggestion.strip())[:1000] if suggestion and suggestion.strip() else None
    now = time.time()
    with con:
        con.execute("INSERT OR REPLACE INTO screen_ratings VALUES (?,?,?,?,?,?,?,?)",
                    (screen_id, rater["rater_id"], None if skipped else choice, int(skipped),
                     int(bool(english_strange)), clean, duration_ms, now))
        if not skipped:
            for cid, v in verdicts.items():
                con.execute("INSERT OR REPLACE INTO candidate_ratings VALUES (?,?,?,?)",
                            (cid, rater["rater_id"], v, int(cid in wd)))


# ── analysis ──────────────────────────────────────────────────────────

def report(con: sqlite3.Connection) -> dict:
    """Aggregates for the admin: per source absolute verdicts and how often
    it was chosen; per rater control accuracy."""
    out: dict = {"by_source": {}, "raters": [], "totals": {}}
    for r in con.execute("""SELECT c.source, s.dialect, cr.verdict, COUNT(*) n, SUM(cr.wrong_dialect) wd
                            FROM candidate_ratings cr JOIN candidates c USING (cand_id)
                            JOIN screens s USING (screen_id) WHERE c.kind != 'control'
                            GROUP BY 1,2,3"""):
        key = f"{r['source']}/{r['dialect']}"
        e = out["by_source"].setdefault(key, {"send": 0, "fix": 0, "no": 0, "wrong_dialect": 0,
                                              "chosen": 0, "shown": 0})
        e[r["verdict"]] = r["n"]
        e["wrong_dialect"] += r["wd"] or 0
    for r in con.execute("""SELECT c.source, s.dialect, COUNT(*) shown,
                              SUM(sr.choice = c.cand_id) chosen
                            FROM screen_ratings sr JOIN screens s USING (screen_id)
                            JOIN candidates c ON c.screen_id = s.screen_id
                            WHERE sr.skipped = 0 AND c.kind != 'control' GROUP BY 1,2"""):
        e = out["by_source"].get(f"{r['source']}/{r['dialect']}")
        if e:
            e["shown"], e["chosen"] = r["shown"], r["chosen"] or 0
    for e in out["by_source"].values():
        judged = e["send"] + e["fix"] + e["no"]
        e["send_pct"] = round(100 * e["send"] / judged, 1) if judged else None
        e["chosen_pct"] = round(100 * e["chosen"] / e["shown"], 1) if e["shown"] else None
    for r in con.execute("""SELECT ra.rater_id, ra.label,
                              (SELECT COUNT(*) FROM screen_ratings x WHERE x.rater_id = ra.rater_id) screens,
                              SUM(c.kind = 'control') controls,
                              SUM(c.kind = 'control' AND cr.verdict = c.expected) controls_ok
                            FROM raters ra LEFT JOIN candidate_ratings cr USING (rater_id)
                            LEFT JOIN candidates c USING (cand_id) GROUP BY ra.rater_id"""):
        out["raters"].append(dict(r))
    out["totals"] = {
        "screens": con.execute("SELECT COUNT(*) FROM screens").fetchone()[0],
        "rated_screens": con.execute("SELECT COUNT(*) FROM screen_ratings").fetchone()[0],
        "none_chosen": con.execute("SELECT COUNT(*) FROM screen_ratings WHERE choice = 'none'").fetchone()[0],
        "english_strange": con.execute("SELECT COUNT(*) FROM screen_ratings WHERE english_strange = 1").fetchone()[0],
    }
    return out


if __name__ == "__main__":  # python -m ongiini.ratings report
    import sys
    print(json.dumps(report(connect()), indent=2) if sys.argv[1:] == ["report"] else __doc__)
