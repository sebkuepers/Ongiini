"""Persistence for the learner_messages chat thread.

Phase 2's mental model: the whole learning UI is a chat conversation
between the learner and their AI coach. This module is the persistence
layer for that thread — append, list, and a small set of utility
queries the API + coach orchestrator rely on.

Same file-shape rules as the rest of the learning package:
  * Per-call ``_conn()`` from db.py.
  * Free-text content in payloads goes through ``pii.sanitize`` at the
    boundary (here, in ``append``) so callers can't accidentally store
    raw emails / IDs / etc. on disk.
  * No global state, no in-process cache — sqlite is the source of
    truth.

Message kinds + payload shapes
──────────────────────────────
  coach_text     → {"text": str}
  learner_text   → {"text": str}            (sanitised by this module)
  lesson         → {"title": str, "body": str, "examples"?: list[str]}
  exercise       → {"card_type": str, "prompt_text": str,
                   "hint_text"?: str, "difficulty"?: int}
  feedback       → {"rating": str, "feedback": str}
  progress       → {"box": int, "total_seen": int,
                   "total_correct": int, "by_box": {int: int}}

The shape is enforced by callers (coach.py, api/learn.py). This module
just persists and replays. Unknown ``kind`` strings are rejected at
write so a typo doesn't silently corrupt the thread.
"""
from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from .. import pii
from .db import (
    MESSAGE_KINDS,
    MSG_COACH_TEXT,
    MSG_EXERCISE,
    MSG_FEEDBACK,
    MSG_LEARNER_TEXT,
    MSG_LESSON,
    _conn,
    _now_iso,
)

log = logging.getLogger("ongiini.learning.messages")


# Every kind that can carry model-authored OR learner-typed free text
# gets its text fields PII-scrubbed before storage. The CLAUDE.md
# contract: "If you add a new persistence path, it MUST go through
# pii.sanitize first." Model output can echo learner-pasted PII (the
# grader sees the learner's typed answer and quotes it in feedback;
# lesson examples are model-authored but could regurgitate prior
# context). Defence in depth.
_TEXT_FIELDS_BY_KIND: dict[str, tuple[str, ...]] = {
    MSG_LEARNER_TEXT:    ("text",),
    MSG_COACH_TEXT:      ("text",),
    MSG_LESSON:          ("title", "body"),
    MSG_EXERCISE:        ("prompt_text", "hint_text"),
    MSG_FEEDBACK:        ("feedback",),
    # MSG_PROGRESS has no free-text fields — it's just counts.
}


def _sanitise_payload(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Return a new payload dict with every known free-text field
    PII-scrubbed. List fields (``examples`` on a lesson) get each
    string element scrubbed in place."""
    out = dict(payload)
    for field in _TEXT_FIELDS_BY_KIND.get(kind, ()):
        value = out.get(field)
        if isinstance(value, str):
            out[field] = pii.sanitize(value)
    # Lesson examples are a list[str]; scrub each entry.
    if kind == MSG_LESSON and isinstance(out.get("examples"), list):
        out["examples"] = [
            pii.sanitize(x) if isinstance(x, str) else x
            for x in out["examples"]
        ]
    # Multi-step lesson cards (steps[]): each step has its own
    # free-text fields. Scrub them all in place so PII can't leak
    # through the new payload shape.
    if kind == MSG_LESSON and isinstance(out.get("steps"), list):
        scrubbed_steps: list[Any] = []
        for step in out["steps"]:
            if not isinstance(step, dict):
                scrubbed_steps.append(step)
                continue
            s = dict(step)
            for field in ("body", "prompt", "answer", "hint"):
                v = s.get(field)
                if isinstance(v, str):
                    s[field] = pii.sanitize(v)
            if isinstance(s.get("examples"), list):
                s["examples"] = [
                    pii.sanitize(x) if isinstance(x, str) else x
                    for x in s["examples"]
                ]
            scrubbed_steps.append(s)
        out["steps"] = scrubbed_steps
    return out


def append(
    *,
    learner_id: str,
    goal_id: str,
    kind: str,
    payload: dict[str, Any],
    card_id: str | None = None,
) -> dict[str, Any]:
    """Append one message to a learner's thread. Returns the persisted
    row (including the generated ``message_id`` and ``created_at``).

    Raises ``ValueError`` for unknown kinds or empty learner/goal ids.
    """
    if not learner_id:
        raise ValueError("learner_id is required")
    if not goal_id:
        raise ValueError("goal_id is required")
    if kind not in MESSAGE_KINDS:
        raise ValueError(f"unknown message kind: {kind!r}")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")

    safe_payload = _sanitise_payload(kind, payload)
    # The docstring promises ValueError on bad input — convert any
    # JSON-serialisation crash into ValueError too, so callers don't
    # need to handle TypeError separately for bytes / datetime / set
    # values that occasionally leak through model output.
    try:
        payload_json = json.dumps(
            safe_payload, separators=(",", ":"), ensure_ascii=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload not JSON-serialisable: {exc}") from exc
    message_id = str(uuid4())
    now = _now_iso()

    with _conn() as c:
        c.execute(
            "INSERT INTO learner_messages "
            "(message_id, learner_id, goal_id, kind, payload_json, "
            "card_id, answered, created_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
            (message_id, learner_id, goal_id, kind, payload_json,
             card_id, now),
        )

    return {
        "message_id": message_id,
        "learner_id": learner_id,
        "goal_id": goal_id,
        "kind": kind,
        "payload": safe_payload,
        "card_id": card_id,
        "answered": False,
        "created_at": now,
    }


def list_for_goal(
    *,
    learner_id: str,
    goal_id: str,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Return the thread for one (learner, goal). The slice is the
    NEWEST ``limit`` messages (so a long-running learner always sees
    their recent history on rehydration), returned in chronological
    order (oldest first) so the frontend can append-render without
    re-sorting.

    Earlier versions selected ``ORDER BY created_at ASC LIMIT 200``,
    which silently dropped recent turns once the thread crossed the
    cap — the chat-first UI would rehydrate ancient history while the
    actual conversation disappeared. The fix is to slice DESC and
    reverse client-side here.
    """
    if not learner_id or not goal_id:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM learner_messages "
            "WHERE learner_id = ? AND goal_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (learner_id, goal_id, limit),
        ).fetchall()
    rows.reverse()
    return [_row_to_dict(r) for r in rows]


def latest_unanswered_exercise(
    *,
    learner_id: str,
    goal_id: str,
) -> dict[str, Any] | None:
    """Return the most recent exercise message that hasn't been answered.

    The coach orchestrator uses this to decide: if the learner types
    something while there's an unanswered exercise on the thread, the
    input is treated as their answer to that card. Otherwise it's a
    free-form question.
    """
    if not learner_id or not goal_id:
        return None
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM learner_messages "
            "WHERE learner_id = ? AND goal_id = ? "
            "AND kind = ? AND answered = 0 "
            "ORDER BY created_at DESC LIMIT 1",
            (learner_id, goal_id, MSG_EXERCISE),
        ).fetchone()
    return _row_to_dict(row) if row else None


def mark_answered(message_id: str) -> None:
    """Flag an exercise message as answered (a grading happened). Called
    by the coach after appending the feedback row. Soft-fails — a
    missing row just becomes a no-op.

    Prefer :func:`claim_exercise` from a grading code path — it's
    atomic and guards against double-grade races. This helper is kept
    for callers that just need to set the flag without race semantics
    (e.g. recovering from a missing-card branch)."""
    if not message_id:
        return
    with _conn() as c:
        c.execute(
            "UPDATE learner_messages SET answered = 1 WHERE message_id = ?",
            (message_id,),
        )


def claim_exercise(message_id: str) -> bool:
    """Atomic "I'm grading this one" claim. Returns True iff THIS call
    is the one that flipped ``answered`` from 0 → 1.

    Two concurrent ``run_turn`` invocations (browser double-tap, retry
    storm) will both read the same unanswered exercise; without this
    atomic step both would call ``grade_answer`` and ``record_attempt``,
    advancing the Leitner box twice and inflating the attempt log. The
    SQL ``WHERE answered = 0`` clause is the lock — the second caller
    gets ``rowcount = 0`` and bails out before the model call.

    Empty / missing id returns False (treated as 'someone else claimed
    it' — safer than silently proceeding)."""
    if not message_id:
        return False
    with _conn() as c:
        cur = c.execute(
            "UPDATE learner_messages SET answered = 1 "
            "WHERE message_id = ? AND answered = 0",
            (message_id,),
        )
        return cur.rowcount == 1


def clear_for_goal(*, learner_id: str, goal_id: str) -> int:
    """Delete every message in one (learner, goal) thread. Used by the
    'restart curriculum' flow. Returns rows removed."""
    if not learner_id or not goal_id:
        return 0
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM learner_messages WHERE learner_id = ? AND goal_id = ?",
            (learner_id, goal_id),
        )
        return cur.rowcount


def recent_text_pairs(
    *,
    learner_id: str,
    goal_id: str,
    max_pairs: int = 6,
) -> list[dict[str, Any]]:
    """Return the most-recent N text messages (coach + learner) for
    feeding into the LLM as conversation context. Newest at the END
    so prompt construction can just join them as-is.

    Excludes lesson / exercise / feedback / progress rows — those are
    surfaced separately via the curriculum + last_card context fields.
    """
    if not learner_id or not goal_id:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM learner_messages "
            "WHERE learner_id = ? AND goal_id = ? "
            "AND kind IN (?, ?) "
            "ORDER BY created_at DESC LIMIT ?",
            (learner_id, goal_id, MSG_COACH_TEXT, MSG_LEARNER_TEXT,
             max_pairs * 2),
        ).fetchall()
    rows.reverse()
    return [_row_to_dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────────
# Row → dict helper
# ──────────────────────────────────────────────────────────────────

def _row_to_dict(row: Any) -> dict[str, Any]:
    """Decode a sqlite Row into a UI-friendly dict.

    ``payload_json`` is parsed back into the dict the caller passed
    into ``append``. Corrupt JSON (shouldn't happen) is surfaced as
    an empty dict so the frontend can render something rather than
    throwing on the malformed row."""
    try:
        payload = json.loads(row["payload_json"]) if row["payload_json"] else {}
    except json.JSONDecodeError:
        log.warning("messages: corrupt payload_json for %s", row["message_id"])
        payload = {}
    return {
        "message_id": row["message_id"],
        "learner_id": row["learner_id"],
        "goal_id": row["goal_id"],
        "kind": row["kind"],
        "payload": payload,
        "card_id": row["card_id"],
        "answered": bool(row["answered"]),
        "created_at": row["created_at"],
    }
