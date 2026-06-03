"""High-level data access for the learning surface.

Module-level functions over the SQLite schema in ``db.py``. Mirrors
the shape of ``ongiini.contributions`` — per-call ``_conn()``, soft
sanity checks, no global state, no in-process cache (SQLite is the
source of truth for everything).

The functions split into three layers:

  * **Identity & lifecycle** — create_anonymous_learner, get_learner,
    delete_learner (the GDPR right-to-erasure path).
  * **Profile / intake** — get_profile, save_profile_field,
    mark_intake_complete. The LLM conducts the conversation; the API
    calls into these after a successful ``intake.validate_field`` to
    persist the captured value.
  * **Goal / curriculum / cards** — get_or_create_active_goal,
    save_curriculum_outline, save_card, record_attempt, next_due_cards.
    The LLM owns content; this layer owns persistence + the Leitner
    arithmetic via ``srs.promote`` / ``srs.next_due_at``.

PII contract: ``record_attempt`` is the only write path that touches
free-text user input. It calls ``pii.sanitize`` before INSERT so the
caller cannot bypass it.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .. import pii
from . import srs
from .db import (
    CARD_TYPES,
    IDENTITY_ANONYMOUS,
    IDENTITY_WHATSAPP,
    RATINGS,
    RATING_CORRECT,
    RATING_PARTIAL,
    RATING_WRONG,
    _conn,
    _now_iso,
)

log = logging.getLogger("ongiini.learning.store")


# ──────────────────────────────────────────────────────────────────
# Identity & lifecycle
# ──────────────────────────────────────────────────────────────────

def create_anonymous_learner() -> str:
    """Create a fresh anonymous learner. Returns the new learner_id
    (UUID v4). Caller is responsible for stashing this in the browser's
    localStorage so subsequent requests can find the row again."""
    learner_id = str(uuid4())
    now = _now_iso()
    with _conn() as c:
        c.execute(
            "INSERT INTO learners (learner_id, identity_type, "
            "created_at, last_active_at) VALUES (?, ?, ?, ?)",
            (learner_id, IDENTITY_ANONYMOUS, now, now),
        )
    log.info("learner created (anonymous) learner_id=%s", learner_id[:8])
    return learner_id


def upsert_whatsapp_learner(hashed_msisdn: str) -> str:
    """Create or refresh a WhatsApp-bound learner. Returns the
    learner_id, which is the conventional ``wa:<hashed_msisdn>`` form
    so the anonymous + whatsapp namespaces don't collide.

    Used by the magic-link / upgrade flows in later phases; safe to
    call repeatedly on the same hash (idempotent on the learner row,
    refreshes last_active_at)."""
    if not hashed_msisdn:
        raise ValueError("hashed_msisdn is required")
    learner_id = f"wa:{hashed_msisdn}"
    now = _now_iso()
    with _conn() as c:
        c.execute(
            "INSERT INTO learners (learner_id, identity_type, "
            "created_at, last_active_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(learner_id) DO UPDATE SET last_active_at = excluded.last_active_at",
            (learner_id, IDENTITY_WHATSAPP, now, now),
        )
    return learner_id


def touch_learner(learner_id: str) -> None:
    """Refresh ``last_active_at``. Soft-fail (logs but doesn't raise) —
    keeping a session live shouldn't bring the request down if the row
    disappeared."""
    if not learner_id:
        return
    try:
        with _conn() as c:
            c.execute(
                "UPDATE learners SET last_active_at = ? WHERE learner_id = ?",
                (_now_iso(), learner_id),
            )
    except Exception as exc:                                # noqa: BLE001
        log.warning("touch_learner failed: %s", exc)


def get_learner(learner_id: str) -> dict[str, Any] | None:
    """Return the learner row as a dict, or None if absent."""
    if not learner_id:
        return None
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM learners WHERE learner_id = ?",
            (learner_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_learner(learner_id: str) -> int:
    """Delete the learner row + cascade. Returns rows removed (0 or 1).
    The GDPR right-to-erasure path — matches the `delete my data`
    contract on WhatsApp."""
    if not learner_id:
        return 0
    with _conn() as c:
        cur = c.execute("DELETE FROM learners WHERE learner_id = ?", (learner_id,))
        return cur.rowcount


# ──────────────────────────────────────────────────────────────────
# Profile / intake
# ──────────────────────────────────────────────────────────────────

def get_profile(learner_id: str) -> dict[str, Any] | None:
    """Return the learner_profiles row as a dict, or None if no row
    exists yet (i.e. intake hasn't even started)."""
    if not learner_id:
        return None
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM learner_profiles WHERE learner_id = ?",
            (learner_id,),
        ).fetchone()
    return dict(row) if row else None


_PROFILE_FIELDS = {"name", "age", "current_level", "objective"}
# Free-text fields go through the PII scrubber before storage. Same
# project-wide rule the WhatsApp + chat surfaces follow: disk + mem0
# see redacted text. Names and the objective sentence are both user-
# typed; numeric / enum fields don't need scrubbing.
_FREE_TEXT_PROFILE_FIELDS = {"name", "objective"}


def save_profile_field(learner_id: str, field: str, value: Any) -> None:
    """Persist a single intake-captured field atomically.

    Whitelists the field name (closes the SQL-injection vector on the
    dynamic UPDATE). Free-text fields are sanitised through
    ``pii.sanitize`` before INSERT — callers can't bypass the contract.
    Wrapped in BEGIN IMMEDIATE / COMMIT so a concurrent ``delete_learner``
    can't sneak between the upsert and the update; raises if the
    UPDATE matches zero rows (the parent learner row disappeared)."""
    if not learner_id:
        raise ValueError("learner_id is required")
    if field not in _PROFILE_FIELDS:
        raise ValueError(f"unknown profile field: {field}")
    if field in _FREE_TEXT_PROFILE_FIELDS and isinstance(value, str):
        value = pii.sanitize(value)
    with _conn() as c:
        c.execute("BEGIN IMMEDIATE")
        try:
            try:
                c.execute(
                    "INSERT INTO learner_profiles (learner_id) VALUES (?) "
                    "ON CONFLICT(learner_id) DO NOTHING",
                    (learner_id,),
                )
            except sqlite3.IntegrityError as exc:
                # Foreign-key constraint failed — the parent learner row
                # doesn't exist (deleted concurrently, or the caller
                # passed a bad id). Rethrow as the friendlier RuntimeError
                # so callers can pattern-match on type rather than
                # parsing 'FOREIGN KEY constraint failed'.
                c.execute("ROLLBACK")
                raise RuntimeError(
                    f"save_profile_field: learner {learner_id} not found"
                ) from exc
            cur = c.execute(
                f"UPDATE learner_profiles SET {field} = ? WHERE learner_id = ?",
                (value, learner_id),
            )
            if cur.rowcount == 0:
                # Belt-and-suspenders: with FOREIGN KEY constraints ON
                # the parent-missing case is already caught above. This
                # branch only fires if foreign keys are disabled, which
                # would be a configuration accident.
                c.execute("ROLLBACK")
                raise RuntimeError(
                    f"save_profile_field: learner {learner_id} not found"
                )
            c.execute("COMMIT")
        except Exception:
            try:
                c.execute("ROLLBACK")
            except Exception:                               # noqa: BLE001
                pass
            raise


def mark_intake_complete(learner_id: str) -> None:
    """Stamp the profile as intake-done. The API calls this when
    ``intake.is_complete(profile)`` flips to True. Idempotent."""
    if not learner_id:
        raise ValueError("learner_id is required")
    with _conn() as c:
        c.execute(
            "INSERT INTO learner_profiles (learner_id, intake_completed_at) "
            "VALUES (?, ?) "
            "ON CONFLICT(learner_id) DO UPDATE SET "
            "intake_completed_at = COALESCE(learner_profiles.intake_completed_at, excluded.intake_completed_at)",
            (learner_id, _now_iso()),
        )


# ──────────────────────────────────────────────────────────────────
# Goal / curriculum
# ──────────────────────────────────────────────────────────────────

def get_or_create_active_goal(
    learner_id: str,
    *,
    language: str = "afrikaans",
    source_language: str = "english",
    current_level: str | None = None,
    context: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Return the active goal row for this learner, creating one if
    none exists. MVP: one active goal per learner. Returns dict-shape
    of the row (including the curriculum_outline JSON column, if any).

    Wrapped in BEGIN IMMEDIATE / COMMIT so two concurrent requests for
    the same learner can't both see "no active goal" and both insert,
    leaving two active goals. The losing concurrent caller will get
    SQLITE_BUSY which surfaces as a sqlite OperationalError — caller
    can retry.

    Free-text ``context`` (the learning objective the magic link
    carried) is PII-scrubbed before storage, same as profile.objective.

    ``title`` is applied ONLY when this call creates a new goal — never
    relabels an existing active goal. The /turn handler passes the
    learner's intake `profile.objective` here so the auto-created goal
    shows up in the drawer with a meaningful name ("job interview at
    SPAR") rather than "Untitled curriculum"."""
    if not learner_id:
        raise ValueError("learner_id is required")
    # Validate the language pair at the boundary — same checks the API
    # request handlers run. Centralised here so callers (including the
    # legacy /turn auto-create path) can't slip an invalid pair in.
    from .skill_renderer import validate_language_pair
    validate_language_pair(source_language, language)
    if isinstance(context, str):
        context = pii.sanitize(context)
    if isinstance(title, str):
        title = pii.sanitize(title).strip()[:80] or None
    with _conn() as c:
        c.execute("BEGIN IMMEDIATE")
        try:
            row = c.execute(
                "SELECT * FROM learning_goals WHERE learner_id = ? "
                "AND status = 'active' ORDER BY created_at DESC LIMIT 1",
                (learner_id,),
            ).fetchone()
            if row:
                c.execute("COMMIT")
                return dict(row)
            goal_id = str(uuid4())
            c.execute(
                "INSERT INTO learning_goals (goal_id, learner_id, language, "
                "source_language, current_level, context, status, title, "
                "created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)",
                (goal_id, learner_id, language, source_language,
                 current_level, context, title, _now_iso()),
            )
            row = c.execute(
                "SELECT * FROM learning_goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
            c.execute("COMMIT")
        except Exception:
            try:
                c.execute("ROLLBACK")
            except Exception:                               # noqa: BLE001
                pass
            raise
    return dict(row)


def list_goals(
    learner_id: str,
    *,
    include_archived: bool = False,
) -> list[dict[str, Any]]:
    """All goals for one learner, newest first. Excludes archived by
    default so the frontend's switcher only surfaces live ones.

    Returns dicts that include the persisted goal columns plus a
    ``has_outline`` boolean derived from ``curriculum_outline IS NOT NULL``
    so the UI can show "Plan ready" vs "Plan pending" without a second
    fetch."""
    if not learner_id:
        return []
    sql = (
        "SELECT goal_id, learner_id, language, source_language, "
        "       current_level, context, status, title, "
        "       archived_at, created_at, outline_updated_at, "
        "       CASE WHEN curriculum_outline IS NULL THEN 0 ELSE 1 END "
        "       AS has_outline "
        "FROM learning_goals WHERE learner_id = ?"
    )
    params: list[Any] = [learner_id]
    if not include_archived:
        sql += " AND status != 'archived'"
    sql += " ORDER BY created_at DESC"
    with _conn() as c:
        rows = c.execute(sql, params).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        d["has_outline"] = bool(d.get("has_outline"))
        out.append(d)
    return out


def create_new_goal(
    learner_id: str,
    *,
    title: str | None = None,
    context: str | None = None,
    language: str = "afrikaans",
    source_language: str = "english",
    current_level: str | None = None,
    activate: bool = True,
) -> dict[str, Any]:
    """Create a fresh learning goal. When ``activate=True`` (the
    default), any existing active goal for this learner is moved to
    ``paused`` so only ONE active goal exists at a time. Returns the
    new goal row.

    Both ``title`` (the user-chosen name like "Job interview at SPAR")
    and ``context`` (the underlying objective) are PII-scrubbed before
    storage — same contract as ``save_profile_field`` for free-text
    profile values.

    Wrapped in BEGIN IMMEDIATE so the demote-then-insert pair can't be
    interleaved with another caller's create."""
    if not learner_id:
        raise ValueError("learner_id is required")
    # Validate the language pair at the boundary.
    from .skill_renderer import validate_language_pair
    validate_language_pair(source_language, language)
    if isinstance(title, str):
        title = pii.sanitize(title).strip() or None
    if isinstance(context, str):
        context = pii.sanitize(context).strip() or None
    new_goal_id = str(uuid4())
    now = _now_iso()
    with _conn() as c:
        c.execute("BEGIN IMMEDIATE")
        try:
            if activate:
                c.execute(
                    "UPDATE learning_goals SET status = 'paused' "
                    "WHERE learner_id = ? AND status = 'active'",
                    (learner_id,),
                )
            status_val = "active" if activate else "paused"
            c.execute(
                "INSERT INTO learning_goals (goal_id, learner_id, language, "
                "source_language, current_level, context, status, title, "
                "created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (new_goal_id, learner_id, language, source_language,
                 current_level, context, status_val, title, now),
            )
            row = c.execute(
                "SELECT * FROM learning_goals WHERE goal_id = ?",
                (new_goal_id,),
            ).fetchone()
            c.execute("COMMIT")
        except Exception:
            try:
                c.execute("ROLLBACK")
            except Exception:                               # noqa: BLE001
                pass
            raise
    return dict(row)


def activate_goal(learner_id: str, goal_id: str) -> dict[str, Any]:
    """Switch the active goal. Atomically demotes the existing active
    goal to ``paused`` and promotes the requested one to ``active``.

    Raises ``RuntimeError`` if the goal doesn't belong to this learner
    or is archived (archived goals can't be re-activated — restart or
    create a new one)."""
    if not learner_id or not goal_id:
        raise ValueError("learner_id and goal_id are required")
    with _conn() as c:
        c.execute("BEGIN IMMEDIATE")
        try:
            row = c.execute(
                "SELECT learner_id, status FROM learning_goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
            if not row or row["learner_id"] != learner_id:
                c.execute("ROLLBACK")
                raise RuntimeError(f"activate_goal: goal {goal_id} not found")
            if row["status"] == "archived":
                c.execute("ROLLBACK")
                raise RuntimeError(
                    "activate_goal: cannot re-activate archived goal; "
                    "create a new one instead"
                )
            # Demote the currently-active goal (no-op if there isn't one,
            # or if it's the same as goal_id and already active).
            c.execute(
                "UPDATE learning_goals SET status = 'paused' "
                "WHERE learner_id = ? AND status = 'active' AND goal_id != ?",
                (learner_id, goal_id),
            )
            c.execute(
                "UPDATE learning_goals SET status = 'active' WHERE goal_id = ?",
                (goal_id,),
            )
            row = c.execute(
                "SELECT * FROM learning_goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
            c.execute("COMMIT")
        except Exception:
            try:
                c.execute("ROLLBACK")
            except Exception:                               # noqa: BLE001
                pass
            raise
    return dict(row)


def restart_goal(learner_id: str, goal_id: str) -> dict[str, Any]:
    """Wipe one curriculum's progress without deleting the goal row.
    Removes all cards (which cascades to attempts + review_state) and
    all messages on the thread. KEEPS the goal row + curriculum_outline
    so the learner gets the same plan back, fresh.

    Validates ownership — a learner can't restart someone else's goal.
    Returns a small summary dict (rows removed per table)."""
    if not learner_id or not goal_id:
        raise ValueError("learner_id and goal_id are required")
    with _conn() as c:
        c.execute("BEGIN IMMEDIATE")
        try:
            row = c.execute(
                "SELECT learner_id FROM learning_goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
            if not row or row["learner_id"] != learner_id:
                c.execute("ROLLBACK")
                raise RuntimeError(f"restart_goal: goal {goal_id} not found")
            # Cards cascade to attempts + card_review_state via FK ON
            # DELETE CASCADE — so a single DELETE on learning_cards is
            # enough to wipe the SRS state too.
            cur_cards = c.execute(
                "DELETE FROM learning_cards WHERE goal_id = ?", (goal_id,),
            )
            cards_deleted = cur_cards.rowcount
            cur_msgs = c.execute(
                "DELETE FROM learner_messages WHERE goal_id = ?", (goal_id,),
            )
            msgs_deleted = cur_msgs.rowcount
            c.execute("COMMIT")
        except Exception:
            try:
                c.execute("ROLLBACK")
            except Exception:                               # noqa: BLE001
                pass
            raise
    return {
        "goal_id": goal_id,
        "cards_deleted": cards_deleted,
        "messages_deleted": msgs_deleted,
    }


def archive_goal(learner_id: str, goal_id: str) -> dict[str, Any]:
    """Soft-delete a goal. Sets ``status='archived'`` and stamps
    ``archived_at``. The goal row and all its content stays on disk so
    historical analytics still work; the switcher just hides it.

    If the archived goal was the active one, leaves the learner with no
    active goal — the frontend should prompt them to pick or create a
    new one. (We don't auto-activate a paused goal: silently changing
    focus is more surprising than helpful.)"""
    if not learner_id or not goal_id:
        raise ValueError("learner_id and goal_id are required")
    now = _now_iso()
    with _conn() as c:
        c.execute("BEGIN IMMEDIATE")
        try:
            row = c.execute(
                "SELECT learner_id FROM learning_goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
            if not row or row["learner_id"] != learner_id:
                c.execute("ROLLBACK")
                raise RuntimeError(f"archive_goal: goal {goal_id} not found")
            c.execute(
                "UPDATE learning_goals SET status = 'archived', "
                "archived_at = ? WHERE goal_id = ?",
                (now, goal_id),
            )
            row = c.execute(
                "SELECT * FROM learning_goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
            c.execute("COMMIT")
        except Exception:
            try:
                c.execute("ROLLBACK")
            except Exception:                               # noqa: BLE001
                pass
            raise
    return dict(row)


def update_goal_title(learner_id: str, goal_id: str, title: str) -> None:
    """Set the human-readable goal name. PII-scrubbed (titles are user-
    typed, e.g. "interview at SPAR"). Raises if the goal isn't this
    learner's."""
    if not learner_id or not goal_id:
        raise ValueError("learner_id and goal_id are required")
    title = pii.sanitize(title or "").strip() or None
    with _conn() as c:
        row = c.execute(
            "SELECT learner_id FROM learning_goals WHERE goal_id = ?",
            (goal_id,),
        ).fetchone()
        if not row or row["learner_id"] != learner_id:
            raise RuntimeError(f"update_goal_title: goal {goal_id} not found")
        c.execute(
            "UPDATE learning_goals SET title = ? WHERE goal_id = ?",
            (title, goal_id),
        )


def save_curriculum_outline(goal_id: str, outline: dict[str, Any]) -> None:
    """Persist the LLM-authored curriculum outline as JSON.

    The shape of ``outline`` is the LLM's design — we don't validate
    its inner schema. We DO validate it's JSON-serialisable so a bad
    payload fails here rather than at the next read.

    Idempotent at the row level: rewrites the column + timestamp."""
    if not goal_id:
        raise ValueError("goal_id is required")
    try:
        outline_json = json.dumps(outline, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"outline is not JSON-serialisable: {exc}") from exc
    with _conn() as c:
        c.execute(
            "UPDATE learning_goals SET curriculum_outline = ?, "
            "outline_updated_at = ? WHERE goal_id = ?",
            (outline_json, _now_iso(), goal_id),
        )


def get_curriculum_outline(goal_id: str) -> dict[str, Any] | None:
    """Return the parsed outline dict, or None if no outline written yet."""
    if not goal_id:
        return None
    with _conn() as c:
        row = c.execute(
            "SELECT curriculum_outline FROM learning_goals WHERE goal_id = ?",
            (goal_id,),
        ).fetchone()
    if not row or not row["curriculum_outline"]:
        return None
    try:
        return json.loads(row["curriculum_outline"])
    except json.JSONDecodeError:
        # Corrupt JSON in the column — treat as absent rather than crash.
        log.warning("get_curriculum_outline: invalid JSON for goal=%s", goal_id)
        return None


# ──────────────────────────────────────────────────────────────────
# Cards
# ──────────────────────────────────────────────────────────────────

def save_card(
    goal_id: str,
    card_type: str,
    prompt_text: str,
    *,
    reference_answer: str | None = None,
    hint_text: str | None = None,
    difficulty: int | None = None,
    module_id: str | None = None,
    topic_id: str | None = None,
    extras: dict[str, Any] | None = None,
) -> str:
    """Persist an LLM-generated card so SRS re-reviews surface the same
    prompt rather than re-rolling it. Returns the new card_id (UUID v4).

    ``module_id`` ties this card back to one of the modules in the
    curriculum outline. ``topic_id`` ties it to one of that module's
    topics so the runtime can enforce "no exercises on untaught
    topics". Both optional for back-compat with cards authored before
    the tags existed, but new cards should include them.

    ``extras`` is the per-card-type structural payload — MC options,
    reorder tokens, dialogue turns, grammar source_sentence, proverb
    cultural_note — stored as a JSON blob so SRS replay can rebuild
    the renderer payload without losing the question's shape.
    """
    if not goal_id:
        raise ValueError("goal_id is required")
    if card_type not in CARD_TYPES:
        raise ValueError(f"unknown card_type: {card_type}")
    if not prompt_text or not prompt_text.strip():
        raise ValueError("prompt_text is required")
    extras_json = json.dumps(extras, ensure_ascii=False) if extras else None
    card_id = str(uuid4())
    with _conn() as c:
        c.execute(
            "INSERT INTO learning_cards (card_id, goal_id, card_type, "
            "prompt_text, reference_answer, hint_text, difficulty, "
            "module_id, topic_id, extras_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (card_id, goal_id, card_type, prompt_text.strip(),
             reference_answer, hint_text, difficulty, module_id,
             topic_id, extras_json, _now_iso()),
        )
    return card_id


def progress_for_modules(
    learner_id: str,
    goal_id: str,
) -> dict[str, dict[str, Any]]:
    """Per-module progress for one goal. Returns
    ``{module_id: {"lessons_given": int, "exercises_emitted": int,
                   "exercises_attempted": int, "exercises_correct": int,
                   "cards_in_module": int,
                   "topics_taught": {topic_id: lesson_count, ...},
                   "topics_drilled": {topic_id: exercise_count, ...}}}``.

    Counts only cards with a non-null ``module_id`` — older un-tagged
    cards are excluded so the numbers match what the curriculum panel
    UI claims to show. The breakdown distinguishes lessons (read +
    acknowledged) from exercises (attempted + graded) so the API can
    answer "Module 1: 5 / 8 cards" with the right semantics.

    The per-topic dicts (``topics_taught`` / ``topics_drilled``) drive
    the teach-then-test pacing rule — the runtime can refuse to drill
    a topic that hasn't been taught yet. Cards with a NULL ``topic_id``
    contribute to the module-level counts but not to the per-topic
    breakdown."""
    from .db import CARD_LESSON, EXERCISE_CARD_TYPES
    if not learner_id or not goal_id:
        return {}
    with _conn() as c:
        rows = c.execute(
            "SELECT lc.module_id AS module_id, lc.topic_id AS topic_id, "
            "       lc.card_type, "
            "       COUNT(DISTINCT lc.card_id) AS n_cards, "
            "       COALESCE(SUM(crs.total_seen), 0) AS attempts_seen, "
            "       COALESCE(SUM(crs.total_correct), 0) AS attempts_correct "
            "FROM learning_cards lc "
            "LEFT JOIN card_review_state crs "
            "       ON crs.card_id = lc.card_id AND crs.learner_id = ? "
            "WHERE lc.goal_id = ? AND lc.module_id IS NOT NULL "
            "GROUP BY lc.module_id, lc.topic_id, lc.card_type",
            (learner_id, goal_id),
        ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        m = r["module_id"]
        d = out.setdefault(m, {
            "lessons_given": 0,
            "exercises_emitted": 0,
            "exercises_attempted": 0,
            "exercises_correct": 0,
            "cards_in_module": 0,
            "topics_taught": {},
            "topics_drilled": {},
        })
        n = int(r["n_cards"] or 0)
        d["cards_in_module"] += n
        topic = r["topic_id"]
        if r["card_type"] == CARD_LESSON:
            d["lessons_given"] += n
            if topic:
                d["topics_taught"][topic] = d["topics_taught"].get(topic, 0) + n
        elif r["card_type"] in EXERCISE_CARD_TYPES:
            d["exercises_emitted"] += n
            d["exercises_attempted"] += int(r["attempts_seen"] or 0)
            d["exercises_correct"] += int(r["attempts_correct"] or 0)
            if topic:
                d["topics_drilled"][topic] = d["topics_drilled"].get(topic, 0) + n
    return out


def get_card(card_id: str) -> dict[str, Any] | None:
    if not card_id:
        return None
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM learning_cards WHERE card_id = ?",
            (card_id,),
        ).fetchone()
    return dict(row) if row else None


def next_due_cards(
    learner_id: str,
    *,
    limit: int = 1,
    goal_id: str | None = None,
    exclude_card_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return cards due for review now, ordered by next_due_at (oldest
    first). Joins card metadata so the caller doesn't need a second
    fetch. Empty list = no due cards (LLM should generate a new one).

    ``goal_id`` scopes to one curriculum — important now that learners
    have multiple goals. ``exclude_card_id`` filters out one specific
    card (the just-answered one) so the coach's SRS replay doesn't
    immediately re-emit the card the learner just got wrong, which
    would feel jarring (Anki-style "Again" surfaces after a couple
    of new cards, not back-to-back)."""
    if not learner_id or limit < 1:
        return []
    now = _now_iso()
    sql = (
        "SELECT lc.*, crs.box, crs.next_due_at, crs.total_seen, "
        "crs.total_correct "
        "FROM card_review_state crs "
        "JOIN learning_cards lc ON lc.card_id = crs.card_id "
        "WHERE crs.learner_id = ? AND crs.next_due_at <= ?"
    )
    params: list[Any] = [learner_id, now]
    if goal_id:
        sql += " AND lc.goal_id = ?"
        params.append(goal_id)
    if exclude_card_id:
        sql += " AND lc.card_id != ?"
        params.append(exclude_card_id)
    # Exercise cards only — lessons don't go through SRS. Drive the
    # IN list from the canonical EXERCISE_CARD_TYPES tuple so cards
    # of new types (cloze, multiple_choice, grammar, dialogue, etc.)
    # also resurface via SRS replay. The earlier hard-coded
    # ('vocab', 'translation', 'production') silently excluded
    # everything added in the card-variety round.
    from .db import EXERCISE_CARD_TYPES
    placeholders = ",".join("?" for _ in EXERCISE_CARD_TYPES)
    sql += f" AND lc.card_type IN ({placeholders})"
    params.extend(EXERCISE_CARD_TYPES)
    sql += " ORDER BY crs.next_due_at ASC LIMIT ?"
    params.append(limit)
    with _conn() as c:
        rows = c.execute(sql, params).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        # Deserialise extras_json so the SRS replay path can rebuild
        # the renderer payload (MC options, reorder tokens, dialogue
        # turns, etc.) instead of dropping the per-type extras.
        raw = d.pop("extras_json", None)
        if isinstance(raw, str) and raw:
            try:
                d["extras"] = json.loads(raw)
            except json.JSONDecodeError:
                log.warning(
                    "store: corrupt extras_json for card %s; ignoring",
                    d.get("card_id"),
                )
                d["extras"] = None
        else:
            d["extras"] = None
        out.append(d)
    return out


# ──────────────────────────────────────────────────────────────────
# Attempts (the PII boundary)
# ──────────────────────────────────────────────────────────────────

def record_attempt(
    *,
    learner_id: str,
    card_id: str,
    user_answer: str,
    ai_feedback: str,
    rating: str,
    hint_used: bool = False,
) -> dict[str, Any]:
    """Record a graded attempt and advance the Leitner state.

    PII contract: ``user_answer`` is sanitised via ``pii.sanitize``
    before INSERT. Callers can't bypass this — same enforcement
    pattern contributions.py uses.

    The Leitner promotion treats 'correct' and 'partial' as success
    (forward motion in MVP). Only 'wrong' demotes to box 1. The card
    review state is created on first attempt and updated on subsequent
    attempts. Returns a small dict summarising the updated state so
    the API can echo it to the frontend without a second read.
    """
    if not learner_id:
        raise ValueError("learner_id is required")
    if not card_id:
        raise ValueError("card_id is required")
    if rating not in RATINGS:
        raise ValueError(f"unknown rating: {rating}")

    sanitised_answer = pii.sanitize(user_answer or "")
    correct = rating in (RATING_CORRECT, RATING_PARTIAL)
    now_iso = _now_iso()
    now_dt = datetime.now(timezone.utc)

    # All three statements (insert attempt → read SRS state → upsert)
    # must run as one atomic unit. Without BEGIN IMMEDIATE, two
    # concurrent attempts for the same (learner, card) can both read
    # the same prior_box, both compute promote(), both upsert — the
    # later writer overwrites the first, and one attempt's progression
    # is silently lost (total_seen undercount, wrong box state).
    with _conn() as c:
        c.execute("BEGIN IMMEDIATE")
        try:
            attempt_id = str(uuid4())
            c.execute(
                "INSERT INTO card_attempts (attempt_id, card_id, learner_id, "
                "user_answer, ai_feedback, rating, hint_used, attempted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (attempt_id, card_id, learner_id, sanitised_answer,
                 ai_feedback, rating, 1 if hint_used else 0, now_iso),
            )

            row = c.execute(
                "SELECT box, total_seen, total_correct FROM card_review_state "
                "WHERE learner_id = ? AND card_id = ?",
                (learner_id, card_id),
            ).fetchone()

            prior_box = row["box"] if row else srs.MIN_BOX
            prior_seen = row["total_seen"] if row else 0
            prior_correct = row["total_correct"] if row else 0

            new_box = srs.promote(prior_box, correct=correct)
            new_due = srs.next_due_at(new_box, now_dt)
            new_seen = prior_seen + 1
            new_correct = prior_correct + (1 if correct else 0)

            c.execute(
                "INSERT INTO card_review_state (learner_id, card_id, box, "
                "next_due_at, last_seen_at, total_seen, total_correct) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(learner_id, card_id) DO UPDATE SET "
                "box = excluded.box, "
                "next_due_at = excluded.next_due_at, "
                "last_seen_at = excluded.last_seen_at, "
                "total_seen = excluded.total_seen, "
                "total_correct = excluded.total_correct",
                (learner_id, card_id, new_box, new_due, now_iso,
                 new_seen, new_correct),
            )
            c.execute("COMMIT")
        except Exception:
            try:
                c.execute("ROLLBACK")
            except Exception:                               # noqa: BLE001
                pass
            raise

    return {
        "attempt_id": attempt_id,
        "rating": rating,
        "new_box": new_box,
        "next_due_at": new_due,
        "total_seen": new_seen,
        "total_correct": new_correct,
    }


# ──────────────────────────────────────────────────────────────────
# Aggregate progress (drives the stats panel on the UI)
# ──────────────────────────────────────────────────────────────────

def progress_for(
    learner_id: str,
    *,
    goal_id: str | None = None,
) -> dict[str, Any]:
    """Return summary stats: total seen, total correct, per-box counts.

    Used by the API to echo into every learn-turn response so the UI
    can update the progress widget without a separate request.

    When ``goal_id`` is provided, the counts are scoped to cards in that
    one curriculum (joined via ``learning_cards.goal_id``). When
    omitted, aggregates across ALL of the learner's cards — used for
    learner-level views like the homepage progress badge."""
    if not learner_id:
        return {"total_seen": 0, "total_correct": 0, "by_box": {}}
    with _conn() as c:
        if goal_id:
            agg = c.execute(
                "SELECT COALESCE(SUM(crs.total_seen), 0) AS seen, "
                "COALESCE(SUM(crs.total_correct), 0) AS correct "
                "FROM card_review_state crs "
                "JOIN learning_cards lc ON lc.card_id = crs.card_id "
                "WHERE crs.learner_id = ? AND lc.goal_id = ?",
                (learner_id, goal_id),
            ).fetchone()
            boxes = c.execute(
                "SELECT crs.box, COUNT(*) AS n FROM card_review_state crs "
                "JOIN learning_cards lc ON lc.card_id = crs.card_id "
                "WHERE crs.learner_id = ? AND lc.goal_id = ? "
                "GROUP BY crs.box",
                (learner_id, goal_id),
            ).fetchall()
        else:
            agg = c.execute(
                "SELECT COALESCE(SUM(total_seen), 0) AS seen, "
                "COALESCE(SUM(total_correct), 0) AS correct "
                "FROM card_review_state WHERE learner_id = ?",
                (learner_id,),
            ).fetchone()
            boxes = c.execute(
                "SELECT box, COUNT(*) AS n FROM card_review_state "
                "WHERE learner_id = ? GROUP BY box",
                (learner_id,),
            ).fetchall()
    return {
        "total_seen": int(agg["seen"] or 0),
        "total_correct": int(agg["correct"] or 0),
        "by_box": {int(r["box"]): int(r["n"]) for r in boxes},
    }
