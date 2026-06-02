"""HTTP endpoints for the learn.ongiini.ai chat-first learning surface.

Phase 2 model: the whole UI is a chat thread between the learner and
the Ongiini coach. Rich cards (lessons / exercises / feedback /
progress) appear as messages inside the thread, alongside plain text
bubbles. The composer at the bottom dispatches into a single
``/turn`` endpoint that the coach orchestrator decides how to route —
answer to the active card, free-form question, or off-topic redirect.

Endpoints (all under ``/v1/learn/``):

  * ``/sessions``    — create or resume an anonymous learner. Returns
    profile completeness, the list of the learner's goals, the active
    goal info, and the active goal's thread so the frontend can
    rehydrate on a cold visit.
  * ``/intake``      — submit one intake answer. Same shape as Phase 1.
  * ``/turn``        — one chat turn. ``text`` is the learner's typed
    message (None means "give me what's next"). Coach decides whether
    to grade, teach, answer a question, or redirect off-topic. Returns
    the new messages to append + updated progress + outline + goal.
  * ``/goals``       — list this learner's goals.
  * ``/goals/new``   — create a new curriculum (activates by default).
  * ``/goals/activate`` — switch the active goal.
  * ``/goals/restart``  — wipe one goal's cards + thread (keep outline).
  * ``/goals/archive``  — soft-delete a goal.
  * ``/clear``       — delete the learner row + cascade (GDPR right-
    to-erasure).
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from owela import Model

from ..config import settings
from ..learning import coach as coach_mod
from ..learning import db, intake, messages as messages_mod, store, tokens

log = logging.getLogger("ongiini.api.learn")


# ──────────────────────────────────────────────────────────────────
# Friendly default intake prompts — the API surfaces these to the
# frontend for the four fixed fields. The LLM owns everything AFTER
# intake.
# ──────────────────────────────────────────────────────────────────

_INTAKE_PROMPTS = {
    intake.FIELD_NAME: "Welcome! What should I call you?",
    intake.FIELD_AGE: "How old are you? (Just so I pitch examples at the right level.)",
    intake.FIELD_LEVEL: (
        "Where would you say your Afrikaans is today — beginner, "
        "elementary, intermediate, or advanced?"
    ),
    intake.FIELD_OBJECTIVE: (
        "Last one — what do you actually want to be able to do in "
        "Afrikaans? A job interview, talking to in-laws, helping the "
        "kids with homework? In one sentence."
    ),
}


def _intake_prompt(field: str | None) -> str | None:
    if not field:
        return None
    return _INTAKE_PROMPTS.get(field)


def _ensure_enabled() -> None:
    """All learn endpoints honor the kill-switch except /clear (data
    deletion is always allowed). Single helper so the message stays
    consistent."""
    if not settings.learn_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Learning surface is temporarily disabled.",
        )


def _active_or_none(learner_id: str) -> dict[str, Any] | None:
    """Return the learner's currently-active goal dict, or None when
    they have no active goal (just archived their last one, never had
    one, etc.). NEVER auto-creates — endpoints that need a fresh goal
    must call ``store.get_or_create_active_goal`` explicitly so the
    side-effect is visible."""
    for g in store.list_goals(learner_id, include_archived=False):
        if g["status"] == "active":
            return g
    return None


# ──────────────────────────────────────────────────────────────────
# Request / response models
# ──────────────────────────────────────────────────────────────────

class SessionRequest(BaseModel):
    """POST /v1/learn/sessions body."""
    learner_id: str | None = Field(default=None, max_length=128)
    token: str | None = Field(default=None, max_length=4096)


class GoalInfo(BaseModel):
    goal_id: str
    title: str | None
    status: str
    language: str
    context: str | None
    has_outline: bool
    archived_at: str | None
    created_at: str


class MessageItem(BaseModel):
    message_id: str
    kind: str
    payload: dict[str, Any]
    card_id: str | None = None
    answered: bool = False
    created_at: str


class SessionResponse(BaseModel):
    learner_id: str
    intake_complete: bool
    next_intake_field: str | None
    next_intake_prompt: str | None
    profile: dict[str, Any] | None
    goals: list[GoalInfo]
    active_goal: GoalInfo | None
    thread: list[MessageItem]
    progress: dict[str, Any]
    curriculum_outline: dict[str, Any] | None


class IntakeRequest(BaseModel):
    learner_id: str = Field(..., max_length=128)
    field: str = Field(..., max_length=32)
    value: Any


class IntakeResponse(BaseModel):
    intake_complete: bool
    next_intake_field: str | None
    next_intake_prompt: str | None
    validation_error: str | None = None
    profile: dict[str, Any] | None


class TurnRequest(BaseModel):
    learner_id: str = Field(..., max_length=128)
    goal_id: str | None = Field(default=None, max_length=128)
    # ``text`` is the learner's typed message. ``None`` is a deliberate
    # "give me what's next" — used by the frontend on first paint after
    # intake completes, and after a graded answer that yielded a
    # transition coach_text but no follow-up exercise (rare).
    text: str | None = Field(default=None, max_length=4000)


class TurnResponse(BaseModel):
    learner_id: str
    goal_id: str
    messages: list[MessageItem]
    progress: dict[str, Any]
    curriculum_outline: dict[str, Any] | None
    goal: GoalInfo


class GoalsRequest(BaseModel):
    learner_id: str = Field(..., max_length=128)
    include_archived: bool = False


class GoalsResponse(BaseModel):
    goals: list[GoalInfo]
    active_goal_id: str | None


class GoalsNewRequest(BaseModel):
    learner_id: str = Field(..., max_length=128)
    title: str | None = Field(default=None, max_length=200)
    context: str | None = Field(default=None, max_length=2000)
    activate: bool = True


class GoalsNewResponse(BaseModel):
    goal: GoalInfo
    goals: list[GoalInfo]


class GoalsActivateRequest(BaseModel):
    learner_id: str = Field(..., max_length=128)
    goal_id: str = Field(..., max_length=128)


class GoalsActivateResponse(BaseModel):
    goal: GoalInfo
    goals: list[GoalInfo]
    thread: list[MessageItem]
    progress: dict[str, Any]
    curriculum_outline: dict[str, Any] | None


class GoalsRestartRequest(BaseModel):
    learner_id: str = Field(..., max_length=128)
    goal_id: str = Field(..., max_length=128)


class GoalsRestartResponse(BaseModel):
    goal: GoalInfo
    cards_deleted: int
    messages_deleted: int


class GoalsArchiveRequest(BaseModel):
    learner_id: str = Field(..., max_length=128)
    goal_id: str = Field(..., max_length=128)


class GoalsArchiveResponse(BaseModel):
    goals: list[GoalInfo]
    active_goal_id: str | None


class ClearRequest(BaseModel):
    learner_id: str = Field(..., max_length=128)


class ClearResponse(BaseModel):
    ok: bool
    rows_deleted: int


# ──────────────────────────────────────────────────────────────────
# Marshalling helpers
# ──────────────────────────────────────────────────────────────────

def _goal_info(row: dict[str, Any]) -> GoalInfo:
    """Coerce a store-row into the API's GoalInfo. Tolerant of dict
    shapes from ``list_goals`` (has_outline pre-computed) and from
    ``get_or_create_active_goal`` (has curriculum_outline column)."""
    has_outline = row.get("has_outline")
    if has_outline is None:
        has_outline = bool(row.get("curriculum_outline"))
    return GoalInfo(
        goal_id=row["goal_id"],
        title=row.get("title"),
        status=row["status"],
        language=row.get("language") or "afrikaans",
        context=row.get("context"),
        has_outline=bool(has_outline),
        archived_at=row.get("archived_at"),
        created_at=row["created_at"],
    )


def _message_item(row: dict[str, Any]) -> MessageItem:
    return MessageItem(
        message_id=row["message_id"],
        kind=row["kind"],
        payload=row["payload"],
        card_id=row.get("card_id"),
        answered=bool(row.get("answered")),
        created_at=row["created_at"],
    )


def _goal_payload_bundle(
    learner_id: str,
    goal: dict[str, Any],
) -> tuple[GoalInfo, list[MessageItem], dict[str, Any], dict[str, Any] | None]:
    """Common (goal, thread, progress, outline) bundle. The /turn,
    /sessions, and /goals/activate endpoints all need exactly this
    shape — factor once so the API stays consistent."""
    goal_info = _goal_info(goal)
    thread = [
        _message_item(m) for m in messages_mod.list_for_goal(
            learner_id=learner_id, goal_id=goal["goal_id"],
        )
    ]
    progress = store.progress_for(learner_id, goal_id=goal["goal_id"])
    outline = store.get_curriculum_outline(goal["goal_id"])
    return goal_info, thread, progress, outline


# ──────────────────────────────────────────────────────────────────
# Router factory
# ──────────────────────────────────────────────────────────────────

def build_router(*, model: Model, skill_content: str) -> APIRouter:
    """Build a FastAPI router with the learning endpoints.

    ``model`` is the shared VLLMGemmaModel instance built once at
    startup. ``skill_content`` is the rendered ``learning-afrikaans``
    SKILL.md, loaded by the lifespan and passed in here so each call
    doesn't re-read it.
    """
    router = APIRouter()

    # ── /sessions ───────────────────────────────────────────────
    @router.post("/sessions", response_model=SessionResponse)
    async def create_or_resume_session(req: SessionRequest) -> SessionResponse:
        _ensure_enabled()

        learner_id: str | None = None
        goal_text: str | None = None

        # Magic-link arrivals carry a signed learner_id + goal_text.
        if req.token:
            payload = tokens.verify(req.token)
            if not payload:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Magic link is invalid or expired.",
                )
            learner_id = str(payload.get("lid") or "")
            goal_text = payload.get("g")
            if learner_id:
                if not store.get_learner(learner_id):
                    learner_id = store.create_anonymous_learner()
                else:
                    store.touch_learner(learner_id)

        # Cold-visit resume: browser passed back its stashed learner_id.
        if not learner_id and req.learner_id:
            row = store.get_learner(req.learner_id)
            if row:
                learner_id = req.learner_id
                store.touch_learner(learner_id)

        # Fresh anonymous learner.
        if not learner_id:
            learner_id = store.create_anonymous_learner()

        # Seed the goal context if we have one from the magic link.
        # Only seed when the learner has NO live goals at all — a paused
        # goal is still a curriculum we should respect; silently spawning
        # a third on every magic-link arrival is the previous bug. If the
        # learner has paused-only state, the frontend should prompt them
        # to pick or restart rather than the API guessing.
        if goal_text and not store.list_goals(learner_id, include_archived=False):
            store.get_or_create_active_goal(learner_id, context=goal_text)

        profile = store.get_profile(learner_id)
        complete = intake.is_complete(profile)
        missing = intake.missing_fields(profile)
        next_field = missing[0] if missing else None

        goals_list = [_goal_info(g) for g in store.list_goals(learner_id)]
        active = _active_or_none(learner_id)

        if active:
            goal_info, thread, progress, outline = _goal_payload_bundle(
                learner_id, active,
            )
        else:
            goal_info = None
            thread = []
            progress = store.progress_for(learner_id)
            outline = None

        return SessionResponse(
            learner_id=learner_id,
            intake_complete=complete,
            next_intake_field=next_field,
            next_intake_prompt=_intake_prompt(next_field),
            profile=profile,
            goals=goals_list,
            active_goal=goal_info,
            thread=thread,
            progress=progress,
            curriculum_outline=outline,
        )

    # ── /intake ─────────────────────────────────────────────────
    @router.post("/intake", response_model=IntakeResponse)
    async def submit_intake(req: IntakeRequest) -> IntakeResponse:
        _ensure_enabled()

        result = intake.validate_field(req.field, req.value)
        if not result.ok:
            profile = store.get_profile(req.learner_id)
            return IntakeResponse(
                intake_complete=False,
                next_intake_field=req.field,
                next_intake_prompt=_intake_prompt(req.field),
                validation_error=result.reason,
                profile=profile,
            )

        try:
            store.save_profile_field(req.learner_id, req.field, result.value)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            )

        profile = store.get_profile(req.learner_id)
        if intake.is_complete(profile):
            store.mark_intake_complete(req.learner_id)
            profile = store.get_profile(req.learner_id)
            return IntakeResponse(
                intake_complete=True,
                next_intake_field=None,
                next_intake_prompt=None,
                profile=profile,
            )

        missing = intake.missing_fields(profile)
        next_field = missing[0] if missing else None
        return IntakeResponse(
            intake_complete=False,
            next_intake_field=next_field,
            next_intake_prompt=_intake_prompt(next_field),
            profile=profile,
        )

    # ── /turn ───────────────────────────────────────────────────
    @router.post("/turn", response_model=TurnResponse)
    async def take_turn(req: TurnRequest) -> TurnResponse:
        _ensure_enabled()

        # Intake gating — the coach assumes a fully-onboarded profile;
        # without it the prompts are useless and we'd burn tokens.
        profile = store.get_profile(req.learner_id)
        if not intake.is_complete(profile):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Intake not complete — finish onboarding first.",
            )

        # Resolve the goal: explicit goal_id (validated for ownership),
        # else the active goal, else get-or-create. ``get_or_create``
        # is the legacy single-curriculum behaviour — preserves it for
        # the simple case where the frontend ignores goal_id.
        if req.goal_id:
            row = store.get_learner(req.learner_id)
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Learner not found.",
                )
            goals = store.list_goals(req.learner_id, include_archived=True)
            match = next(
                (g for g in goals if g["goal_id"] == req.goal_id), None,
            )
            if not match:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Goal not found.",
                )
            if match["status"] == "archived":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Goal is archived — re-create or activate "
                    "another curriculum.",
                )
            goal = match
        else:
            active = _active_or_none(req.learner_id)
            if active:
                goal = active
            else:
                goal = store.get_or_create_active_goal(req.learner_id)

        new_msgs = await coach_mod.run_turn(
            learner_id=req.learner_id,
            goal_id=goal["goal_id"],
            user_text=req.text,
            model=model,
            skill_content=skill_content,
        )

        # Re-read the goal so any side-effects (outline written, title
        # set) surface in the response.
        fresh = next(
            (g for g in store.list_goals(req.learner_id, include_archived=True)
             if g["goal_id"] == goal["goal_id"]),
            goal,
        )
        progress = store.progress_for(req.learner_id, goal_id=goal["goal_id"])
        outline = store.get_curriculum_outline(goal["goal_id"])

        return TurnResponse(
            learner_id=req.learner_id,
            goal_id=goal["goal_id"],
            messages=[_message_item(m) for m in new_msgs],
            progress=progress,
            curriculum_outline=outline,
            goal=_goal_info(fresh),
        )

    # ── /goals ──────────────────────────────────────────────────
    @router.post("/goals", response_model=GoalsResponse)
    async def list_goals_endpoint(req: GoalsRequest) -> GoalsResponse:
        _ensure_enabled()
        goals = store.list_goals(
            req.learner_id, include_archived=req.include_archived,
        )
        active = next((g for g in goals if g["status"] == "active"), None)
        return GoalsResponse(
            goals=[_goal_info(g) for g in goals],
            active_goal_id=active["goal_id"] if active else None,
        )

    # ── /goals/new ──────────────────────────────────────────────
    @router.post("/goals/new", response_model=GoalsNewResponse)
    async def create_goal(req: GoalsNewRequest) -> GoalsNewResponse:
        _ensure_enabled()
        if not store.get_learner(req.learner_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Learner not found.",
            )
        new_goal = store.create_new_goal(
            req.learner_id,
            title=req.title,
            context=req.context,
            activate=req.activate,
        )
        return GoalsNewResponse(
            goal=_goal_info(new_goal),
            goals=[_goal_info(g) for g in store.list_goals(req.learner_id)],
        )

    # ── /goals/activate ─────────────────────────────────────────
    @router.post("/goals/activate", response_model=GoalsActivateResponse)
    async def activate_goal_endpoint(
        req: GoalsActivateRequest,
    ) -> GoalsActivateResponse:
        _ensure_enabled()
        try:
            activated = store.activate_goal(req.learner_id, req.goal_id)
        except RuntimeError as exc:
            # Cross-tenant / archived / missing — all collapse to 404
            # to avoid leaking which path was wrong.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            )

        goal_info, thread, progress, outline = _goal_payload_bundle(
            req.learner_id, activated,
        )
        return GoalsActivateResponse(
            goal=goal_info,
            goals=[_goal_info(g) for g in store.list_goals(req.learner_id)],
            thread=thread,
            progress=progress,
            curriculum_outline=outline,
        )

    # ── /goals/restart ──────────────────────────────────────────
    @router.post("/goals/restart", response_model=GoalsRestartResponse)
    async def restart_goal_endpoint(
        req: GoalsRestartRequest,
    ) -> GoalsRestartResponse:
        _ensure_enabled()
        try:
            summary = store.restart_goal(req.learner_id, req.goal_id)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            )
        # Re-read the goal row so the response carries the up-to-date
        # title + status (the underlying restart only wiped content).
        row = next(
            (g for g in store.list_goals(req.learner_id, include_archived=True)
             if g["goal_id"] == req.goal_id),
            None,
        )
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Goal not found after restart.",
            )
        return GoalsRestartResponse(
            goal=_goal_info(row),
            cards_deleted=summary["cards_deleted"],
            messages_deleted=summary["messages_deleted"],
        )

    # ── /goals/archive ──────────────────────────────────────────
    @router.post("/goals/archive", response_model=GoalsArchiveResponse)
    async def archive_goal_endpoint(
        req: GoalsArchiveRequest,
    ) -> GoalsArchiveResponse:
        _ensure_enabled()
        try:
            store.archive_goal(req.learner_id, req.goal_id)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            )
        goals = store.list_goals(req.learner_id)
        active = next((g for g in goals if g["status"] == "active"), None)
        return GoalsArchiveResponse(
            goals=[_goal_info(g) for g in goals],
            active_goal_id=active["goal_id"] if active else None,
        )

    # ── /clear ──────────────────────────────────────────────────
    @router.post("/clear", response_model=ClearResponse)
    async def clear_learner(req: ClearRequest) -> ClearResponse:
        # Always allowed — data deletion is independent of feature flags.
        n = store.delete_learner(req.learner_id)
        return ClearResponse(ok=True, rows_deleted=n)

    return router
