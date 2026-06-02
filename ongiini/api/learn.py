"""HTTP endpoints for the learn.ongiini.ai surface.

Five POST routes, all under ``/v1/learn/``:

  * ``/sessions`` — create a fresh anonymous learner or resume from a
    magic-link token. Returns the learner_id + a snapshot of profile
    completeness and the next intake field (if any).
  * ``/intake`` — submit one intake answer (name / age / level /
    objective). Validates the SHAPE through ``intake.validate_field``,
    persists via ``store.save_profile_field``, and returns the next
    field to ask for (or ``intake_complete=True``).
  * ``/next-card`` — return a card to study now. If the SRS queue has
    something due, surface it; otherwise ask the model to generate a
    new one (and, on the very first learning turn, design the
    curriculum outline before generating).
  * ``/answer`` — grade a submitted answer (model call), persist the
    attempt + updated Leitner state, return the rating + feedback +
    new progress snapshot.
  * ``/clear`` — delete the learner row and cascade (GDPR / "delete
    my data" parity).

The intake prompts are hard-coded friendly defaults — Sebastian's
explicit decision: the four fields are always the same so the LLM
doesn't need to design them; the LLM owns everything AFTER intake
(curriculum design, card authoring, grading).
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from owela import Model

from ..config import settings
from ..learning import cards as cards_mod
from ..learning import context as ctx_mod
from ..learning import curriculum, db, grading, intake, store, tokens
from ..learning.llm import ModelOutputError

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


# ──────────────────────────────────────────────────────────────────
# Request / response models
# ──────────────────────────────────────────────────────────────────

class SessionRequest(BaseModel):
    """POST /v1/learn/sessions body."""
    # Existing browser-side learner_id from localStorage (cold-visit
    # resume). If absent and no token, we create a fresh anonymous row.
    learner_id: str | None = Field(default=None, max_length=128)
    # Optional magic-link token (Phase 2 — issued by the chat-side
    # offer flow). Verified; on success the learner row is upserted
    # by the embedded learner_id and any goal_text is carried over.
    token: str | None = Field(default=None, max_length=4096)


class SessionResponse(BaseModel):
    learner_id: str
    intake_complete: bool
    next_intake_field: str | None
    next_intake_prompt: str | None
    profile: dict[str, Any] | None


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


class NextCardRequest(BaseModel):
    learner_id: str = Field(..., max_length=128)


class NextCardResponse(BaseModel):
    card_id: str
    card_type: str
    prompt_text: str
    hint_text: str | None = None
    difficulty: int | None = None
    progress: dict[str, Any]
    box: int   # current Leitner box for this card (1 for a freshly-generated card)


class AnswerRequest(BaseModel):
    learner_id: str = Field(..., max_length=128)
    card_id: str = Field(..., max_length=128)
    answer: str = Field(..., max_length=4000)
    hint_used: bool = False


class AnswerResponse(BaseModel):
    rating: str
    feedback: str
    new_box: int
    next_due_at: str
    progress: dict[str, Any]


class ClearRequest(BaseModel):
    learner_id: str = Field(..., max_length=128)


class ClearResponse(BaseModel):
    ok: bool
    rows_deleted: int


# ──────────────────────────────────────────────────────────────────
# Router factory
# ──────────────────────────────────────────────────────────────────

def build_router(*, model: Model, skill_content: str) -> APIRouter:
    """Build a FastAPI router with the learning endpoints.

    ``model`` is the shared VLLMGemmaModel instance built once at
    startup. ``skill_content`` is the rendered ``learning-afrikaans``
    SKILL.md (markdown body without the YAML frontmatter), loaded by
    the lifespan and passed in here so each call doesn't re-read it.
    """
    router = APIRouter()

    # ── /sessions ───────────────────────────────────────────────
    @router.post("/sessions", response_model=SessionResponse)
    async def create_or_resume_session(req: SessionRequest) -> SessionResponse:
        if not settings.learn_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Learning surface is temporarily disabled.",
            )

        learner_id: str | None = None
        goal_text: str | None = None

        # Magic-link arrivals (Phase 2 — verified token carries the
        # learner_id + goal_text). Verify before trusting either.
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
                # The learner_id was issued by us (signed); just refresh
                # last_active_at. If the row doesn't exist yet (new
                # magic-link with no prior intake), it'll be created
                # below by the create_anonymous_learner fallback.
                if not store.get_learner(learner_id):
                    # The token referred to a learner row that doesn't
                    # exist — could be a magic link for a brand-new
                    # anonymous learner. Create it.
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
        if goal_text:
            store.get_or_create_active_goal(learner_id, context=goal_text)

        profile = store.get_profile(learner_id)
        complete = intake.is_complete(profile)
        missing = intake.missing_fields(profile)
        next_field = missing[0] if missing else None

        return SessionResponse(
            learner_id=learner_id,
            intake_complete=complete,
            next_intake_field=next_field,
            next_intake_prompt=_intake_prompt(next_field),
            profile=profile,
        )

    # ── /intake ─────────────────────────────────────────────────
    @router.post("/intake", response_model=IntakeResponse)
    async def submit_intake(req: IntakeRequest) -> IntakeResponse:
        if not settings.learn_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Learning surface is temporarily disabled.",
            )

        # Validate the SHAPE — semantic correctness is the LLM's job
        # later, but the field name + value type must be storable.
        result = intake.validate_field(req.field, req.value)
        if not result.ok:
            profile = store.get_profile(req.learner_id)
            missing = intake.missing_fields(profile)
            return IntakeResponse(
                intake_complete=False,
                next_intake_field=req.field,
                next_intake_prompt=_intake_prompt(req.field),
                validation_error=result.reason,
                profile=profile,
            )

        # Persist (PII-scrubbed inside store.save_profile_field for
        # free-text fields).
        try:
            store.save_profile_field(req.learner_id, req.field, result.value)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            )

        # Recompute completeness.
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

    # ── /next-card ──────────────────────────────────────────────
    @router.post("/next-card", response_model=NextCardResponse)
    async def get_next_card(req: NextCardRequest) -> NextCardResponse:
        if not settings.learn_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Learning surface is temporarily disabled.",
            )

        profile = store.get_profile(req.learner_id)
        if not intake.is_complete(profile):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Intake not complete yet — finish onboarding first.",
            )

        # 1) Anything due in the SRS queue? Surface that first.
        due = store.next_due_cards(req.learner_id, limit=1)
        if due:
            d = due[0]
            return NextCardResponse(
                card_id=d["card_id"],
                card_type=d["card_type"],
                prompt_text=d["prompt_text"],
                hint_text=d.get("hint_text"),
                difficulty=d.get("difficulty"),
                progress=store.progress_for(req.learner_id),
                box=int(d.get("box") or 1),
            )

        # 2) Nothing due — get/create the goal, ensure outline exists,
        # then ask the model to author a new card.
        goal = store.get_or_create_active_goal(req.learner_id)
        ctx = ctx_mod.build_learner_context(
            req.learner_id, goal_id=goal["goal_id"]
        )

        # On the very first learning turn there's no outline yet —
        # design one before generating the card.
        if not ctx.curriculum_outline:
            try:
                outline = await curriculum.design_outline(
                    ctx, model=model, skill_content=skill_content,
                )
            except ModelOutputError as exc:
                log.warning("design_outline failed: %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Couldn't design your curriculum just now — try again in a moment.",
                )
            store.save_curriculum_outline(goal["goal_id"], outline)
            # Rebuild the context so the card prompt sees the freshly-
            # written outline.
            ctx = ctx_mod.build_learner_context(
                req.learner_id, goal_id=goal["goal_id"]
            )

        try:
            card_payload = await cards_mod.generate_card(
                ctx, model=model, skill_content=skill_content,
            )
        except ModelOutputError as exc:
            log.warning("generate_card failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Couldn't generate a card just now — try again in a moment.",
            )

        # Persist the card so a future SRS re-review surfaces the same
        # prompt rather than re-rolling.
        card_id = store.save_card(
            goal["goal_id"],
            card_payload["card_type"],
            card_payload["prompt_text"],
            reference_answer=card_payload.get("reference_answer"),
            hint_text=card_payload.get("hint_text"),
            difficulty=card_payload.get("difficulty"),
        )

        return NextCardResponse(
            card_id=card_id,
            card_type=card_payload["card_type"],
            prompt_text=card_payload["prompt_text"],
            hint_text=card_payload.get("hint_text"),
            difficulty=card_payload.get("difficulty"),
            progress=store.progress_for(req.learner_id),
            box=1,    # brand-new card starts in box 1
        )

    # ── /answer ─────────────────────────────────────────────────
    @router.post("/answer", response_model=AnswerResponse)
    async def submit_answer(req: AnswerRequest) -> AnswerResponse:
        if not settings.learn_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Learning surface is temporarily disabled.",
            )

        card = store.get_card(req.card_id)
        if not card:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Card not found.",
            )
        # Establish goal_id so the LLM sees the curriculum outline.
        goal = store.get_or_create_active_goal(req.learner_id)
        ctx = ctx_mod.build_learner_context(
            req.learner_id, goal_id=goal["goal_id"]
        )

        try:
            grading_payload = await grading.grade_answer(
                ctx, card=card, user_answer=req.answer,
                hint_used=req.hint_used,
                model=model, skill_content=skill_content,
            )
        except ModelOutputError as exc:
            log.warning("grade_answer failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Couldn't grade your answer just now — try again in a moment.",
            )

        # record_attempt updates the Leitner state + persists the
        # PII-scrubbed user_answer.
        attempt = store.record_attempt(
            learner_id=req.learner_id,
            card_id=req.card_id,
            user_answer=req.answer,
            ai_feedback=grading_payload["feedback"],
            rating=grading_payload["rating"],
            hint_used=req.hint_used,
        )

        return AnswerResponse(
            rating=attempt["rating"],
            feedback=grading_payload["feedback"],
            new_box=attempt["new_box"],
            next_due_at=attempt["next_due_at"],
            progress=store.progress_for(req.learner_id),
        )

    # ── /clear ──────────────────────────────────────────────────
    @router.post("/clear", response_model=ClearResponse)
    async def clear_learner(req: ClearRequest) -> ClearResponse:
        # Always allowed even when learn_enabled is False — let users
        # delete their data regardless of feature flags.
        n = store.delete_learner(req.learner_id)
        return ClearResponse(ok=True, rows_deleted=n)

    return router
