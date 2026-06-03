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
from typing import Callable

from ..learning import coach as coach_mod
from ..learning import conversation as conversation_mod
from ..learning import context as ctx_mod
from ..learning import (
    db, intake, intake_parser, messages as messages_mod, store, tokens,
)
from ..learning import skill_renderer as skill_renderer_mod

log = logging.getLogger("ongiini.api.learn")


# ──────────────────────────────────────────────────────────────────
# Friendly default intake prompts — the API surfaces these to the
# frontend for the four fixed fields. The LLM owns everything AFTER
# intake. The level + objective prompts reference the TARGET language
# the learner picked on the landing page; if no target is known yet
# (cold magic-link arrival), they fall back to a neutral phrasing.
# ──────────────────────────────────────────────────────────────────

# Display names for the {target} placeholder — keys are the lowercase
# canonical language values used in the goal record.
_TARGET_DISPLAY = {
    "afrikaans": "Afrikaans",
    "english": "English",
    "german": "German",
    "oshiwambo": "Oshiwambo",
}

_INTAKE_PROMPTS_TEMPLATE = {
    intake.FIELD_NAME: "What should I call you?",
    intake.FIELD_AGE: "How old are you? (Just so I pitch examples at the right level.)",
    intake.FIELD_LEVEL: (
        "Where would you say your {target} is today — beginner, "
        "elementary, intermediate, or advanced?"
    ),
    intake.FIELD_OBJECTIVE: (
        "Last one — what do you want to focus on for {target}? A "
        "situation you're preparing for (a job interview, talking to "
        "colleagues, your in-laws) OR a specific topic you want to "
        "nail (past tenses, modal verbs, restaurant vocabulary). One "
        "sentence."
    ),
}


def _intake_prompt(field: str | None, target_language: str | None = None) -> str | None:
    if not field:
        return None
    raw = _INTAKE_PROMPTS_TEMPLATE.get(field)
    if raw is None:
        return None
    if "{target}" not in raw:
        return raw
    target = (target_language or "").strip().lower()
    display = _TARGET_DISPLAY.get(target) or "the language you want to learn"
    return raw.replace("{target}", display)


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
    # The target language the learner just picked on the landing page,
    # before any goal exists. Used to phrase intake prompts in terms of
    # the right language ("Where would you say your German is today")
    # instead of always falling back to Afrikaans. Optional — magic-link
    # arrivals and cold resumes may not have it.
    target_language: str | None = Field(default=None, max_length=32)


class GoalInfo(BaseModel):
    goal_id: str
    title: str | None
    status: str
    # ``language`` is the TARGET language the learner is studying.
    # Kept as ``language`` (not ``target_language``) for back-compat
    # with the existing frontend state machine. ``source_language`` is
    # the new field — what the learner already speaks well.
    language: str
    source_language: str = "english"
    current_level: str | None = None
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
    # Pending target language picked on the landing page so the NEXT
    # intake prompt (after the parsed answer is saved) references the
    # right language. Mirrors SessionRequest.target_language.
    target_language: str | None = Field(default=None, max_length=32)


class IntakeResponse(BaseModel):
    intake_complete: bool
    next_intake_field: str | None
    next_intake_prompt: str | None
    # When the LLM can't confidently extract a value from free-text,
    # it returns a natural-voice follow-up here. The frontend renders
    # this as a normal coach bubble (no "Hmm — validation error" stiff
    # template). Either ``clarify_text`` OR ``validation_error`` is set;
    # ``validation_error`` is kept for the rare defence-in-depth case
    # where the LLM extracts something the shape validator rejects.
    clarify_text: str | None = None
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


class ChatRequest(BaseModel):
    """POST /v1/learn/chat body — conversation mode (Track C)."""
    learner_id: str = Field(..., max_length=128)
    goal_id: str | None = Field(default=None, max_length=128)
    # Required: the learner's turn IN TARGET LANGUAGE at their level.
    # An empty text isn't useful here (unlike /turn which uses None
    # as "what's next") so the field is required.
    text: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    """The coach's reply + structured Notes block + the persisted
    chat-message history rows the frontend appended to its thread."""
    learner_id: str
    goal_id: str
    # Persisted message rows (MSG_CHAT_LEARNER, MSG_CHAT_COACH,
    # MSG_CHAT_NOTES) the frontend appends to its chat thread in
    # order.
    messages: list[dict[str, Any]]


class ModuleProgress(BaseModel):
    """Per-module progress for the slim bar under the topbar + per-
    module badges in the curriculum panel."""
    module_id: str
    title: str | None = None
    status: str | None = None
    estimated_cards: int | None = None
    # Counters from the join.
    lessons_given: int = 0
    exercises_emitted: int = 0
    exercises_attempted: int = 0
    exercises_correct: int = 0
    # cards_in_module is exercises_emitted + lessons_given — convenience
    # so the frontend can format "5 / 8" without doing the math.
    cards_in_module: int = 0


class TurnResponse(BaseModel):
    learner_id: str
    goal_id: str
    messages: list[MessageItem]
    progress: dict[str, Any]
    curriculum_outline: dict[str, Any] | None
    goal: GoalInfo
    # The learner's full non-archived goals list, refreshed every turn
    # so the drawer's "My curriculums" stays in sync. Without this the
    # drawer is stuck at whatever was last returned by /sessions /
    # /goals/new / /goals/activate — the auto-created goal that /turn
    # spawned on intake completion was invisible there.
    goals: list[GoalInfo] = []
    # Per-module rollups so the frontend can render the slim bar +
    # the curriculum-panel badges. Ordered to match the outline.
    module_progress: list[ModuleProgress] = []
    # The module the learner is actively working on (status=='in_progress'
    # in the outline). The slim bar reads this to know what to display.
    active_module_id: str | None = None


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
    # Language pair. Defaults preserve the prior Afrikaans-from-English
    # behaviour so older frontends without the pair UI keep working.
    # New frontends always send both — server validates the pair via
    # skill_renderer.validate_language_pair inside store.create_new_goal.
    language: str = Field(default="afrikaans", max_length=32)
    source_language: str = Field(default="english", max_length=32)
    current_level: str | None = Field(default=None, max_length=32)
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
    # Mirrors TurnResponse so the frontend can drop in the slim
    # progress bar + curriculum-panel badges for the newly-activated
    # goal without waiting for the first /turn. Without these the bar
    # showed the previous goal's data until something triggered a
    # /turn refresh.
    module_progress: list[ModuleProgress] = []
    active_module_id: str | None = None


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
        source_language=row.get("source_language") or "english",
        current_level=row.get("current_level"),
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


def _module_progress_from_outline(
    learner_id: str,
    goal_id: str,
    outline: dict[str, Any] | None,
) -> tuple[list[ModuleProgress], str | None]:
    """Build the per-module progress list ordered to match the outline
    + identify the active module (status == 'in_progress'). Returns
    ([], None) when there's no outline yet (intake just finished, the
    first /turn is about to design the curriculum).

    The denominator displayed to the learner is the outline's
    ``estimated_cards`` — what the LLM said it'd take when it drew the
    plan, so the learner's expectations match what they saw. The
    frontend caps the visible bar at 100% if actual emissions exceed
    the estimate."""
    if not outline:
        return [], None
    digest = store.progress_for_modules(learner_id, goal_id)
    out: list[ModuleProgress] = []
    active: str | None = None
    for m in outline.get("modules", []) or []:
        if not isinstance(m, dict):
            continue
        mod_id = m.get("id")
        if not isinstance(mod_id, str):
            continue
        d = digest.get(mod_id, {})
        if m.get("status") == "in_progress" and active is None:
            active = mod_id
        out.append(ModuleProgress(
            module_id=mod_id,
            title=m.get("title") if isinstance(m.get("title"), str) else None,
            status=m.get("status") if isinstance(m.get("status"), str) else None,
            estimated_cards=(
                int(m["estimated_cards"])
                if isinstance(m.get("estimated_cards"), int)
                else None
            ),
            lessons_given=int(d.get("lessons_given", 0)),
            exercises_emitted=int(d.get("exercises_emitted", 0)),
            exercises_attempted=int(d.get("exercises_attempted", 0)),
            exercises_correct=int(d.get("exercises_correct", 0)),
            cards_in_module=int(d.get("cards_in_module", 0)),
        ))
    return out, active


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

def build_router(
    *,
    model: Model,
    skill_renderer: Callable[[str, str], str] | None = None,
) -> APIRouter:
    """Build a FastAPI router with the learning endpoints.

    ``model`` is the shared VLLMGemmaModel instance built once at
    startup. ``skill_renderer`` is a callable
    ``(source_language, target_language) -> skill_text`` — invoked
    per turn so the right per-pair skill markdown reaches the LLM.
    Defaults to ``skill_renderer_mod.render_skill_for_pair`` which
    reads from the bundled core template + per-language anchor files;
    tests can pass a stub.
    """
    if skill_renderer is None:
        skill_renderer = skill_renderer_mod.render_skill_for_pair
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

        # Prefer the active goal's target language (resume flow); fall
        # back to whatever the landing page passed (pending pick); the
        # template defaults to a neutral phrasing if neither is set.
        prompt_target = (
            (goal_info.language if goal_info else None)
            or req.target_language
        )

        return SessionResponse(
            learner_id=learner_id,
            intake_complete=complete,
            next_intake_field=next_field,
            next_intake_prompt=_intake_prompt(next_field, prompt_target),
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

        # LLM step — interpret the learner's free text BEFORE running
        # the shape validator. Without this the validator's mechanical
        # rejection ("age must be a positive integer") leaks through to
        # the chat as a bot bubble, which violates "the LLM owns
        # content; the deterministic layer owns shape/persistence".
        # The parser returns either a clean value to validate or a
        # natural-voice follow-up to surface as a coach bubble.
        if not isinstance(req.value, str):
            # Legacy callers (or magic-link prefills) may pass typed
            # primitives — go straight to the shape validator in that
            # case, no LLM call needed.
            extracted_value: Any = req.value
            clarify: str | None = None
        else:
            parser_result = await intake_parser.parse_intake_answer(
                field=req.field, user_text=req.value, model=model,
            )
            if intake_parser.CLARIFY_KEY in parser_result:
                profile = store.get_profile(req.learner_id)
                return IntakeResponse(
                    intake_complete=False,
                    next_intake_field=req.field,
                    next_intake_prompt=_intake_prompt(req.field, req.target_language),
                    clarify_text=parser_result[intake_parser.CLARIFY_KEY],
                    profile=profile,
                )
            extracted_value = parser_result.get(intake_parser.VALUE_KEY, req.value)
            clarify = None

        result = intake.validate_field(req.field, extracted_value)
        if not result.ok:
            # Defence in depth — the LLM produced something the shape
            # validator rejects. Surface as a polite clarify so the user
            # never sees the raw "must be a positive integer" string.
            profile = store.get_profile(req.learner_id)
            return IntakeResponse(
                intake_complete=False,
                next_intake_field=req.field,
                next_intake_prompt=_intake_prompt(req.field, req.target_language),
                clarify_text="Sorry — could you say that another way?",
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
            next_intake_prompt=_intake_prompt(next_field, req.target_language),
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
                # First /turn after intake — auto-create a goal AND
                # seed its title from the learner's stated objective.
                # Without this the new goal lands in the drawer as
                # "Untitled curriculum" because /turn was the only
                # creation path that didn't accept a title.
                title_seed: str | None = None
                if profile and isinstance(profile.get("objective"), str):
                    title_seed = str(profile.get("objective"))
                goal = store.get_or_create_active_goal(
                    req.learner_id, title=title_seed,
                )

        # Render the per-pair skill once per turn from the goal row.
        # Defaults exist on existing data via the backfill so this is
        # always callable; the renderer also validates the pair.
        skill_text = skill_renderer(
            source=goal.get("source_language") or "english",
            target=goal.get("language") or "afrikaans",
        )
        new_msgs = await coach_mod.run_turn(
            learner_id=req.learner_id,
            goal_id=goal["goal_id"],
            user_text=req.text,
            model=model,
            skill_content=skill_text,
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
        mod_progress, active_mod_id = _module_progress_from_outline(
            req.learner_id, goal["goal_id"], outline,
        )

        # Refresh the drawer's goal list every turn — without this,
        # auto-creation of the first goal (via the fallback above)
        # never gets surfaced to the drawer until the user explicitly
        # creates or switches a goal.
        goals_list = [_goal_info(g) for g in store.list_goals(req.learner_id)]

        return TurnResponse(
            learner_id=req.learner_id,
            goal_id=goal["goal_id"],
            messages=[_message_item(m) for m in new_msgs],
            progress=progress,
            curriculum_outline=outline,
            goal=_goal_info(fresh),
            goals=goals_list,
            module_progress=mod_progress,
            active_module_id=active_mod_id,
        )

    # ── /chat ─── Track C: conversation mode ───────────────────
    @router.post("/chat", response_model=ChatResponse)
    async def chat_turn_endpoint(req: ChatRequest) -> ChatResponse:
        """One conversation turn. The coach replies in the goal's
        TARGET LANGUAGE at the learner's level, plus a Notes block
        (corrections + new high-frequency words). Persists three
        messages to the thread (learner / coach / notes) so the chat
        rehydrates on reload."""
        _ensure_enabled()

        # Same intake gate as /turn — without a profile the prompts
        # are useless and we'd burn tokens.
        profile = store.get_profile(req.learner_id)
        if not intake.is_complete(profile):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Intake not complete — finish onboarding first.",
            )

        # Resolve the goal — same logic as /turn so chat and cards
        # share the same active-curriculum semantics.
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
                    detail="Goal is archived.",
                )
            goal = match
        else:
            active = _active_or_none(req.learner_id)
            if active is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=("No active curriculum — pick a track first "
                            "(chat needs a target language)."),
                )
            goal = active

        goal_id = goal["goal_id"]

        # Render the skill + build the learner context. Context reuse
        # is intentional — chat sees the same CEFR calibration, the
        # same error_patterns, the same focus the cards see.
        skill_text = skill_renderer(
            source=goal.get("source_language") or "english",
            target=goal.get("language") or "afrikaans",
        )
        ctx = ctx_mod.build_learner_context(
            req.learner_id, goal_id=goal_id,
        )

        # Build conversation memory from prior MSG_CHAT_* rows. The
        # full thread list is used (not the recent_cards window) so
        # the conversation can carry across long sessions.
        thread = messages_mod.list_for_goal(
            learner_id=req.learner_id, goal_id=goal_id, limit=200,
        )
        history = conversation_mod.build_history_from_messages(thread)

        # Persist the learner turn FIRST so a model-side failure
        # doesn't lose the input.
        learner_msg = messages_mod.append(
            learner_id=req.learner_id, goal_id=goal_id,
            kind=db.MSG_CHAT_LEARNER,
            payload={"text": req.text},
        )

        turn = await conversation_mod.chat_turn(
            ctx,
            user_text=req.text,
            history=history,
            model=model,
            skill_content=skill_text,
        )

        # Soft-fail surface: an empty reply means the model crashed
        # or returned malformed JSON. Emit a friendly coach_text in
        # SOURCE language asking the learner to try again rather
        # than fabricating a target-language reply.
        if not turn.reply:
            error_msg = messages_mod.append(
                learner_id=req.learner_id, goal_id=goal_id,
                kind=db.MSG_COACH_TEXT,
                payload={
                    "text": "Sorry — I couldn't reply just now. Try "
                            "rephrasing or sending again.",
                    "meta": {"error": "chat_turn_failed"},
                },
            )
            return ChatResponse(
                learner_id=req.learner_id,
                goal_id=goal_id,
                messages=[_message_item(learner_msg), _message_item(error_msg)],
            )

        coach_msg = messages_mod.append(
            learner_id=req.learner_id, goal_id=goal_id,
            kind=db.MSG_CHAT_COACH,
            payload={"reply": turn.reply},
        )
        out_messages = [
            _message_item(learner_msg),
            _message_item(coach_msg),
        ]
        # Notes block — only emit when there's something worth
        # surfacing. A clean turn with no corrections + no new words
        # doesn't clutter the thread with an empty Notes bubble.
        if turn.corrections or turn.new_words:
            notes_msg = messages_mod.append(
                learner_id=req.learner_id, goal_id=goal_id,
                kind=db.MSG_CHAT_NOTES,
                payload={
                    "corrections": turn.corrections,
                    "new_words": turn.new_words,
                },
            )
            out_messages.append(_message_item(notes_msg))
        return ChatResponse(
            learner_id=req.learner_id,
            goal_id=goal_id,
            messages=out_messages,
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
        try:
            new_goal = store.create_new_goal(
                req.learner_id,
                title=req.title,
                context=req.context,
                language=req.language,
                source_language=req.source_language,
                current_level=req.current_level,
                activate=req.activate,
            )
        except ValueError as exc:
            # Unsupported language / same source-target collapses to 422.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
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
        mod_progress, active_mod_id = _module_progress_from_outline(
            req.learner_id, activated["goal_id"], outline,
        )
        return GoalsActivateResponse(
            goal=goal_info,
            goals=[_goal_info(g) for g in store.list_goals(req.learner_id)],
            thread=thread,
            progress=progress,
            curriculum_outline=outline,
            module_progress=mod_progress,
            active_module_id=active_mod_id,
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
