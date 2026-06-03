"""End-to-end tests for the /v1/learn/* endpoints (Phase 2).

Drives a FastAPI TestClient against the real router built with a
FakeModel. Each test isolates state via the ``temp_db`` fixture
(patches ``settings.data_dir`` to a fresh tmp_path) so they can run
in any order without leakage.

What's covered:
  * /sessions: cold visit, resume, post-intake rehydrate of thread +
    active goal
  * /intake: validation + completion
  * /turn: gating (intake required), goal resolution (explicit /
    active / get-or-create), 404 / 409 paths, model invocation
  * /goals: list / new / activate / restart / archive lifecycle
  * /clear: data deletion

The FakeModel here is the same shape coach.py + classifier tests use:
a queued list of canned responses consumed in order. Tests stub the
classifier's verdicts inline so the routing decisions are
deterministic without needing a live LLM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from owela import ModelRequest, ModelResponse

from ongiini.api.learn import build_router
from ongiini.learning import db, store


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    from ongiini import config
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    db.warmup()
    return tmp_path / "learning.sqlite"


@dataclass
class FakeModel:
    """Returns the next response in ``responses`` for each successive
    call, defaulting to ``response`` when the queue is empty. Captures
    every request so tests can assert on what the coach asked."""
    response: str = ""
    responses: list[str] = field(default_factory=list)
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, req: ModelRequest) -> ModelResponse:
        self.requests.append(req)
        body = self.responses.pop(0) if self.responses else self.response
        return ModelResponse(
            content=body, tool_calls=[], finish_reason="stop",
            tokens_in=5, tokens_out=5, cached_tokens=0, raw=None,
        )


_OUTLINE = json.dumps({
    "summary": "Job-interview Afrikaans in 2 weeks.",
    "modules": [
        {"id": "mod-1", "title": "Greetings", "status": "in_progress",
         "topics": [
             {"id": "t1", "title": "Greetings basics", "kind": "lesson"},
             {"id": "t2", "title": "Drill: greetings", "kind": "practice"},
         ]},
    ],
})
_EXERCISE = json.dumps({
    "prompt_text": "How do you say 'thank you'?",
    "reference_answer": "dankie",
})
# Lesson content payload — steps[] only, no scaffolding (the coach
# attaches card_type/module_id/topic_id from the selector's pick).
_LESSON = json.dumps({
    "title": "Time-of-day greetings",
    "steps": [
        {"kind": "concept", "body": "Greetings vary by time of day."},
        {"kind": "example", "body": "Examples:",
         "examples": ["goeie môre", "goeie naand"]},
    ],
})
# Critic-approves-on-iter-1 response — slots between outline and the
# next card in queues that exercise the curriculum-design path.
_CRITIC_READY = json.dumps({"ready": True, "score": 9, "issues": []})


def _stub_skill_renderer(*, source: str, target: str) -> str:
    """Test-time replacement for the per-pair skill renderer — emits a
    short marker string so tests can assert on which pair was selected
    without dragging the real template + anchor files into every
    test fixture. Mirrors the production renderer's keyword-only
    signature so the call sites stay honest."""
    return f"SKILL source={source} target={target}"


def _client(fake_model: FakeModel | None = None) -> TestClient:
    """Build a TestClient against a fresh FastAPI app with the learn
    router mounted at /v1/learn — same path layout production uses.

    The intake LLM parser is invoked on every /intake POST with a
    string value, so 4 canned ``{"value": ...}`` responses are PRE-
    PENDED to the FakeModel queue. Tests that exercise a specific
    /turn or grading flow continue to control their own response order
    by passing a FakeModel — those responses run AFTER the intake
    prefix the helper installs."""
    app = FastAPI()
    if fake_model is None:
        fake_model = FakeModel(responses=list(_INTAKE_PARSER_RESPONSES))
    else:
        fake_model.responses = (
            list(_INTAKE_PARSER_RESPONSES) + fake_model.responses
        )
    router = build_router(model=fake_model, skill_renderer=_stub_skill_renderer)
    app.include_router(router, prefix="/v1/learn")
    return TestClient(app)


_INTAKE_PARSER_RESPONSES = [
    json.dumps({"value": "Sebastian"}),
    json.dumps({"value": 35}),
    json.dumps({"value": "beginner"}),
    json.dumps({"value": "job interview at SPAR"}),
]


def _finish_intake(client: TestClient, learner_id: str) -> None:
    """Convenience — drive the four intake POSTs so /turn isn't gated.

    The intake LLM is in front of the validator now (free-text →
    {value} or {clarify}); each POST consumes one model response from
    the queue. Tests that only need intake done can call this; tests
    that explicitly exercise intake should set up FakeModel responses
    themselves."""
    for field_name, value in [
        ("name", "Sebastian"),
        ("age", "35"),    # str so it routes through the parser
        ("current_level", "beginner"),
        ("objective", "job interview at SPAR"),
    ]:
        r = client.post(
            "/v1/learn/intake",
            json={"learner_id": learner_id, "field": field_name, "value": value},
        )
        assert r.status_code == 200, r.text


# ============================================================
# /sessions
# ============================================================

def test_sessions_cold_visit_creates_anonymous_learner(temp_db):
    client = _client()
    r = client.post("/v1/learn/sessions", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["learner_id"]) == 36
    assert body["intake_complete"] is False
    assert body["next_intake_field"] == "name"
    assert body["next_intake_prompt"]
    # Cold visit → no goals, no thread, empty progress.
    assert body["goals"] == []
    assert body["active_goal"] is None
    assert body["thread"] == []


def test_sessions_resume_with_known_learner_id(temp_db):
    client = _client()
    learner_id = store.create_anonymous_learner()
    r = client.post("/v1/learn/sessions", json={"learner_id": learner_id})
    assert r.status_code == 200
    assert r.json()["learner_id"] == learner_id


def test_sessions_magic_link_does_not_spawn_third_goal_when_paused_exists(
    temp_db, monkeypatch,
):
    """Code-review IMPORTANT: a magic-link arrival used to silently
    create a brand-new active goal whenever no active goal existed —
    including the case where the learner already had a paused goal.
    Lock in: if ANY non-archived goal exists, the magic link does
    NOT seed a new one (the frontend should prompt the user to pick
    or restart instead)."""
    from ongiini import config
    from ongiini.learning import tokens
    monkeypatch.setattr(config.settings, "learn_token_secret", "test-secret")

    client = _client()
    learner_id = store.create_anonymous_learner()
    # Learner has one paused goal (no active).
    g_paused = store.create_new_goal(
        learner_id, title="Old plan", activate=False,
    )
    assert store.list_goals(learner_id)[0]["status"] == "paused"

    tok = tokens.sign(learner_id=learner_id, goal_text="Interview prep")
    r = client.post("/v1/learn/sessions", json={"token": tok})
    assert r.status_code == 200, r.text

    # Goal count must still be 1 — the paused one, not auto-promoted,
    # not joined by a brand-new active one.
    goals = store.list_goals(learner_id, include_archived=True)
    assert len(goals) == 1
    assert goals[0]["goal_id"] == g_paused["goal_id"]
    assert goals[0]["status"] == "paused"


def test_sessions_resume_includes_thread_after_intake(temp_db):
    """After intake + a /turn that produced messages, /sessions must
    rehydrate the chat thread + active goal — that's how the frontend
    paints the page on a cold visit instead of starting from blank."""
    from ongiini.learning import messages as msg_mod
    client = _client()
    s = client.post("/v1/learn/sessions", json={}).json()
    learner_id = s["learner_id"]
    _finish_intake(client, learner_id)

    # Plant a goal + a couple of messages directly so we don't need a
    # live model call here.
    goal = store.get_or_create_active_goal(learner_id)
    msg_mod.append(
        learner_id=learner_id, goal_id=goal["goal_id"],
        kind=db.MSG_COACH_TEXT, payload={"text": "Welcome back."},
    )
    store.save_curriculum_outline(goal["goal_id"], {"summary": "x", "modules": []})

    r = client.post("/v1/learn/sessions", json={"learner_id": learner_id})
    body = r.json()
    assert body["intake_complete"] is True
    assert body["active_goal"]["goal_id"] == goal["goal_id"]
    assert body["active_goal"]["has_outline"] is True
    assert len(body["thread"]) == 1
    assert body["thread"][0]["kind"] == db.MSG_COACH_TEXT
    assert body["curriculum_outline"] == {"summary": "x", "modules": []}


# ============================================================
# /intake
# ============================================================

def test_intake_clarify_surfaces_natural_language_followup(temp_db):
    """The fix to Sebastian's "this is dumb" complaint: when the user
    types "#46" or "I dont know anything", the LLM intermediary writes
    a natural-voice clarify question. The frontend surfaces it as a
    coach bubble. NO mechanical "age must be a positive integer"
    string ever reaches the user."""
    fm = FakeModel(responses=[
        json.dumps({"clarify": "No worries — could you give me your age as a number?"}),
    ])
    # Bypass _INTAKE_PARSER_RESPONSES prefix — we want this exact response.
    fm.responses = list(fm.responses)
    app = FastAPI()
    router = build_router(model=fm, skill_renderer=_stub_skill_renderer)
    app.include_router(router, prefix="/v1/learn")
    client = TestClient(app)

    learner_id = store.create_anonymous_learner()
    r = client.post(
        "/v1/learn/intake",
        json={"learner_id": learner_id, "field": "age", "value": "#46"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["intake_complete"] is False
    assert body["next_intake_field"] == "age"
    assert body["clarify_text"]
    assert "age" in body["clarify_text"].lower()
    # Nothing got persisted — defence in depth.
    profile = store.get_profile(learner_id)
    assert (profile or {}).get("age") is None


def test_intake_defence_in_depth_when_llm_value_fails_validator(temp_db):
    """If the LLM extracts a value the shape validator rejects (e.g.
    hallucinated age 999), surface a polite clarify text rather than
    the mechanical reason. The validator's reason is still attached for
    server-side debugging."""
    fm = FakeModel(responses=[json.dumps({"value": 999})])
    app = FastAPI()
    router = build_router(model=fm, skill_renderer=_stub_skill_renderer)
    app.include_router(router, prefix="/v1/learn")
    client = TestClient(app)

    learner_id = store.create_anonymous_learner()
    r = client.post(
        "/v1/learn/intake",
        json={"learner_id": learner_id, "field": "age", "value": "I am 999"},
    )
    body = r.json()
    # User sees the natural clarify, not the raw validator string.
    assert body["clarify_text"]
    assert "must be between" not in body["clarify_text"]
    # Server-side reason still recorded for debugging.
    assert body["validation_error"]
    assert "between" in body["validation_error"]
    # Nothing persisted.
    assert (store.get_profile(learner_id) or {}).get("age") is None


def test_intake_completion_flips_flag(temp_db):
    client = _client()
    s = client.post("/v1/learn/sessions", json={}).json()
    _finish_intake(client, s["learner_id"])
    r = client.post("/v1/learn/sessions", json={"learner_id": s["learner_id"]})
    assert r.json()["intake_complete"] is True


def test_intake_objective_prompt_invites_both_situation_and_topic(temp_db):
    """The intake prompt was rephrased to invite both situational
    framing ('job interview') AND specific-topic framing ('past
    tenses', 'modal verbs'). Lock in the new phrasing so a future
    edit that drops the topic-focus invitation regresses loudly."""
    client = _client()
    s = client.post("/v1/learn/sessions", json={
        "target_language": "german",
    }).json()
    # Walk to the objective step.
    for field_name, raw in [
        ("name", "Sebastian"),
        ("age", "35"),
        ("current_level", "beginner"),
    ]:
        r = client.post(
            "/v1/learn/intake",
            json={"learner_id": s["learner_id"], "field": field_name,
                  "value": raw, "target_language": "german"},
        )
        assert r.status_code == 200
        body = r.json()
    # Last hop's response carries the objective prompt for the
    # frontend to render. Confirm the new wording invites both
    # framings — situational AND topic.
    prompt = body["next_intake_prompt"]
    assert "situation" in prompt.lower()
    assert "topic" in prompt.lower()
    # The previous "what do you actually want to be able to do" wording
    # is gone — would over-bias toward situational framing.
    assert "actually want to be able to do" not in prompt.lower()


# ============================================================
# /turn — the chat-first entry point
# ============================================================

def test_turn_returns_409_before_intake(temp_db):
    client = _client()
    learner_id = store.create_anonymous_learner()
    r = client.post(
        "/v1/learn/turn",
        json={"learner_id": learner_id, "text": "hi"},
    )
    assert r.status_code == 409


def test_turn_response_includes_goals_list_after_auto_create(temp_db):
    """When the FIRST /turn after intake auto-creates a goal, the
    response must include the refreshed goals[] so the drawer shows
    the new curriculum. Previously /turn returned only `goal`
    (singular) and the drawer stayed at 'No curriculums yet.'"""
    fm = FakeModel(responses=[_OUTLINE, _CRITIC_READY, _EXERCISE])
    client = _client(fm)
    s = client.post("/v1/learn/sessions", json={}).json()
    _finish_intake(client, s["learner_id"])

    r = client.post(
        "/v1/learn/turn",
        json={"learner_id": s["learner_id"], "text": None},
    )
    body = r.json()
    # The auto-created goal is in the list, with title seeded from
    # profile.objective so the drawer doesn't render it as
    # "Untitled curriculum".
    assert "goals" in body
    assert len(body["goals"]) == 1
    assert body["goals"][0]["goal_id"] == body["goal_id"]
    assert body["goals"][0]["title"] == "job interview at SPAR"
    assert body["goals"][0]["status"] == "active"


def test_goals_activate_includes_module_progress_for_target_goal(temp_db):
    """Switching curriculums used to leave the slim progress bar stuck
    on the previous goal's module because /goals/activate didn't carry
    module_progress + active_module_id. Lock in that the new response
    fields are populated AND scoped to the activated goal."""
    client = _client()
    s = client.post("/v1/learn/sessions", json={}).json()
    _finish_intake(client, s["learner_id"])

    # Goal A with an outline that has its own modules.
    goal_a = store.get_or_create_active_goal(s["learner_id"])
    store.save_curriculum_outline(goal_a["goal_id"], {
        "summary": "A",
        "modules": [
            {"id": "a-1", "title": "A first", "status": "in_progress",
             "estimated_cards": 8},
        ],
    })

    # Goal B becomes active (demoting A to paused).
    goal_b = store.create_new_goal(s["learner_id"], title="B")
    store.save_curriculum_outline(goal_b["goal_id"], {
        "summary": "B",
        "modules": [
            {"id": "b-1", "title": "B first", "status": "in_progress",
             "estimated_cards": 4},
        ],
    })

    r = client.post(
        "/v1/learn/goals/activate",
        json={"learner_id": s["learner_id"], "goal_id": goal_a["goal_id"]},
    )
    body = r.json()
    # The response carries A's module_progress + active_module_id,
    # NOT B's — that was the bug.
    assert body["active_module_id"] == "a-1"
    assert len(body["module_progress"]) == 1
    assert body["module_progress"][0]["module_id"] == "a-1"
    assert body["module_progress"][0]["estimated_cards"] == 8


def test_turn_no_text_no_goal_id_designs_outline_and_emits_lesson(temp_db):
    """Fresh learner → outline designed → selector picks first lesson
    topic → frontend gets MSG_LESSON. (Used to be MSG_EXERCISE when
    the LLM picked card_type; now teach-first is the rule.)"""
    fm = FakeModel(responses=[_OUTLINE, _CRITIC_READY, _LESSON])
    client = _client(fm)
    s = client.post("/v1/learn/sessions", json={}).json()
    _finish_intake(client, s["learner_id"])

    r = client.post(
        "/v1/learn/turn",
        json={"learner_id": s["learner_id"], "text": None},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["messages"]) == 1
    assert body["messages"][0]["kind"] == db.MSG_LESSON
    assert body["curriculum_outline"]["summary"]
    # The goal info has the outline flag set after the outline was
    # written — the frontend's "Plan ready" badge depends on this.
    assert body["goal"]["has_outline"] is True


def test_turn_with_explicit_goal_id_uses_that_goal(temp_db):
    """The chat-first frontend may have a curriculum switcher — passing
    goal_id lets it scope a turn to a non-active goal (e.g. previewing
    a paused curriculum). Lock in that explicit goal_id is honored."""
    fm = FakeModel(responses=[_OUTLINE, _CRITIC_READY, _EXERCISE])
    client = _client(fm)
    s = client.post("/v1/learn/sessions", json={}).json()
    _finish_intake(client, s["learner_id"])

    g_paused = store.create_new_goal(
        s["learner_id"], title="Paused one", activate=False,
    )
    r = client.post(
        "/v1/learn/turn",
        json={
            "learner_id": s["learner_id"],
            "goal_id": g_paused["goal_id"],
            "text": None,
        },
    )
    assert r.status_code == 200
    assert r.json()["goal_id"] == g_paused["goal_id"]


def test_turn_with_unknown_goal_id_returns_404(temp_db):
    client = _client()
    s = client.post("/v1/learn/sessions", json={}).json()
    _finish_intake(client, s["learner_id"])
    r = client.post(
        "/v1/learn/turn",
        json={
            "learner_id": s["learner_id"],
            "goal_id": "00000000-0000-0000-0000-000000000000",
            "text": None,
        },
    )
    assert r.status_code == 404


def test_turn_with_archived_goal_returns_409(temp_db):
    client = _client()
    s = client.post("/v1/learn/sessions", json={}).json()
    _finish_intake(client, s["learner_id"])
    goal = store.get_or_create_active_goal(s["learner_id"])
    store.archive_goal(s["learner_id"], goal["goal_id"])
    r = client.post(
        "/v1/learn/turn",
        json={
            "learner_id": s["learner_id"],
            "goal_id": goal["goal_id"],
            "text": None,
        },
    )
    assert r.status_code == 409


# ============================================================
# /goals — list / new / activate / restart / archive
# ============================================================

def test_goals_list_includes_only_non_archived_by_default(temp_db):
    client = _client()
    s = client.post("/v1/learn/sessions", json={}).json()
    _finish_intake(client, s["learner_id"])
    g1 = store.get_or_create_active_goal(s["learner_id"])
    g2 = store.create_new_goal(s["learner_id"], title="other")
    store.archive_goal(s["learner_id"], g1["goal_id"])

    r = client.post("/v1/learn/goals", json={"learner_id": s["learner_id"]})
    body = r.json()
    assert len(body["goals"]) == 1
    assert body["goals"][0]["goal_id"] == g2["goal_id"]
    assert body["active_goal_id"] == g2["goal_id"]

    r2 = client.post(
        "/v1/learn/goals",
        json={"learner_id": s["learner_id"], "include_archived": True},
    )
    assert len(r2.json()["goals"]) == 2


def test_goals_new_accepts_language_pair_and_level(temp_db):
    """The frontend's new-curriculum modal sends source + target +
    level. Lock in that the API persists all three and surfaces them
    in the response GoalInfo."""
    client = _client()
    s = client.post("/v1/learn/sessions", json={}).json()
    _finish_intake(client, s["learner_id"])
    r = client.post(
        "/v1/learn/goals/new",
        json={
            "learner_id": s["learner_id"],
            "title": "Berlin trip",
            "language": "german",
            "source_language": "english",
            "current_level": "elementary",
        },
    )
    assert r.status_code == 200, r.text
    goal = r.json()["goal"]
    assert goal["language"] == "german"
    assert goal["source_language"] == "english"
    assert goal["current_level"] == "elementary"


def test_goals_new_rejects_same_source_and_target(temp_db):
    """Validation runs at the store boundary and surfaces as 422 —
    the frontend never lets the user pick source == target, but a
    direct API caller could try."""
    client = _client()
    s = client.post("/v1/learn/sessions", json={}).json()
    _finish_intake(client, s["learner_id"])
    r = client.post(
        "/v1/learn/goals/new",
        json={
            "learner_id": s["learner_id"],
            "title": "x",
            "language": "english",
            "source_language": "english",
        },
    )
    assert r.status_code == 422


def test_turn_renders_skill_for_goal_language_pair(temp_db):
    """The /turn handler must invoke the skill_renderer with the
    goal's (source, target) so the LLM sees the right per-pair
    prompt. The stub renderer encodes the pair in the returned text
    — we can spy on what the coach saw via the FakeModel's captured
    requests."""
    fm = FakeModel(responses=[_OUTLINE, _CRITIC_READY, _EXERCISE])
    client = _client(fm)
    s = client.post("/v1/learn/sessions", json={}).json()
    _finish_intake(client, s["learner_id"])

    # Create a German-from-Afrikaans goal explicitly.
    r = client.post(
        "/v1/learn/goals/new",
        json={
            "learner_id": s["learner_id"],
            "title": "Berlin trip",
            "language": "german",
            "source_language": "afrikaans",
        },
    )
    goal_id = r.json()["goal"]["goal_id"]

    # First /turn for that goal — the prompt should carry the stub
    # marker for the pair we set.
    client.post(
        "/v1/learn/turn",
        json={"learner_id": s["learner_id"], "goal_id": goal_id, "text": None},
    )
    # Find the curriculum-design call in fm.requests (system prompt
    # contains the marker).
    sys_prompts = [r.messages[0]["content"] for r in fm.requests]
    assert any("source=afrikaans target=german" in sp for sp in sys_prompts)


def test_goals_new_creates_and_activates_demoting_previous(temp_db):
    client = _client()
    s = client.post("/v1/learn/sessions", json={}).json()
    _finish_intake(client, s["learner_id"])
    g1 = store.get_or_create_active_goal(s["learner_id"])
    r = client.post(
        "/v1/learn/goals/new",
        json={
            "learner_id": s["learner_id"],
            "title": "Interview prep",
            "context": "hospitality interview",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["goal"]["status"] == "active"
    # The list response now reflects the swap.
    statuses = {g["goal_id"]: g["status"] for g in body["goals"]}
    assert statuses[g1["goal_id"]] == "paused"
    assert statuses[body["goal"]["goal_id"]] == "active"


def test_goals_new_without_activate_stays_paused(temp_db):
    client = _client()
    s = client.post("/v1/learn/sessions", json={}).json()
    _finish_intake(client, s["learner_id"])
    store.get_or_create_active_goal(s["learner_id"])
    r = client.post(
        "/v1/learn/goals/new",
        json={
            "learner_id": s["learner_id"], "title": "later",
            "activate": False,
        },
    )
    assert r.json()["goal"]["status"] == "paused"


def test_goals_activate_swaps_active(temp_db):
    client = _client()
    s = client.post("/v1/learn/sessions", json={}).json()
    _finish_intake(client, s["learner_id"])
    g1 = store.get_or_create_active_goal(s["learner_id"])
    g2 = store.create_new_goal(s["learner_id"], title="other")    # active now

    r = client.post(
        "/v1/learn/goals/activate",
        json={"learner_id": s["learner_id"], "goal_id": g1["goal_id"]},
    )
    body = r.json()
    assert body["goal"]["goal_id"] == g1["goal_id"]
    assert body["goal"]["status"] == "active"
    statuses = {g["goal_id"]: g["status"] for g in body["goals"]}
    assert statuses[g2["goal_id"]] == "paused"


def test_goals_activate_unknown_returns_404(temp_db):
    client = _client()
    s = client.post("/v1/learn/sessions", json={}).json()
    r = client.post(
        "/v1/learn/goals/activate",
        json={
            "learner_id": s["learner_id"],
            "goal_id": "00000000-0000-0000-0000-000000000000",
        },
    )
    assert r.status_code == 404


def test_goals_restart_wipes_content_keeps_outline(temp_db):
    from ongiini.learning import messages as msg_mod
    client = _client()
    s = client.post("/v1/learn/sessions", json={}).json()
    _finish_intake(client, s["learner_id"])
    goal = store.get_or_create_active_goal(s["learner_id"])
    store.save_curriculum_outline(goal["goal_id"], {"summary": "x", "modules": []})
    card_id = store.save_card(goal["goal_id"], db.CARD_VOCAB, "x?")
    msg_mod.append(
        learner_id=s["learner_id"], goal_id=goal["goal_id"],
        kind=db.MSG_COACH_TEXT, payload={"text": "hi"},
    )

    r = client.post(
        "/v1/learn/goals/restart",
        json={"learner_id": s["learner_id"], "goal_id": goal["goal_id"]},
    )
    body = r.json()
    assert body["cards_deleted"] == 1
    assert body["messages_deleted"] == 1
    # Outline persists; the learner gets the same plan back fresh.
    assert store.get_curriculum_outline(goal["goal_id"]) == {
        "summary": "x", "modules": [],
    }


def test_goals_archive_removes_from_default_list(temp_db):
    client = _client()
    s = client.post("/v1/learn/sessions", json={}).json()
    _finish_intake(client, s["learner_id"])
    goal = store.get_or_create_active_goal(s["learner_id"])

    r = client.post(
        "/v1/learn/goals/archive",
        json={"learner_id": s["learner_id"], "goal_id": goal["goal_id"]},
    )
    assert r.status_code == 200
    assert r.json()["goals"] == []
    assert r.json()["active_goal_id"] is None


# ============================================================
# /clear
# ============================================================

def test_clear_deletes_learner_and_cascades(temp_db):
    client = _client()
    s = client.post("/v1/learn/sessions", json={}).json()
    learner_id = s["learner_id"]
    _finish_intake(client, learner_id)
    store.get_or_create_active_goal(learner_id)

    r = client.post("/v1/learn/clear", json={"learner_id": learner_id})
    assert r.json()["ok"] is True
    assert r.json()["rows_deleted"] == 1
    # Learner row + goals row are gone (cascade on delete).
    assert store.get_learner(learner_id) is None
    assert store.list_goals(learner_id, include_archived=True) == []


# ============================================================
# Feature flag gating
# ============================================================

def test_endpoints_503_when_learn_disabled(temp_db, monkeypatch):
    """The kill-switch must catch every endpoint EXCEPT /clear (data
    deletion is allowed regardless of flags)."""
    from ongiini import config
    monkeypatch.setattr(config.settings, "learn_enabled", False)
    client = _client()

    assert client.post("/v1/learn/sessions", json={}).status_code == 503
    assert client.post(
        "/v1/learn/intake",
        json={"learner_id": "x", "field": "name", "value": "y"},
    ).status_code == 503
    assert client.post(
        "/v1/learn/turn", json={"learner_id": "x"},
    ).status_code == 503
    assert client.post(
        "/v1/learn/goals", json={"learner_id": "x"},
    ).status_code == 503
    # /clear still works — locked in.
    assert client.post(
        "/v1/learn/clear", json={"learner_id": "x"},
    ).status_code == 200
