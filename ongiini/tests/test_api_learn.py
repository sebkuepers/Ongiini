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
    "card_type": "vocab",
    "prompt_text": "How do you say 'thank you'?",
    "reference_answer": "dankie",
})


def _client(fake_model: FakeModel | None = None) -> TestClient:
    """Build a TestClient against a fresh FastAPI app with the learn
    router mounted at /v1/learn — same path layout production uses."""
    app = FastAPI()
    model = fake_model or FakeModel()
    router = build_router(model=model, skill_content="SKILL")
    app.include_router(router, prefix="/v1/learn")
    return TestClient(app)


def _finish_intake(client: TestClient, learner_id: str) -> None:
    """Convenience — drive the four intake POSTs so /turn isn't gated."""
    for field_name, value in [
        ("name", "Sebastian"),
        ("age", 35),
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

def test_intake_validation_error_does_not_persist(temp_db):
    client = _client()
    learner_id = store.create_anonymous_learner()
    r = client.post(
        "/v1/learn/intake",
        json={"learner_id": learner_id, "field": "age", "value": "not a number"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["validation_error"]
    assert body["intake_complete"] is False
    # Profile should not have been touched.
    profile = store.get_profile(learner_id)
    assert (profile or {}).get("age") is None


def test_intake_completion_flips_flag(temp_db):
    client = _client()
    s = client.post("/v1/learn/sessions", json={}).json()
    _finish_intake(client, s["learner_id"])
    r = client.post("/v1/learn/sessions", json={"learner_id": s["learner_id"]})
    assert r.json()["intake_complete"] is True


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


def test_turn_no_text_no_goal_id_designs_outline_and_emits_exercise(temp_db):
    fm = FakeModel(responses=[_OUTLINE, _EXERCISE])
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
    assert body["messages"][0]["kind"] == db.MSG_EXERCISE
    assert body["curriculum_outline"]["summary"]
    # The goal info has the outline flag set after the outline was
    # written — the frontend's "Plan ready" badge depends on this.
    assert body["goal"]["has_outline"] is True


def test_turn_with_explicit_goal_id_uses_that_goal(temp_db):
    """The chat-first frontend may have a curriculum switcher — passing
    goal_id lets it scope a turn to a non-active goal (e.g. previewing
    a paused curriculum). Lock in that explicit goal_id is honored."""
    fm = FakeModel(responses=[_OUTLINE, _EXERCISE])
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
