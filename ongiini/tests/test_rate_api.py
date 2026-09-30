"""HTTP-level tests for /v1/rate."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ongiini import ratings
from ongiini.api.rate import build_router
from ongiini.config import settings


def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    app = FastAPI()
    app.include_router(build_router(), prefix="/v1/rate")
    return TestClient(app)


def test_session_and_answer_flow(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    con = ratings.connect()
    ratings.load_tasks(con, [{"task_id": "a", "round": "r1", "item_id": 1, "dialect": "oshindonga",
                              "english": "Hello", "text": "Wa lalapo", "kind": "ref",
                              "source": "kaarina", "priority": 1, "target": 1}])
    token = ratings.add_rater(con, "Tester", ["oshindonga"])
    con.close()

    assert c.post("/v1/rate/session", json={"token": "x" * 20}).status_code == 401
    s = c.post("/v1/rate/session", json={"token": token}).json()
    assert s["label"] == "Tester" and s["task"]["task_id"] == "a"
    assert "kaarina" not in str(s) and "ref" not in s["task"]
    r = c.post("/v1/rate/answer", json={"token": token, "task_id": "a", "verdict": "correct"}).json()
    assert r["progress"]["done"] == 1 and r["task"] is None
    bad = c.post("/v1/rate/answer", json={"token": token, "task_id": "a", "verdict": "great"})
    assert bad.status_code == 400
