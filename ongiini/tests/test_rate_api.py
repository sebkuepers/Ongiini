"""HTTP-level tests for /v1/rate."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ongiini import ratings
from ongiini.api.rate import build_router
from ongiini.config import settings


def test_session_and_answer_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    app = FastAPI()
    app.include_router(build_router(), prefix="/v1/rate")
    c = TestClient(app)
    con = ratings.connect()
    ratings.load_screens(con, [{"screen_id": "a", "round": "r1", "item_id": 1, "dialect": "oshindonga",
        "english": "Hello", "priority": 1, "target": 1, "candidates": [
            {"cand_id": "a1", "text": "Wa lalapo", "kind": "ref", "source": "kaarina"},
            {"cand_id": "a2", "text": "Halo", "kind": "model", "source": "claude-opus-5"}]}])
    token = ratings.add_rater(con, "Tester", ["oshindonga"])
    con.close()

    assert c.post("/v1/rate/session", json={"token": "x" * 20}).status_code == 401
    s = c.post("/v1/rate/session", json={"token": token}).json()
    assert s["label"] == "Tester" and s["screen"]["screen_id"] == "a"
    assert "kaarina" not in str(s) and "ref" not in str(s["screen"]["candidates"])
    bad = c.post("/v1/rate/answer", json={"token": token, "screen_id": "a",
                                          "verdicts": {"a1": "send"}, "choice": "a1"})
    assert bad.status_code == 400
    ok = c.post("/v1/rate/answer", json={"token": token, "screen_id": "a",
                                         "verdicts": {"a1": "send", "a2": "no"}, "choice": "a1"}).json()
    assert ok["progress"]["done"] == 1 and ok["screen"] is None
