"""HTTP-level tests for /v1/rate (v3)."""
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
    base = {"round": "r1", "dialect": "oshindonga", "english": "Hello"}
    ratings.load_items(con, [
        {**base, "item_key": "a", "sentence_id": 1, "text": "Wa lalapo", "grp": "random_ref",
         "kind": "ref", "source": "kaarina"},
        {**base, "item_key": "p", "sentence_id": 9, "text": "x", "grp": "practice", "kind": "practice",
         "source": "kaarina+err:number", "expected": "wrong", "explanation": "Number differs."}])
    token = ratings.add_rater(con, "Tester", ["oshindonga"])
    con.close()

    assert c.post("/v1/rate/session", json={"token": "x" * 20}).status_code == 401
    s = c.post("/v1/rate/session", json={"token": token}).json()
    assert s["label"] == "Tester" and s["item"]["item_key"] == "a"
    assert s["practice"][0]["expected"] == "wrong"
    assert "kaarina" not in str(s["item"])
    bad = c.post("/v1/rate/answer", json={"token": token, "item_key": "a", "verdict": "great"})
    assert bad.status_code == 400
    ok = c.post("/v1/rate/answer", json={"token": token, "item_key": "a", "verdict": "almost",
                                         "issues": ["word_choice"], "suggestion": "Wa lala po"}).json()
    assert ok["progress"] == {"done": 1, "everyone": 1} and ok["item"] is None
