"""/v1/chat/feedback: per-reply language ratings on experimental.ongiini.ai."""
from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ongiini import pii
from ongiini.api.chat import build_router
from ongiini.config import settings
from ongiini.memory.session_memory import SessionStore

SID = "0b6c2a54-7e1f-4c3a-9d2e-5f8a1b2c3d4e"


def _client(tmp_path, monkeypatch, enabled=True):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "chat_feedback_enabled", enabled)
    store = SessionStore()
    store.append_turn(SID, "Translate: good morning", "Wa lele po")
    store.append_turn(SID, "Mail me at jane@example.com", "Ondi ku shangela")
    app = FastAPI()
    app.include_router(build_router(store, shared=None, pii_sanitiser=pii.sanitize), prefix="/v1")  # as mounted in main.py
    return TestClient(app)


def test_rating_stores_scrubbed_pair_with_model(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    (tmp_path / "feedback").mkdir()
    (tmp_path / "feedback" / "model.txt").write_text("Ongiini Ombuto (T4b31)\n")
    assert c.post("/v1/chat/feedback", json={"session_id": SID, "turn": 1, "rating": "wrong"}).json() == {"ok": True}
    assert c.post("/v1/chat/feedback", json={"session_id": SID, "turn": 0, "rating": "correct"}).status_code == 200
    rows = [json.loads(line) for line in (tmp_path / "feedback" / "language_ratings.jsonl").read_text().splitlines()]
    assert [r["turn"] for r in rows] == [1, 0]
    assert rows[0]["rating"] == "wrong" and rows[0]["answer"] == "Ondi ku shangela"
    assert "jane@example.com" not in rows[0]["question"] and "REDACTED" in rows[0]["question"]
    assert rows[1]["question"] == "Translate: good morning" and rows[1]["model"] == "Ongiini Ombuto (T4b31)"
    assert SID not in json.dumps(rows) and len(rows[0]["session"]) == 16


def test_rejects_unknown_turn_bad_rating_and_bad_session(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.post("/v1/chat/feedback", json={"session_id": SID, "turn": 2, "rating": "correct"}).status_code == 404
    assert c.post("/v1/chat/feedback", json={"session_id": SID, "turn": 0, "rating": "great"}).status_code == 422
    bad = "0B6C2A54-7E1F-4C3A-9D2E-5F8A1B2C3D4E"
    assert c.post("/v1/chat/feedback", json={"session_id": bad, "turn": 0, "rating": "correct"}).status_code == 400
    assert not (tmp_path / "feedback" / "language_ratings.jsonl").exists()


def test_endpoint_absent_unless_enabled(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, enabled=False)
    assert c.post("/v1/chat/feedback", json={"session_id": SID, "turn": 0, "rating": "correct"}).status_code == 404
