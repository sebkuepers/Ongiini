"""Tests for the hosted Meyabase service (serve.py), with the model mocked.

Needs Meyabase's ``translation`` package on the path, as in the image:

    MEYABASE_SRC=/path/to/meyabase-translate-inference pytest deploy/meyabase/test_serve.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SRC = os.environ.get("MEYABASE_SRC")
if not SRC:
    pytest.skip("MEYABASE_SRC not set", allow_module_level=True)
sys.path.insert(0, SRC)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402

import serve  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    calls = []

    def fake_translate(text, *, model_id=None, direction=None, **params):
        calls.append((text, model_id, params))
        return [{"translation_text": f"<{model_id.split('/')[-1]}>{text[::-1]}"}]

    monkeypatch.setattr(serve.translator, "translate", fake_translate)
    serve._hits.clear()
    c = TestClient(serve.app)
    c.calls = calls
    return c


def test_hf_endpoint_shape(client):
    r = client.post("/", json={"inputs": "abc", "model_id": "meyabase/en-ng-translation",
                               "parameters": {"num_beams": 99}})
    assert r.status_code == 200 and r.json() == [{"translation_text": "<en-ng-translation>cba"}]
    assert client.calls[-1][2] == {}                         # caller parameters are ignored


def test_direction_and_translate_route(client):
    assert client.post("/", json={"inputs": "x", "direction": "ng-en"}).status_code == 200
    r = client.post("/translate", json={"text": "ab", "direction": "en-ng"})
    assert r.json() == {"translation": "<en-ng-translation>ba"}


def test_rejections(client):
    assert client.post("/", json={"inputs": "x", "model_id": "evil/model"}).status_code == 400
    assert client.post("/", json={"inputs": "x", "model_id": ""}).status_code == 400   # site warm-up call
    assert client.post("/", json={"inputs": "x" * (serve.MAX_CHARS + 1),
                                  "direction": "en-ng"}).status_code == 413
    assert client.post("/", json={"inputs": "  ", "direction": "en-ng"}).status_code == 422


def test_rate_limit(client):
    for _ in range(serve.RATE_PER_MIN):
        assert client.post("/", json={"inputs": "a", "direction": "en-ng"},
                           headers={"cf-connecting-ip": "1.2.3.4"}).status_code == 200
    assert client.post("/", json={"inputs": "a", "direction": "en-ng"},
                       headers={"cf-connecting-ip": "1.2.3.4"}).status_code == 429
    assert client.post("/", json={"inputs": "a", "direction": "en-ng"},
                       headers={"cf-connecting-ip": "5.6.7.8"}).status_code == 200


def test_cors_only_for_meyabase(client):
    ok = client.options("/", headers={"Origin": "https://translate.meyabase.com",
                                      "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "https://translate.meyabase.com"
    bad = client.options("/", headers={"Origin": "https://evil.example",
                                       "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


def test_health(client):
    assert client.get("/health").json()["directions"] == ["en-ng", "ng-en"]


def test_text_never_logged(client, caplog):
    caplog.set_level("INFO", logger="meyabase")
    client.post("/", json={"inputs": "secret sentence", "direction": "en-ng"})
    assert "secret" not in caplog.text and "chars=15" in caplog.text
