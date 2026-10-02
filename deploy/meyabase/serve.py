"""Meyabase translation service, hosted on the DGX Spark for Meyabase Platforms.

A thin, defensive HTTP layer around Meyabase's own inference code
(github.com/axelmukwena/meyabase-translate-inference, ``translation.core``),
which the Dockerfile copies in at a pinned commit. Two request shapes:

* ``POST /`` — the HuggingFace Inference Endpoint format their website
  already sends: ``{"inputs": "...", "model_id": "meyabase/en-ng-translation"}``
  (or ``"direction": "en-ng"``) → ``[{"translation_text": "..."}]``. Their
  site only has to swap the endpoint URL.
* ``POST /translate`` — their FastAPI shape ``{"text", "direction"}`` →
  ``{"translation": "..."}``.

Guards, because this runs on our production box: one request translates at
a time, a bounded wait queue, a per-client rate limit, a length cap, no
caller-supplied generation parameters, CORS limited to their site. Request
text is never logged — only direction, length and duration.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from translation.config import DIRECTIONS, resolve_model_id
from translation.core import Translator, extract_translations

MAX_CHARS = int(os.environ.get("MAX_CHARS", "1000"))
RATE_PER_MIN = int(os.environ.get("RATE_PER_MIN", "30"))
MAX_WAITING = int(os.environ.get("MAX_WAITING", "16"))
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS", "https://translate.meyabase.com").split(",") if o.strip()]

log = logging.getLogger("meyabase")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Meyabase Translation (hosted by Ongiini AI)", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["content-type", "authorization"], max_age=3600)

translator = Translator()
_gpu = threading.Lock()                     # one translation at a time
_waiting = 0
_waiting_lock = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _client(request: Request) -> str:
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")


def _rate_limit(client: str) -> None:
    now = time.monotonic()
    with _hits_lock:
        q = _hits[client]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_PER_MIN:
            raise HTTPException(status_code=429, detail="Too many requests, please wait a minute.")
        q.append(now)
        if len(_hits) > 10_000:             # bound memory: drop idle clients
            for k in [k for k, v in _hits.items() if not v]:
                del _hits[k]


def _run(text: str, model_id: str) -> list[dict]:
    global _waiting
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=422, detail="Provide non-empty text.")
    if len(text) > MAX_CHARS:
        raise HTTPException(status_code=413, detail=f"Text is longer than {MAX_CHARS} characters.")
    with _waiting_lock:
        if _waiting >= MAX_WAITING:
            raise HTTPException(status_code=503, detail="Busy, please try again shortly.")
        _waiting += 1
    t0 = time.monotonic()
    try:
        with _gpu:
            prediction = translator.translate(text, model_id=model_id)
    finally:
        with _waiting_lock:
            _waiting -= 1
    log.info("translate model=%s chars=%d ms=%d", model_id.split("/")[-1], len(text),
             int(1000 * (time.monotonic() - t0)))
    return prediction


def _resolve(direction: str | None, model_id: str | None) -> str:
    try:
        return resolve_model_id(direction=direction or None, model_id=model_id or None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


class EndpointIn(BaseModel):
    inputs: str
    model_id: str | None = None
    direction: str | None = None
    parameters: dict | None = None          # accepted for compatibility, ignored


class TranslateIn(BaseModel):
    text: str
    direction: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "directions": sorted(DIRECTIONS)}


@app.post("/")
def endpoint(body: EndpointIn, request: Request) -> list[dict]:
    model_id = _resolve(body.direction, body.model_id)
    _rate_limit(_client(request))
    return _run(body.inputs, model_id)


@app.post("/translate")
def translate(body: TranslateIn, request: Request) -> dict:
    model_id = _resolve(body.direction, None)
    _rate_limit(_client(request))
    return {"translation": extract_translations(_run(body.text, model_id))}
