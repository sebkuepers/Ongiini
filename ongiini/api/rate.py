"""HTTP endpoints for the ongiini.ai/rate/ translation-rating surface.

Endpoints (under ``/v1/rate/``), both POST with the invite token in the
body (never in a URL the server logs):
  * ``/session`` — validate the token; return label, dialects, progress,
    the onboarding practice examples (with their expected answer and
    feedback) and the first item — or ``needs_profile`` on a first visit
  * ``/profile`` — store dialect(s) + first language, then as /session
  * ``/answer``  — store one verdict (+ optional issues, better
    translation, or a can't-judge reason) and return the next item

Real items never reveal whether a translation is the human reference, a
model or a control. See ongiini/ratings.py for assignment and storage.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import ratings

log = logging.getLogger("ongiini.api.rate")


class SessionIn(BaseModel):
    token: str = Field(min_length=8, max_length=128)


class ProfileIn(SessionIn):
    dialects: list[str] = Field(min_length=1, max_length=2)
    first_language: str


class AnswerIn(SessionIn):
    item_key: str = Field(min_length=1, max_length=64)
    verdict: str
    cant_reason: str | None = None
    issues: list[str] | None = Field(default=None, max_length=6)
    suggestion: str | None = Field(default=None, max_length=1000)
    duration_ms: int | None = Field(default=None, ge=0, le=3_600_000)


def build_router() -> APIRouter:
    router = APIRouter()

    def rater_or_401(con, token: str):
        rater = ratings.rater_for(con, token)
        if rater is None:
            raise HTTPException(status_code=401, detail="invalid or expired invite link")
        return rater

    def state(con, rater) -> dict:
        out = {"label": rater["label"], "dialects": [d for d in rater["dialects"].split(",") if d],
               "progress": ratings.progress(con, rater["rater_id"]),
               "needs_profile": ratings.needs_profile(rater)}
        if not out["needs_profile"]:
            out |= {"practice": ratings.practice_items(con, rater), "item": ratings.next_item(con, rater)}
        return out

    @router.post("/session")
    def session(body: SessionIn) -> dict:
        con = ratings.connect()
        try:
            return state(con, rater_or_401(con, body.token))
        finally:
            con.close()

    @router.post("/profile")
    def profile(body: ProfileIn) -> dict:
        con = ratings.connect()
        try:
            rater = rater_or_401(con, body.token)
            try:
                rater = ratings.set_profile(con, rater, body.dialects, body.first_language)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
            return state(con, rater)
        finally:
            con.close()

    @router.post("/answer")
    def answer(body: AnswerIn) -> dict:
        con = ratings.connect()
        try:
            rater = rater_or_401(con, body.token)
            try:
                ratings.record(con, rater, body.item_key, body.verdict,
                               cant_reason=body.cant_reason, issues=body.issues,
                               suggestion=body.suggestion, duration_ms=body.duration_ms)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
            return {"progress": ratings.progress(con, rater["rater_id"]),
                    "item": ratings.next_item(con, rater)}
        finally:
            con.close()

    return router
