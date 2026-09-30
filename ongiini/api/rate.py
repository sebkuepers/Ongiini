"""HTTP endpoints for the ongiini.ai/rate/ translation-rating surface.

Endpoints (under ``/v1/rate/``), both POST with the invite token in the
body (never in a URL the server logs):
  * ``/session`` — validate the token, return the rater's label, dialects,
    progress and the first task
  * ``/answer``  — store one verdict (+ optional better translation) and
    return the next task

Tasks never reveal whether a candidate is the human reference, a model
or a control. See ongiini/ratings.py for assignment and storage.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import ratings

log = logging.getLogger("ongiini.api.rate")


class SessionIn(BaseModel):
    token: str = Field(min_length=8, max_length=128)


class AnswerIn(SessionIn):
    task_id: str = Field(min_length=1, max_length=64)
    verdict: str
    suggestion: str | None = Field(default=None, max_length=1000)
    duration_ms: int | None = Field(default=None, ge=0, le=3_600_000)


def build_router() -> APIRouter:
    router = APIRouter()

    def rater_or_401(con, token: str):
        rater = ratings.rater_for(con, token)
        if rater is None:
            raise HTTPException(status_code=401, detail="invalid or expired invite link")
        return rater

    @router.post("/session")
    def session(body: SessionIn) -> dict:
        con = ratings.connect()
        try:
            rater = rater_or_401(con, body.token)
            return {"label": rater["label"], "dialects": rater["dialects"].split(","),
                    "progress": ratings.progress(con, rater["rater_id"]),
                    "task": ratings.next_task(con, rater)}
        finally:
            con.close()

    @router.post("/answer")
    def answer(body: AnswerIn) -> dict:
        con = ratings.connect()
        try:
            rater = rater_or_401(con, body.token)
            try:
                ratings.record(con, rater, body.task_id, body.verdict, body.suggestion,
                               body.duration_ms)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
            return {"progress": ratings.progress(con, rater["rater_id"]),
                    "task": ratings.next_task(con, rater)}
        finally:
            con.close()

    return router
