"""Tests for StateGatedClassifier.

The gate is the structural defence against silent saves of false-positive
classifier output. These tests cover every gating decision plus the
pass-through paths."""
from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from owela import ClassifierResult, DEPTH_DEEP, DEPTH_SHALLOW, InboundMessage
from ongiini import contributions
from ongiini.routers.state_gated_classifier import StateGatedClassifier


MSISDN = "264811234567"


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "contributions.sqlite"
    monkeypatch.setattr(contributions, "_db_path", lambda: db)
    monkeypatch.setattr(contributions.settings, "contributions_hash_salt", "test-salt")
    contributions.warmup()
    yield


class _FakeInner:
    """Inner classifier returning a fixed result. Lets us probe gate
    behaviour without involving vLLM."""

    def __init__(self, result: ClassifierResult) -> None:
        self.result = result
        self.calls = 0

    async def classify(self, msg: InboundMessage) -> ClassifierResult:
        self.calls += 1
        return self.result


def _msg(text: str = "ondi ya nawa", msisdn: str = MSISDN) -> InboundMessage:
    return InboundMessage(
        user_id=msisdn,
        msg_id="t",
        text=text,
        content_parts=[{"type": "text", "text": text}],
    )


def _backdate_pending(h: str, minutes_ago: int) -> None:
    old = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
           ).isoformat(timespec="seconds")
    conn = sqlite3.connect(contributions._db_path())
    conn.execute(
        "UPDATE contributors SET pending_set_at = ? WHERE contributor_hash = ?",
        (old, h),
    )
    conn.commit()
    conn.close()


# ── CONTRIBUTE_SAVE gating ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_blocked_when_no_pending():
    inner = _FakeInner(ClassifierResult(verdict="CONTRIBUTE_SAVE", depth=DEPTH_SHALLOW))
    gated = StateGatedClassifier(inner)
    result = await gated.classify(_msg())
    assert result.verdict == "NONE"
    assert result.depth == DEPTH_SHALLOW
    assert result.attrs["redirected_from"] == "CONTRIBUTE_SAVE"
    assert result.attrs["redirect_reason"] == "no_active_contribute_flow"


@pytest.mark.asyncio
async def test_save_blocked_when_pending_is_stale():
    contributions.seed_tasks([{"source_en": "x", "category": "c", "seed_id": 1}])
    h = contributions.hash_msisdn(MSISDN)
    contributions.set_pending_save(h, 1, "Oshindonga")
    _backdate_pending(h, contributions.PENDING_SAVE_TTL_MIN + 1)
    inner = _FakeInner(ClassifierResult(verdict="CONTRIBUTE_SAVE", depth=DEPTH_SHALLOW))
    gated = StateGatedClassifier(inner)
    result = await gated.classify(_msg())
    assert result.verdict == "NONE"
    assert result.attrs["redirected_from"] == "CONTRIBUTE_SAVE"


@pytest.mark.asyncio
async def test_save_passes_when_pending_is_fresh():
    contributions.seed_tasks([{"source_en": "x", "category": "c", "seed_id": 1}])
    h = contributions.hash_msisdn(MSISDN)
    contributions.set_pending_save(h, 1, "Oshindonga")
    inner = _FakeInner(ClassifierResult(verdict="CONTRIBUTE_SAVE", depth=DEPTH_SHALLOW))
    gated = StateGatedClassifier(inner)
    result = await gated.classify(_msg())
    assert result.verdict == "CONTRIBUTE_SAVE"
    assert "redirected_from" not in (result.attrs or {})


# ── CONTRIBUTE_NEXT gating ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_next_blocked_without_pending_or_awaiting():
    inner = _FakeInner(ClassifierResult(verdict="CONTRIBUTE_NEXT", depth=DEPTH_SHALLOW))
    gated = StateGatedClassifier(inner)
    result = await gated.classify(_msg())
    assert result.verdict == "NONE"
    assert result.attrs["redirected_from"] == "CONTRIBUTE_NEXT"


@pytest.mark.asyncio
async def test_next_passes_when_awaiting_followup_set():
    h = contributions.hash_msisdn(MSISDN)
    contributions.set_awaiting_followup(h)
    inner = _FakeInner(ClassifierResult(verdict="CONTRIBUTE_NEXT", depth=DEPTH_SHALLOW))
    gated = StateGatedClassifier(inner)
    result = await gated.classify(_msg())
    assert result.verdict == "CONTRIBUTE_NEXT"


@pytest.mark.asyncio
async def test_next_passes_when_pending_set():
    contributions.seed_tasks([{"source_en": "x", "category": "c", "seed_id": 1}])
    h = contributions.hash_msisdn(MSISDN)
    contributions.set_pending_save(h, 1, "Oshindonga")
    inner = _FakeInner(ClassifierResult(verdict="CONTRIBUTE_NEXT", depth=DEPTH_SHALLOW))
    gated = StateGatedClassifier(inner)
    result = await gated.classify(_msg())
    assert result.verdict == "CONTRIBUTE_NEXT"


# ── CONTRIBUTE_SKIP / DECLINE gating ─────────────────────────────


@pytest.mark.asyncio
async def test_skip_blocked_without_pending():
    inner = _FakeInner(ClassifierResult(verdict="CONTRIBUTE_SKIP", depth=DEPTH_SHALLOW))
    gated = StateGatedClassifier(inner)
    result = await gated.classify(_msg())
    assert result.verdict == "NONE"


@pytest.mark.asyncio
async def test_decline_blocked_without_pending_or_awaiting():
    inner = _FakeInner(ClassifierResult(verdict="CONTRIBUTE_DECLINE", depth=DEPTH_SHALLOW))
    gated = StateGatedClassifier(inner)
    result = await gated.classify(_msg())
    assert result.verdict == "NONE"


@pytest.mark.asyncio
async def test_decline_passes_when_awaiting_followup_set():
    h = contributions.hash_msisdn(MSISDN)
    contributions.set_awaiting_followup(h)
    inner = _FakeInner(ClassifierResult(verdict="CONTRIBUTE_DECLINE", depth=DEPTH_SHALLOW))
    gated = StateGatedClassifier(inner)
    result = await gated.classify(_msg())
    assert result.verdict == "CONTRIBUTE_DECLINE"


# ── State-independent verdicts always pass through ──────────────


@pytest.mark.asyncio
async def test_invite_always_passes():
    inner = _FakeInner(ClassifierResult(verdict="CONTRIBUTE_INVITE", depth=DEPTH_SHALLOW))
    gated = StateGatedClassifier(inner)
    result = await gated.classify(_msg())
    assert result.verdict == "CONTRIBUTE_INVITE"


@pytest.mark.asyncio
async def test_dialect_always_passes():
    inner = _FakeInner(ClassifierResult(verdict="CONTRIBUTE_DIALECT", depth=DEPTH_SHALLOW))
    gated = StateGatedClassifier(inner)
    result = await gated.classify(_msg())
    assert result.verdict == "CONTRIBUTE_DIALECT"


@pytest.mark.asyncio
async def test_stats_always_passes():
    inner = _FakeInner(ClassifierResult(verdict="CONTRIBUTE_STATS", depth=DEPTH_SHALLOW))
    gated = StateGatedClassifier(inner)
    result = await gated.classify(_msg())
    assert result.verdict == "CONTRIBUTE_STATS"


# ── Non-contribute verdicts always pass through ─────────────────


@pytest.mark.asyncio
async def test_search_passes_through_unchanged():
    inner = _FakeInner(ClassifierResult(verdict="SEARCH", depth=DEPTH_DEEP))
    gated = StateGatedClassifier(inner)
    result = await gated.classify(_msg())
    assert result.verdict == "SEARCH"
    assert result.depth == DEPTH_DEEP


@pytest.mark.asyncio
async def test_none_passes_through_unchanged():
    inner = _FakeInner(ClassifierResult(verdict="NONE", depth=DEPTH_SHALLOW))
    gated = StateGatedClassifier(inner)
    result = await gated.classify(_msg())
    assert result.verdict == "NONE"
    # Should not have a redirected_from since it wasn't redirected
    assert "redirected_from" not in (result.attrs or {})


# ── attrs preservation ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_redirect_preserves_classifier_attrs():
    """The wrapper must keep any attrs the inner classifier provided
    so the trace records full provenance, not just the gate's reason."""
    inner = _FakeInner(ClassifierResult(
        verdict="CONTRIBUTE_SAVE", depth=DEPTH_SHALLOW,
        attrs={"confidence": "high", "reasoning": "looked like oshindonga"},
    ))
    gated = StateGatedClassifier(inner)
    result = await gated.classify(_msg())
    assert result.verdict == "NONE"
    assert result.attrs["confidence"] == "high"
    assert result.attrs["reasoning"] == "looked like oshindonga"
    assert result.attrs["redirected_from"] == "CONTRIBUTE_SAVE"


@pytest.mark.asyncio
async def test_redirect_preserves_token_counts_for_billing():
    """BillingHook reads tokens_in/out from the result — redirected
    results must carry these through or billing under-counts."""
    inner = _FakeInner(ClassifierResult(
        verdict="CONTRIBUTE_SAVE", depth=DEPTH_SHALLOW,
        tokens_in=100, tokens_out=50, cached_tokens=80,
    ))
    gated = StateGatedClassifier(inner)
    result = await gated.classify(_msg())
    assert result.tokens_in == 100
    assert result.tokens_out == 50
    assert result.cached_tokens == 80
