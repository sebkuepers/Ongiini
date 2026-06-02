"""Unit tests for SessionStore + SessionMemoryProvider.

Cover the LRU + TTL invariants of the store, the recordings against the
MemoryProvider protocol, and a couple of the cross-cutting concurrency
scenarios the code-review flagged (the get-then-mutate race that the
single-lock fix closes)."""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from owela import InboundMessage
from ongiini.memory.session_memory import (
    SessionMemoryProvider,
    SessionState,
    SessionStore,
)


SID_A = "11111111-1111-4111-8111-111111111111"
SID_B = "22222222-2222-4222-8222-222222222222"
SID_C = "33333333-3333-4333-8333-333333333333"


# ─── SessionStore basics ───────────────────────────────────────────


def test_get_or_create_creates_on_first_access():
    store = SessionStore(max_sessions=10, ttl_minutes=60)
    s = store.get_or_create(SID_A)
    assert isinstance(s, SessionState)
    assert s.history == []
    assert s.tokens_used == 0


def test_get_or_create_returns_same_object_on_repeat():
    store = SessionStore(max_sessions=10, ttl_minutes=60)
    s1 = store.get_or_create(SID_A)
    s2 = store.get_or_create(SID_A)
    assert s1 is s2  # live in-store object reused


def test_append_turn_writes_user_then_assistant():
    store = SessionStore(max_sessions=10, ttl_minutes=60)
    store.append_turn(SID_A, "hi there", "hello back")
    history = store.snapshot_history(SID_A)
    assert history == [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello back"},
    ]


def test_append_turn_auto_creates_session():
    store = SessionStore(max_sessions=10, ttl_minutes=60)
    # Session doesn't exist yet
    assert store.peek(SID_A) is None
    store.append_turn(SID_A, "hi", "hello")
    assert store.peek(SID_A) is not None
    assert len(store.snapshot_history(SID_A)) == 2


def test_touch_tokens_adds_to_running_total():
    store = SessionStore(max_sessions=10, ttl_minutes=60)
    store.get_or_create(SID_A)
    assert store.touch_tokens(SID_A, 100) == 100
    assert store.touch_tokens(SID_A, 50) == 150
    assert store.peek(SID_A).tokens_used == 150


def test_touch_tokens_no_op_for_missing_session():
    store = SessionStore(max_sessions=10, ttl_minutes=60)
    # No session created — touch returns 0 and doesn't crash
    assert store.touch_tokens(SID_A, 100) == 0
    assert store.peek(SID_A) is None


def test_delete_drops_session():
    store = SessionStore(max_sessions=10, ttl_minutes=60)
    store.append_turn(SID_A, "hi", "hello")
    assert store.delete(SID_A) is True
    assert store.peek(SID_A) is None
    assert store.delete(SID_A) is False  # idempotent


def test_snapshot_history_returns_copy():
    """Caller mutating the returned list must not affect the live
    state — the snapshot is taken under the lock and is independent."""
    store = SessionStore(max_sessions=10, ttl_minutes=60)
    store.append_turn(SID_A, "hi", "hello")
    snap = store.snapshot_history(SID_A)
    snap.append({"role": "user", "content": "BOGUS"})
    # Live store unaffected
    real = store.snapshot_history(SID_A)
    assert real == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


# ─── LRU eviction ──────────────────────────────────────────────────


def test_lru_evicts_oldest_when_over_max():
    store = SessionStore(max_sessions=2, ttl_minutes=60)
    store.get_or_create(SID_A)
    store.get_or_create(SID_B)
    # Adding a third evicts the LRU (A — it was created first and never
    # touched since).
    store.get_or_create(SID_C)
    assert store.peek(SID_A) is None
    assert store.peek(SID_B) is not None
    assert store.peek(SID_C) is not None


def test_lru_bumps_recency_on_get_or_create():
    """A re-accessed session moves to MRU so the next eviction takes
    the OTHER one instead."""
    store = SessionStore(max_sessions=2, ttl_minutes=60)
    store.get_or_create(SID_A)
    store.get_or_create(SID_B)
    # Touch A → becomes MRU, B becomes LRU
    store.get_or_create(SID_A)
    # Adding C evicts B (LRU), keeps A
    store.get_or_create(SID_C)
    assert store.peek(SID_A) is not None
    assert store.peek(SID_B) is None
    assert store.peek(SID_C) is not None


def test_lru_bumps_recency_on_append_turn():
    store = SessionStore(max_sessions=2, ttl_minutes=60)
    store.get_or_create(SID_A)
    store.get_or_create(SID_B)
    store.append_turn(SID_A, "hi", "hello")
    store.get_or_create(SID_C)
    assert store.peek(SID_A) is not None
    assert store.peek(SID_B) is None


# ─── TTL eviction ──────────────────────────────────────────────────


def test_ttl_evicts_stale_sessions(monkeypatch):
    """A session whose last_used_at is older than ttl_minutes is
    dropped on the next access. We backdate manually rather than
    sleeping in the test."""
    store = SessionStore(max_sessions=10, ttl_minutes=30)
    state = store.get_or_create(SID_A)
    # Backdate this session's last_used_at past the TTL.
    state.last_used_at = datetime.now(timezone.utc) - timedelta(minutes=31)
    # Any subsequent access triggers the sweep — touch a different
    # session id so we observe what happens to SID_A specifically.
    store.get_or_create(SID_B)
    assert store.peek(SID_A) is None
    assert store.peek(SID_B) is not None


def test_ttl_keeps_fresh_sessions():
    store = SessionStore(max_sessions=10, ttl_minutes=30)
    store.get_or_create(SID_A)
    # Brand-new session, accessed once. Should survive the sweep.
    store.get_or_create(SID_B)
    assert store.peek(SID_A) is not None
    assert store.peek(SID_B) is not None


# ─── Concurrency (the get-then-append race the review flagged) ─────


def test_concurrent_append_no_lost_turns():
    """20 threads each append a turn to the same session id. We rely
    on the single-lock fix in append_turn to ensure every (user,
    assistant) pair lands. The check is that 20 × 2 = 40 history
    entries exist at the end."""
    store = SessionStore(max_sessions=10, ttl_minutes=60)

    def worker(i):
        store.append_turn(SID_A, f"user-{i}", f"bot-{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    history = store.snapshot_history(SID_A)
    assert len(history) == 40
    # Every (user-N, bot-N) pair is present (order may vary across
    # threads but each turn comes through atomically).
    users = {h["content"] for h in history if h["role"] == "user"}
    bots = {h["content"] for h in history if h["role"] == "assistant"}
    assert users == {f"user-{i}" for i in range(20)}
    assert bots == {f"bot-{i}" for i in range(20)}


# ─── SessionMemoryProvider ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_assemble_messages_basic_shape():
    store = SessionStore(max_sessions=10, ttl_minutes=60)
    provider = SessionMemoryProvider(
        system_prompt="SYSTEM_PROMPT_HERE",
        store=store,
        skills=None,
        pii_sanitiser=None,
    )
    msg = InboundMessage(
        user_id=SID_A,
        msg_id="x",
        text="hello",
        content_parts=[{"type": "text", "text": "hello"}],
    )
    out = await provider.assemble_messages(msg, policy=None, prior_steps=[])
    # Order: system prompt → date anchor → (no skills) → user message
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "SYSTEM_PROMPT_HERE"
    # Date anchor is a system message containing "Right now in Namibia"
    assert any(
        m["role"] == "system" and "Right now in Namibia" in (m.get("content") or "")
        for m in out
    )
    # Final entry is the user turn
    assert out[-1]["role"] == "user"


@pytest.mark.asyncio
async def test_assemble_messages_includes_history():
    store = SessionStore(max_sessions=10, ttl_minutes=60)
    store.append_turn(SID_A, "earlier user", "earlier bot")
    history_snapshot = store.snapshot_history(SID_A)

    provider = SessionMemoryProvider(
        system_prompt="SYS",
        store=store,
        skills=None,
    )
    msg = InboundMessage(
        user_id=SID_A,
        msg_id="y",
        text="now",
        content_parts=[{"type": "text", "text": "now"}],
        history=history_snapshot,
    )
    out = await provider.assemble_messages(msg, policy=None, prior_steps=[])
    # The history entries land between the system block and the current
    # user message.
    contents = [m["content"] for m in out]
    assert "earlier user" in contents
    assert "earlier bot" in contents
    assert out[-1] == {"role": "user", "content": "now"}


@pytest.mark.asyncio
async def test_record_turn_writes_to_store():
    store = SessionStore(max_sessions=10, ttl_minutes=60)
    provider = SessionMemoryProvider(system_prompt="SYS", store=store)
    await provider.record_turn(SID_A, "hello", "hi back")
    assert store.snapshot_history(SID_A) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi back"},
    ]


@pytest.mark.asyncio
async def test_record_turn_applies_pii_sanitiser():
    store = SessionStore(max_sessions=10, ttl_minutes=60)
    provider = SessionMemoryProvider(
        system_prompt="SYS",
        store=store,
        pii_sanitiser=lambda s: s.replace("email@example.com", "[REDACTED]"),
    )
    await provider.record_turn(SID_A, "my email@example.com", "thanks for sharing")
    history = store.snapshot_history(SID_A)
    assert "email@example.com" not in history[0]["content"]
    assert "[REDACTED]" in history[0]["content"]


@pytest.mark.asyncio
async def test_record_image_turn_uses_placeholder():
    store = SessionStore(max_sessions=10, ttl_minutes=60)
    provider = SessionMemoryProvider(system_prompt="SYS", store=store)
    await provider.record_image_turn(SID_A, caption="a maize leaf", reply="I see yellow spots")
    history = store.snapshot_history(SID_A)
    assert history[0]["content"].startswith("[image attached]")
    assert "maize leaf" in history[0]["content"]


@pytest.mark.asyncio
async def test_delete_all_clears_session():
    store = SessionStore(max_sessions=10, ttl_minutes=60)
    provider = SessionMemoryProvider(system_prompt="SYS", store=store)
    await provider.record_turn(SID_A, "hi", "hello")
    removed = await provider.delete_all(SID_A)
    assert removed is True
    assert store.peek(SID_A) is None


@pytest.mark.asyncio
async def test_list_all_returns_history():
    store = SessionStore(max_sessions=10, ttl_minutes=60)
    provider = SessionMemoryProvider(system_prompt="SYS", store=store)
    await provider.record_turn(SID_A, "hi", "hello")
    facts = await provider.list_all(SID_A)
    assert len(facts) == 2
    assert facts[0]["role"] == "user"


def test_format_facts_empty_returns_friendly_string():
    provider = SessionMemoryProvider(
        system_prompt="SYS",
        store=SessionStore(),
    )
    assert "fresh session" in provider.format_facts([]).lower()


def test_format_facts_renders_turns_as_dialog():
    provider = SessionMemoryProvider(
        system_prompt="SYS",
        store=SessionStore(),
    )
    facts = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = provider.format_facts(facts)
    assert "You: hi" in out
    assert "Me:" in out and "hello" in out
