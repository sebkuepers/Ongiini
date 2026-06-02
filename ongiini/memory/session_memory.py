"""Session-scoped, in-process memory for the chat.ongiini.ai endpoint.

WhatsApp users get the full Ongiini memory contract: a per-msisdn JSON
file (~50 turns rolling), mem0 long-term facts, source-index URL
persistence. Anonymous web sessions get **none** of that. The browser
session IS the memory boundary: open the tab → fresh conversation;
close the tab → it's gone.

This module provides two pieces:

  * ``SessionStore`` — a process-local LRU+TTL dict of session_id →
    SessionState. No disk write, no mem0, no IPC.
  * ``SessionMemoryProvider`` — an Owela ``MemoryProvider`` impl that
    reads/writes the store and assembles the model's context block
    matching the shape ``OngiiniMemoryProvider.assemble_messages``
    produces, minus the mem0 facts block and minus the source-index
    block.

The contract is documented on chat.ongiini.ai itself ("memory only for
the browser session") so this module's behaviour is observable to the
user.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from owela import InboundMessage, PlanStep, Policy, SkillRegistry, Step

log = logging.getLogger("ongiini.memory.session")


@dataclass
class SessionState:
    """One in-flight session's history + accounting."""
    history: list[dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_used_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class SessionStore:
    """Process-local LRU+TTL session store.

    Thread-safe via a single mutex (sessions are touched from the
    FastAPI request workers, which run on the asyncio loop, but the
    background eviction sweep can fire from any worker — keep the
    invariant explicit). Concurrent reads + writes are infrequent
    enough that lock contention isn't a concern at chat-MVP scale.

    Behaviour:

      * ``get_or_create`` returns a SessionState, creating it if
        missing. Bumps ``last_used_at`` so the TTL sweep doesn't drop a
        warm session.
      * ``touch_tokens`` adds N tokens to the running total and bumps
        last_used_at.
      * ``delete`` drops a session entirely.
      * Inserts beyond ``max_sessions`` evict the LRU entry to keep
        memory bounded.
      * Every ``get_or_create`` / ``record_turn`` also runs the TTL
      * sweep so we don't need a background thread or task.
    """

    def __init__(
        self,
        *,
        max_sessions: int = 10_000,
        ttl_minutes: int = 360,
    ) -> None:
        self._max = max_sessions
        self._ttl = timedelta(minutes=ttl_minutes)
        self._lock = threading.Lock()
        # OrderedDict so we can use move_to_end for LRU semantics and
        # popitem(last=False) for the oldest-first eviction.
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()

    # ─── operations ─────────────────────────────────────────────────

    def get_or_create(self, session_id: str) -> SessionState:
        """Return the session, creating it on first access.

        Bumps ``last_used_at`` and runs the TTL sweep. If creating would
        push us over ``max_sessions``, evicts the LRU entry first.

        Note: the returned SessionState is the live in-store object.
        Callers that need to read its fields without lock-coordinated
        access should use ``snapshot_history`` instead; callers that
        mutate it must do so under ``self._lock`` (see ``append_turn``
        for the safe pattern).
        """
        with self._lock:
            return self._get_or_create_locked(session_id)

    def _get_or_create_locked(self, session_id: str) -> SessionState:
        """Inner helper — callers MUST hold ``self._lock`` already."""
        now = datetime.now(timezone.utc)
        self._sweep_expired_locked(now)
        state = self._sessions.get(session_id)
        if state is None:
            if len(self._sessions) >= self._max:
                # Pop the oldest (LRU) entry to make room.
                self._sessions.popitem(last=False)
            state = SessionState(created_at=now, last_used_at=now)
            self._sessions[session_id] = state
        else:
            state.last_used_at = now
            self._sessions.move_to_end(session_id)
        return state

    def peek(self, session_id: str) -> SessionState | None:
        """Read the session without bumping last_used_at. Used by the
        token-cap check before deciding whether to accept a request."""
        with self._lock:
            return self._sessions.get(session_id)

    def snapshot_history(self, session_id: str) -> list[dict[str, Any]]:
        """Return a deep-copy of the session's current history, taken
        atomically under the lock. Use this from any code path that
        iterates history outside the lock (avoids ``list changed size
        during iteration`` from a concurrent ``append_turn``)."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return []
            return [dict(turn) for turn in state.history]

    def append_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """Append a (user, assistant) pair to the session history.
        Auto-creates the session if it doesn't exist yet.

        Single lock acquisition: get-or-create + history mutation +
        timestamp bump all happen under one ``with self._lock``. A
        concurrent ``get_or_create`` for the same session_id can't
        evict the SessionState mid-operation and leave us appending to
        an orphaned object.
        """
        with self._lock:
            state = self._get_or_create_locked(session_id)
            state.history.append({"role": "user", "content": user_text})
            state.history.append({"role": "assistant", "content": assistant_text})
            state.last_used_at = datetime.now(timezone.utc)

    def touch_tokens(self, session_id: str, tokens: int) -> int:
        """Add ``tokens`` to the session's running total. Returns the
        new total. No-op (returns 0) if the session doesn't exist."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return 0
            state.tokens_used += max(0, tokens)
            state.last_used_at = datetime.now(timezone.utc)
            self._sessions.move_to_end(session_id)
            return state.tokens_used

    def delete(self, session_id: str) -> bool:
        """Drop the session entirely. Returns True if anything was
        actually removed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def stats(self) -> dict[str, int]:
        """Diagnostic snapshot for the /healthz / metrics endpoint.
        Cheap to call — held under the lock for one read."""
        with self._lock:
            return {
                "sessions": len(self._sessions),
                "max_sessions": self._max,
            }

    # ─── internal ───────────────────────────────────────────────────

    def _sweep_expired_locked(self, now: datetime) -> int:
        """Drop any sessions whose ``last_used_at`` is older than the
        TTL. Caller must hold the lock. Returns the number of evictions
        for observability."""
        cutoff = now - self._ttl
        # Iterate keys snapshot — modifying the dict mid-iteration is
        # unsupported. Sweep is O(n) but n is bounded by max_sessions
        # and runs only when a request lands.
        expired = [
            sid for sid, st in self._sessions.items()
            if st.last_used_at < cutoff
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
        if expired:
            log.info("session TTL sweep removed %d expired sessions", len(expired))
        return len(expired)


# ── Owela MemoryProvider for sessions ─────────────────────────────


# Namibia Central Africa Time. The web-chat audience isn't necessarily
# Namibian, but the bot's content and system prompt are tuned for
# Namibia — keeping the same date anchor avoids drift.
_NAMIBIA_TZ = timezone(timedelta(hours=2))


def _today_in_namibia_prompt() -> str:
    """Same date anchor as `OngiiniMemoryProvider`. Kept local so the
    session provider stays decoupled from the WhatsApp provider — both
    happen to use the same anchor today, but if either diverges this is
    the seam."""
    now = datetime.now(_NAMIBIA_TZ)
    return (
        f"Right now in Namibia it is {now.strftime('%A, %d %B %Y, %H:%M')} "
        f"(Central Africa Time / UTC+2, no DST).\n"
        "Anchor all 'soon', 'upcoming', 'recent', 'this week', 'next week', "
        "'open right now', 'still open', 'tonight' reasoning to THIS date "
        "and time.\n"
        "When web_search results include dated events, compare the event "
        "date to today: events BEFORE today have already happened — never "
        "present them as upcoming. If all results are about past events, "
        "say so plainly instead of pretending they're scheduled."
    )


class SessionMemoryProvider:
    """Owela `MemoryProvider` for the chat.ongiini.ai endpoint.

    Reads/writes only the in-process `SessionStore`. No disk, no mem0.
    The model's view of context is intentionally simpler than the
    WhatsApp path: system prompt + skill manifest + date anchor +
    session history + current message. No long-term facts (we don't
    have any), no source-index (the SourceIndexHook is dropped in the
    chat runtime).
    """

    def __init__(
        self,
        system_prompt: str,
        *,
        store: SessionStore,
        skills: SkillRegistry | None = None,
        pii_sanitiser: Callable[[str], str] | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self._store = store
        self._skills = skills
        # PII sanitiser is optional — for sessions the user accepts
        # that whatever they type is processed by the model. We still
        # sanitise on record_turn so the in-memory snapshot of history
        # doesn't carry raw emails/IDs/cards visible to a future
        # ``whats_in_my_memory`` query. If None, content is stored
        # verbatim.
        self._sanitise = pii_sanitiser

    async def assemble_messages(
        self,
        msg: InboundMessage,
        policy: Policy,
        prior_steps: list[Step],
    ) -> list[dict[str, Any]]:
        """Build the OpenAI-style messages list for this turn.

        Order matches the WhatsApp provider as far as practical so the
        model sees an identical preamble between the two transports
        (cache-friendly for vLLM prefix-cache):

          1. SYSTEM_PROMPT
          2. Skill manifest (when skills are registered)
          3. Today's date anchor
          4. Plan injection (if SEARCH_DEEP fired the planner)
          5. msg.history (the session's rolling conversation)
          6. Current user message
        """
        if msg.has_image and msg.content_parts:
            user_content: Any = msg.content_parts
        else:
            user_content = msg.text

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if self._skills is not None:
            manifest = self._skills.manifest()
            if manifest:
                messages.append({"role": "system", "content": manifest})
        messages.append({"role": "system", "content": _today_in_namibia_prompt()})

        plan_msg = self._extract_plan_message(prior_steps)
        if plan_msg:
            messages.append({"role": "system", "content": plan_msg})

        messages.extend(msg.history)
        messages.append({"role": "user", "content": user_content})
        return messages

    async def record_turn(
        self,
        user_id: str,
        user_text: str,
        reply: str,
    ) -> None:
        """Append this turn to the session's in-memory history.

        Soft-fail: never raises. PII sanitisation is applied to both
        sides of the turn if a sanitiser was wired in, so the stored
        history doesn't carry raw emails / ID numbers / card numbers.
        """
        try:
            user_stored = self._sanitise(user_text) if self._sanitise else user_text
            reply_stored = self._sanitise(reply) if self._sanitise else reply
            self._store.append_turn(user_id, user_stored, reply_stored)
        except Exception as exc:                       # noqa: BLE001 — soft-fail
            log.warning("session record_turn failed for %s: %s", user_id[:8], exc)

    async def record_image_turn(
        self,
        user_id: str,
        caption: str,
        reply: str,
    ) -> None:
        """Image-bearing turn — store the caption + placeholder, drop
        the bytes. Matches the WhatsApp provider's placeholder shape so
        downstream code that walks history doesn't have to branch."""
        placeholder = "[image attached]"
        if caption:
            cleaned_caption = self._sanitise(caption) if self._sanitise else caption
            placeholder = f"{placeholder} {cleaned_caption}"
        try:
            reply_stored = self._sanitise(reply) if self._sanitise else reply
            self._store.append_turn(user_id, placeholder, reply_stored)
        except Exception as exc:                       # noqa: BLE001
            log.warning("session record_image_turn failed for %s: %s", user_id[:8], exc)

    async def delete_all(self, user_id: str) -> bool:
        """Wipe this session. Soft-fail: any error returns False."""
        try:
            return self._store.delete(user_id)
        except Exception as exc:                       # noqa: BLE001
            log.warning("session delete_all failed for %s: %s", user_id[:8], exc)
            return False

    async def list_all(self, user_id: str) -> list[dict[str, Any]]:
        """Return the session's current history. The
        ``whats_in_my_memory`` tool surfaces this to the user when they
        ask 'what do you remember about me?'.

        Shape matches what the WhatsApp provider returns — a list of
        ``{role, content}`` dicts the formatter can render. Use
        ``SessionStore.snapshot_history`` so the iteration happens
        under the store's lock (no torn reads if append_turn races)."""
        return self._store.snapshot_history(user_id)

    def format_facts(self, facts: list[dict[str, Any]]) -> str:
        """Render the session history as a human-readable string for
        the ``whats_in_my_memory`` tool. We don't have typed facts (the
        WA path's [PROFILE]/[PREFERENCE]/etc tags don't exist here), so
        just enumerate the turns plainly."""
        if not facts:
            return "Nothing yet — this is a fresh session."
        lines = []
        for turn in facts:
            role = turn.get("role", "?")
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                lines.append(f"You: {content}")
            elif role == "assistant":
                lines.append(f"Me:  {content}")
        return "\n".join(lines) if lines else "Nothing yet — this is a fresh session."

    @staticmethod
    def _extract_plan_message(prior_steps: list[Step]) -> str:
        """Pull the latest PlanStep's plan_text and format it for
        the model. Mirrors `OngiiniMemoryProvider._extract_plan_message`
        — if either diverges this is the seam to copy from."""
        for step in reversed(prior_steps):
            if isinstance(step, PlanStep) and step.plan_text:
                return (
                    "Pre-search context (search results will appear "
                    "in the conversation below):\n"
                    f"{step.plan_text}"
                )
        return ""
