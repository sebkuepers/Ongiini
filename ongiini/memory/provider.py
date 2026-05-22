"""Owela ``MemoryProvider`` implementation for Ongiini.

Wraps the two existing memory tiers:

  - ``webhook.ongiini.memory.short_term`` — short-term JSON files per user (~50 turns
    verbatim, capped at memory_window*2)
  - ``webhook.ongiini.memory.long_term`` — long-term mem0 vector store with typed facts
    extracted by an LLM, retrieved by similarity to the current query

``assemble_messages`` is the single point that builds the model's view
of context for one turn. The order is:

  1. SYSTEM_PROMPT (constant per-app)
  2. mem0 long-term memory block (only if non-empty; injected as its
     own system message to keep the static SYSTEM_PROMPT prefix-cached)
  3. conversation history (passed in via InboundMessage.history)
  4. the current user message (text or multipart)

``record_turn`` writes to both tiers. Long-term mem0 calls go through
``asyncio.to_thread`` because mem0's API is synchronous and the
embedding step does CPU work we don't want pinning the event loop.

The short-term and long-term backends are injected at construction so
tests can substitute simple fakes. Production wires the real
``webhook.ongiini.memory.short_term`` and ``webhook.ongiini.memory.long_term`` modules.

Summarisation (folding old turns into a rolling system summary when
history grows large) is NOT done here — the application is responsible
for calling ``llm.maybe_summarize`` on the history BEFORE passing it as
``InboundMessage.history``. Pragmatic choice: summarisation needs a
model call, and threading the Model through the MemoryProvider couples
two responsibilities. v1 may promote it to a Hook; for now it stays
where it is in the FastAPI handler.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from owela import InboundMessage, PlanStep, Policy, Step

log = logging.getLogger("ongiini.memory_provider")


# Namibia uses Central Africa Time (UTC+2) year-round; no DST. The
# container clock runs in UTC, so we shift explicitly when we want to
# present "today" to the model — otherwise after 22:00 CAT we'd already
# be on tomorrow's UTC date.
_NAMIBIA_TZ = timezone(timedelta(hours=2))


def _today_in_namibia_prompt() -> str:
    """Short system message anchoring the model to today's real date AND time.

    Critical for SEARCH replies: web results commonly include events
    dated in the past, and a date-blind model presents them as
    "upcoming". Without this anchor the model defaults to its training-
    cutoff sense of "now" (typically 2024-25 for current Gemma builds),
    which is months stale.

    Time matters too — "is the bank open right now", "what time does X
    close", "is it past sundown" all need the current local time.
    Namibia uses Central Africa Time (UTC+2) year-round, no DST.
    """
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


class ShortTermBackend(Protocol):
    """Duck-typed contract for the short-term JSON-file memory module."""
    def load(self, user_id: str) -> list[dict[str, Any]]: ...
    def save(self, user_id: str, messages: list[dict[str, Any]]) -> None: ...
    def delete(self, user_id: str) -> bool: ...


class LongTermBackend(Protocol):
    """Duck-typed contract for the mem0 long-term memory module."""
    def search(self, user_id: str, query: str, limit: int) -> list[dict[str, Any]]: ...
    def add_turn(self, user_id: str, user_content: Any, assistant_text: str) -> None: ...
    def add_image_turn(self, user_id: str, caption: str, assistant_text: str) -> None: ...
    def list_all(self, user_id: str) -> list[dict[str, Any]]: ...
    def delete_all(self, user_id: str) -> bool: ...
    def format_relevant(self, memories: list[dict[str, Any]]) -> str: ...


class OngiiniMemoryProvider:
    """Two-tier memory: JSON short-term + mem0 long-term.

    The short-term and long-term backends are injected so this module
    can be unit-tested without importing mem0 (which transitively pulls
    in torch / sentence-transformers). Production wires
    ``webhook.ongiini.memory.short_term`` and ``webhook.ongiini.memory.long_term``.
    """

    def __init__(
        self,
        system_prompt: str,
        *,
        short_term: ShortTermBackend,
        long_term: LongTermBackend,
        mem0_search_limit: int = 5,
    ) -> None:
        self.system_prompt = system_prompt
        self._short = short_term
        self._long = long_term
        self.mem0_search_limit = mem0_search_limit

    async def assemble_messages(
        self,
        msg: InboundMessage,
        policy: Policy,
        prior_steps: list[Step],
    ) -> list[dict[str, Any]]:
        # Long-term vector search. ``mem.search`` returns [] on any
        # error, so the assembled list still works if mem0 is down or
        # warming up.
        relevant = await asyncio.to_thread(
            self._long.search, msg.user_id, msg.text, self.mem0_search_limit,
        )
        memory_block = self._long.format_relevant(relevant)

        # Pick the right user content shape. For image-bearing turns the
        # OpenAI multipart list is what the model needs to see; for text
        # only we send a plain string so prompts stay maximally cacheable.
        if msg.has_image and msg.content_parts:
            user_content: Any = msg.content_parts
        else:
            user_content = msg.text

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        # Today's date as its own short system message. Goes AFTER the
        # static system prompt so the prefix cache still hits everything
        # above it. The date itself rotates daily — fine, the cache miss
        # cost on ~80 tokens is negligible.
        messages.append({"role": "system", "content": _today_in_namibia_prompt()})
        if memory_block:
            # Separate system message rather than concatenated into
            # SYSTEM_PROMPT — the per-user mem0 block is the only
            # variable here, so the static system stays prefix-cached.
            messages.append({"role": "system", "content": memory_block})

        # Plan injection — if the executor ran the Planner phase BEFORE
        # the act loop (only happens on SEARCH_DEEP turns when
        # policy.enable_planner is True), surface the plan as its own
        # system message so the model enters the act loop with the
        # decomposition in scope. We look at prior_steps rather than
        # taking the plan as a separate parameter so this provider stays
        # decoupled from the Planner protocol.
        plan_msg = self._extract_plan_message(prior_steps)
        if plan_msg:
            messages.append({"role": "system", "content": plan_msg})

        messages.extend(msg.history)
        messages.append({"role": "user", "content": user_content})
        return messages

    @staticmethod
    def _extract_plan_message(prior_steps: list[Step]) -> str:
        """Pull the latest PlanStep's text out of the step list and
        format it as a system-message body. Empty plan_text means the
        Planner soft-failed; treat as no plan."""
        for step in reversed(prior_steps):
            if isinstance(step, PlanStep) and step.plan_text:
                return (
                    "Your plan for this turn (built before the search "
                    "step). Follow it as a guide — adjust if a search "
                    "result reveals something the plan didn't anticipate:\n"
                    f"{step.plan_text}"
                )
        return ""

    async def record_turn(
        self,
        user_id: str,
        user_text: str,
        reply: str,
    ) -> None:
        """Persist this turn to both tiers.

        Short-term writes happen synchronously (the file write is cheap
        and we want the next turn to see this turn's history). Long-term
        mem0 add happens via ``asyncio.to_thread`` because mem0 makes its
        own LLM call (~2 round-trips for fact extraction + reconciliation)
        and we don't want that pinning the event loop.

        Both writes are best-effort — broken persistence must not crash
        a successful reply.
        """
        # Short-term: append to the rolling history.
        try:
            history = self._short.load(user_id)
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": reply})
            self._short.save(user_id, history)
        except Exception as exc:                       # noqa: BLE001
            log.warning("short-term memory.save failed for %s: %s", user_id, exc)

        # Long-term: feed mem0 in a thread (sync API + embedding work).
        try:
            await asyncio.to_thread(self._long.add_turn, user_id, user_text, reply)
        except Exception as exc:                       # noqa: BLE001
            log.warning("long-term mem.add_turn failed for %s: %s", user_id, exc)

    async def record_image_turn(
        self,
        user_id: str,
        caption: str,
        reply: str,
    ) -> None:
        """Same as ``record_turn`` but for image-bearing inbound messages.

        The image bytes are NOT useful to mem0's extraction LLM (they
        balloon the prompt by kilobytes of base64). ``mem.add_image_turn``
        synthesises a text-only "[image attached] <caption>" message that
        the extractor handles cleanly. The assistant's reply contains
        its own description of the image, which serves as ground truth
        for fact extraction.

        The short-term tier still gets the verbatim caption + reply.
        Image bytes are NOT persisted to short-term either — the next
        turn won't see the original image. That's intentional and
        documented on the privacy page.
        """
        # Match the original placeholder format: "[image attached]"
        # alone if no caption, else "[image attached] <caption>". Tested
        # against eval cases that look at the short-term file shape.
        placeholder = "[image attached]"
        if caption:
            placeholder = f"{placeholder} {caption}"
        try:
            history = self._short.load(user_id)
            history.append({"role": "user", "content": placeholder})
            history.append({"role": "assistant", "content": reply})
            self._short.save(user_id, history)
        except Exception as exc:                       # noqa: BLE001
            log.warning("short-term memory.save (image) failed for %s: %s", user_id, exc)

        try:
            await asyncio.to_thread(self._long.add_image_turn, user_id, caption, reply)
        except Exception as exc:                       # noqa: BLE001
            log.warning("long-term mem.add_image_turn failed for %s: %s", user_id, exc)

    async def delete_all(self, user_id: str) -> bool:
        """Wipe both tiers. Returns True if either tier had data to delete.

        Privacy-critical: if one tier raises, we MUST still try the
        other. A failure in short-term cannot leak long-term data and
        vice versa.
        """
        short_removed = False
        try:
            short_removed = self._short.delete(user_id)
        except Exception as exc:                       # noqa: BLE001
            log.warning("short-term memory.delete failed for %s: %s", user_id, exc)
        long_removed = False
        try:
            long_removed = await asyncio.to_thread(self._long.delete_all, user_id)
        except Exception as exc:                       # noqa: BLE001
            log.warning("long-term mem.delete_all failed for %s: %s", user_id, exc)
        return short_removed or long_removed

    async def list_all(self, user_id: str) -> list[dict[str, Any]]:
        """Return the long-term facts. Short-term raw history is
        surfaced separately by the ``whats_in_my_memory`` tool, which
        calls ``memory.load`` directly — keeping both lookups behind
        this single method would force a less natural return shape."""
        return await asyncio.to_thread(self._long.list_all, user_id)

    def format_facts(self, facts: list[dict[str, Any]]) -> str:
        """Render long-term facts grouped by [TAG] for the
        ``whats_in_my_memory`` tool. Delegates to the long-term
        backend's tag-aware formatter; falls back to a flat bullet list
        if the backend doesn't expose one."""
        formatter = getattr(self._long, "format_grouped_by_tag", None)
        if callable(formatter):
            return formatter(facts)
        lines: list[str] = []
        for f in facts:
            text = (f.get("memory") if isinstance(f, dict) else None) or ""
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines)
