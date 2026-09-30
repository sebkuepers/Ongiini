"""Owela ``MemoryProvider`` implementation for Ongiini.

Wraps the two existing memory tiers:

  - ``webhook.ongiini.memory.short_term`` — short-term JSON files per user (~50 turns
    verbatim, capped at memory_window*2)
  - ``webhook.ongiini.memory.long_term`` — long-term mem0 vector store with typed facts
    extracted by an LLM, retrieved by similarity to the current query

``assemble_messages`` is the single point that builds the model's view
of context for one turn. The order is chosen for vLLM's prefix cache —
everything that is stable across a user's turns comes first, everything
that changes every turn comes last:

  1. system prompt: core + the policy's ``prompt_sections`` (+ the
     first-turn section when the history has no assistant message yet)
  2. skill manifest (always-loaded skills) + any ``skill:<name>``
     sections the policy asks for
  3. conversation history (passed in via InboundMessage.history)
  4. ONE turn-context system message: date/time anchor, relevant mem0
     facts, planner context, previously cited sources, degrade note
  5. the current user message (text or multipart)

Before 2026-09-30 the minute-precision date anchor sat between the
system prompt and the history, so the whole history was re-prefilled on
every turn.

``record_turn`` writes to both tiers. Long-term mem0 calls go through
``asyncio.to_thread`` because mem0's API is synchronous and the
embedding step does CPU work we don't want pinning the event loop.

The short-term and long-term backends are injected at construction so
tests can substitute simple fakes. Production wires the real
``webhook.ongiini.memory.short_term`` and ``webhook.ongiini.memory.long_term`` modules.

Summarisation (folding old turns into a rolling system summary when
history grows large) is NOT done here — the application is responsible
for calling ``summary.maybe_summarize`` on the history BEFORE passing it
as ``InboundMessage.history``. Pragmatic choice: summarisation needs a
model call, and threading the Model through the MemoryProvider couples
two responsibilities. v1 may promote it to a Hook; for now it stays
where it is in the FastAPI handler.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Protocol

from owela import DegradeStep, InboundMessage, PlanStep, Policy, SkillRegistry, Step

log = logging.getLogger("ongiini.memory_provider")


# Namibia uses Central Africa Time (UTC+2) year-round; no DST. The
# container clock runs in UTC, so we shift explicitly when we want to
# present "today" to the model — otherwise after 22:00 CAT we'd already
# be on tomorrow's UTC date.
_NAMIBIA_TZ = timezone(timedelta(hours=2))


SECTION_FIRST_TURN = "first_turn"
SKILL_SECTION_PREFIX = "skill:"


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
        system_prompt: str = "",
        *,
        prompt_builder: Callable[[frozenset[str]], str] | None = None,
        short_term: ShortTermBackend,
        long_term: LongTermBackend,
        mem0_search_limit: int = 5,
        mem0_inject_limit: int = 3,
        mem0_min_score: float = 0.3,
        source_index_loader: Callable[[str], list[dict[str, Any]]] | None = None,
        source_index_formatter: Callable[[list[dict[str, Any]]], str] | None = None,
        source_index_deleter: Callable[[str], bool] | None = None,
        skills: SkillRegistry | None = None,
    ) -> None:
        if not system_prompt and prompt_builder is None:
            raise ValueError("OngiiniMemoryProvider needs system_prompt or prompt_builder")
        self.system_prompt = system_prompt
        # prompt_builder(sections) → system prompt text for this turn.
        # Wins over the static ``system_prompt`` when given.
        self._prompt_builder = prompt_builder
        self._short = short_term
        self._long = long_term
        self.mem0_search_limit = mem0_search_limit
        self.mem0_inject_limit = mem0_inject_limit
        self.mem0_min_score = mem0_min_score
        # v1.6-B source-index: optional third memory tier that persists
        # cited URLs across turns. Injected as callables so tests can
        # substitute fakes without touching the on-disk store; None
        # cleanly disables the feature.
        self._load_source_index = source_index_loader
        self._format_source_index = source_index_formatter
        self._delete_source_index = source_index_deleter
        # Skills are referenced from the message-assembly path so the
        # manifest (and any always-loaded content) lands in the system
        # prompt. None / empty registry → no manifest injection.
        self._skills = skills

    async def assemble_messages(
        self,
        msg: InboundMessage,
        policy: Policy,
        prior_steps: list[Step],
    ) -> list[dict[str, Any]]:
        sections = set(policy.prompt_sections)
        if not any(m.get("role") == "assistant" for m in msg.history):
            # No reply from us yet: the EU AI Act disclosure + welcome line.
            sections.add(SECTION_FIRST_TURN)

        # --- stable prefix: system prompt, skills, history ---------------
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt_for(sections)},
        ]
        for block in self._skill_blocks(sections):
            messages.append({"role": "system", "content": block})
        messages.extend(msg.history)

        # --- per-turn context, directly before the user message ----------
        context = await self._turn_context(msg, prior_steps)
        messages.append({"role": "system", "content": context})

        # Image-bearing turns need the multipart list; text-only turns send
        # a plain string.
        if msg.has_image and msg.content_parts:
            user_content: Any = msg.content_parts
        else:
            user_content = msg.text
        messages.append({"role": "user", "content": user_content})
        return messages

    def _system_prompt_for(self, sections: set[str]) -> str:
        if self._prompt_builder is not None:
            return self._prompt_builder(frozenset(sections))
        return self.system_prompt

    def _skill_blocks(self, sections: set[str]) -> list[str]:
        """The manifest of always-loaded skills (static, cache-stable),
        then the full content of every ``skill:<name>`` section the policy
        requests. On-demand skills are not advertised: no policy exposes a
        ``load_skill`` tool, so the policy decides when a skill is needed."""
        if self._skills is None:
            return []
        blocks: list[str] = []
        manifest = self._skills.manifest(include_on_demand=False)
        if manifest:
            blocks.append(manifest)
        for section in sorted(sections):
            if not section.startswith(SKILL_SECTION_PREFIX):
                continue
            skill = self._skills.get(section[len(SKILL_SECTION_PREFIX):])
            if skill is not None and skill.load != "always":
                blocks.append(f"## Skill: {skill.name}\n\n{skill.content}")
        return blocks

    async def _turn_context(self, msg: InboundMessage, prior_steps: list[Step]) -> str:
        """Everything that changes turn to turn, as one system message."""
        parts = [_today_in_namibia_prompt()]
        facts = await self._relevant_facts(msg)
        if facts:
            parts.append(facts)
        plan_msg = self._extract_plan_message(prior_steps)
        if plan_msg:
            parts.append(plan_msg)
        # v1.6-B: URLs cited in earlier turns, so "give me sources" works
        # past the rolling-summary horizon.
        si_msg = self._format_source_index_message(msg.user_id)
        if si_msg:
            parts.append(si_msg)
        degrade_msg = self._extract_degrade_message(prior_steps)
        if degrade_msg:
            parts.append(degrade_msg)
        return "\n\n".join(parts)

    async def _relevant_facts(self, msg: InboundMessage) -> str:
        """Top mem0 facts for this message: similarity above
        ``mem0_min_score``, at most ``mem0_inject_limit``. Hits without a
        score (backends that don't report one) are kept. ``search``
        returns [] on any error, so a down mem0 just means no facts."""
        hits = await asyncio.to_thread(
            self._long.search, msg.user_id, msg.text, self.mem0_search_limit,
        )
        kept = [
            h for h in hits
            if not isinstance(h, dict)
            or not isinstance(h.get("score"), (int, float))
            or h["score"] >= self.mem0_min_score
        ][: self.mem0_inject_limit]
        return self._long.format_relevant(kept)

    @staticmethod
    def _extract_degrade_message(prior_steps: list[Step]) -> str:
        for step in prior_steps:
            if isinstance(step, DegradeStep):
                tools = ", ".join(step.missing_tools) or "a required tool"
                return (
                    f"Live tools unavailable this turn ({tools}). Answer from "
                    "general knowledge, keep it short, say plainly that you "
                    "could not check current information, and never present "
                    "prices, dates, schedules or other changing facts as current."
                )
        return ""

    def _format_source_index_message(self, user_id: str) -> str:
        """Read the per-user source index and format it as a system
        block. Returns empty string when the feature is disabled (no
        loader injected) or the index is empty."""
        if self._load_source_index is None or self._format_source_index is None:
            return ""
        try:
            entries = self._load_source_index(user_id)
        except Exception as exc:                       # noqa: BLE001 — soft-fail
            log.warning("source_index load failed for %s: %s", user_id, exc)
            return ""
        try:
            return self._format_source_index(entries)
        except Exception as exc:                       # noqa: BLE001
            log.warning("source_index format failed for %s: %s", user_id, exc)
            return ""

    @staticmethod
    def _extract_plan_message(prior_steps: list[Step]) -> str:
        """Pull the latest PlanStep's ``plan_text`` (the ``facts_known``
        prose, post v1.3) and format it as a context-priming system
        message body.

        v1.3 removed the imperative tool-steering wrapper: the executor
        now synthesises searches deterministically from
        ``PlanStep.queries``, so the model no longer needs prose
        instructions on which tool to call. What remains is the
        planner's pre-search context — facts the model can rely on
        without searching. Surface it neutrally as additional context.

        Empty ``plan_text`` means the planner soft-failed or had no
        context to add; return empty string and skip injection.
        """
        for step in reversed(prior_steps):
            if isinstance(step, PlanStep) and step.plan_text:
                return (
                    "Pre-search context (search results will appear "
                    "in the conversation below):\n"
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
        """Wipe all tiers. Returns True if any tier had data to delete.

        Privacy-critical: if one tier raises, we MUST still try the
        others. A failure in one tier cannot leak data from another.
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
        source_removed = False
        if self._delete_source_index is not None:
            try:
                source_removed = self._delete_source_index(user_id)
            except Exception as exc:                   # noqa: BLE001
                log.warning("source_index.delete failed for %s: %s", user_id, exc)
        return short_removed or long_removed or source_removed

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
