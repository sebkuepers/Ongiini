"""MemoryProvider protocol — context assembly + persistence.

This is the single point of truth for "what context does the model see
on this turn." The Ongiini impl combines:
  - the application's static SYSTEM_PROMPT
  - mem0 long-term semantic memory (injected as a separate system msg
    to keep the static prompt prefix-cached)
  - the short-term conversation history (passed in via InboundMessage)
  - the current user message

A future impl might use a different memory stack (LangChain memory,
straight Redis, no memory at all). Owela does not care — it just calls
``assemble_messages`` and trusts the result.

The provider also owns the admin operations needed by the
``delete_my_data`` and ``whats_in_my_memory`` tools, since they touch
the same underlying storage.

Note that ``record_turn`` is called by the built-in
``MemoryRecordingHook`` in ``owela.hooks_builtin``, NOT by the executor
directly. Applications that want persistence add that hook to their
``HookRegistry``; applications that don't simply omit it.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .policy import Policy
from .step import Step
from .transport import InboundMessage


@runtime_checkable
class MemoryProvider(Protocol):
    """Context assembly + storage. Implementations are application-owned."""

    async def assemble_messages(
        self,
        msg: InboundMessage,
        policy: Policy,
        prior_steps: list[Step],
    ) -> list[dict[str, Any]]:
        """Return the complete OpenAI-style messages list for the upcoming
        model call. The returned list is what the model sees verbatim —
        the executor does NOT add a system prompt itself.

        **Ownership / mutation contract:** the returned list is OWNED by
        the executor for the rest of the turn. The executor appends
        assistant + tool messages to it during the act loop. Impls
        should not retain a reference and expect it to stay stable. If
        you need to observe the full conversation including the act-
        loop tool fan-out, look at ``prior_steps`` plus the steps fired
        by the hook layer — ``assemble_messages`` is called exactly
        ONCE at the start of the turn.

        ``prior_steps`` includes the RouterStep at minimum. Use it if
        you want assembly to depend on the classifier verdict (e.g.
        skip mem0 lookup on ADMIN turns).
        """
        ...

    async def record_turn(
        self,
        user_id: str,
        user_text: str,
        reply: str,
    ) -> None:
        """Persist this turn for future context. Soft-fail: should never
        raise (the hook wrapping the call also catches, but defence in
        depth)."""
        ...

    async def delete_all(self, user_id: str) -> bool:
        """Wipe all memory tiers for this user. Returns True if anything
        was actually deleted."""
        ...

    async def list_all(self, user_id: str) -> list[dict[str, Any]]:
        """Return all stored facts/turns for this user, for the
        ``whats_in_my_memory`` tool. Format is impl-specific."""
        ...

    def format_facts(self, facts: list[dict[str, Any]]) -> str:
        """Render the result of ``list_all`` as a human-friendly string
        for surfacing to the user via the ``whats_in_my_memory`` tool.

        Default impl returns a bullet list of ``memory`` fields. Impls
        with typed/tagged facts (e.g. mem0's [PROFILE]/[PREFERENCE]/...
        tags) should override to group by tag."""
        ...
