"""Backwards-compat shim — wraps Owela so the existing eval scripts keep working.

The old hand-rolled ``respond()`` loop is GONE. What remains here is:

  - ``SYSTEM_PROMPT`` re-export from ``system_prompt`` for any caller
    that still imports it via this path.
  - ``maybe_summarize`` / ``summarize_turns`` re-export from ``summary``
    for callers that still import them via this path.
  - ``LLMResult`` dataclass + ``respond()`` wrapper — translates the
    old function-call shape into ``Agent.handle()`` so the parity-smoke
    eval scripts in ``webhook/tests/`` keep working without rewrites.

Follow-up cleanup (v0.1): migrate eval scripts to call ``Agent.handle``
directly, then delete this module entirely. Out of scope for the
initial Owela migration commit.
"""

from __future__ import annotations

from dataclasses import dataclass

from owela import InboundMessage, ReplyStep, ToolRegistry, ToolStep

# Re-exports — keep the import paths the same so older callers don't break.
from .summary import (                              # noqa: F401 — re-exported
    SUMMARY_PREFIX as _SUMMARY_PREFIX, maybe_summarize, summarize_turns,
)
from .system_prompt import SYSTEM_PROMPT            # noqa: F401 — re-exported
from .tools_pkg import ALL_TOOLS as _ALL_TOOLS

# Eval / bench scripts (notably ``vllm_bench.py``) imported the old
# module-level ``TOOLS`` list of OpenAI tool schemas. Reconstruct the
# same shape from the Owela tool registry so those scripts keep
# working without rewrites.
TOOLS = ToolRegistry(list(_ALL_TOOLS)).schemas()


@dataclass
class LLMResult:
    """The shape the eval scripts expect. Maps 1:1 from Owela's
    ``HandleResult`` + the underlying steps list."""
    reply: str
    tokens_in: int
    tokens_out: int
    used_search: bool
    used_web_search: bool = False
    used_fetch_url: bool = False
    deleted_data: bool = False
    used_whats_in_my_memory: bool = False
    used_my_token_usage: bool = False
    used_lookup_ongiini_docs: bool = False


# Lazy singleton — built on first use, reused thereafter. Eval scripts
# may import this module without setting up the Runtime; we defer
# construction until ``respond`` is actually called.
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        from .ongiini_runtime import build_agent
        _agent = build_agent()
    return _agent


async def respond(history, user_content, msisdn: str) -> LLMResult:
    """Backwards-compat wrapper around ``Agent.handle``.

    Eval scripts use this signature: ``respond(history, text_or_multipart,
    msisdn) -> LLMResult``. New code should call ``agent.handle`` directly.
    """
    agent = _get_agent()

    if isinstance(user_content, str):
        text = user_content
        content_parts: list[dict] = [{"type": "text", "text": text}]
        has_image = False
    else:
        # Multipart with image — extract the text part for routing and
        # the full multipart for the model.
        text_parts = [
            (p.get("text") or "")
            for p in user_content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        text = " ".join(t for t in text_parts if t)
        content_parts = user_content
        has_image = any(
            isinstance(p, dict) and p.get("type") == "image_url"
            for p in user_content
        )

    msg = InboundMessage(
        user_id=msisdn,
        msg_id="",
        text=text,
        content_parts=content_parts,
        has_image=has_image,
        history=list(history),
    )
    result = await agent.handle(msg)

    # Derive the legacy LLMResult fields from the steps list.
    tool_names_fired: set[str] = set()
    tokens_in = 0
    tokens_out = 0
    for s in result.steps:
        if hasattr(s, "tool_name"):
            tn = getattr(s, "tool_name", "")
            if tn:
                tool_names_fired.add(tn)
        if hasattr(s, "tokens_in"):
            tokens_in += s.tokens_in
        if hasattr(s, "tokens_out"):
            tokens_out += s.tokens_out

    return LLMResult(
        reply=result.reply_text,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        used_search="web_search" in tool_names_fired
                    or "fetch_url" in tool_names_fired
                    or "fetch_urls" in tool_names_fired,
        used_web_search="web_search" in tool_names_fired,
        used_fetch_url="fetch_url" in tool_names_fired or "fetch_urls" in tool_names_fired,
        deleted_data="delete_my_data" in tool_names_fired,
        used_whats_in_my_memory="whats_in_my_memory" in tool_names_fired,
        used_my_token_usage="my_token_usage" in tool_names_fired,
        used_lookup_ongiini_docs="lookup_ongiini_docs" in tool_names_fired,
    )
