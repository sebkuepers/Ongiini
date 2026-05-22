"""Backwards-compat shim for tests/eval scripts that pre-date Owela.

The old ``app.llm.respond(history, content, msisdn) -> LLMResult`` API
is gone (the live code path is ``ongiini.runtime.build_agent().handle``).
For tests that haven't been rewritten to use the Owela API directly,
this module recreates the old signature on top of the new runtime so
each script needs only a one-line import change.

New code should NOT use this — call ``Agent.handle`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from owela import InboundMessage, ToolStep


@dataclass
class LLMResult:
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


_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        from ongiini.runtime import build_agent
        _agent = build_agent()
    return _agent


async def respond(history, user_content, msisdn: str) -> LLMResult:
    agent = _get_agent()
    if isinstance(user_content, str):
        text = user_content
        content_parts: list[dict] = [{"type": "text", "text": text}]
        has_image = False
    else:
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

    fired: set[str] = set()
    tokens_in = tokens_out = 0
    for s in result.steps:
        if isinstance(s, ToolStep) and s.tool_name:
            fired.add(s.tool_name)
        tokens_in += getattr(s, "tokens_in", 0) or 0
        tokens_out += getattr(s, "tokens_out", 0) or 0

    return LLMResult(
        reply=result.reply_text,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        used_search=bool(fired & {"web_search", "fetch_url", "fetch_urls"}),
        used_web_search="web_search" in fired,
        used_fetch_url=("fetch_url" in fired) or ("fetch_urls" in fired),
        deleted_data="delete_my_data" in fired,
        used_whats_in_my_memory="whats_in_my_memory" in fired,
        used_my_token_usage="my_token_usage" in fired,
        used_lookup_ongiini_docs="lookup_ongiini_docs" in fired,
    )
