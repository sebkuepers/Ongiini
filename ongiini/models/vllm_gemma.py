"""Owela ``Model`` implementation for Gemma 4 26B served by vLLM.

What's special here, vs. just hitting AsyncOpenAI directly:

  1. **Prefix-cache–aware token reporting.** vLLM running with
     ``--enable-prompt-tokens-details`` populates
     ``usage.prompt_tokens_details.cached_tokens``. We subtract those
     from ``prompt_tokens`` so the static SYSTEM_PROMPT / product.md
     overhead doesn't keep eating the user's monthly allowance after
     the first warm-up request.

  2. **Gemma 4 reasoning knobs.** ``enable_thinking`` and
     ``reasoning_budget`` are passed via ``extra_body.chat_template_kwargs``.
     The Policy decides whether to enable thinking on this turn; the
     adapter just propagates.

  3. **Single-attempt semantics.** No internal retries. The webhook's
     fire-and-forget pattern + Meta's webhook redelivery handle the
     retry case at the right layer.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from openai import AsyncOpenAI

from owela import Model, ModelRequest, ModelResponse

log = logging.getLogger("ongiini.models.vllm_gemma")


# Gemma 4 chat-template special tokens. These should be stripped or
# extracted into ``msg.reasoning`` by vLLM's ``--reasoning-parser gemma4``
# — but live trace 2026-05-23 caught the parser letting them through
# into ``msg.content``. Forms observed in the wild:
#   - ``<|channel|>``        (double-pipe variant — standard form)
#   - ``<|channel>``         (single-pipe + ``>`` — partial / malformed)
#   - ``<channel|>`` etc.
# The pattern matches any ``<|...>`` or ``<...|>`` shape; we don't
# enforce the closing-pipe so partial tokens get cleaned too.
_GEMMA_SPECIAL_TOKEN_RE = re.compile(r"<\|?[a-zA-Z0-9_\-]+\|?>")


def _strip_gemma_reasoning_leak(text: str) -> tuple[str, int]:
    """Remove Gemma 4 special tokens from a content string and, if any
    were present, also strip a leaked reasoning preamble (everything up
    to the first paragraph break) when the cleaned text starts with
    obvious thought-channel detritus.

    Returns ``(cleaned_text, num_tokens_stripped)``. The integer count
    is 0 on a clean reply, positive when the scrub fired — the model
    adapter forwards it via ``ModelResponse.attrs["reasoning_leak_stripped"]``
    so the TracingHook can surface it in the per-turn trace and
    operators can monitor recurrence without grepping raw replies.

    The preamble-strip is gated on having seen at least one ``<|...|>``
    token in the input — that's the strong signal that a reasoning
    leak occurred, so we trust the heuristic. A legitimate reply that
    happens to start with the word "thought" but had no special tokens
    won't be touched.
    """
    if not text:
        return text, 0
    matches = _GEMMA_SPECIAL_TOKEN_RE.findall(text)
    leak_count = len(matches)
    cleaned = _GEMMA_SPECIAL_TOKEN_RE.sub("", text).strip()
    if leak_count > 0:
        # The reasoning leak pattern starts with "thought" or "Wait" or
        # similar reasoning preamble, then a paragraph break, then the
        # actual answer. Find the first ``\n\n`` and discard the
        # preamble. If no clean break exists, leave the (still-token-
        # stripped) text — don't risk eating the whole reply.
        head_lower = cleaned[:50].lower()
        if (
            head_lower.startswith("thought")
            or head_lower.startswith("wait")
            or head_lower.startswith("(wait")
        ):
            idx = cleaned.find("\n\n")
            if idx != -1:
                cleaned = cleaned[idx + 2:].lstrip()
    return cleaned, leak_count


def _billable_from_usage(usage_obj: Any) -> tuple[int, int, int]:
    """Extract (billable_in, completion, cached) from a vLLM/OpenAI usage object.

    Returns ``(0, 0, 0)`` when ``usage_obj`` is None. Falls back gracefully
    when ``prompt_tokens_details`` is absent (non-vLLM backends or vLLM
    without ``--enable-prompt-tokens-details``)."""
    if usage_obj is None:
        return 0, 0, 0
    prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
    cached = 0
    details = getattr(usage_obj, "prompt_tokens_details", None)
    if details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    billable_in = max(0, prompt_tokens - cached)
    return billable_in, completion_tokens, cached


class VLLMGemmaModel(Model):
    """Gemma 4 via vLLM's OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        temperature: float = 0.6,
        max_tokens: int = 1500,
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.base_url = base_url
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Caller can inject a client for tests; default is a fresh AsyncOpenAI.
        self._client = client or AsyncOpenAI(base_url=base_url, api_key="not-needed")

    async def complete(self, req: ModelRequest) -> ModelResponse:
        # Gemma 4 reasoning knobs travel via extra_body so they reach the
        # vLLM chat template.
        chat_template_kwargs: dict[str, Any] = {"enable_thinking": req.enable_thinking}
        if req.enable_thinking and req.policy.reasoning_budget is not None:
            chat_template_kwargs["reasoning_budget"] = req.policy.reasoning_budget

        resp = await self._client.chat.completions.create(
            model=self.model_id,
            messages=req.messages,
            tools=req.tools or None,
            tool_choice=req.tool_choice,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"chat_template_kwargs": chat_template_kwargs},
        )

        billable_in, completion, cached = _billable_from_usage(resp.usage)
        choice = resp.choices[0] if resp.choices else None
        msg = choice.message if choice else None

        # Content: prefer .content; if empty (Gemma reasoning mode ate
        # the budget), fall back to the reasoning text so the user gets
        # *something* instead of "Sorry, couldn't reply". The executor
        # itself decides what to do when content is empty; here we just
        # surface what we have.
        #
        # vLLM's --reasoning-parser gemma4 puts the chain-of-thought in
        # ``msg.reasoning`` (NOT ``reasoning_content``). Older vLLM/parser
        # combinations expose it as ``reasoning_content``; check both
        # for forwards/backwards compatibility, and also peek at
        # ``model_extra`` since the OpenAI SDK stores unknown response
        # fields there if the typed model didn't define them.
        content = (getattr(msg, "content", "") or "").strip()
        # Defensive scrub: Gemma 4 reasoning channel tokens (and the
        # "thought" preamble they sometimes drag with them) can leak
        # into ``msg.content`` when vLLM's --reasoning-parser gemma4
        # mis-routes them. Live trace 2026-05-23 showed
        # ``thought<|channel><|channel>thought la l'une des langues...``
        # in front of an otherwise excellent reply. Strip the special
        # tokens unconditionally; only strip the preamble paragraph
        # when we ALSO detected leaked tokens (to avoid eating
        # legitimate replies that happen to start with "thought").
        # v1.4 audit: track count so the trace can show recurrence.
        content, leak_count_primary = _strip_gemma_reasoning_leak(content)
        leak_count_total = leak_count_primary
        if not content:
            reasoning = ""
            for candidate in (
                getattr(msg, "reasoning", None),
                getattr(msg, "reasoning_content", None),
            ):
                if candidate:
                    reasoning = str(candidate).strip()
                    if reasoning:
                        break
            if not reasoning:
                extra = getattr(msg, "model_extra", None) or {}
                for k in ("reasoning", "reasoning_content"):
                    v = extra.get(k) if isinstance(extra, dict) else None
                    if v:
                        reasoning = str(v).strip()
                        if reasoning:
                            break
            if reasoning:
                log.warning(
                    "Gemma returned empty content with %d chars of reasoning — "
                    "surfacing reasoning text as content",
                    len(reasoning),
                )
                content, leak_count_fallback = _strip_gemma_reasoning_leak(reasoning)
                leak_count_total += leak_count_fallback

        tool_calls_raw = getattr(msg, "tool_calls", None) or []
        tool_calls: list[dict[str, Any]] = []
        for tc in tool_calls_raw:
            fn = getattr(tc, "function", None)
            tool_calls.append({
                "id": getattr(tc, "id", ""),
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": getattr(fn, "name", "") if fn else "",
                    "arguments": getattr(fn, "arguments", "") if fn else "",
                },
            })

        attrs: dict[str, Any] = {}
        if leak_count_total > 0:
            # v1.4 audit: scrub detected leaked Gemma 4 channel tokens.
            # Surface the count so TracingHook can record it and the
            # `trace_query.py reasoning-leak-count` CLI can spot
            # recurrence without operators having to grep raw replies.
            attrs["reasoning_leak_stripped"] = leak_count_total

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", "") if choice else "",
            tokens_in=billable_in,
            tokens_out=completion,
            cached_tokens=cached,
            raw=resp,
            attrs=attrs,
        )
