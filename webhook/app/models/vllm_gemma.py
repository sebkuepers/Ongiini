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
from typing import Any

from openai import AsyncOpenAI

from owela import Model, ModelRequest, ModelResponse

log = logging.getLogger("ongiini.models.vllm_gemma")


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
        # the budget), fall back to reasoning_content so the user gets
        # *something* instead of "Sorry, couldn't reply". The executor
        # itself decides what to do when content is empty; here we just
        # surface what we have.
        content = (getattr(msg, "content", "") or "").strip()
        if not content:
            reasoning = (getattr(msg, "reasoning_content", "") or "").strip()
            if reasoning:
                log.warning(
                    "Gemma returned empty content with %d chars of reasoning — "
                    "surfacing reasoning text as content",
                    len(reasoning),
                )
                content = reasoning

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

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", "") if choice else "",
            tokens_in=billable_in,
            tokens_out=completion,
            cached_tokens=cached,
            raw=resp,
        )
