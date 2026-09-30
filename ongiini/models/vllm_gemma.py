"""Owela ``Model`` implementation for Gemma 4 26B served by vLLM.

What's special here, vs. just hitting AsyncOpenAI directly:

  1. **Prefix-cache–aware token reporting.** vLLM running with
     ``--enable-prompt-tokens-details`` populates
     ``usage.prompt_tokens_details.cached_tokens``. We subtract those
     from ``prompt_tokens`` so the static SYSTEM_PROMPT / product.md
     overhead doesn't keep eating the user's monthly allowance after
     the first warm-up request.

  2. **Gemma 4 reasoning knobs.** ``ModelRequest.thinking`` maps to
     ``chat_template_kwargs.enable_thinking`` (+ ``reasoning_budget``).
     Thinking gets its own token budget ON TOP of the reply budget, so
     reasoning can no longer consume the answer (the root cause of the
     2026-05-23 leak: a 500-token thinking budget inside a shared
     1500-token cap). ``thinking="low"`` also adds a one-line
     low-effort hint to the first system message, per Google's Gemma 4
     guidance.

  3. **One sanitising path.** Leaked channel tokens and truncated raw
     reasoning are scrubbed here, for every caller — the act loop and
     the auxiliary calls (classifier, planner, reviewer, summariser) all
     go through ``complete``.

  4. **Single-attempt semantics, bounded.** No internal retries, but a
     per-request timeout (default 90 s, overridable via
     ``ModelRequest.timeout_s``) so a hung engine surfaces as an error
     the executor can turn into a fallback reply instead of minutes of
     silence.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from openai import AsyncOpenAI

from owela import THINKING_LOW, THINKING_OFF, Model, ModelRequest, ModelResponse

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


# Production 2026-05-23 (turn 18:33): user asked "You speak Oshiwango?".
# DOCS policy → critique on → thinking on. Model started reasoning about
# which rule applies, hit max_tokens=1500 mid-thought, never emitted the
# closing thinking tag, and shipped 5570 chars of raw chain-of-thought
# directly to a user via WhatsApp. The special-token-based stripper
# above missed it because no tokens were ever emitted — the leak was
# pure raw thinking text in msg.content.
#
# This is a SECOND defensive layer: heuristically detect raw thinking
# WITHOUT relying on special tokens. Strict gating (multiple
# co-occurring signals required) to avoid eating legitimate replies.
_RAW_THINKING_OPENING_RE = re.compile(
    r"^\s*(?:"
    r"the\s+user\s+(?:is\s+asking|wants?|said|wrote|wrote\s+a|asked|previously|just|seems|appears)|"
    r"looking\s+at\s+(?:the|this|the\s+user)|"
    r"i\s+(?:should|will|need\s+to|must|have\s+to|am\s+going\s+to|'ll)\s+"
    r"(?:consider|think|reason|check|look|first|analyse|analyze|use|follow|respond|reply)|"
    r"let\s+me\s+(?:think|consider|reason|analyse|analyze|check|look)|"
    r"step\s+\d+|"
    r"first[,:]"
    r")",
    re.IGNORECASE,
)
_RAW_THINKING_DETRITUS_PATTERNS = (
    re.compile(r"self[-\s]?correction\s*:", re.IGNORECASE),
    re.compile(r"^\s*\*\s{3}", re.MULTILINE),     # "*   " bullet, indented reasoning style
    re.compile(r"^\s{2,}\*\s", re.MULTILINE),     # indented "*  " bullet
    re.compile(r"\b(?:instruction|skill\s+documentation|the\s+(?:rule|pattern|skill)\s+says)\b",
               re.IGNORECASE),
    re.compile(r"\b(?:wait\b.*?actually|but\s+wait\b|hmm[,.]\s+let\s+me)", re.IGNORECASE),
)


# Google's Gemma 4 guidance: a short low-effort instruction in the system
# turn cuts thinking tokens (~20 %) without switching thinking off.
_LOW_THINKING_HINT = "Reasoning effort: LOW. Think briefly, then answer."
_LOW_THINKING_MAX_BUDGET = 256

DEFAULT_TIMEOUT_S = 90.0


def _detect_truncated_thinking_leak(
    content: str,
    finish_reason: str,
    enable_thinking: bool,
) -> bool:
    """Heuristic detector for raw chain-of-thought that leaked into
    msg.content with NO special tokens to anchor on.

    Triggers only when MULTIPLE strong signals co-occur, so legitimate
    replies that happen to start with "The user said..." or contain a
    bulleted list don't get eaten.

    Strongest signal: ``finish_reason == "length"`` AND
    ``enable_thinking == True`` — means the model was thinking, got
    cut off mid-stream, and never finished. If the truncated content
    ALSO looks like reasoning detritus → almost certainly a leak.
    """
    if not content:
        return False
    truncated_while_thinking = (
        finish_reason == "length" and enable_thinking
    )
    starts_like_reasoning = bool(_RAW_THINKING_OPENING_RE.search(content[:200]))
    detritus_hits = sum(
        1 for p in _RAW_THINKING_DETRITUS_PATTERNS if p.search(content)
    )
    # Strict gating: only fire if the truncation signal is present AND
    # the content also looks like reasoning. Falls back to detritus-
    # heavy detection when the model thought-then-stopped-naturally
    # but still leaked (rarer but possible).
    if truncated_while_thinking and (starts_like_reasoning or detritus_hits >= 2):
        return True
    # Without the truncation signal, require very strong evidence.
    if starts_like_reasoning and detritus_hits >= 3:
        return True
    return False


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


def _with_low_thinking_hint(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` whose first system message ends with
    the low-effort thinking hint. Never mutates the caller's list — the
    executor owns it and appends to it across the act loop."""
    out = list(messages)
    for i, m in enumerate(out):
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            out[i] = {**m, "content": m["content"] + "\n\n" + _LOW_THINKING_HINT}
            return out
    return [{"role": "system", "content": _LOW_THINKING_HINT}] + out


class VLLMGemmaModel(Model):
    """Gemma 4 via vLLM's OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        temperature: float = 0.6,
        max_tokens: int = 1500,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.base_url = base_url
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Caller can inject a client for tests; default is a fresh AsyncOpenAI
        # with a bounded timeout (the SDK default is 600 s).
        self._client = client or AsyncOpenAI(
            base_url=base_url, api_key="not-needed", timeout=timeout_s,
        )

    async def complete(self, req: ModelRequest) -> ModelResponse:
        thinking_on = req.thinking != THINKING_OFF
        # vLLM 0.20 + gemma4 parser: with thinking on AND a named
        # tool_choice, the whole output can land in the reasoning channel;
        # content is then None and the tool-call parser dies on
        # `assert content is not None` (HTTP 500 — found in the 2026-09-30
        # eval). A forced call only has to pick arguments, so it runs
        # without thinking; the compose call after it still thinks.
        forced_call = bool(req.tools) and isinstance(req.tool_choice, dict)
        thinking_suppressed = thinking_on and forced_call
        if thinking_suppressed:
            thinking_on = False
        budget = req.thinking_budget if thinking_on else None
        if budget is not None and req.thinking == THINKING_LOW:
            budget = min(budget, _LOW_THINKING_MAX_BUDGET)
        # Gemma 4 reasoning knobs travel via extra_body so they reach the
        # vLLM chat template.
        chat_template_kwargs: dict[str, Any] = {"enable_thinking": thinking_on}
        if budget is not None:
            chat_template_kwargs["reasoning_budget"] = budget

        messages = req.messages
        if req.thinking == THINKING_LOW and thinking_on:
            messages = _with_low_thinking_hint(messages)

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature if req.temperature is None else req.temperature,
            # Thinking gets its own budget on top of the reply's.
            "max_tokens": (req.max_tokens or self.max_tokens) + (budget or 0),
            "extra_body": {"chat_template_kwargs": chat_template_kwargs},
        }
        if req.tools:
            kwargs["tools"] = req.tools
            kwargs["tool_choice"] = req.tool_choice
        if req.response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        if req.timeout_s is not None:
            kwargs["timeout"] = req.timeout_s

        resp = await self._client.chat.completions.create(**kwargs)

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
        # Only when thinking was requested: with thinking off, any reasoning
        # text is a stray thought channel (Gemma 4 occasionally emits one
        # anyway) and must never become the reply.
        if not content and thinking_on:
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

        finish_reason = getattr(choice, "finish_reason", "") if choice else ""

        attrs: dict[str, Any] = {}
        if thinking_suppressed:
            attrs["thinking_suppressed_for_forced_tool"] = True
        if leak_count_total > 0:
            # v1.4 audit: scrub detected leaked Gemma 4 channel tokens.
            # Surface the count so TracingHook can record it and the
            # `trace_query.py reasoning-leak-count` CLI can spot
            # recurrence without operators having to grep raw replies.
            attrs["reasoning_leak_stripped"] = leak_count_total

        # Second-layer defence (added 2026-05-23 after production leak
        # shipped 5570 chars of raw chain-of-thought via WhatsApp):
        # detect raw thinking text that leaked WITHOUT special tokens.
        # Most common cause: thinking-mode truncation by max_tokens —
        # the model never gets to emit its closing thinking tag, so
        # the special-token stripper has nothing to anchor on.
        # We only fire on no-tool-call replies (a leaked-thinking turn
        # that ALSO somehow produced tool calls is a much weirder beast
        # and we don't want to blow away its tool output).
        if not tool_calls and _detect_truncated_thinking_leak(
            content, finish_reason, thinking_on,
        ):
            # Drop the leaked reasoning. An empty reply makes the transport
            # send its own "couldn't come up with a reply" text; the trace
            # records the block so recurrence is visible.
            log.warning(
                "truncated/raw thinking detected in reply (finish_reason=%s, "
                "thinking=%s, content_len=%d) — dropping it to avoid leaking "
                "chain-of-thought to the user",
                finish_reason, req.thinking, len(content),
            )
            content = ""
            attrs["truncated_thinking_blocked"] = True

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            tokens_in=billable_in,
            tokens_out=completion,
            cached_tokens=cached,
            raw=resp,
            attrs=attrs,
        )
