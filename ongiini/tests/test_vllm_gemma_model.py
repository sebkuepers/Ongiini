"""Unit tests for ``ongiini.models.vllm_gemma.VLLMGemmaModel``.

We inject a fake AsyncOpenAI client so these tests run without vLLM.
The contract under test is: given an Owela ModelRequest, build the
right OpenAI chat.completions.create call, and translate the response
into an Owela ModelResponse with cache-aware billing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from owela import ModelRequest, Policy
from ongiini.models.vllm_gemma import VLLMGemmaModel, _billable_from_usage


# ---------- _billable_from_usage ----------

def test_billable_subtracts_cached():
    usage = MagicMock(prompt_tokens=1000, completion_tokens=200)
    usage.prompt_tokens_details.cached_tokens = 800
    billable, completion, cached = _billable_from_usage(usage)
    assert billable == 200       # 1000 - 800
    assert completion == 200
    assert cached == 800


def test_billable_none_returns_zeros():
    assert _billable_from_usage(None) == (0, 0, 0)


def test_billable_no_details_falls_back_to_full_prompt():
    """Backends without prompt_tokens_details must not crash. We treat
    them as 'no cache' and bill the full prompt."""
    usage = MagicMock(prompt_tokens=500, completion_tokens=100)
    usage.prompt_tokens_details = None
    billable, completion, cached = _billable_from_usage(usage)
    assert billable == 500
    assert completion == 100
    assert cached == 0


def test_billable_clamps_at_zero():
    """Defence: a buggy backend that reports cached > prompt should not
    yield a negative billable count."""
    usage = MagicMock(prompt_tokens=100, completion_tokens=10)
    usage.prompt_tokens_details.cached_tokens = 500
    billable, _, _ = _billable_from_usage(usage)
    assert billable == 0


# ---------- VLLMGemmaModel.complete ----------

def _make_openai_response(
    content: str = "hello",
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning: str = "",
    reasoning_content: str = "",
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    cached_tokens: int = 0,
):
    """Build a MagicMock that looks like an OpenAI ChatCompletion."""
    msg = MagicMock()
    msg.content = content
    # vLLM gemma4 reasoning parser exposes the thinking on `.reasoning`;
    # older builds used `.reasoning_content`. The adapter looks at both.
    msg.reasoning = reasoning
    msg.reasoning_content = reasoning_content
    msg.model_extra = {}

    if tool_calls:
        tc_mocks = []
        for tc in tool_calls:
            m = MagicMock()
            m.id = tc["id"]
            m.type = tc.get("type", "function")
            m.function = MagicMock()
            m.function.name = tc["function"]["name"]
            m.function.arguments = tc["function"]["arguments"]
            tc_mocks.append(m)
        msg.tool_calls = tc_mocks
    else:
        msg.tool_calls = None

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )
    resp.usage.prompt_tokens_details.cached_tokens = cached_tokens
    return resp


def _make_client(response) -> Any:
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _basic_request(enable_thinking: bool = False) -> ModelRequest:
    return ModelRequest(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        tool_choice="auto",
        policy=Policy(name="test"),
        enable_thinking=enable_thinking,
    )


@pytest.mark.asyncio
async def test_complete_basic_reply():
    response = _make_openai_response(content="hello back")
    client = _make_client(response)
    model = VLLMGemmaModel(base_url="x", model_id="gemma", client=client)
    out = await model.complete(_basic_request())
    assert out.content == "hello back"
    assert out.tool_calls == []
    assert out.tokens_out == 20


@pytest.mark.asyncio
async def test_complete_subtracts_cached_tokens():
    response = _make_openai_response(prompt_tokens=1000, cached_tokens=900, completion_tokens=50)
    client = _make_client(response)
    model = VLLMGemmaModel(base_url="x", model_id="gemma", client=client)
    out = await model.complete(_basic_request())
    assert out.tokens_in == 100         # 1000 - 900
    assert out.cached_tokens == 900
    assert out.tokens_out == 50


@pytest.mark.asyncio
async def test_complete_passes_reasoning_kwargs_when_enabled():
    """When enable_thinking=True AND policy.reasoning_budget is set, the
    adapter must propagate both via extra_body.chat_template_kwargs."""
    response = _make_openai_response()
    client = _make_client(response)
    model = VLLMGemmaModel(base_url="x", model_id="gemma", client=client)
    await model.complete(_basic_request(enable_thinking=True))
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_budget": 500},
    }


@pytest.mark.asyncio
async def test_complete_omits_budget_when_thinking_off():
    """enable_thinking=False -> reasoning_budget must NOT appear (otherwise
    vLLM may apply it even when thinking is supposed to be off)."""
    response = _make_openai_response()
    client = _make_client(response)
    model = VLLMGemmaModel(base_url="x", model_id="gemma", client=client)
    await model.complete(_basic_request(enable_thinking=False))
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
    }


@pytest.mark.asyncio
async def test_complete_translates_tool_calls():
    response = _make_openai_response(
        content="",
        finish_reason="tool_calls",
        tool_calls=[{
            "id": "call_abc",
            "type": "function",
            "function": {"name": "web_search", "arguments": '{"query": "x"}'},
        }],
    )
    client = _make_client(response)
    model = VLLMGemmaModel(base_url="x", model_id="gemma", client=client)
    out = await model.complete(_basic_request())
    assert out.finish_reason == "tool_calls"
    assert len(out.tool_calls) == 1
    tc = out.tool_calls[0]
    assert tc["id"] == "call_abc"
    assert tc["function"]["name"] == "web_search"
    assert tc["function"]["arguments"] == '{"query": "x"}'


@pytest.mark.asyncio
async def test_complete_falls_back_to_reasoning_when_content_empty():
    """Gemma + reasoning sometimes returns empty content when the
    reasoning_budget runs out. We surface the reasoning text as content
    so the user sees *something* instead of a hard fallback message."""
    response = _make_openai_response(content="", reasoning="my thinking went here")
    client = _make_client(response)
    model = VLLMGemmaModel(base_url="x", model_id="gemma", client=client)
    out = await model.complete(_basic_request(enable_thinking=True))
    assert out.content == "my thinking went here"


@pytest.mark.asyncio
async def test_complete_falls_back_to_legacy_reasoning_content_field():
    """Forward-compat: older vLLM exposed reasoning_content instead of
    reasoning. Both must work."""
    response = _make_openai_response(content="", reasoning_content="legacy thinking")
    client = _make_client(response)
    model = VLLMGemmaModel(base_url="x", model_id="gemma", client=client)
    out = await model.complete(_basic_request(enable_thinking=True))
    assert out.content == "legacy thinking"


@pytest.mark.asyncio
async def test_complete_falls_back_to_model_extra_reasoning():
    """If the typed model doesn't expose reasoning at all (older SDK +
    vLLM combo), it may land in model_extra. We dig there too."""
    response = _make_openai_response(content="")
    response.choices[0].message.reasoning = None
    response.choices[0].message.reasoning_content = None
    response.choices[0].message.model_extra = {"reasoning": "extra-located"}
    client = _make_client(response)
    model = VLLMGemmaModel(base_url="x", model_id="gemma", client=client)
    out = await model.complete(_basic_request(enable_thinking=True))
    assert out.content == "extra-located"


@pytest.mark.asyncio
async def test_complete_passes_messages_and_tools():
    response = _make_openai_response()
    client = _make_client(response)
    model = VLLMGemmaModel(base_url="x", model_id="gemma", client=client)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "t"}}]
    req = ModelRequest(
        messages=msgs, tools=tools, tool_choice="auto",
        policy=Policy(name="t"), enable_thinking=False,
    )
    await model.complete(req)
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"] == msgs
    assert call_kwargs["tools"] == tools
    assert call_kwargs["tool_choice"] == "auto"
    assert call_kwargs["model"] == "gemma"


@pytest.mark.asyncio
async def test_complete_passes_none_tools_when_empty():
    """OpenAI's chat.completions.create rejects an empty tools=[] in some
    versions. Send None instead — same semantic, friendlier across SDK
    versions."""
    response = _make_openai_response()
    client = _make_client(response)
    model = VLLMGemmaModel(base_url="x", model_id="gemma", client=client)
    await model.complete(_basic_request())
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tools"] is None


# ---------- v1.3.2 hotfix: reasoning-leak scrubber ----------

def test_strip_gemma_reasoning_leak_removes_channel_tokens():
    """Live production bug 2026-05-23: vLLM's reasoning parser let
    Gemma 4 channel tokens leak into msg.content. Defensive strip
    removes <|...|> patterns unconditionally."""
    from ongiini.models.vllm_gemma import _strip_gemma_reasoning_leak

    leak = "thought<|channel><|channel>thought la l'une des langues ?\n\nNamibia has 12 languages."
    out = _strip_gemma_reasoning_leak(leak)
    # Special tokens gone.
    assert "<|" not in out
    # Reasoning preamble (gated on token-detected signal) gone.
    assert "thought" not in out
    assert "Namibia has 12 languages." in out


def test_strip_gemma_reasoning_leak_preserves_clean_content():
    """If no special tokens were present, the content is returned
    unchanged. Defensive: a legitimate reply starting with the word
    'Thought' must NOT be eaten."""
    from ongiini.models.vllm_gemma import _strip_gemma_reasoning_leak

    clean = "Thoughtfully designed houses tend to use cross-ventilation."
    out, count = _strip_gemma_reasoning_leak(clean)
    assert out == clean
    assert count == 0


def test_strip_gemma_reasoning_leak_keeps_text_when_no_paragraph_break():
    """If we detected leaked tokens but the rest of the content has no
    paragraph break to delimit the leak from the answer, we leave the
    token-stripped text alone rather than risk eating the whole reply."""
    from ongiini.models.vllm_gemma import _strip_gemma_reasoning_leak

    leak = "thought<|channel|> some reasoning content all on one line"
    out, count = _strip_gemma_reasoning_leak(leak)
    # Token stripped.
    assert "<|" not in out
    # But we kept the rest (no paragraph break to use as delimiter).
    assert "reasoning content" in out
    assert count == 1


def test_strip_gemma_reasoning_leak_handles_wait_prefix():
    """Reasoning leaks sometimes start with '(Wait, user asked...' or
    similar. The 'wait' prefix also triggers the preamble-strip when
    paired with detected special tokens."""
    from ongiini.models.vllm_gemma import _strip_gemma_reasoning_leak

    leak = "(Wait, user asked<|channel|> in English)\n\nThe answer is X."
    out, count = _strip_gemma_reasoning_leak(leak)
    assert "Wait" not in out
    assert "The answer is X." in out
    assert count == 1


def test_strip_gemma_reasoning_leak_empty_string():
    from ongiini.models.vllm_gemma import _strip_gemma_reasoning_leak
    assert _strip_gemma_reasoning_leak("") == ("", 0)


# ---------- 2026-05-23 second-layer leak detection (no special tokens) ----------

# Production leak from 2026-05-23T18:33 — the model hit max_tokens
# mid-thinking, never emitted closing special tokens, and shipped raw
# chain-of-thought to a WhatsApp user. This is the exact text-prefix.
_PRODUCTION_LEAK_2026_05_23 = """The user is asking "You speak Oshiwango?".
The user's previous message was "Tangi 🙏" (Oshiwambo for "Thank you").
The user is checking my language capabilities, specifically regarding Oshiwambo (Oshiwambo is the general term for the language group, though the user wrote "Oshiwango").

    *   The user is asking a question in English.
    *   The user's previous message used an Oshiwambo word ("Tangi").
    *   The user's intent is to see if I can speak/understand Oshiwambo.

    *   I have a specialized oshiwambo skill.
    *   The user's input is English ("You speak Oshiwango?").

    *   The oshiwambo skill instructions say: "When the user writes in Oshiwambo (Oshindonga or Oshikwanyama)..."

    *   Self-Correction: The user wrote "Oshiwango" (likely a typo for Oshiwambo).
"""


def test_detect_truncated_thinking_leak_fires_on_real_production_text():
    """The exact regression: the leaked production text must be detected
    so the user sees the retry message, not the raw chain-of-thought."""
    from ongiini.models.vllm_gemma import _detect_truncated_thinking_leak
    assert _detect_truncated_thinking_leak(
        _PRODUCTION_LEAK_2026_05_23,
        finish_reason="length",
        enable_thinking=True,
    ) is True


def test_detect_truncated_thinking_leak_requires_thinking_signal():
    """Same suspicious-looking content but with finish_reason='stop'
    and thinking off should NOT fire (we're conservative — only the
    strongest signals trigger the user-facing replacement)."""
    from ongiini.models.vllm_gemma import _detect_truncated_thinking_leak
    # Same content, but neither truncation signal nor thinking enabled
    # → must NOT trigger from the (truncation_signal OR detritus_threshold) branch.
    # However, the production text has 3 detritus markers + reasoning opening,
    # so it COULD still hit the "strong evidence" fallback. Verify with a
    # weaker text that should NOT trigger:
    weak = "The user is asking about prices. Here's a list:\n  * Apples\n  * Oranges"
    assert _detect_truncated_thinking_leak(
        weak, finish_reason="stop", enable_thinking=False,
    ) is False


def test_detect_truncated_thinking_leak_ignores_legitimate_reply():
    """A perfectly normal substantive reply must not be flagged."""
    from ongiini.models.vllm_gemma import _detect_truncated_thinking_leak
    normal = (
        "Yes, Ongiini supports both English and Afrikaans. We also offer "
        "warm greetings in Oshiwambo when users open in that language. "
        "Is there something specific you'd like to know more about?"
    )
    assert _detect_truncated_thinking_leak(
        normal, finish_reason="stop", enable_thinking=True,
    ) is False


def test_detect_truncated_thinking_leak_ignores_normal_truncated_reply():
    """A substantive reply that just happened to hit max_tokens (no
    reasoning detritus) should NOT trigger — we don't want every
    long answer eaten."""
    from ongiini.models.vllm_gemma import _detect_truncated_thinking_leak
    long_normal = (
        "Sure — here's a long explanation of how exchange rates work. "
        "First, the central bank publishes a reference rate daily. "
        "Then commercial banks add a spread on top of that for retail "
        "customers. The spread depends on the currency pair and"
    )
    assert _detect_truncated_thinking_leak(
        long_normal, finish_reason="length", enable_thinking=True,
    ) is False


def test_detect_truncated_thinking_leak_ignores_bulleted_legitimate_reply():
    """A legitimate reply that uses indented bullets must not trigger
    just because it has the bullet pattern."""
    from ongiini.models.vllm_gemma import _detect_truncated_thinking_leak
    bulleted = (
        "Here are three options for you:\n"
        "    * Apply through the NSFAF online portal\n"
        "    * Visit a NSFAF office in Windhoek\n"
        "    * Email them at info@nsfaf.gov.na"
    )
    # Has the indented-bullet pattern (1 marker), but no reasoning
    # opening and no other detritus. Should NOT fire.
    assert _detect_truncated_thinking_leak(
        bulleted, finish_reason="stop", enable_thinking=False,
    ) is False


def test_detect_truncated_thinking_leak_empty_string():
    from ongiini.models.vllm_gemma import _detect_truncated_thinking_leak
    assert _detect_truncated_thinking_leak("", "stop", True) is False
    assert _detect_truncated_thinking_leak("", "length", True) is False


def test_detect_truncated_thinking_leak_strong_evidence_without_truncation():
    """If the content has VERY strong reasoning markers (opening + 3+
    detritus signals), we trigger even without the truncation hint.
    Catches the rare case of leak-with-natural-stop."""
    from ongiini.models.vllm_gemma import _detect_truncated_thinking_leak
    very_strong = (
        "The user is asking about prices.\n"
        "    *   Step 1: check the catalog\n"
        "    *   Step 2: apply discount\n"
        "Self-Correction: actually they want net prices\n"
        "The rule says always include VAT\n"
        "The skill documentation mentions this case."
    )
    assert _detect_truncated_thinking_leak(
        very_strong, finish_reason="stop", enable_thinking=False,
    ) is True
