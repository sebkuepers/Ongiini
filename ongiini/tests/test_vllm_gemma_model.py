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
