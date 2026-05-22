"""Unit tests for OngiiniPlanner."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from owela import InboundMessage, PlanStep, Policy
from ongiini.planner import OngiiniPlanner


def _msg(text: str = "compare home loan rates at 3 banks") -> InboundMessage:
    return InboundMessage(
        user_id="+264u", msg_id="m", text=text,
        content_parts=[{"type": "text", "text": text}],
    )


def _client_returning(content: str, prompt_tokens: int = 300, cached: int = 280) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock(message=msg)
    resp = MagicMock(choices=[choice])
    resp.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=180)
    resp.usage.prompt_tokens_details.cached_tokens = cached
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


@pytest.mark.asyncio
async def test_planner_returns_planstep_with_text():
    plan_body = (
        "FACTS I ALREADY KNOW:\n"
        "- Namibia has 4 commercial banks (Bank Windhoek, FNB, Standard Bank, Nedbank).\n\n"
        "FACTS TO LOOK UP:\n"
        "- Current home loan rates at each bank.\n"
        "- Whether rates vary by tenure / down payment.\n\n"
        "SEARCH PLAN:\n"
        "- Search 'Namibia home loan rates 2026'; if thin, search each bank individually.\n"
        "PLAN_DONE\n"
    )
    client = _client_returning(plan_body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    step = await planner.plan(_msg(), Policy(name="search_deep"), [])

    assert isinstance(step, PlanStep)
    assert "FACTS I ALREADY KNOW" in step.plan_text
    assert "PLAN_DONE" not in step.plan_text   # sentinel stripped
    assert step.tokens_in == 20                # 300 - 280 cached
    assert step.tokens_out == 180
    assert step.cached_tokens == 280
    assert step.ended_at is not None


@pytest.mark.asyncio
async def test_planner_strips_content_after_sentinel():
    """Sometimes Gemma keeps writing after PLAN_DONE — drop the tail."""
    body = "FACTS I ALREADY KNOW:\n- thing\nPLAN_DONE\nThen something extra here that shouldn't appear."
    client = _client_returning(body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    step = await planner.plan(_msg(), Policy(name="search_deep"), [])
    assert "extra here" not in step.plan_text
    assert "thing" in step.plan_text


@pytest.mark.asyncio
async def test_planner_empty_text_returns_empty_plan_without_call():
    """Defensive: no question = no plan, no LLM call."""
    client = _client_returning("FACTS...PLAN_DONE")
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    step = await planner.plan(_msg(text=""), Policy(name="search_deep"), [])
    assert step.plan_text == ""
    client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_planner_timeout_returns_empty_plan():
    async def slow(*args, **kwargs):
        await asyncio.sleep(10.0)
        return None

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = slow
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client, timeout_s=0.05)
    step = await planner.plan(_msg(), Policy(name="search_deep"), [])
    assert step.plan_text == ""
    assert step.attrs.get("error") == "timeout"


@pytest.mark.asyncio
async def test_planner_exception_returns_empty_plan():
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("vllm down"))
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    step = await planner.plan(_msg(), Policy(name="search_deep"), [])
    assert step.plan_text == ""
    assert "vllm down" in step.attrs.get("error", "")


@pytest.mark.asyncio
async def test_planner_prompt_contains_question_and_sentinel_instruction():
    """Lock the prompt contract: the question goes through, the PLAN_DONE
    sentinel instruction is present, max_tokens is set."""
    client = _client_returning("ok\nPLAN_DONE")
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    await planner.plan(_msg("compare three banks"), Policy(name="search_deep"), [])

    call_kwargs = client.chat.completions.create.call_args.kwargs
    sent = call_kwargs["messages"][0]["content"]
    assert "compare three banks" in sent
    assert "PLAN_DONE" in sent
    assert call_kwargs["max_tokens"] == 280


@pytest.mark.asyncio
async def test_planner_prompt_includes_tool_plan_section():
    """The TOOL PLAN section is the v1.2 fix for Gemma not escalating
    to fetch_url / fetch_urls on its own. The prompt must explicitly
    describe when to use each tool by question shape."""
    client = _client_returning("ok\nPLAN_DONE")
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    await planner.plan(_msg("compare three banks"), Policy(name="search_deep"), [])
    sent = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    # All four sections must be in the prompt template.
    assert "FACTS I ALREADY KNOW:" in sent
    assert "FACTS TO LOOK UP:" in sent
    assert "SEARCH PLAN:" in sent
    assert "TOOL PLAN:" in sent
    # And the specific tool-escalation guidance shapes are present.
    assert "fetch_urls" in sent and "fetch_url" in sent
    assert "COMPARISON" in sent
    assert "VERBATIM" in sent or "SPECIFIC DATA" in sent
