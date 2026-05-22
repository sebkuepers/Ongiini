"""Unit tests for OngiiniReviewer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from owela import CritiqueStep, InboundMessage, Policy, ReviseStep, ToolStep
from ongiini.reviewer import OngiiniReviewer, _extract_fail_reasons


def _msg(text: str = "compare bank rates") -> InboundMessage:
    return InboundMessage(
        user_id="+264u", msg_id="m", text=text,
        content_parts=[{"type": "text", "text": text}],
    )


def _client_returning(content: str, prompt_tokens: int = 600, cached: int = 500, completion_tokens: int = 80) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock(message=msg)
    resp = MagicMock(choices=[choice])
    resp.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    resp.usage.prompt_tokens_details.cached_tokens = cached
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


def _tool_step(name: str, result: str = "search result here") -> ToolStep:
    s = ToolStep(tool_name=name, result_len=len(result))
    s.attrs["result"] = result
    return s


# ---------- critique() ----------

@pytest.mark.asyncio
async def test_critique_parses_pass_verdict():
    body = (
        "1. OK\n2. OK\n3. OK\n4. OK\n5. OK\n6. OK\n"
        "\nVERDICT: PASS\n"
    )
    client = _client_returning(body)
    rev = OngiiniReviewer(base_url="x", model_id="g", client=client)
    step = await rev.critique(_msg(), "draft text", [], Policy(name="search_deep"))
    assert isinstance(step, CritiqueStep)
    assert step.verdict == "PASS"
    assert step.reasons == []
    assert step.tokens_in == 100   # 600 - 500


@pytest.mark.asyncio
async def test_critique_parses_revise_with_reasons():
    body = (
        "1. OK\n"
        "2. FAIL: claims a hackathon in October 2025 but tool results don't mention it\n"
        "3. OK\n"
        "4. FAIL: search came back thin but draft pretends to be confident\n"
        "5. OK\n"
        "6. OK\n"
        "\nVERDICT: REVISE\n"
    )
    client = _client_returning(body)
    rev = OngiiniReviewer(base_url="x", model_id="g", client=client)
    step = await rev.critique(
        _msg("hackathons in Windhoek?"),
        "There is a hackathon on Oct 15 2025...",
        [_tool_step("web_search", "tavily came back empty")],
        Policy(name="search_deep"),
    )
    assert step.verdict == "REVISE"
    assert len(step.reasons) == 2
    assert "October 2025" in step.reasons[0]
    assert "thin" in step.reasons[1]


@pytest.mark.asyncio
async def test_critique_empty_draft_short_circuits_to_pass():
    """No draft = nothing to critique. The transport will use its own
    empty-body fallback message."""
    client = _client_returning("VERDICT: REVISE")
    rev = OngiiniReviewer(base_url="x", model_id="g", client=client)
    step = await rev.critique(_msg(), "   ", [], Policy(name="search_deep"))
    assert step.verdict == "PASS"
    client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_critique_unparseable_output_defaults_to_pass():
    """If we can't find a VERDICT line, ship the draft. Don't block on a
    flaky reviewer."""
    client = _client_returning("the critique was eaten by reasoning tokens...")
    rev = OngiiniReviewer(base_url="x", model_id="g", client=client)
    step = await rev.critique(_msg(), "draft", [], Policy(name="search_deep"))
    assert step.verdict == "PASS"


@pytest.mark.asyncio
async def test_critique_timeout_falls_back_to_pass():
    async def slow(*args, **kwargs):
        await asyncio.sleep(10.0)
        return None
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = slow
    rev = OngiiniReviewer(base_url="x", model_id="g", client=client, critique_timeout_s=0.05)
    step = await rev.critique(_msg(), "draft", [], Policy(name="search_deep"))
    assert step.verdict == "PASS"
    assert step.attrs.get("error") == "timeout"


@pytest.mark.asyncio
async def test_critique_exception_falls_back_to_pass():
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("vllm down"))
    rev = OngiiniReviewer(base_url="x", model_id="g", client=client)
    step = await rev.critique(_msg(), "draft", [], Policy(name="search_deep"))
    assert step.verdict == "PASS"
    assert "vllm down" in step.attrs.get("error", "")


@pytest.mark.asyncio
async def test_critique_includes_tool_results_in_prompt():
    """The prompt MUST include the tool_results block so the critique
    LLM can actually verify claims against the ground truth."""
    client = _client_returning("VERDICT: PASS")
    rev = OngiiniReviewer(base_url="x", model_id="g", client=client)
    await rev.critique(
        _msg("ex rate?"),
        "Today the USD/NAD rate is 18.50.",
        [_tool_step("web_search", "USD/NAD: 18.42 on 2026-05-22")],
        Policy(name="search_shallow"),
    )
    sent = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "USD/NAD: 18.42" in sent
    assert "Today the USD/NAD rate is 18.50." in sent
    assert "web_search" in sent


@pytest.mark.asyncio
async def test_critique_truncates_long_tool_results():
    """The prompt must cap each tool result so a giant fetch_url doesn't
    blow up the critique prompt size."""
    big_result = "x" * 5000
    client = _client_returning("VERDICT: PASS")
    rev = OngiiniReviewer(base_url="x", model_id="g", client=client)
    await rev.critique(
        _msg(), "draft",
        [_tool_step("fetch_url", big_result)],
        Policy(name="search_shallow"),
    )
    sent = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "[…truncated]" in sent
    # Should NOT contain the full 5000 x's
    assert sent.count("x") < 3000


# ---------- revise() ----------

@pytest.mark.asyncio
async def test_revise_returns_revised_text():
    client = _client_returning("Here's a corrected version of the reply.")
    rev = OngiiniReviewer(base_url="x", model_id="g", client=client)
    critique = CritiqueStep(verdict="REVISE", reasons=["claim X not grounded"])
    step = await rev.revise(_msg(), "original draft", critique, [], Policy(name="search_deep"))
    assert isinstance(step, ReviseStep)
    assert step.attrs["revised_reply"] == "Here's a corrected version of the reply."


@pytest.mark.asyncio
async def test_revise_timeout_falls_back_to_original():
    """If revise times out, the user gets the original draft — better
    than the empty-body fallback."""
    async def slow(*args, **kwargs):
        await asyncio.sleep(20.0)
        return None
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = slow
    rev = OngiiniReviewer(base_url="x", model_id="g", client=client, revise_timeout_s=0.05)
    critique = CritiqueStep(verdict="REVISE", reasons=["x"])
    step = await rev.revise(_msg(), "original", critique, [], Policy(name="search_deep"))
    assert step.attrs["revised_reply"] == "original"
    assert step.attrs.get("error") == "timeout"


@pytest.mark.asyncio
async def test_revise_empty_output_falls_back_to_original():
    """If Gemma reasons-out-the-budget and returns empty content on the
    revise pass, we fall back to the original draft — better than the
    transport's empty-body fallback."""
    client = _client_returning("")
    rev = OngiiniReviewer(base_url="x", model_id="g", client=client)
    critique = CritiqueStep(verdict="REVISE", reasons=["x"])
    step = await rev.revise(_msg(), "original draft", critique, [], Policy(name="search_deep"))
    assert step.attrs["revised_reply"] == "original draft"


@pytest.mark.asyncio
async def test_revise_passes_reasons_to_prompt():
    client = _client_returning("revised text")
    rev = OngiiniReviewer(base_url="x", model_id="g", client=client)
    critique = CritiqueStep(
        verdict="REVISE",
        reasons=["claim about Oct 2025 isn't in search results", "no citation"],
    )
    await rev.revise(_msg(), "original", critique, [], Policy(name="search_deep"))
    sent = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Oct 2025" in sent
    assert "no citation" in sent
    assert "original" in sent   # original draft is in the prompt


@pytest.mark.asyncio
async def test_revise_falls_back_to_raw_critique_if_no_reasons_extracted():
    """When critique's raw text is opaque (no parseable FAIL: lines),
    the revise prompt should still get something to work with."""
    client = _client_returning("rewritten")
    rev = OngiiniReviewer(base_url="x", model_id="g", client=client)
    critique = CritiqueStep(verdict="REVISE", reasons=[])
    critique.attrs["raw_critique"] = "Something is off but I can't say what specifically"
    await rev.revise(_msg(), "draft", critique, [], Policy(name="search_deep"))
    sent = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Something is off" in sent


# ---------- helper ----------

def test_extract_fail_reasons():
    body = (
        "1. OK\n"
        "2. FAIL: confabulated date\n"
        "3. OK\n"
        "4) FAIL: language mismatch\n"   # different decoration
        "- FAIL: another one\n"
        "VERDICT: REVISE\n"
    )
    reasons = _extract_fail_reasons(body)
    assert reasons == ["confabulated date", "language mismatch", "another one"]


def test_extract_fail_reasons_handles_no_fails():
    body = "1. OK\n2. OK\nVERDICT: PASS\n"
    assert _extract_fail_reasons(body) == []


# ---------- verdict regex robustness (review I2 fix) ----------

@pytest.mark.asyncio
async def test_critique_ignores_prompt_echo_of_verdict_template():
    """Gemma sometimes echoes the prompt's 'VERDICT: PASS / VERDICT:
    REVISE' instruction block in its output. The regex must only match
    a real verdict line (start-of-line, on its own), not an instruction
    echo. AND it must take the LAST occurrence — final word wins."""
    body = (
        "Let me work through this — the dimensions to check are\n"
        "1. answers question — VERDICT: PASS or VERDICT: REVISE — that kind of thing.\n"
        "\n"
        "Now my actual critique:\n"
        "1. OK\n"
        "2. FAIL: confabulated\n"
        "3. OK\n"
        "4. OK\n"
        "5. OK\n"
        "6. OK\n"
        "\n"
        "VERDICT: REVISE\n"
    )
    client = _client_returning(body)
    rev = OngiiniReviewer(base_url="x", model_id="g", client=client)
    step = await rev.critique(_msg(), "draft", [], Policy(name="search_deep"))
    assert step.verdict == "REVISE"   # not PASS from the echo


@pytest.mark.asyncio
async def test_critique_last_verdict_wins():
    """If the model truly says PASS first then changes its mind to
    REVISE, the LAST anchored line is the final answer."""
    body = (
        "VERDICT: PASS\n"
        "\n"
        "Wait, on second thought I missed something.\n"
        "\n"
        "VERDICT: REVISE\n"
    )
    client = _client_returning(body)
    rev = OngiiniReviewer(base_url="x", model_id="g", client=client)
    step = await rev.critique(_msg(), "draft", [], Policy(name="search_deep"))
    assert step.verdict == "REVISE"


# ---------- None guard (review I3 fix) ----------

@pytest.mark.asyncio
async def test_critique_handles_none_msg_text():
    """An image-only inbound has msg.text=None. Must not put the literal
    string "None" into the critique prompt — short-circuit to PASS
    instead (no point critiquing a draft against an empty question)."""
    msg = InboundMessage(
        user_id="u", msg_id="m", text="",     # explicitly empty/None-like
        content_parts=[{"type": "image_url", "image_url": {"url": "data:..."}}],
        has_image=True,
    )
    client = _client_returning("VERDICT: REVISE")
    rev = OngiiniReviewer(base_url="x", model_id="g", client=client)
    step = await rev.critique(msg, "Some draft", [], Policy(name="search_shallow"))
    assert step.verdict == "PASS"
    client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_revise_strips_none_msg_text():
    """Revise must not put literal 'None' into the prompt either."""
    msg = InboundMessage(
        user_id="u", msg_id="m", text="",   # empty
        content_parts=[], has_image=False,
    )
    client = _client_returning("revised reply")
    rev = OngiiniReviewer(base_url="x", model_id="g", client=client)
    critique = CritiqueStep(verdict="REVISE", reasons=["x"])
    await rev.revise(msg, "original", critique, [], Policy(name="search_deep"))
    sent = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    # The prompt's "User's original question:" line should be followed
    # by an empty string, not the literal "None"
    assert "None" not in sent.split("User's original question:")[-1].split("\n")[0]
