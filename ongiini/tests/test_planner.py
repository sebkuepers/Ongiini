"""Unit tests for OngiiniPlanner — v1.3 JSON-emitting contract."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from owela import InboundMessage, PlanStep, Policy, QueryVariant
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


# ---------- Happy path ----------

@pytest.mark.asyncio
async def test_planner_returns_planstep_with_facts_and_queries():
    """The planner now emits JSON. The parser extracts ``facts_known``
    into ``plan_text`` and a list of QueryVariants into ``queries``."""
    body = json.dumps({
        "facts_known": "Namibia has 4 major commercial banks: Bank Windhoek, FNB Namibia, Nedbank, Standard Bank.",
        "queries": [
            {"query": "Bank Windhoek home loan rate 2026", "topic": "general", "time_range": None},
            {"query": "FNB Namibia home loan rate 2026",  "topic": "general", "time_range": None},
            {"query": "Nedbank Namibia home loan rate 2026", "topic": "general", "time_range": None},
            {"query": "Namibia prime lending rate", "topic": "general", "time_range": "month"},
        ],
    }) + "\nPLAN_DONE"
    client = _client_returning(body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    step = await planner.plan(_msg(), Policy(name="search_deep"), [])

    assert isinstance(step, PlanStep)
    assert "Bank Windhoek" in step.plan_text
    assert len(step.queries) == 4
    assert step.queries[0].query == "Bank Windhoek home loan rate 2026"
    # "general" topic is the default; not stored in extra (only deviations are).
    assert "topic" not in step.queries[0].extra
    # The 4th query has a time_range bias.
    assert step.queries[3].extra == {"time_range": "month"}
    assert step.tokens_in == 20      # 300 - 280 cached
    assert step.tokens_out == 180
    assert step.cached_tokens == 280
    assert step.ended_at is not None


@pytest.mark.asyncio
async def test_planner_parses_news_topic():
    """News-shaped queries set topic='news' which lands in extra."""
    body = json.dumps({
        "facts_known": "Namibian medicine supply chain has been under stress in 2025.",
        "queries": [
            {"query": "Namibia medicine shortage 2026", "topic": "news", "time_range": "month"},
        ],
    }) + "\nPLAN_DONE"
    client = _client_returning(body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    step = await planner.plan(_msg("medicine shortage news"), Policy(name="search_deep"), [])
    assert len(step.queries) == 1
    q = step.queries[0]
    assert q.extra == {"topic": "news", "time_range": "month"}


@pytest.mark.asyncio
async def test_planner_handles_facts_none_string():
    """When the model writes 'none' for facts_known (per the prompt
    instruction), the parser normalises to empty string."""
    body = json.dumps({
        "facts_known": "none",
        "queries": [{"query": "BoN exchange rate today",
                     "topic": "general", "time_range": "day"}],
    }) + "\nPLAN_DONE"
    client = _client_returning(body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    step = await planner.plan(_msg(), Policy(name="search_deep"), [])
    assert step.plan_text == ""
    assert step.queries[0].extra == {"time_range": "day"}


# ---------- Sentinel + stray text handling ----------

@pytest.mark.asyncio
async def test_planner_strips_text_after_plan_done():
    """The model occasionally appends text after PLAN_DONE — ignore it."""
    body = (
        json.dumps({"facts_known": "x", "queries": [{"query": "q"}]})
        + "\nPLAN_DONE\nThen something extra that shouldn't be parsed."
    )
    client = _client_returning(body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    step = await planner.plan(_msg(), Policy(name="search_deep"), [])
    assert "extra" not in step.plan_text
    assert step.queries[0].query == "q"


@pytest.mark.asyncio
async def test_planner_tolerates_prose_before_json():
    """If the model prefixes the JSON with a stray sentence, the parser
    still locates the first balanced JSON object and reads it."""
    body = (
        "Here is my plan:\n\n"
        + json.dumps({"facts_known": "f", "queries": [{"query": "q"}]})
        + "\nPLAN_DONE"
    )
    client = _client_returning(body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    step = await planner.plan(_msg(), Policy(name="search_deep"), [])
    assert step.plan_text == "f"
    assert step.queries[0].query == "q"


# ---------- Soft-fail (parse failure → empty plan) ----------

@pytest.mark.asyncio
async def test_planner_soft_fails_on_malformed_json():
    """Malformed JSON → empty plan_text + empty queries. The executor
    falls back to model-driven turn 1 (no fan-out)."""
    body = "this isn't json at all\nPLAN_DONE"
    client = _client_returning(body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    step = await planner.plan(_msg(), Policy(name="search_deep"), [])
    assert step.plan_text == ""
    assert step.queries == []


@pytest.mark.asyncio
async def test_planner_soft_fails_on_missing_queries_key():
    """JSON without a queries key → empty queries (facts_known still
    parsed if present)."""
    body = json.dumps({"facts_known": "some context"}) + "\nPLAN_DONE"
    client = _client_returning(body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    step = await planner.plan(_msg(), Policy(name="search_deep"), [])
    assert step.plan_text == "some context"
    assert step.queries == []


@pytest.mark.asyncio
async def test_planner_skips_malformed_query_entries():
    """The parser is tolerant: a malformed entry is skipped, good ones
    are kept."""
    body = json.dumps({
        "facts_known": "ok",
        "queries": [
            {"query": "good one", "topic": "general", "time_range": None},
            {"not_a_query": True},        # malformed — no query key
            {"query": "", "topic": "general"},  # empty query string — skipped
            {"query": "another good one"},
        ],
    }) + "\nPLAN_DONE"
    client = _client_returning(body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    step = await planner.plan(_msg(), Policy(name="search_deep"), [])
    assert [q.query for q in step.queries] == ["good one", "another good one"]


@pytest.mark.asyncio
async def test_planner_caps_at_five_queries():
    """Even if the model emits 10 queries (against prompt instruction),
    the parser silently caps at 5."""
    body = json.dumps({
        "facts_known": "x",
        "queries": [{"query": f"q{i}"} for i in range(10)],
    }) + "\nPLAN_DONE"
    client = _client_returning(body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    step = await planner.plan(_msg(), Policy(name="search_deep"), [])
    assert len(step.queries) == 5
    assert step.queries[0].query == "q0"
    assert step.queries[-1].query == "q4"


# ---------- Empty input / network failures ----------

@pytest.mark.asyncio
async def test_planner_empty_question_returns_empty_plan_without_call():
    """Defensive: no question = no plan, no LLM call."""
    client = _client_returning("any")
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    step = await planner.plan(_msg(text=""), Policy(name="search_deep"), [])
    assert step.plan_text == ""
    assert step.queries == []
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
    assert step.queries == []
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
    assert step.queries == []
    assert "vllm down" in step.attrs.get("error", "")


# ---------- Prompt contract ----------

@pytest.mark.asyncio
async def test_planner_ignores_unknown_json_keys():
    """The parser only reads ``facts_known`` and ``queries`` — any
    extra keys the model emits (e.g. ``reasoning``) are silently
    ignored."""
    body = json.dumps({
        "facts_known": "ok",
        "queries": [{"query": "q1"}],
        "reasoning": "I thought hard about this",
        "score": 0.95,
    }) + "\nPLAN_DONE"
    client = _client_returning(body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    step = await planner.plan(_msg(), Policy(name="search_deep"), [])
    assert step.plan_text == "ok"
    assert [q.query for q in step.queries] == ["q1"]


@pytest.mark.asyncio
async def test_planner_handles_facts_known_explicit_null():
    """When the JSON has ``"facts_known": null`` (not missing, not
    the string 'none'), the parser normalises to empty string."""
    body = json.dumps({
        "facts_known": None,
        "queries": [{"query": "q"}],
    }) + "\nPLAN_DONE"
    client = _client_returning(body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    step = await planner.plan(_msg(), Policy(name="search_deep"), [])
    assert step.plan_text == ""
    assert step.queries[0].query == "q"


@pytest.mark.asyncio
async def test_planner_prompt_carries_question_and_sentinel():
    """Locks the prompt contract: the question goes through, the
    PLAN_DONE sentinel is present, max_tokens is set, JSON shape is
    documented."""
    body = json.dumps({"facts_known": "x", "queries": []}) + "\nPLAN_DONE"
    client = _client_returning(body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)
    await planner.plan(_msg("compare three banks"), Policy(name="search_deep"), [])

    call_kwargs = client.chat.completions.create.call_args.kwargs
    sent = call_kwargs["messages"][0]["content"]
    assert "compare three banks" in sent
    assert "PLAN_DONE" in sent
    # The prompt MUST instruct JSON output (the executor depends on it).
    assert '"queries"' in sent
    assert '"facts_known"' in sent
    # Per-shape guidance is what makes the planner produce the right
    # number of entity queries — confirm the COMPARISON branch exists.
    assert "COMPARISON" in sent
    assert call_kwargs["max_tokens"] == 320


# ---------- v1.3.1: conversation context resolution ----------

@pytest.mark.asyncio
async def test_planner_prompt_includes_recent_history_block_when_present():
    """When msg.history has prior turns, the planner prompt includes a
    'Conversation just before this question' block so the model can
    resolve pronouns like 'them', 'this', 'compare them'."""
    body = json.dumps({"facts_known": "x", "queries": [{"query": "q"}]}) + "\nPLAN_DONE"
    client = _client_returning(body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)

    msg = InboundMessage(
        user_id="+264u", msg_id="m", text="compare them",
        content_parts=[{"type": "text", "text": "compare them"}],
        history=[
            {"role": "user", "content": "How many datacenters in Namibia?"},
            {"role": "assistant", "content": "There are 6 main ones: Paratus, FNB, ..."},
        ],
    )
    await planner.plan(msg, Policy(name="search_deep"), [])
    sent = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS USER: How many datacenters in Namibia?" in sent
    assert "PREVIOUS REPLY:" in sent
    assert "Paratus" in sent
    # And the actual question is still in the prompt.
    assert "compare them" in sent


@pytest.mark.asyncio
async def test_planner_prompt_unchanged_when_history_empty():
    """When msg.history is empty (first-turn query), the prompt MUST be
    byte-identical to v1.3 — preserves prefix-cache hits."""
    body = json.dumps({"facts_known": "x", "queries": [{"query": "q"}]}) + "\nPLAN_DONE"
    client = _client_returning(body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)

    msg = InboundMessage(
        user_id="+264u", msg_id="m", text="a question",
        content_parts=[{"type": "text", "text": "a question"}],
        history=[],
    )
    await planner.plan(msg, Policy(name="search_deep"), [])
    sent = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    # No history block when history is empty.
    assert "Conversation just before this question" not in sent
    assert "PREVIOUS USER" not in sent


@pytest.mark.asyncio
async def test_planner_history_caps_long_messages():
    """Very long previous messages get capped to ~400 chars — we just
    need pronoun context, not a full transcript."""
    body = json.dumps({"facts_known": "x", "queries": []}) + "\nPLAN_DONE"
    client = _client_returning(body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)

    long_reply = "y" * 2000
    msg = InboundMessage(
        user_id="+264u", msg_id="m", text="follow-up",
        content_parts=[{"type": "text", "text": "follow-up"}],
        history=[
            {"role": "user", "content": "x" * 2000},
            {"role": "assistant", "content": long_reply},
        ],
    )
    await planner.plan(msg, Policy(name="search_deep"), [])
    sent = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    # Truncation marker present.
    assert "…" in sent
    # Slice the history block: from "Conversation just before" to the
    # next section ("Plan the search BEFORE..."). Should be ~900 chars
    # (cap of 400 chars per snippet × 2 snippets + framing), far less
    # than the 4000 chars of raw history we passed in.
    history_section_start = sent.index("Conversation just before")
    next_section = sent.index("Plan the search BEFORE", history_section_start)
    history_section = sent[history_section_start:next_section]
    assert len(history_section) < 1500


@pytest.mark.asyncio
async def test_planner_history_skips_non_string_content():
    """Image-bearing prior turns have list[dict] content_parts; the
    planner should silently skip those (no value for query decomp)."""
    body = json.dumps({"facts_known": "x", "queries": []}) + "\nPLAN_DONE"
    client = _client_returning(body)
    planner = OngiiniPlanner(base_url="x", model_id="gemma", client=client)

    msg = InboundMessage(
        user_id="+264u", msg_id="m", text="next question",
        content_parts=[{"type": "text", "text": "next question"}],
        history=[
            {"role": "user", "content": [{"type": "image_url", "image_url": "..."}]},
            {"role": "assistant", "content": "I see a chart..."},
        ],
    )
    await planner.plan(msg, Policy(name="search_deep"), [])
    sent = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    # The text-only assistant reply was surfaced (it's a useful string);
    # the image-bearing user message was silently skipped.
    assert "I see a chart" in sent
    # No fallback marker for the skipped image-bearing turn.
    assert "image_url" not in sent
