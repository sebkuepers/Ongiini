"""Tests for the Ongiini tool catalogue."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from owela import ToolContext
from owela.tools import ToolRegistry
from ongiini.tools import (
    ALL_TOOLS, delete_my_data, fetch_url, fetch_urls,
    lookup_ongiini_docs, my_token_usage, web_search,
    whats_in_my_memory,
)


def _ctx(user_id: str = "+264u") -> ToolContext:
    from owela.transport import InboundMessage
    msg = InboundMessage(user_id=user_id, msg_id="m", text="t", content_parts=[])
    return ToolContext(user_id=user_id, runtime=MagicMock(), msg=msg)


# ---------- Registry construction ----------

def test_all_tools_register_in_registry():
    reg = ToolRegistry(list(ALL_TOOLS))
    schemas = reg.schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == {
        "web_search", "fetch_url", "fetch_urls", "delete_my_data",
        "whats_in_my_memory", "my_token_usage", "lookup_ongiini_docs",
    }


def test_fetch_urls_schema_is_array():
    """fetch_urls is the new fan-out tool. The schema must declare an
    array param so the model emits a JSON list, not a single string."""
    spec = fetch_urls.__owela_tool__
    assert spec.parameters["properties"]["urls"]["type"] == "array"
    assert spec.parameters["properties"]["urls"]["items"]["type"] == "string"


def test_admin_tools_have_no_schema_params():
    """ToolContext-bearing tools must exclude ctx from the schema —
    the model doesn't see the runtime."""
    for fn in (delete_my_data, whats_in_my_memory, my_token_usage, lookup_ongiini_docs):
        spec = fn.__owela_tool__
        assert spec.parameters == {"type": "object", "properties": {}}


def test_context_tools_have_needs_context_flag():
    for fn in (delete_my_data, whats_in_my_memory, my_token_usage):
        assert fn.__owela_tool__.needs_context is True
    # lookup_ongiini_docs is context-less (returns static product.md).
    assert lookup_ongiini_docs.__owela_tool__.needs_context is False


# ---------- fetch_urls ----------

@pytest.mark.asyncio
async def test_fetch_urls_batches_via_extract_endpoint():
    """v1.3: fetch_urls calls Tavily /extract ONCE with a URL list,
    instead of N parallel HTTP calls. The tool delegates to the search
    layer's extract_urls(), which returns a dict {url: text_or_error}."""
    async def fake_extract(urls):
        return {u: f"body of {u}" for u in urls}

    with patch(
        "ongiini.tools.ongiini_tools._extract_urls_impl",
        new=AsyncMock(side_effect=fake_extract),
    ) as mock_extract:
        result = await fetch_urls(["https://a.example", "https://b.example"])
    # One batched call, not two.
    assert mock_extract.call_count == 1
    assert mock_extract.call_args.args[0] == ["https://a.example", "https://b.example"]
    assert "## https://a.example" in result
    assert "body of https://a.example" in result
    assert "## https://b.example" in result
    assert "body of https://b.example" in result


@pytest.mark.asyncio
async def test_fetch_urls_caps_at_five():
    """Tool-level cap is 5 URLs even though Tavily /extract accepts up
    to 20 — prompt-budget reasons (the model gets the top 5)."""
    urls = [f"https://{i}.example" for i in range(10)]

    async def fake_extract(passed_urls):
        return {u: "x" for u in passed_urls}

    with patch(
        "ongiini.tools.ongiini_tools._extract_urls_impl",
        new=AsyncMock(side_effect=fake_extract),
    ) as mock_extract:
        await fetch_urls(urls)
    # ONE batched call, with the FIRST 5 URLs (not the last 5).
    assert mock_extract.call_count == 1
    passed = mock_extract.call_args.args[0]
    assert passed == [f"https://{i}.example" for i in range(5)]


@pytest.mark.asyncio
async def test_fetch_urls_empty_list_returns_friendly_message():
    out = await fetch_urls([])
    assert "no urls" in out.lower()


@pytest.mark.asyncio
async def test_fetch_urls_per_url_failure_inlined_in_result():
    """One URL fails → its block shows the error, others show their
    content. The batched /extract call returns per-URL outcomes in
    one dict."""
    async def fake_extract(urls):
        return {
            "https://good.example": "body of https://good.example",
            "https://bad.example": "Failed to fetch https://bad.example: simulated failure",
        }

    with patch(
        "ongiini.tools.ongiini_tools._extract_urls_impl",
        new=AsyncMock(side_effect=fake_extract),
    ):
        result = await fetch_urls(["https://good.example", "https://bad.example"])
    assert "body of https://good.example" in result
    assert "Failed to fetch https://bad.example" in result


# ---------- delete_my_data ----------

@pytest.mark.asyncio
async def test_delete_my_data_calls_memory_delete():
    ctx = _ctx("+264u")
    ctx.runtime.memory.delete_all = AsyncMock(return_value=True)
    result = await delete_my_data(ctx)
    ctx.runtime.memory.delete_all.assert_awaited_once_with("+264u")
    assert "privacy model" in result.lower()
    assert "reset" in result.lower()


@pytest.mark.asyncio
async def test_delete_my_data_empty_user_gets_friendly_message():
    ctx = _ctx("+264new")
    ctx.runtime.memory.delete_all = AsyncMock(return_value=False)
    result = await delete_my_data(ctx)
    assert "clean slate" in result.lower()


# ---------- whats_in_my_memory ----------

@pytest.mark.asyncio
async def test_whats_in_my_memory_returns_empty_message_for_clean_user():
    ctx = _ctx("+264u")
    ctx.runtime.memory.list_all = AsyncMock(return_value=[])
    with patch("ongiini.tools.ongiini_tools._memory.load", return_value=[]):
        result = await whats_in_my_memory(ctx)
    assert "empty" in result.lower()


@pytest.mark.asyncio
async def test_whats_in_my_memory_includes_long_term_facts():
    ctx = _ctx("+264u")
    facts = [{"memory": "[PROFILE] Lives in Oshakati"}]
    ctx.runtime.memory.list_all = AsyncMock(return_value=facts)
    # The provider's format_facts method handles tag-grouped rendering;
    # tools call THAT (not a private _long attribute).
    ctx.runtime.memory.format_facts = MagicMock(return_value="About you:\n  - Lives in Oshakati")
    with patch("ongiini.tools.ongiini_tools._memory.load", return_value=[]):
        result = await whats_in_my_memory(ctx)
    assert "Long-term memory (1 facts" in result
    assert "Lives in Oshakati" in result
    ctx.runtime.memory.format_facts.assert_called_once_with(facts)


# ---------- lookup_ongiini_docs ----------

@pytest.mark.asyncio
async def test_lookup_docs_falls_back_when_file_missing():
    """A missing product.md must not crash — return the deployment-bug
    advisory message instead."""
    from pathlib import Path
    from ongiini.tools import ongiini_tools as ot

    # Reset cache + redirect the docs path to a path that definitely
    # doesn't exist. Restore after.
    original_path = ot._PRODUCT_DOCS_PATH
    original_cache = ot._product_docs_cache
    try:
        ot._PRODUCT_DOCS_PATH = Path("/nonexistent/owela-test-product.md")
        ot._product_docs_cache = None
        result = await lookup_ongiini_docs()
    finally:
        ot._PRODUCT_DOCS_PATH = original_path
        ot._product_docs_cache = original_cache

    assert "ongiini.ai/product.md" in result


# ---------- web_search / fetch_url ----------

@pytest.mark.asyncio
async def test_web_search_returns_text_and_urls_tuple():
    """v1.3: web_search tool returns (text, {"urls": [...]}) so the
    @tool registry can attach the URL list to the ToolStep.attrs
    without exposing the structured form to the model."""
    with patch(
        "ongiini.tools.ongiini_tools._web_search_impl",
        new=AsyncMock(return_value=("search results here", ["https://a.com", "https://b.com"])),
    ) as mock:
        result = await web_search("test query")
    # _web_search_impl is called with the keyword args topic + time_range.
    mock.assert_awaited_once()
    call = mock.call_args
    assert call.args == ("test query",)
    assert call.kwargs == {"topic": "general", "time_range": None}
    # The tool wraps the (text, urls) impl response into (text, {"urls": urls}).
    assert isinstance(result, tuple)
    assert result[0] == "search results here"
    assert result[1] == {"urls": ["https://a.com", "https://b.com"]}


@pytest.mark.asyncio
async def test_web_search_forwards_topic_and_time_range_kwargs():
    """The planner emits topic + time_range per QueryVariant; when the
    executor synthesises a tool call with those args, the tool must
    forward them to the search impl."""
    with patch(
        "ongiini.tools.ongiini_tools._web_search_impl",
        new=AsyncMock(return_value=("news block", [])),
    ) as mock:
        await web_search("medicine shortage", topic="news", time_range="week")
    call = mock.call_args
    assert call.kwargs == {"topic": "news", "time_range": "week"}


@pytest.mark.asyncio
async def test_web_search_empty_time_range_normalised_to_none():
    """Tool signature uses str default '' instead of None (Owela's
    @tool decoder doesn't support Optional[str] yet). The tool
    normalises an empty string to None before calling the impl."""
    with patch(
        "ongiini.tools.ongiini_tools._web_search_impl",
        new=AsyncMock(return_value=("ok", [])),
    ) as mock:
        await web_search("q", topic="general", time_range="")
    assert mock.call_args.kwargs == {"topic": "general", "time_range": None}


@pytest.mark.asyncio
async def test_fetch_url_delegates_to_impl():
    with patch(
        "ongiini.tools.ongiini_tools._fetch_url_impl",
        new=AsyncMock(return_value="page body"),
    ) as mock:
        result = await fetch_url("https://example.com/x")
    mock.assert_awaited_once_with("https://example.com/x")
    assert result == "page body"


# ---------- my_token_usage ----------

@pytest.mark.asyncio
async def test_my_token_usage_summarises_stats():
    ctx = _ctx("+264u")
    fake_stats = {
        "month": "2026-05",
        "messages": 12,
        "tokens_total": 50000,
        "limit": 1_000_000,
        "percent_used": 5.0,
        "breakdown": {
            "chat": {"tokens_in": 30000, "tokens_out": 15000},
            "memory": {"tokens_in": 3000, "tokens_out": 1500},
            "summary": {"tokens_in": 400, "tokens_out": 100},
        },
    }
    with patch("ongiini.tools.ongiini_tools._usage.summary_for", return_value=fake_stats):
        result = await my_token_usage(ctx)
    assert "2026-05" in result
    assert "12 messages" in result
    assert "50000 tokens total" in result
    assert "45000 tokens" in result    # chat sum
