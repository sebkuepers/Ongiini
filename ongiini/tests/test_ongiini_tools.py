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
async def test_fetch_urls_fans_out_in_parallel():
    with patch(
        "ongiini.tools.ongiini_tools._fetch_url_impl",
        new=AsyncMock(side_effect=lambda u: f"body of {u}"),
    ) as mock_fetch:
        result = await fetch_urls(["https://a.example", "https://b.example"])
    assert mock_fetch.call_count == 2
    assert "## https://a.example" in result
    assert "body of https://a.example" in result
    assert "## https://b.example" in result
    assert "body of https://b.example" in result


@pytest.mark.asyncio
async def test_fetch_urls_caps_at_five():
    urls = [f"https://{i}.example" for i in range(10)]
    with patch(
        "ongiini.tools.ongiini_tools._fetch_url_impl",
        new=AsyncMock(return_value="x"),
    ) as mock_fetch:
        await fetch_urls(urls)
    assert mock_fetch.call_count == 5     # silent cap
    # Lock against accidental `urls[-5:]` — must keep the FIRST 5.
    called_urls = [c.args[0] for c in mock_fetch.call_args_list]
    assert called_urls == [f"https://{i}.example" for i in range(5)]


@pytest.mark.asyncio
async def test_fetch_urls_empty_list_returns_friendly_message():
    out = await fetch_urls([])
    assert "no urls" in out.lower()


@pytest.mark.asyncio
async def test_fetch_urls_one_failure_does_not_block_others():
    async def maybe_fail(url: str) -> str:
        if "bad" in url:
            raise RuntimeError("simulated failure")
        return f"body of {url}"

    with patch(
        "ongiini.tools.ongiini_tools._fetch_url_impl",
        new=AsyncMock(side_effect=maybe_fail),
    ):
        result = await fetch_urls(["https://good.example", "https://bad.example"])
    assert "body of https://good.example" in result
    assert "fetch failed" in result


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
async def test_web_search_delegates_to_impl():
    with patch(
        "ongiini.tools.ongiini_tools._web_search_impl",
        new=AsyncMock(return_value="search results here"),
    ) as mock:
        result = await web_search("test query")
    mock.assert_awaited_once_with("test query")
    assert result == "search results here"


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
