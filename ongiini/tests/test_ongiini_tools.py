"""Tests for the Ongiini tool catalogue."""

from __future__ import annotations

from datetime import datetime
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
        "load_skill",
        # v2 contribute tools — classifier-forced, not model-chosen
        "contribute_invite_check", "contribute_set_dialect",
        "contribute_next", "contribute_save", "contribute_skip",
        "contribute_decline", "contribute_stats",
        # broadcast opt-out — classifier-forced, not model-chosen
        "opt_out_broadcast",
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
    # _web_search_impl is called with kwargs topic + time_range +
    # include_raw_content (v1.3.1 added the latter — defaults True
    # because the tool's signature default is True; SEARCH_DEEP overrides
    # to False via Policy.planner_query_default_args).
    mock.assert_awaited_once()
    call = mock.call_args
    assert call.args == ("test query",)
    assert call.kwargs == {
        "topic": "general", "time_range": None,
        "include_raw_content": True,
    }
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
    assert call.kwargs == {
        "topic": "news", "time_range": "week",
        "include_raw_content": True,
    }


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
    assert mock.call_args.kwargs == {
        "topic": "general", "time_range": None,
        "include_raw_content": True,
    }


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
async def test_my_token_usage_returns_structured_payload():
    """The tool now returns JSON the model uses to compose a friendly
    month-summary. Lead fields are the user-relevant ones (messages,
    active_days, themes); raw token numbers stay in the infrastructure
    layer."""
    import json as _json
    ctx = _ctx("+264u")
    fake_stats = {
        "month": "2026-05",
        "messages": 12,
        "tokens_total": 50000,
        "limit": 1_000_000,
        "percent_used": 5.0,
        "breakdown": {},
    }
    with patch("ongiini.tools.ongiini_tools._usage.summary_for", return_value=fake_stats), \
         patch("ongiini.tools.ongiini_tools._activity_patterns",
               return_value={"active_days": 5, "peak_day_of_week": "Tuesday",
                             "peak_hour_band": "evenings"}), \
         patch("ongiini.tools.ongiini_tools._media_counts",
               return_value={"images": 3, "voice_notes": 1}), \
         patch("ongiini.tools.ongiini_tools._top_user_facts",
               return_value=["[SITUATION] Working on non-profit setup",
                             "[PROFILE] Software engineer in Namibia"]):
        result = await my_token_usage(ctx)
    payload = _json.loads(result)
    assert payload["month"] == "2026-05"
    assert payload["messages"] == 12
    assert payload["active_days"] == 5
    assert payload["peak_day_of_week"] == "Tuesday"
    assert payload["peak_hour_band"] == "evenings"
    assert payload["images_shared"] == 3
    assert payload["voice_notes"] == 1
    assert len(payload["top_topics_from_memory"]) == 2
    # The model — not the tool — composes the natural-language reply;
    # raw token numbers should NOT appear in the payload (they're not
    # user-relevant).
    assert "tokens_total" not in payload
    assert "limit" not in payload


@pytest.mark.asyncio
async def test_my_token_usage_handles_empty_user_gracefully():
    """A brand-new user with no usage.log lines, no memory file, no mem0
    facts should still get a valid JSON payload with zero/null values."""
    import json as _json
    ctx = _ctx("+264u")
    fake_stats = {"month": "2026-05", "messages": 0,
                  "tokens_total": 0, "limit": 1_000_000,
                  "percent_used": 0.0, "breakdown": {}}
    with patch("ongiini.tools.ongiini_tools._usage.summary_for", return_value=fake_stats), \
         patch("ongiini.tools.ongiini_tools._activity_patterns",
               return_value={"active_days": 0, "peak_day_of_week": None,
                             "peak_hour_band": None}), \
         patch("ongiini.tools.ongiini_tools._media_counts",
               return_value={"images": 0, "voice_notes": 0}), \
         patch("ongiini.tools.ongiini_tools._top_user_facts", return_value=[]):
        result = await my_token_usage(ctx)
    payload = _json.loads(result)
    assert payload["messages"] == 0
    assert payload["active_days"] == 0
    assert payload["peak_day_of_week"] is None
    assert payload["top_topics_from_memory"] == []


def test_activity_patterns_parses_usage_log(tmp_path, monkeypatch):
    """Sanity-check the usage.log parser identifies the right user and
    correctly tallies active days + day-of-week + hour band."""
    from ongiini.tools.ongiini_tools import _activity_patterns
    log_file = tmp_path / "usage.log"
    # 3 chat turns on Tuesday + Wednesday in evenings, 1 on Monday afternoon
    # Note: 2026-05-04 is Monday, 05 Tue, 06 Wed
    log_file.write_text(
        "2026-05-04T14:00:00 | 264811111111 | tokens_in=100 tokens_out=10 | search=no | kind=chat\n"
        "2026-05-05T19:00:00 | 264811111111 | tokens_in=100 tokens_out=10 | search=no | kind=chat\n"
        "2026-05-05T20:00:00 | 264811111111 | tokens_in=100 tokens_out=10 | search=no | kind=chat\n"
        "2026-05-06T19:30:00 | 264811111111 | tokens_in=100 tokens_out=10 | search=no | kind=chat\n"
        # Different user — should be ignored
        "2026-05-06T19:30:00 | 264899999999 | tokens_in=100 tokens_out=10 | search=no | kind=chat\n"
        # memory-kind line — not user-facing activity, ignored
        "2026-05-06T19:30:00 | 264811111111 | tokens_in=100 tokens_out=10 | search=no | kind=memory\n"
    )
    monkeypatch.setattr("ongiini.tools.ongiini_tools._usage.LOG_PATH", log_file)
    # Freeze "now" to a date in May 2026 so the month filter picks up our lines
    monkeypatch.setattr(
        "ongiini.tools.ongiini_tools.datetime",
        type("D", (), {
            "now": staticmethod(lambda tz=None: datetime(2026, 5, 30, tzinfo=tz)),
            "fromisoformat": staticmethod(datetime.fromisoformat),
        }),
    )
    result = _activity_patterns("264811111111")
    assert result["active_days"] == 3   # Mon, Tue, Wed
    # Tuesday has 2 entries — highest
    assert result["peak_day_of_week"] == "Tuesday"
    # 19/20 UTC = evening band
    assert result["peak_hour_band"] == "evenings"


def test_media_counts_walks_user_memory(monkeypatch):
    """Image and voice markers in short-term memory are the only signal
    we have for 'documents reviewed / voice notes' — verify the count
    matches what's in the memory file."""
    from ongiini.tools.ongiini_tools import _media_counts
    fake_history = [
        {"role": "user", "content": "[image attached] hello"},
        {"role": "assistant", "content": "I see the photo…"},
        {"role": "user", "content": "[image attached]"},
        {"role": "user", "content": "[voice note] hey"},
        {"role": "assistant", "content": "Got your message"},
        # Assistant turns aren't counted even if they mention markers
        {"role": "assistant", "content": "[image attached] ← shouldn't count"},
    ]
    with patch("ongiini.tools.ongiini_tools._memory.load", return_value=fake_history):
        result = _media_counts("264811111111")
    assert result["images"] == 2
    assert result["voice_notes"] == 1


def test_top_user_facts_skips_quote_entries(monkeypatch):
    """[QUOTE] mem0 entries are verbatim utterance snapshots — they
    repeat what the user said but don't name a domain. Skip them
    so the bot's topic narrative anchors to [SITUATION] / [PROFILE]
    / [GOAL] tags instead."""
    from ongiini.tools.ongiini_tools import _top_user_facts
    fake_facts = [
        {"memory": "[QUOTE] 'I want to try something different'"},
        {"memory": "[SITUATION] Studying nursing at Atlantic Institute"},
        {"memory": "[PROFILE] Lives in Walvis Bay"},
        {"memory": "[QUOTE] 'Help me with my CV'"},
        {"memory": "[GOAL] Wants to apply for scholarship"},
    ]
    # Patch list_all on the already-imported module so the lazy import
    # inside _top_user_facts picks up the fake.
    import sys, types
    if "ongiini.memory.long_term" not in sys.modules:
        fake_module = types.ModuleType("ongiini.memory.long_term")
        sys.modules["ongiini.memory.long_term"] = fake_module
    monkeypatch.setattr(
        sys.modules["ongiini.memory.long_term"],
        "list_all", lambda _msi: fake_facts, raising=False,
    )
    result = _top_user_facts("264811111111", n=8)
    assert len(result) == 3
    assert all("[QUOTE]" not in t for t in result)
