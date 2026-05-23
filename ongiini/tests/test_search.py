"""Tests for the search.py Tavily wrapper — v1.3 search quality overhaul.

These are unit tests against the Tavily adapter shape: no live network.
Tavily's POST is mocked via patching httpx.AsyncClient.post.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ongiini import search as _search


# ---------- _TTLCache ----------

def test_ttl_cache_get_miss_returns_none():
    c = _search._TTLCache(ttl_seconds=60.0)
    assert c.get(("missing",)) is None


def test_ttl_cache_put_then_get_returns_value():
    c = _search._TTLCache(ttl_seconds=60.0)
    c.put(("k",), "value")
    assert c.get(("k",)) == "value"


def test_ttl_cache_evicts_after_ttl():
    """Entries past their TTL are not returned and are popped lazily on get."""
    c = _search._TTLCache(ttl_seconds=0.01)  # 10ms TTL
    c.put(("k",), "value")
    time.sleep(0.02)
    assert c.get(("k",)) is None
    # Lazy eviction: the get() call should have popped it.
    assert ("k",) not in c._store


def test_ttl_cache_hard_cap_drops_oldest_when_soft_max_breached_with_no_expired():
    """If soft_max is exceeded AND no entries are expired, the oldest
    inserted entry is evicted (FIFO via Python 3.7+ dict ordering)."""
    c = _search._TTLCache(ttl_seconds=60.0, soft_max=3)
    c.put(("a",), 1)
    c.put(("b",), 2)
    c.put(("c",), 3)
    c.put(("d",), 4)   # breaches soft_max
    # 'a' was the oldest; it must have been dropped.
    assert c.get(("a",)) is None
    assert c.get(("d",)) == 4


def test_ttl_cache_clear_empties_store():
    c = _search._TTLCache(ttl_seconds=60.0)
    c.put(("k",), 1)
    c.clear()
    assert c.get(("k",)) is None


# ---------- _canonical_url ----------

def test_canonical_url_strips_trailing_slash_from_non_root_path():
    assert _search._canonical_url("https://example.com/foo/") == "https://example.com/foo"


def test_canonical_url_keeps_root_path_slash():
    assert _search._canonical_url("https://example.com/") == "https://example.com/"


def test_canonical_url_lowercases_host():
    assert _search._canonical_url("HTTPS://EXAMPLE.COM/Path") == "https://example.com/Path"
    # Path keeps its case (URLs are path-case-sensitive on most servers).


def test_canonical_url_keeps_query_string():
    assert _search._canonical_url("https://example.com/q?x=1&y=2") == "https://example.com/q?x=1&y=2"


def test_canonical_url_drops_fragment_but_keeps_query():
    """urllib.parse separates the fragment from the path; we don't
    reattach it. Two URLs that differ only by fragment should canonicalise
    to the same key (browsers resolve fragments client-side anyway)."""
    assert _search._canonical_url("https://example.com/x#section") == "https://example.com/x"
    assert _search._canonical_url("https://example.com/x") == "https://example.com/x"


def test_canonical_url_passes_through_malformed():
    """A URL we can't parse falls through verbatim — better than crashing
    in a cache lookup helper."""
    assert _search._canonical_url("not a url") == "not a url"


# ---------- web_search ----------

@pytest.mark.asyncio
async def test_web_search_returns_text_and_urls_tuple(monkeypatch):
    """web_search posts to Tavily, parses results, returns (text, urls)."""
    _search._SEARCH_CACHE.clear()

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(return_value={
        "answer": "Some summary.",
        "results": [
            {"title": "T1", "content": "snippet 1", "url": "https://a.example/x",
             "raw_content": "full text of A"},
            {"title": "T2", "content": "snippet 2", "url": "https://b.example/y",
             "raw_content": ""},
        ],
    })

    monkeypatch.setattr(_search.settings, "tavily_api_key", "fake-key")
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake_response)) as mock_post:
        text, urls = await _search.web_search("hello")

    assert "Some summary." in text
    assert "snippet 1" in text
    assert "full text of A" in text
    assert urls == ["https://a.example/x", "https://b.example/y"]

    # Confirm advanced params were sent.
    payload = mock_post.call_args.kwargs["json"]
    assert payload["search_depth"] == "advanced"
    assert payload["include_raw_content"] is True
    assert payload["chunks_per_source"] == 3
    assert payload["country"] == "namibia"
    assert payload["include_answer"] is True
    # time_range NOT in payload when None — Tavily treats absent key as no restriction.
    assert "time_range" not in payload


@pytest.mark.asyncio
async def test_web_search_includes_time_range_when_provided(monkeypatch):
    _search._SEARCH_CACHE.clear()
    monkeypatch.setattr(_search.settings, "tavily_api_key", "fake-key")
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(return_value={"results": []})

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake_response)) as mock_post:
        await _search.web_search("hello", time_range="week", topic="news")

    payload = mock_post.call_args.kwargs["json"]
    assert payload["time_range"] == "week"
    assert payload["topic"] == "news"


@pytest.mark.asyncio
async def test_web_search_invalid_topic_falls_back_to_general(monkeypatch):
    _search._SEARCH_CACHE.clear()
    monkeypatch.setattr(_search.settings, "tavily_api_key", "fake-key")
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(return_value={"results": []})

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake_response)) as mock_post:
        await _search.web_search("q", topic="nonsense", time_range="invalid")

    payload = mock_post.call_args.kwargs["json"]
    assert payload["topic"] == "general"   # silently coerced
    assert "time_range" not in payload      # invalid → None → omitted


@pytest.mark.asyncio
async def test_web_search_no_api_key_returns_friendly_message(monkeypatch):
    monkeypatch.setattr(_search.settings, "tavily_api_key", "")
    text, urls = await _search.web_search("hello")
    assert "not configured" in text.lower()
    assert urls == []


@pytest.mark.asyncio
async def test_web_search_cache_returns_fresh_list_copy(monkeypatch):
    """The cache stores an immutable URL tuple; each cache hit returns
    a fresh mutable list so callers can mutate without poisoning future hits."""
    _search._SEARCH_CACHE.clear()
    monkeypatch.setattr(_search.settings, "tavily_api_key", "fake-key")
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(return_value={
        "results": [{"title": "T", "content": "x", "url": "https://a/", "raw_content": ""}],
    })

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake_response)):
        _, urls_first = await _search.web_search("same q")
    # Mutate the first caller's returned list.
    urls_first.append("https://injected")

    # Second call hits the cache — must NOT see the injected URL.
    with patch("httpx.AsyncClient.post", new=AsyncMock()):
        _, urls_second = await _search.web_search("same q")
    assert "https://injected" not in urls_second
    assert urls_second == ["https://a/"]


# ---------- extract_urls (SSRF + batching) ----------

@pytest.mark.asyncio
async def test_extract_urls_rejects_internal_hostnames(monkeypatch):
    """SSRF guard: localhost / 127.0.0.1 / private nets refused BEFORE
    forwarding to Tavily. The error lands in the returned dict."""
    monkeypatch.setattr(_search.settings, "tavily_api_key", "fake-key")
    _search._EXTRACT_CACHE.clear()

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(return_value={
        "results": [{"url": "https://good.example.com/page", "raw_content": "ok"}],
    })

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake_response)) as mock_post:
        result = await _search.extract_urls([
            "http://localhost/x",
            "http://192.168.1.1/admin",
            "https://good.example.com/page",
        ])

    # Tavily was called with only the safe URL.
    assert mock_post.call_count == 1
    sent_urls = mock_post.call_args.kwargs["json"]["urls"]
    assert sent_urls == ["https://good.example.com/page"]

    # Both blocked URLs land in the result dict as error strings.
    assert "Refusing to fetch internal hostname" in result["http://localhost/x"]
    assert "Refusing to fetch private/internal" in result["http://192.168.1.1/admin"]
    # And the safe URL came back with its content.
    assert "ok" in result["https://good.example.com/page"]


@pytest.mark.asyncio
async def test_extract_urls_no_api_key_returns_per_url_messages(monkeypatch):
    monkeypatch.setattr(_search.settings, "tavily_api_key", "")
    result = await _search.extract_urls(["https://a/", "https://b/"])
    assert "not configured" in result["https://a/"].lower()
    assert "not configured" in result["https://b/"].lower()


@pytest.mark.asyncio
async def test_extract_urls_handles_tavily_failure(monkeypatch):
    """If Tavily 5xx's, every URL in the batch lands as failure (the
    /extract endpoint is atomic per request — no partial recovery)."""
    monkeypatch.setattr(_search.settings, "tavily_api_key", "fake-key")
    _search._EXTRACT_CACHE.clear()

    with patch(
        "httpx.AsyncClient.post",
        new=AsyncMock(side_effect=httpx.HTTPStatusError("500", request=None, response=None)),
    ):
        result = await _search.extract_urls(["https://x/"])
    assert "Failed to fetch" in result["https://x/"]


@pytest.mark.asyncio
async def test_extract_urls_truncates_long_content(monkeypatch):
    monkeypatch.setattr(_search.settings, "tavily_api_key", "fake-key")
    _search._EXTRACT_CACHE.clear()

    big = "x" * (_search.FETCH_MAX_CHARS + 5000)
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(return_value={
        "results": [{"url": "https://big/", "raw_content": big}],
    })
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake_response)):
        result = await _search.extract_urls(["https://big/"])
    body = result["https://big/"]
    assert "truncated at" in body
    # The truncation marker is appended; length is the cap + marker length.
    assert len(body) < _search.FETCH_MAX_CHARS + 200    # generous bound
