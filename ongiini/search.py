"""Tavily web search + URL extraction with Namibia bias.

v1.3 changes (search-quality overhaul):

  - ``web_search`` returns ``(text, urls)``: the formatted snippet block
    for the model AND the structured URL list for the executor's
    auto-follow-up synthesis machinery. Existing string-only callers
    can still get just the text via convenience helpers below.

  - ``search_depth: "advanced"`` and ``extract_depth: "advanced"`` —
    Tavily's higher-quality tiers. Cost ~2x credits each; the upgraded
    plan absorbs this.

  - ``include_raw_content`` + ``chunks_per_source`` on /search return
    substantially more page content per result. Combined with
    multi-query fan-out, the model gets dense, attributable evidence
    before any fetch even fires.

  - ``max_results`` bumped to 10 (was 5) so the URL ranker downstream
    has a real candidate pool to apply diversity to.

  - Optional ``topic`` (general | news) and ``time_range``
    (day/week/month/year) per call — the planner picks these per query
    variant so news / recency-sensitive queries route correctly.

  - ``extract_urls(urls)`` batches up to 5 URLs into ONE Tavily
    ``/extract`` call. ``fetch_url`` becomes a thin convenience
    wrapper that calls ``extract_urls([url])``.

  - ``FETCH_MAX_CHARS`` bumped 6000 → 12000. Gemma 4's context window
    is 128K; page text is gold tokens and we were being conservative
    for no reason. The executor's per-result truncation is the right
    place to cap model-visible context.

  - In-process TTL cache (60s) on ``(query, topic, time_range)`` for
    /search and (longer TTL, 120s) on canonical URLs for /extract.
    Two users asking "BoN exchange rate today" within a minute share
    the result.

SSRF guard (``_safe_url``) is unchanged — defence-in-depth refusing to
forward internal/private addresses to Tavily even though Tavily would
itself reject them.
"""

from __future__ import annotations

import ipaddress
import logging
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from .config import settings

log = logging.getLogger("ongiini.search")

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"

# Per-URL maximum characters returned from /extract. Bumped from 6000 →
# 12000 because Gemma 4 has 128K context — the executor's per-result
# truncation handles model-visible bounding. Reviewers see the full
# pre-truncation text from ``ToolStep.attrs["result"]``.
FETCH_MAX_CHARS = 12000

# Default /search params. Each is overridable per-call via web_search()
# kwargs.
#
# v1.3.1 reduced max_results 10→6 and chunks_per_source 3→1 — for
# SEARCH_DEEP the auto-follow-up to fetch_urls supplies real depth,
# so the search response only needs to give the URL ranker enough
# candidates (6 results × 1 chunk is plenty). Cuts per-search payload
# size ~5× without losing the diversity needed for top-per-host
# selection.
_DEFAULT_MAX_RESULTS = 6
_DEFAULT_CHUNKS_PER_SOURCE = 1
_DEFAULT_COUNTRY = "namibia"
_VALID_TOPICS = {"general", "news"}
_VALID_TIME_RANGES = {None, "day", "week", "month", "year"}

# Tavily /extract endpoint hard cap is 20 URLs per call, but the
# fetch_urls tool caps at 5 for prompt-budget reasons. Keep them in
# sync.
_EXTRACT_BATCH_MAX = 20

# SSRF block list — refuse to forward these to Tavily even though Tavily
# would itself reject them. Defence-in-depth.
_BLOCKED_HOSTNAMES = {
    "localhost",
    "spark-dccf",
    "spark-dccf.local",
    "host.docker.internal",
}
_BLOCKED_IP_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),    # AWS/GCP/Azure metadata
    ipaddress.ip_network("100.64.0.0/10"),     # carrier-grade NAT / Tailscale
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _safe_url(url: str) -> tuple[bool, str]:
    """Return (ok, reason). Reject internal / private / non-http(s) URLs."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "URL could not be parsed."

    if parsed.scheme not in ("http", "https"):
        return False, f"URL must use http or https (got {parsed.scheme!r})."

    host = (parsed.hostname or "").lower()
    if not host:
        return False, "URL has no hostname."

    if host in _BLOCKED_HOSTNAMES:
        return False, "Refusing to fetch internal hostname."

    try:
        ip = ipaddress.ip_address(host)
        for net in _BLOCKED_IP_NETS:
            if ip in net:
                return False, f"Refusing to fetch private/internal address {ip}."
    except ValueError:
        pass

    return True, ""


# --------------------------------- TTL cache ---------------------------------

class _TTLCache:
    """In-process cache with per-entry TTL. Not thread-safe (we're in
    one asyncio loop per process) and not LRU — we trim opportunistically
    when the store exceeds a soft cap.

    Use case: two webhook calls within ~60 seconds asking the same
    question hit Tavily once. Hit rate matters for tail latency on
    common queries (BoN exchange rate, prime rate, etc.).
    """

    def __init__(self, ttl_seconds: float, soft_max: int = 200) -> None:
        self._ttl = ttl_seconds
        self._soft_max = soft_max
        self._store: dict[tuple, tuple[float, Any]] = {}

    def get(self, key: tuple) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expiry, value = entry
        if time.monotonic() > expiry:
            self._store.pop(key, None)
            return None
        return value

    def put(self, key: tuple, value: Any) -> None:
        self._store[key] = (time.monotonic() + self._ttl, value)
        if len(self._store) > self._soft_max:
            self._evict_expired()
            # Hard cap: if expiry-based eviction didn't free space (high
            # churn, nothing expired yet), drop oldest insertions until
            # back under the soft_max. Python 3.7+ dicts preserve
            # insertion order so this is correct FIFO.
            while len(self._store) > self._soft_max:
                self._store.pop(next(iter(self._store)), None)

    def _evict_expired(self) -> None:
        now = time.monotonic()
        stale = [k for k, (exp, _) in self._store.items() if exp <= now]
        for k in stale:
            self._store.pop(k, None)

    def clear(self) -> None:
        """For tests. Production code shouldn't need this."""
        self._store.clear()


# 60s on /search — Namibia-relevant facts (exchange rates, news) move
# slowly enough that a 60s window is safe for cache aliasing.
_SEARCH_CACHE = _TTLCache(ttl_seconds=60.0)
# 120s on /extract — page bodies are even more stable. Extracted text
# costs more credits than search snippets, so cache longer.
_EXTRACT_CACHE = _TTLCache(ttl_seconds=120.0)


def _canonical_url(url: str) -> str:
    """Best-effort canonicalisation for cache keys + dedup.

    Strips trailing slash from the path, drops a stray fragment, keeps
    the query string intact (URLs with query params are meaningfully
    different from those without). Not a full RFC 3986 normalisation —
    just enough to dedupe trivial duplicates from search results.

    Malformed inputs (no scheme / no host) are returned verbatim — the
    cache layer will just key them as-is, never colliding with real
    URLs.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    if not parsed.scheme or not parsed.netloc:
        return url
    path = parsed.path.rstrip("/") if len(parsed.path) > 1 else parsed.path
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    rebuilt = f"{scheme}://{netloc}{path}"
    if parsed.query:
        rebuilt = f"{rebuilt}?{parsed.query}"
    return rebuilt


# --------------------------------- /search ---------------------------------

async def web_search(
    query: str,
    *,
    topic: str = "general",
    time_range: str | None = None,
    max_results: int = _DEFAULT_MAX_RESULTS,
    include_raw_content: bool = True,
) -> tuple[str, list[str]]:
    """Query Tavily and return (formatted_text, urls).

    The text block is what the model sees: each result rendered as
    ``[N] title\\n<content>\\n<url>`` with Tavily's AI-generated
    ``Summary:`` line first if present.

    The URL list is the executor's hook for auto-follow-up synthesis —
    the rule-based fetch_urls escalation reads URLs from
    ``ToolStep.attrs["urls"]`` (the tool stashes this list there).

    ``topic`` is ``"general"`` (default) or ``"news"``. The planner
    sets ``"news"`` for breaking-events queries.

    ``time_range`` is ``None`` / ``"day"`` / ``"week"`` / ``"month"`` /
    ``"year"``. The planner sets a recency window for queries with
    explicit recency words ("today", "this week", "latest").

    ``include_raw_content`` defaults True (helpful for SEARCH_SHALLOW
    where no fetch_urls follow-up runs). SEARCH_DEEP sets this False
    via ``Policy.planner_query_default_args`` — the auto-followup to
    fetch_urls supplies real depth, so embedding raw_content in the
    search response too is redundant ~10× tokens.
    """
    if not settings.tavily_api_key:
        return "Web search is not configured.", []

    topic = topic if topic in _VALID_TOPICS else "general"
    if time_range not in _VALID_TIME_RANGES:
        time_range = None

    cache_key = (query.strip().lower(), topic, time_range, bool(include_raw_content))
    cached = _SEARCH_CACHE.get(cache_key)
    if cached is not None:
        # Return a fresh mutable list copy of the cached URL tuple so
        # the caller can mutate without poisoning future hits.
        cached_text, cached_urls = cached
        return cached_text, list(cached_urls)

    payload: dict[str, Any] = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": max_results,
        # "advanced" returns substantially more content per snippet than
        # "basic" (~2x Tavily credits, worth it). Live-test showed Gemma 4
        # was prematurely satisfied by thin "basic" snippets and never
        # escalated to fetch_urls; advanced gives the model enough
        # context to either answer directly OR realise it still needs
        # the full page.
        "search_depth": "advanced",
        "chunks_per_source": _DEFAULT_CHUNKS_PER_SOURCE,
        "country": _DEFAULT_COUNTRY,
        # Tavily's AI summary at the head of the result block. Helpful
        # for single-source SHALLOW queries; for DEEP queries the
        # executor's auto-follow-up to fetch_urls supplies the real
        # depth.
        "include_answer": True,
        # Full cleaned page text per result. SEARCH_SHALLOW gets this
        # (no fetch_urls follow-up — raw content from the search IS the
        # depth). SEARCH_DEEP turns this off because fetch_urls auto-
        # followup supplies the deep read, and embedding raw_content
        # in the search response would double-fetch the same pages.
        "include_raw_content": bool(include_raw_content),
        "topic": topic,
    }
    if time_range is not None:
        payload["time_range"] = time_range

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(TAVILY_SEARCH_URL, json=payload)
        r.raise_for_status()
        data = r.json()

    text = _format_search_results(data, max_results=max_results)
    urls: list[str] = []
    for res in data.get("results", [])[:max_results]:
        url = (res.get("url") or "").strip()
        if url:
            urls.append(url)

    # Cache the URL list as an immutable tuple so the cached value
    # cannot be poisoned by downstream mutation (a caller sorting or
    # appending to the returned list would otherwise affect every
    # subsequent cache hit).
    _SEARCH_CACHE.put(cache_key, (text, tuple(urls)))
    return text, list(urls)


def _format_search_results(data: dict[str, Any], *, max_results: int) -> str:
    """Render Tavily's JSON into the snippet block the model reads."""
    parts: list[str] = []
    if answer := data.get("answer"):
        parts.append(f"Summary: {answer}")
    for i, res in enumerate(data.get("results", [])[:max_results], start=1):
        title = (res.get("title") or "").strip()
        content = (res.get("content") or "").strip()
        url = (res.get("url") or "").strip()
        raw = (res.get("raw_content") or "").strip()
        # Compose per-result block: header / snippet / full text (if any) / url.
        block = f"[{i}] {title}\n{content}"
        if raw and raw != content:
            # Cap per-result raw content so a single huge page can't
            # dominate the snippet block. Per-call total truncation
            # happens further down the pipeline (executor + reviewer).
            if len(raw) > FETCH_MAX_CHARS:
                raw = raw[:FETCH_MAX_CHARS] + f"\n[truncated at {FETCH_MAX_CHARS} chars]"
            block += f"\n\nFull text:\n{raw}"
        block += f"\n{url}"
        parts.append(block)
    return "\n\n".join(parts) if parts else "No results found."


# --------------------------------- /extract ---------------------------------

async def extract_urls(urls: list[str]) -> dict[str, str]:
    """Batched /extract: fetch the cleaned full text of N URLs in ONE
    Tavily call. Returns a dict ``{url: text_or_error_string}``.

    Tavily's endpoint accepts up to 20 URLs per call; we let callers
    pass any list and silently respect that cap.

    SSRF guard: each URL is checked with ``_safe_url`` BEFORE we
    forward to Tavily. Blocked URLs land in the returned dict with an
    explanatory error string so the caller can still see what got
    rejected.

    Caches per-URL (canonical) for 120s.
    """
    if not settings.tavily_api_key:
        return {u: "URL fetch is not configured." for u in urls}

    if not urls:
        return {}

    # Cap defensively.
    if len(urls) > _EXTRACT_BATCH_MAX:
        urls = urls[:_EXTRACT_BATCH_MAX]

    out: dict[str, str] = {}
    to_fetch: list[str] = []
    # Split: cache hits, SSRF rejects, real fetches.
    for u in urls:
        u_clean = (u or "").strip()
        if not u_clean:
            continue
        ok, reason = _safe_url(u_clean)
        if not ok:
            out[u_clean] = f"Error: {reason}"
            continue
        cached = _EXTRACT_CACHE.get((_canonical_url(u_clean),))
        if cached is not None:
            out[u_clean] = cached
            continue
        to_fetch.append(u_clean)

    if to_fetch:
        payload = {
            "api_key": settings.tavily_api_key,
            "urls": to_fetch,
            # "advanced" extraction returns cleaner page text (better
            # at stripping nav / footers / cookie banners) at ~2x
            # Tavily credit cost. Worth it because the next thing we
            # do is feed this directly into the model's context.
            "extract_depth": "advanced",
            "format": "text",
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(TAVILY_EXTRACT_URL, json=payload)
                r.raise_for_status()
                data = r.json()
        except Exception as exc:                       # noqa: BLE001
            log.warning("extract_urls Tavily call failed: %s", exc)
            for u in to_fetch:
                out[u] = f"Failed to fetch {u}: {exc}"
            return out

        # Successes
        for res in data.get("results", []) or []:
            url = (res.get("url") or "").strip()
            raw = (res.get("raw_content") or "").strip()
            if not url:
                continue
            if not raw:
                out[url] = f"Page returned no readable content: {url}"
                continue
            if len(raw) > FETCH_MAX_CHARS:
                raw = raw[:FETCH_MAX_CHARS] + f"\n\n[truncated at {FETCH_MAX_CHARS} characters]"
            text = f"Fetched: {url}\n\n{raw}"
            out[url] = text
            _EXTRACT_CACHE.put((_canonical_url(url),), text)

        # Failures
        for fail in data.get("failed_results", []) or []:
            url = (fail.get("url") or "").strip()
            reason = (fail.get("error") or "unknown error").strip()
            if url and url not in out:
                out[url] = f"Failed to fetch {url}: {reason}"

        # Any URLs we asked for but Tavily didn't return at all.
        for u in to_fetch:
            if u not in out:
                out[u] = f"No content extracted from {u}."

    return out


async def fetch_url(url: str, max_chars: int = FETCH_MAX_CHARS) -> str:
    """Convenience wrapper over ``extract_urls`` for single-URL callers.

    Returns the formatted ``Fetched: <url>\\n\\n<text>`` block, or an
    error string on failure. ``max_chars`` is for backwards compatibility
    with the v0 signature; the actual extraction cap is the module-level
    ``FETCH_MAX_CHARS``.
    """
    url = (url or "").strip()
    if not url:
        return "Error: no URL provided."
    results = await extract_urls([url])
    return results.get(url, f"No content extracted from {url}.")
