import httpx

from .config import settings

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"

# Generous but bounded — ~1500 tokens of page text per fetch is plenty for
# WhatsApp replies and keeps the prompt small.
FETCH_MAX_CHARS = 6000


async def web_search(query: str, max_results: int = 5) -> str:
    """Query Tavily with a Namibia bias and return a compact text block.

    Tavily's `country` parameter weights results toward sources from that
    country. We use it unconditionally because the audience is in Namibia —
    even queries that are not obviously local (e.g. "exchange rate", "current
    weather") almost always mean *in Namibia* for our users.
    """
    if not settings.tavily_api_key:
        return "Web search is not configured."

    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "country": "namibia",
        "include_answer": True,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(TAVILY_SEARCH_URL, json=payload)
        r.raise_for_status()
        data = r.json()

    parts: list[str] = []
    if answer := data.get("answer"):
        parts.append(f"Summary: {answer}")
    for i, res in enumerate(data.get("results", [])[:max_results], start=1):
        title = res.get("title", "").strip()
        content = res.get("content", "").strip()
        url = res.get("url", "").strip()
        parts.append(f"[{i}] {title}\n{content}\n{url}")
    return "\n\n".join(parts) if parts else "No results found."


async def fetch_url(url: str, max_chars: int = FETCH_MAX_CHARS) -> str:
    """Fetch and clean the full text of a single web page via Tavily extract.

    The model should call this after a `web_search` when a snippet looks
    promising but is too short to fully answer the question.
    """
    if not settings.tavily_api_key:
        return "URL fetch is not configured."

    url = (url or "").strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return "Error: URL must start with http:// or https://"

    payload = {
        "api_key": settings.tavily_api_key,
        "urls": [url],
        "extract_depth": "basic",
        "format": "text",
    }
    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.post(TAVILY_EXTRACT_URL, json=payload)
        r.raise_for_status()
        data = r.json()

    results = data.get("results") or []
    if not results:
        failed = data.get("failed_results") or []
        if failed:
            reason = (failed[0] or {}).get("error", "unknown error")
            return f"Failed to fetch {url}: {reason}"
        return f"No content extracted from {url}."

    res = results[0] or {}
    raw = (res.get("raw_content") or "").strip()
    if not raw:
        return f"Page returned no readable content: {url}"

    if len(raw) > max_chars:
        raw = raw[:max_chars] + f"\n\n[truncated at {max_chars} characters]"

    return f"Fetched: {url}\n\n{raw}"
