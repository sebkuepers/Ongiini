import httpx

from .config import settings

TAVILY_URL = "https://api.tavily.com/search"


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
        r = await client.post(TAVILY_URL, json=payload)
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
