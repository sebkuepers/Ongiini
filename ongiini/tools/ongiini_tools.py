"""Ongiini's tool catalogue — all @tool-decorated functions.

Tools that need access to per-request state (the user's msisdn, the
runtime's memory provider, etc.) take a ``ToolContext`` as their first
parameter. The @tool decorator excludes ToolContext from the OpenAI
schema so the model never sees it.

Tool descriptions are the same body the old hand-rolled dict schemas
in ``llm.py`` carried — these are the prompts the model reads when
deciding whether to call. Don't shorten without re-running the eval.

External dependencies (Tavily for web_search/fetch_url, the product
docs file for lookup_ongiini_docs, usage.py for my_token_usage) are
imported lazily / used via runtime context where possible, so the
test harness can mock cleanly.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from owela import ToolContext, tool

from .. import usage as _usage
from ..memory import short_term as _memory
from ..search import extract_urls as _extract_urls_impl
from ..search import fetch_url as _fetch_url_impl
from ..search import web_search as _web_search_impl

log = logging.getLogger("ongiini.tools")


# Tools cap fetch_urls at 5 for prompt-budget reasons (Tavily's /extract
# accepts up to 20 per batch — see ongiini/search.py).
_FETCH_URLS_CAP = 5


# ----------------------------- search tools -----------------------------

@tool(
    name="web_search",
    description=(
        "Search the web for current or local information. Call this BEFORE "
        "answering any factual question that touches Namibia — places, "
        "businesses, organisations, ministries, schools, hospitals, prices, "
        "fees, opening hours, news, exchange rates, current events. ALSO "
        "ALWAYS call for existence/naming questions: 'are there any X in "
        "Namibia?', 'which companies provide Y?', 'name a few Z', 'give me "
        "2-3 examples'. Your training data is stale on Namibian specifics; "
        "never answer those from memory. Do NOT call for pure science, "
        "definitions, schoolwork explanations, generic how-tos with no "
        "local angle, or questions about Ongiini itself (use "
        "lookup_ongiini_docs instead). After the tool returns, cite at "
        "least one full deep URL (not the publication homepage) on its "
        "own line before your next-step question."
    ),
    params={
        "query": "Search query in natural language. Be specific.",
        "topic": (
            "Tavily search topic. 'general' for most queries (default). "
            "'news' for breaking events / current-affairs queries — usually "
            "the planner pre-selects this for you."
        ),
        "time_range": (
            "Restrict to recent results: 'day', 'week', 'month', 'year'. "
            "Leave empty for no time restriction. Usually pre-selected by "
            "the planner for recency-sensitive queries."
        ),
    },
)
async def web_search(
    query: str,
    topic: str = "general",
    time_range: str = "",
) -> tuple[str, dict[str, Any]]:
    """Wrap ``_web_search_impl`` and surface the URL list to the
    executor via the tuple-return contract.

    Returning ``(text, attrs_dict)`` lets the @tool registry attach
    ``{"urls": [...]}`` to the ToolStep.attrs — the executor's auto-
    follow-up synthesis machinery reads URLs from there to escalate
    from search to fetch_urls without an LLM call. The model only
    sees ``text``.
    """
    text, urls = await _web_search_impl(
        query, topic=topic, time_range=time_range or None,
    )
    return text, {"urls": urls}


@tool(
    name="fetch_url",
    description=(
        "Read the full cleaned text of a single web page. Call this after a "
        "`web_search` whenever you need more than a short summary — especially "
        "for VERBATIM TEXT requests (a constitutional article, a law section, "
        "a contract clause, an exact quote from a press release). Search snippets "
        "are routinely TRUNCATED and will omit qualifying clauses; only fetching "
        "the full page gives you the actual wording. Pass exactly one URL from a "
        "previous search result. If you are about to quote anything as verbatim, "
        "you MUST have called fetch_url first — no exceptions."
    ),
    params={"url": "The full URL to fetch (must start with http:// or https://)."},
)
async def fetch_url(url: str) -> str:
    return await _fetch_url_impl(url)


@tool(
    name="fetch_urls",
    description=(
        "Fetch up to 5 URLs IN PARALLEL and return their cleaned text. Use "
        "this INSTEAD OF calling `fetch_url` repeatedly when a question needs "
        "to compare information from multiple sources — typical case is a "
        "research-shaped query that returned 3+ promising URLs from "
        "`web_search` and you want to read several of them. Each page's "
        "text is labelled by URL in the output so you can cite which "
        "fact came from which source. Pass a list of full URLs."
    ),
    params={
        "urls": (
            "List of up to 5 full URLs (each must start with http:// or "
            "https://). Pass URLs verbatim from previous search results."
        ),
    },
)
async def fetch_urls(urls: list[str]) -> str:
    """Batched fetch — ONE Tavily ``/extract`` call returns all results.

    Saves N-1 HTTP round trips vs the old asyncio.gather-per-URL
    approach. Tavily handles parallelism server-side. Per-URL failures
    land as inline ``[fetch failed: ...]`` markers so the model can
    work with whichever pages came back.
    """
    if not urls:
        return "No URLs supplied."
    if len(urls) > _FETCH_URLS_CAP:
        urls = urls[:_FETCH_URLS_CAP]    # cap silently — model gets the top N

    results = await _extract_urls_impl(urls)

    parts: list[str] = []
    for url in urls:
        body = results.get(url, f"[fetch failed: no result for {url}]")
        parts.append(f"## {url}\n{body}")
    return "\n\n".join(parts)


# ----------------------------- admin tools -----------------------------

@tool(
    name="delete_my_data",
    description=(
        "Wipe EVERYTHING Ongiini has stored about this user — both the recent "
        "conversation history AND every long-term fact ever extracted. Call this "
        "when the user asks to delete their data, forget what they've said, "
        "wipe their record, or any equivalent in English or Afrikaans "
        "(e.g. 'delete my data', 'forget everything', 'vergeet alles', "
        "'wis my data'). Takes no arguments."
    ),
)
async def delete_my_data(ctx: ToolContext) -> str:
    removed = await ctx.runtime.memory.delete_all(ctx.user_id)
    if removed:
        return (
            "Done. The user's short-term conversation history AND every "
            "stored long-term fact about them have been wiped. IMPORTANT — "
            "your reply MUST make the privacy model explicit so the user "
            "understands what just happened. Tell them clearly, in their "
            "language: (1) their data is now deleted; (2) if they close "
            "WhatsApp now and don't message you again, you'll have nothing "
            "about them — they walk away clean; (3) deletion is a RESET, "
            "not an opt-out — the moment they send a new message, you'll "
            "start remembering again, because that's how conversational "
            "memory works; (4) they can run 'delete my data' any time to "
            "reset again. Warm tone, no legalese, no corporate hedging."
        )
    return (
        "There was nothing stored for this user — short-term history and "
        "long-term memory are both empty. Confirm to them in a friendly "
        "one-liner that there was nothing to delete (e.g. 'You're already "
        "a clean slate — nothing was stored about you'), and mention that "
        "any new message will start a fresh memory record."
    )


@tool(
    name="whats_in_my_memory",
    description=(
        "Surface EVERYTHING currently stored about THIS user across BOTH memory "
        "tiers: the long-term facts mem0 has extracted across all prior "
        "conversations (location, language preference, projects, recurring "
        "topics) PLUS the recent short-term conversation history. Call this "
        "whenever the user asks 'what do you remember about me?', 'what have "
        "you stored?', 'show me my data', 'wat onthou jy oor my?', 'wat het "
        "julle gestoor?' or any equivalent. After the tool returns, present "
        "the result naturally — lead with the durable facts in your own words, "
        "then only mention recent chat if it adds something. Never dump raw "
        "JSON or the bullet list verbatim. Takes no arguments."
    ),
)
async def whats_in_my_memory(ctx: ToolContext) -> str:
    # Short-term goes via the memory module directly (the provider
    # protocol intentionally doesn't expose raw history because most
    # callers shouldn't see it).
    stored = _memory.load(ctx.user_id)
    long_term = await ctx.runtime.memory.list_all(ctx.user_id)

    if not stored and not long_term:
        return (
            "Memory for this user is currently empty — either this is the "
            "first message, or they recently asked to have it deleted."
        )

    parts: list[str] = []
    if long_term:
        parts.append(
            f"Long-term memory ({len(long_term)} facts about this user):"
        )
        parts.append(ctx.runtime.memory.format_facts(long_term))

    if stored:
        if parts:
            parts.append("")
        parts.append(
            f"Recent conversation ({len(stored)} entries, oldest first):"
        )
        for m in stored:
            role = m.get("role", "?")
            content = (m.get("content") or "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            content = (content or "").strip()
            if len(content) > 240:
                content = content[:240] + "…"
            parts.append(f"- [{role}] {content}")
    return "\n".join(parts)


@tool(
    name="my_token_usage",
    description=(
        "Look up THIS user's PERSONAL token usage for the current calendar "
        "month — how many tokens THEY have spent so far, how close THEY are "
        "to running out. Call this ONLY when the user asks about their own "
        "current usage ('how many tokens have I used?', 'am I close to the "
        "limit?', 'hoeveel tokens het ek gebruik?', 'hoe naby is ek aan die "
        "limiet?'). For policy questions about the limit itself, what counts "
        "against it, how the system is designed, or any general question "
        "about Ongiini's pricing / quotas, use `lookup_ongiini_docs` instead "
        "— that's the canonical product knowledge. After this tool returns, "
        "summarise the numbers in plain language, don't dump them as a list "
        "or JSON. Mention that the counter resets on the 1st of each month. "
        "Takes no arguments."
    ),
)
async def my_token_usage(ctx: ToolContext) -> str:
    stats = _usage.summary_for(ctx.user_id)
    brk = stats.get("breakdown") or {}
    chat = brk.get("chat") or {"tokens_in": 0, "tokens_out": 0}
    mem_t = brk.get("memory") or {"tokens_in": 0, "tokens_out": 0}
    sum_t = brk.get("summary") or {"tokens_in": 0, "tokens_out": 0}
    chat_total = chat["tokens_in"] + chat["tokens_out"]
    mem_total = mem_t["tokens_in"] + mem_t["tokens_out"]
    sum_total = sum_t["tokens_in"] + sum_t["tokens_out"]
    return (
        f"Token usage for this user, month {stats['month']} (UTC):\n"
        f"- {stats['messages']} messages so far this month\n"
        f"- {stats['tokens_total']} tokens total of {stats['limit']} "
        f"monthly allowance ({stats['percent_used']}% used)\n"
        f"  · chat (replies, incl. any images you sent): {chat_total} tokens\n"
        f"  · long-term memory updates: {mem_total} tokens\n"
        f"  · rolling-summary compressions: {sum_total} tokens\n"
        f"Everything counts toward the monthly allowance, including the "
        f"behind-the-scenes memory work. Counter resets on the 1st of next month."
    )


# ----------------------------- docs lookup -----------------------------

_PRODUCT_DOCS_PATH = Path(__file__).resolve().parents[1] / "knowledge" / "product.md"
_product_docs_cache: str | None = None


def _load_product_docs() -> str:
    """Return the canonical product.md, loaded once per process.

    Missing file → soft fallback so the container keeps serving even
    if a deploy mis-ordered product.md regeneration."""
    global _product_docs_cache
    if _product_docs_cache is not None:
        return _product_docs_cache
    try:
        _product_docs_cache = _PRODUCT_DOCS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.warning(
            "product knowledge file missing at %s — lookup_ongiini_docs will "
            "return the fallback message until the next deploy.",
            _PRODUCT_DOCS_PATH,
        )
        _product_docs_cache = (
            "The product knowledge file isn't available right now — this "
            "is a deployment bug. Tell the user you can't look up the "
            "canonical answer at the moment and point them at "
            "https://ongiini.ai/product.md, which has the same content."
        )
    return _product_docs_cache


@tool(
    name="lookup_ongiini_docs",
    description=(
        "Look up authoritative information about Ongiini itself — what it is, "
        "how it works, what's stored / how long, what languages are supported, "
        "the monthly token limit and what counts against it, who runs the "
        "project, where the hardware is, why a German number, GDPR / EU AI Act "
        "status, Privacy Policy clauses, Terms of Service clauses, Imprint — "
        "ANY 'meta' question about Ongiini as a service. Returns the full "
        "canonical product knowledge as markdown (FAQ + Privacy Policy + Terms "
        "+ Imprint), regenerated from the website on every deployment. "
        "Always call this BEFORE answering a question about Ongiini itself; "
        "do not guess from memory. After the tool returns, paraphrase the "
        "relevant section in the user's own language (EN or AF), keep it "
        "conversational, do not paste raw markdown back to the user. Takes "
        "no arguments — the whole doc is returned at once, so a single call "
        "per turn is enough."
    ),
)
async def lookup_ongiini_docs() -> str:
    return _load_product_docs()


# ----------------------------- registry -----------------------------

# The canonical list passed to ToolRegistry at runtime build time. Order
# is the same the model sees in the function-call schemas list; first
# tool listed gets a slight prior. Ordered most-used first.
ALL_TOOLS: tuple[Any, ...] = (
    web_search,
    fetch_url,
    fetch_urls,
    delete_my_data,
    whats_in_my_memory,
    my_token_usage,
    lookup_ongiini_docs,
)
