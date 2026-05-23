"""Planner — pre-act-loop query decomposition for SEARCH_DEEP turns.

v1.3 changes the planner's job: instead of writing PROSE that tries to
steer the model's tool selection (which Gemma 4 routinely ignored —
see live-test history), the planner now emits **structured query
variants** that the executor materialises into parallel tool calls.

Output shape (JSON, terminated by the literal token ``PLAN_DONE``)::

    {
      "facts_known": "1-2 sentence context the model can rely on without searching",
      "queries": [
        {"query": "Bank Windhoek home loan rate 2026", "topic": "general", "time_range": null},
        {"query": "FNB Namibia home loan rate 2026",  "topic": "general", "time_range": null},
        {"query": "Nedbank Namibia home loan rate 2026", "topic": "general", "time_range": null},
        {"query": "Namibia prime lending rate", "topic": "general", "time_range": "month"}
      ]
    }
    PLAN_DONE

The executor reads ``PlanStep.queries`` and synthesises one parallel
``web_search`` call per variant — no LLM step between plan and fan-out.
``PlanStep.plan_text`` carries the prose ``facts_known`` for the
MemoryProvider to inject as context.

Soft-fail contract: any timeout, JSON parse failure, or other error
returns ``PlanStep(plan_text="", queries=[])``. Empty queries means
the executor falls back to letting the model pick the search itself
on turn 1 (gated by ``policy.first_tool``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI

from owela import InboundMessage, PlanStep, Policy, QueryVariant, Step

log = logging.getLogger("ongiini.planner")


# Prompt: ask Gemma for JSON only, with explicit examples per question
# shape. Examples are load-bearing — Gemma 4 emits much cleaner JSON
# with concrete shape templates than with abstract schema descriptions.
_PLAN_PROMPT = """You're about to help a user in Namibia answer this question on WhatsApp:

  {question}
{recent_history}
Plan the search BEFORE any tool runs. Output ONLY a JSON object in this
exact shape (no extra text before or after), then the literal token
PLAN_DONE on its own line.

Shape:
{{
  "facts_known": "1-2 sentences of background you can rely on confidently without searching (e.g. general geography, well-known institutions). Use 'none' if nothing.",
  "queries": [
    {{"query": "<specific search query>", "topic": "general" | "news", "time_range": null | "day" | "week" | "month" | "year"}}
  ]
}}

How to pick queries:

- COMPARISON questions ("compare X, Y, Z", "best 3 banks for...", "cheapest
  medical aid"): emit ONE query per entity AND one context query. Each
  query names the specific entity + the specific fact. 3-5 queries total.
  Example for "compare home loan rates at 3 Namibian banks":
  [
    {{"query": "Bank Windhoek home loan interest rate 2026", "topic": "general", "time_range": null}},
    {{"query": "FNB Namibia home loan rate 2026", "topic": "general", "time_range": null}},
    {{"query": "Nedbank Namibia home loan rate 2026", "topic": "general", "time_range": null}},
    {{"query": "Namibia prime lending rate 2026", "topic": "general", "time_range": "month"}}
  ]

- NEWS / CURRENT EVENTS ("what's happening with...", "is there a strike",
  "any new policy on..."): topic "news", time_range "week" or "month".
  1-2 queries. Example for "Namibian medicine shortage latest":
  [
    {{"query": "Namibia medicine shortage 2026", "topic": "news", "time_range": "month"}},
    {{"query": "Namibia health ministry response medicine shortage", "topic": "news", "time_range": "month"}}
  ]

- VERBATIM / SPECIFIC DATA (an exact price, law section, exact schedule):
  1-2 specific queries; the executor will fetch the source pages
  automatically. Example for "what does Article 16 of the Namibian
  constitution say":
  [
    {{"query": "Article 16 Namibian constitution full text", "topic": "general", "time_range": null}}
  ]

- SINGLE-SOURCE LOOKUPS ("exchange rate today", "BoN repo rate"):
  1 query. Recency-sensitive ones use time_range "day" or "week".
  Example for "BoN exchange rate today":
  [
    {{"query": "Bank of Namibia exchange rate today USD ZAR", "topic": "general", "time_range": "day"}}
  ]

Cap at 5 queries maximum. Keep queries SHORT and ENTITY-SPECIFIC — they
become real search engine queries; padding hurts recall.

PRONOUN RESOLUTION: if the user refers to entities mentioned earlier
in the conversation ("them", "those", "the same", "what about it",
"compare them", "tell me more about the third one"), INFER which
entities they mean from the recent-history block above and generate
ONE query per entity. Do NOT emit an empty queries list because the
question is ambiguous in isolation — that's exactly when the history
context is most valuable. Example: if the previous reply listed
[Paratus, IT Guru, MTN Windhoek] and the user now says "compare
them", emit three queries — one per provider — not zero.

End with PLAN_DONE.
"""


# Output cap. JSON shape with up to 5 query objects fits comfortably
# under 300 completion tokens. Generous to allow for verbose entity
# names ("Bank Windhoek Namibia home loan plus first-time buyer...").
_PLAN_MAX_TOKENS = 320

# Latency budget. If Gemma can't produce a plan in this time, the
# planner returns an empty PlanStep and the executor proceeds without
# fan-out — same shape as a v0 SEARCH_DEEP turn.
_TIMEOUT_S = 4.0

# Soft cap on parsed queries — the executor would dispatch all of
# them but Tavily credits and prompt-budget concerns favour staying
# small. The prompt asks for ≤5; we enforce it on the parser side
# too in case the model emits more.
_MAX_QUERIES = 5


class OngiiniPlanner:
    """Calls Gemma with a planning prompt before the act loop.

    Constructed with the vLLM endpoint + model id; tests inject a
    fake AsyncOpenAI client. The prefix of the prompt is byte-stable
    across requests so vLLM's prefix cache hits on every call after
    warm-up (the variable suffix is just the user's question).
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        *,
        client: AsyncOpenAI | None = None,
        timeout_s: float = _TIMEOUT_S,
        max_tokens: int = _PLAN_MAX_TOKENS,
    ) -> None:
        self.model_id = model_id
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self._client = client or AsyncOpenAI(base_url=base_url, api_key="not-needed")

    async def plan(
        self,
        msg: InboundMessage,
        policy: Policy,
        prior_steps: list[Step],
    ) -> PlanStep:
        started = time.monotonic()
        step = PlanStep(started_at=started)

        question = (msg.text or "").strip()
        if not question:
            # Defensive — caller (the executor) only invokes us when
            # policy.enable_planner is True, which today only fires on
            # SEARCH_DEEP. SEARCH_DEEP needs a question by definition.
            # If we somehow got here without one, return an empty plan.
            step.ended_at = time.monotonic()
            return step

        # v1.3.1: surface the last 2 conversation turns so the planner
        # can resolve pronouns and follow-up references ("compare them",
        # "what about X"). Empty history → empty block, prompt is
        # byte-identical to v1.3 (preserves prefix-cache on first-turn
        # queries).
        recent_history = _format_recent_history(msg.history)

        try:
            resp = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self.model_id,
                    messages=[{
                        "role": "user",
                        "content": _PLAN_PROMPT.format(
                            question=question,
                            recent_history=recent_history,
                        ),
                    }],
                    temperature=0.3,
                    max_tokens=self.max_tokens,
                ),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            log.warning(
                "planner timed out after %ss — proceeding without plan", self.timeout_s,
            )
            step.ended_at = time.monotonic()
            step.attrs["error"] = "timeout"
            return step
        except Exception as exc:                       # noqa: BLE001 — soft-fail
            log.warning("planner failed (%s) — proceeding without plan", exc)
            step.ended_at = time.monotonic()
            step.attrs["error"] = str(exc)
            return step

        billable_in, completion, cached = _billable(resp.usage)
        step.tokens_in = billable_in
        step.tokens_out = completion
        step.cached_tokens = cached

        raw = ""
        if resp.choices:
            raw = (resp.choices[0].message.content or "").strip()

        facts_known, queries = _parse_plan(raw)
        step.plan_text = facts_known
        step.queries = queries
        step.ended_at = time.monotonic()
        return step


def _format_recent_history(history: list[dict[str, Any]]) -> str:
    """Format the last user+assistant exchange (or just the previous
    user message) into a prompt-friendly block. Returns "" when there's
    nothing useful — keeps the prompt byte-stable for prefix-cache hits
    on first-turn queries.

    We take only the LAST 2 entries (the immediately preceding turn).
    Pulling deeper history adds tokens without helping resolution —
    follow-up pronouns almost always point at the IMMEDIATELY prior
    turn, not three turns back.
    """
    if not history:
        return ""
    # Find last user message and the assistant response that followed
    # it (if any). Walk backwards, keeping the last assistant + last
    # user before it.
    last_assistant: str | None = None
    last_user: str | None = None
    for entry in reversed(history):
        role = entry.get("role")
        content = entry.get("content")
        # Only handle plain string content (image-bearing turns use
        # list[dict] content_parts; skip those for the planner — the
        # image isn't useful for query decomposition).
        if not isinstance(content, str):
            continue
        if role == "assistant" and last_assistant is None:
            last_assistant = content.strip()
        elif role == "user" and last_user is None:
            last_user = content.strip()
            break    # we have the immediately prior user turn

    if not last_user and not last_assistant:
        return ""

    parts = ["", "Conversation just before this question (for resolving"
             " pronouns like 'them', 'this', 'what about'):"]
    if last_user:
        # Cap to ~400 chars — we just need pronoun context.
        snippet = last_user[:400] + ("…" if len(last_user) > 400 else "")
        parts.append(f"  PREVIOUS USER: {snippet}")
    if last_assistant:
        snippet = last_assistant[:400] + ("…" if len(last_assistant) > 400 else "")
        parts.append(f"  PREVIOUS REPLY: {snippet}")
    parts.append("")
    return "\n".join(parts)


def _parse_plan(raw: str) -> tuple[str, list[QueryVariant]]:
    """Parse the planner's JSON response.

    Returns ``(facts_known, queries)`` — empty strings/lists on any
    parse failure (soft-fail). The executor treats empty queries as
    "no fan-out — model picks the first query itself."

    Tolerant: ignores text before/after the JSON object, accepts
    malformed queries (skips them while keeping good ones), and
    silently caps at ``_MAX_QUERIES``.
    """
    if not raw:
        return "", []

    # Strip everything from PLAN_DONE onward. Gemma occasionally
    # appends commentary.
    if "PLAN_DONE" in raw:
        raw = raw.split("PLAN_DONE", 1)[0].rstrip()

    # Find first balanced { } block. We don't trust regex with
    # nested braces; manual depth tracking is reliable.
    start = raw.find("{")
    if start == -1:
        return "", []
    depth = 0
    end = -1
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return "", []

    try:
        obj = json.loads(raw[start:end])
    except json.JSONDecodeError as exc:
        log.warning("planner JSON parse failed: %s", exc)
        return "", []

    if not isinstance(obj, dict):
        return "", []

    facts_known_raw = obj.get("facts_known")
    facts_known = facts_known_raw.strip() if isinstance(facts_known_raw, str) else ""
    if facts_known.lower() == "none":
        facts_known = ""

    queries: list[QueryVariant] = []
    raw_queries = obj.get("queries")
    if isinstance(raw_queries, list):
        for q in raw_queries:
            if not isinstance(q, dict):
                continue
            text = q.get("query")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            extra: dict[str, Any] = {}
            # Topic — only "news" needs to land in extra; "general" is
            # the Tavily default and the web_search tool's default,
            # so omitting it from extra is both correct AND keeps the
            # synthesized tool_call args slim.
            topic = q.get("topic")
            if isinstance(topic, str) and topic.strip() == "news":
                extra["topic"] = "news"
            # Time range — null / missing / invalid → omitted (no
            # time restriction).
            tr = q.get("time_range")
            if isinstance(tr, str) and tr.strip() in ("day", "week", "month", "year"):
                extra["time_range"] = tr.strip()
            queries.append(QueryVariant(query=text, extra=extra))
            if len(queries) >= _MAX_QUERIES:
                break

    return facts_known, queries


def _billable(usage_obj: Any) -> tuple[int, int, int]:
    """Same prefix-cache-aware billing logic as the model + classifier."""
    if usage_obj is None:
        return 0, 0, 0
    prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
    cached = 0
    details = getattr(usage_obj, "prompt_tokens_details", None)
    if details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    return max(0, prompt_tokens - cached), completion_tokens, cached
