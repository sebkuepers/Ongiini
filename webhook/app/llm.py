import json
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from .config import settings
from .search import web_search

SYSTEM_PROMPT = """You are Ongiini, a friendly AI assistant built for people in Namibia.

- Respond in the language the user wrote in. You speak English and Afrikaans fluently.
- Be concise and warm. Default to short, mobile-friendly replies (a few sentences).
- Use the web_search tool whenever the answer depends on current information
  (weather, news, prices, sports, recent events, opening hours, exchange rates, etc.).
- If you are unsure whether information is current, search.
- Never invent facts. If you don't know and can't search, say so honestly.
- You are aware that you are reached via WhatsApp; keep formatting plain text
  (no Markdown, no tables) and avoid very long messages.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use this whenever the user asks "
                "about weather, news, prices, exchange rates, sports, recent events, "
                "opening hours, or anything else that may have changed recently."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query in natural language.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

client = AsyncOpenAI(base_url=settings.vllm_base_url, api_key="not-needed")


@dataclass
class LLMResult:
    reply: str
    tokens_in: int
    tokens_out: int
    used_search: bool


async def respond(history: list[dict[str, Any]], user_text: str) -> LLMResult:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    used_search = False
    tokens_in = 0
    tokens_out = 0

    for _ in range(4):
        resp = await client.chat.completions.create(
            model=settings.vllm_model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.6,
            max_tokens=600,
        )
        if resp.usage:
            tokens_in += resp.usage.prompt_tokens or 0
            tokens_out += resp.usage.completion_tokens or 0

        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []

        if not tool_calls:
            return LLMResult(
                reply=(msg.content or "").strip() or "Sorry, I couldn't come up with a reply.",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                used_search=used_search,
            )

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "web_search":
                used_search = True
                result = await web_search(args.get("query", ""))
            else:
                result = f"Unknown tool: {name}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    return LLMResult(
        reply="Sorry, I'm having trouble answering that right now.",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        used_search=used_search,
    )
