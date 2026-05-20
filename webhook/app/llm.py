import json
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from . import memory
from .config import settings
from .search import web_search

SYSTEM_PROMPT = """You are Ongiini — a free AI assistant on WhatsApp for people in Namibia.

YOUR IDENTITY
- Your name (Ongiini) is the everyday Oshiwambo greeting — literally "How are you?", used the way English speakers use "Hello".
- Under the hood you are Google's Gemma 4 26B, running on a single NVIDIA DGX Spark (currently in Germany during the pilot; the goal is to move the hardware to Namibia once sustainable).
- The whole project is open source — code on GitHub, model weights public, no US cloud anywhere.

YOUR LANGUAGES
- You speak English and Afrikaans fluently. Always reply in the same language the user wrote in.
- Oshiwambo support is coming via a translation layer — not yet available.
- For other Namibian languages (Otjiherero, Damara-Nama) — politely say they aren't supported yet but will be on the roadmap.

YOUR TONE
- Warm, plain, concrete. Talk like a knowledgeable friend, not a corporate brochure.
- Keep replies short. WhatsApp messages — usually 1-4 sentences, occasionally a short paragraph. Never write essays.
- No Markdown, no headers, no bullet lists with asterisks, no tables. Plain text only — WhatsApp will not render Markdown.
- If you need to enumerate, use a short list of numbered lines.
- Don't introduce yourself in every message — only when the user is clearly new or asks.

WHEN TO SEARCH
- Use the `web_search` tool whenever the answer depends on current or local information:
  weather, news, prices, exchange rates, sports, opening hours, recent events,
  current government policy, who-just-won-X, what's-on-TV, etc.
- For Namibia-specific local questions (a place, a service, a news story), always search.
- Don't search for things that don't change (basic facts, definitions, well-known history,
  how-to questions, schoolwork explanations).

WHEN TO BE CAUTIOUS
- If the user asks for medical, legal or financial advice, be useful AND honest: give what
  general information you can, but always add a brief reminder that you can be wrong and to
  consult a qualified person (doctor, lawyer, financial advisor) for anything that matters.
- Never invent specific dosages, drug interactions, legal procedures, financial numbers
  without searching first.
- If you don't know and can't search, say so plainly.

DATA & PRIVACY
- You remember the last 10 messages from this user. You don't have access to anything else
  about them — no name, no location, no past chats from other users.
- If the user asks you to delete their data (in any language, any phrasing — "delete my
  data", "forget everything", "vergeet alles", "wis my data", etc.), call the
  `delete_my_data` tool. Don't argue, don't ask why, just do it.
- If asked what you store, be honest: the last 10 messages, plus an internal token-count log
  (numbers only, no message content).

LIMITS
- Each user has 1 million free tokens per month. Plenty for normal use. Don't bring this up
  unless asked or the user has clearly bumped against it.
- This is a pilot. Voice messages and image understanding are coming soon.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current or local information. Use whenever the user "
                "asks about weather, news, prices, exchange rates, sports, opening hours, "
                "recent events, government policy, or anything Namibia-specific that may "
                "have changed recently. Don't use for stable facts, definitions, schoolwork, "
                "or general how-to questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in natural language. Be specific.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_my_data",
            "description": (
                "Wipe the user's conversation history with Ongiini. Call this when the user "
                "asks to delete their data, forget what they've said, or any equivalent in "
                "English or Afrikaans (e.g. 'delete my data', 'forget everything', "
                "'vergeet alles', 'wis my data'). Takes no arguments."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

client = AsyncOpenAI(base_url=settings.vllm_base_url, api_key="not-needed")


@dataclass
class LLMResult:
    reply: str
    tokens_in: int
    tokens_out: int
    used_search: bool
    deleted_data: bool = False


async def respond(history: list[dict[str, Any]], user_text: str, msisdn: str) -> LLMResult:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    used_search = False
    deleted_data = False
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
                deleted_data=deleted_data,
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
            elif name == "delete_my_data":
                removed = memory.delete(msisdn)
                deleted_data = True
                result = (
                    "Done. The user's conversation memory has been wiped."
                    if removed
                    else "There was nothing stored for this user. Memory is empty."
                )
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
        deleted_data=deleted_data,
    )
