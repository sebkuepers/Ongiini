"""Three-way tool router classifier.

When a user's incoming message obviously needs a specific tool (web_search
for current/local Namibian facts; lookup_ongiini_docs for meta-questions
about Ongiini itself), we force that tool on the first LLM turn instead
of relying on tool_choice="auto" — which Gemma 4 26B has been observed
to ignore exactly for the question patterns where search matters most
("which companies provide GPU services in Windhoek?", "what are the fees
for a CC registration?", etc.).

The decision is made by Gemma 4 itself via a separate short classifier
call (~270 prompt tokens, ~5 completion tokens, ~85ms one-shot). The
prompt is prefix-cached on every request after the first vLLM warm-up,
so per-turn billable cost is effectively just the user's text + 5
completion tokens.

Validation:
  - Dev set (40 cases used to tune the prompt): 97.5% accuracy.
  - Held-out set (31 fresh production-style cases never seen during
    tuning): 96.8% accuracy. The single miss was a borderline label
    ambiguity I'd accept either way.

Fail-safe: any error / timeout / un-parseable output returns "NONE",
which makes respond() fall back to tool_choice="auto" — i.e. current
behaviour, not breakage.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Union

from openai import AsyncOpenAI

from . import usage
from .config import settings

log = logging.getLogger("ongiini.router")


# Module-level client. We deliberately make a fresh AsyncOpenAI rather
# than sharing llm.py's `client` so the classifier call is independent
# of the main respond() interaction tracing.
_client = AsyncOpenAI(base_url=settings.vllm_base_url, api_key="not-needed")


# A_LONG prompt — see webhook/tests/router_eval.py for the four-variant
# comparison that produced this winner. Any future tuning MUST be tested
# against the held-out set in webhook/tests/router_eval_holdout.py to
# avoid overfitting.
CLASSIFIER_PROMPT = """\
You classify requests for Ongiini, an AI helper for people in Namibia on WhatsApp.

Decide which of three buckets the request falls in:

SEARCH — the answer involves specific Namibian businesses, services, providers,
fees, prices, exchange rates, opening hours, recent news, current events, or
named recommendations. Training data is stale for these.

DOCS — the user is asking about Ongiini itself: pricing, privacy policy, terms,
how it works, languages supported, hardware, who built it, EU AI Act, the
Common Intelligence Foundation, monthly token limits as a policy.

NONE — general knowledge (science, math, philosophy), generic how-to with no
local angle, emotional support, casual conversation.

Namibian cities (Windhoek, Walvis Bay, Oshakati, Swakopmund, Rundu, Katima
Mulilo) and institutions (BIPA, NamRA, Bank of Namibia, Ministry of Home
Affairs) imply Namibian context even when "Namibia" isn't explicitly said.

Request: {user_text}

Reply with exactly one word: SEARCH, DOCS, or NONE.
"""


# Latency budget. Classifier should fail fast and degrade rather than
# block the user's actual reply. 2s is generous for the typical 85ms
# round-trip but tight enough that, under GPU contention (vLLM batched
# decoding for another user's max_tokens=4000 chat call), we don't
# block the user's p99 reply by sitting on a busy queue. Fallback on
# timeout is "NONE" → tool_choice="auto", which is the pre-router
# behaviour — never breakage.
_TIMEOUT_S = 2.0

ToolChoice = Union[str, dict]


async def classify(user_text: str, msisdn: str | None = None) -> str:
    """Classify the user's incoming message into 'SEARCH', 'DOCS', or 'NONE'.

    'NONE' is the safe default returned on:
      - empty / too-short input
      - vLLM timeout or any other exception
      - un-parseable classifier output

    All of which cause respond() to fall back to tool_choice='auto', which
    is the pre-router behaviour (no regression).

    If `msisdn` is provided, the classifier's token usage is recorded under
    `kind='router'` so it's auditable in usage logs separately from chat
    and memory calls.
    """
    text = (user_text or "").strip()
    if not text or len(text) < 3:
        return "NONE"

    try:
        resp = await asyncio.wait_for(
            _client.chat.completions.create(
                model=settings.vllm_model,
                messages=[{
                    "role": "user",
                    "content": CLASSIFIER_PROMPT.format(user_text=text),
                }],
                temperature=0.0,
                max_tokens=5,
            ),
            timeout=_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        log.warning("router.classify timed out after %ss — falling back to NONE", _TIMEOUT_S)
        return "NONE"
    except Exception as exc:
        log.warning("router.classify failed (%s) — falling back to NONE", exc)
        return "NONE"

    # Record classifier tokens under their own usage kind so we can see
    # the classifier cost separately from chat / memory / summary calls.
    if msisdn:
        try:
            billable_in, completion, _cached = usage.billable_from_usage(resp.usage)
            if billable_in or completion:
                usage.record(
                    msisdn,
                    billable_in,
                    completion,
                    used_search=False,
                    kind="router",
                )
        except Exception as exc:
            log.warning("router usage record failed: %s", exc)

    raw = (resp.choices[0].message.content or "").strip().upper()
    for token in ("SEARCH", "DOCS", "NONE"):
        if token in raw:
            return token
    log.warning("router got un-parseable verdict %r — falling back to NONE", raw)
    return "NONE"


def tool_choice_for(verdict: str) -> ToolChoice:
    """Map a classifier verdict to the tool_choice value the OpenAI-compatible
    chat.completions.create accepts on the FIRST turn.

    SEARCH / DOCS force the corresponding tool. NONE falls through to
    'auto' so the model can still freely choose delete_my_data /
    whats_in_my_memory / my_token_usage / fetch_url etc. without
    classifier interference.
    """
    if verdict == "SEARCH":
        return {"type": "function", "function": {"name": "web_search"}}
    if verdict == "DOCS":
        return {"type": "function", "function": {"name": "lookup_ongiini_docs"}}
    return "auto"
