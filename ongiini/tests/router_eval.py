"""Empirical benchmark for using Gemma 4 itself as the request router classifier.

(Note: the word "eval" in this file refers to evaluation/benchmark, not the
Python builtin `eval()` — no arbitrary code execution happens here.)

Two phases:
  1. Calibrate to the 3-way routing decision we actually need in production:
     SEARCH — force web_search        (local/current/Namibian specific facts)
     DOCS   — force lookup_ongiini_docs (asking about Ongiini itself)
     NONE   — tool_choice=auto        (general knowledge, conversation, emotion)
  2. Try four prompt variants of different lengths and find the shortest
     one that still hits near-100% accuracy — every token spared is per-turn
     latency + cost saved on the classifier hop.

Run inside the webhook container:

    docker cp webhook/tests/router_eval.py ongiini-webhook:/data/router_eval.py
    docker exec ongiini-webhook python3 /data/router_eval.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, "/app")

from openai import AsyncOpenAI
from ongiini.config import settings


# ────────────────────────────────────────────────────────────────────
# 40 test cases. expected: "SEARCH" | "DOCS" | "NONE"
# ────────────────────────────────────────────────────────────────────

@dataclass
class Case:
    text: str
    expected: str
    category: str


CASES: list[Case] = [
    # ─── NAMIBIAN FACTUAL → SEARCH ─────────────────────────────────
    Case("are there any datacenters in Namibia?",                     "SEARCH", "namibian_fact"),
    Case("which companies provide GPU services in Windhoek?",         "SEARCH", "namibian_fact"),
    Case("how do I register a business at BIPA?",                     "SEARCH", "namibian_fact"),
    Case("what are the fees for a CC registration in Namibia?",       "SEARCH", "namibian_fact"),
    Case("what does Namibia's AI strategy say?",                      "SEARCH", "namibian_fact"),
    Case("name some Namibian internet providers",                     "SEARCH", "namibian_fact"),
    Case("what data protection laws does Namibia have?",              "SEARCH", "namibian_fact"),
    Case("where is the nearest hospital in Oshakati?",                "SEARCH", "namibian_fact"),
    Case("how do I apply for a Namibian passport?",                   "SEARCH", "namibian_fact"),
    Case("which banks operate in Walvis Bay?",                        "SEARCH", "namibian_fact"),
    # ─── CURRENT DATA → SEARCH ─────────────────────────────────────
    Case("what's the weather in Windhoek today?",                     "SEARCH", "current_data"),
    Case("current rand-dollar exchange rate",                         "SEARCH", "current_data"),
    Case("what's the news in Namibia today?",                         "SEARCH", "current_data"),
    Case("Bank of Namibia repo rate now",                             "SEARCH", "current_data"),
    # ─── PURE SCIENCE → NONE ──────────────────────────────────────
    Case("what is photosynthesis?",                                   "NONE",   "science"),
    Case("explain machine learning to me",                            "NONE",   "science"),
    Case("how do I solve x squared plus 4x plus 4 equals zero?",      "NONE",   "science"),
    Case("what is the difference between mitosis and meiosis?",       "NONE",   "science"),
    Case("what does democracy mean?",                                 "NONE",   "science"),
    # ─── EMOTIONAL / CASUAL → NONE ─────────────────────────────────
    Case("I'm feeling really stressed about my exam",                 "NONE",   "emotional"),
    Case("I had a hard day at work",                                  "NONE",   "emotional"),
    Case("tell me a joke",                                            "NONE",   "casual"),
    Case("I love Namibian music",                                     "NONE",   "casual"),
    # ─── ONGIINI SELF → DOCS ───────────────────────────────────────
    Case("what is my monthly token limit?",                           "DOCS",   "ongiini_self"),
    Case("where is my data stored?",                                  "DOCS",   "ongiini_self"),
    Case("how does Ongiini work?",                                    "DOCS",   "ongiini_self"),
    Case("what languages do you support?",                            "DOCS",   "ongiini_self"),
    Case("what's your privacy policy?",                               "DOCS",   "ongiini_self"),
    Case("who built Ongiini?",                                        "DOCS",   "ongiini_self"),
    # ─── AFRIKAANS ─────────────────────────────────────────────────
    Case("is daar enige datasentrums in Namibië?",                    "SEARCH", "af_namibian"),
    Case("wat is fotosintese?",                                       "NONE",   "af_science"),
    Case("wat is die wisselkoers vir die Namibiese dollar?",          "SEARCH", "af_current"),
    Case("ek voel gestres oor my eksamen",                            "NONE",   "af_emotional"),
    Case("hoe registreer ek 'n besigheid by BIPA?",                   "SEARCH", "af_namibian"),
    Case("hoe werk Ongiini?",                                         "DOCS",   "af_ongiini"),
    # ─── EDGE CASES ────────────────────────────────────────────────
    Case("I'm in Windhoek and feeling stressed",                      "NONE",   "edge_loc_emotion"),
    Case("what is the capital of Namibia?",                           "NONE",   "edge_known_fact"),
    Case("is the universe infinite?",                                 "NONE",   "edge_philosophy"),
    Case("can you help me write a CV?",                               "NONE",   "edge_generic_howto"),
    Case("recommend a book about African history",                    "SEARCH", "edge_recommend"),
]


# ────────────────────────────────────────────────────────────────────
# Four prompt versions, from verbose to minimal.
# ────────────────────────────────────────────────────────────────────

PROMPT_A_LONG = """\
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


PROMPT_B_MEDIUM = """\
You classify requests for Ongiini, an AI helper for people in Namibia.

SEARCH — needs fresh web info (Namibian businesses, services, fees, prices,
news, current events, named recommendations). Training data is stale.
DOCS — about Ongiini itself (pricing, privacy, terms, how it works, hardware,
who runs it, language coverage).
NONE — general knowledge, math, philosophy, generic how-to, casual chat,
emotional support.

Namibian places (Windhoek, Walvis Bay, Oshakati, etc.) and institutions
(BIPA, NamRA) imply Namibian context.

Request: {user_text}

Reply: SEARCH, DOCS, or NONE.
"""


PROMPT_C_SHORT = """\
Classifier for an AI helper for Namibia.

SEARCH — fresh web info on Namibian places, businesses, services, fees, news.
DOCS — about the assistant itself.
NONE — everything else.

Request: {user_text}

Reply: SEARCH, DOCS, or NONE.
"""


PROMPT_D_MICRO = """\
Route this:
SEARCH — Namibian facts, businesses, news, prices.
DOCS — about Ongiini itself.
NONE — anything else.

Request: {user_text}

Reply: SEARCH, DOCS, or NONE.
"""


PROMPT_E_TIGHT = """\
You classify requests for Ongiini, an AI helper for people in Namibia.

SEARCH — fresh web info needed: Namibian businesses, services, providers,
fees, prices, exchange rates, laws, policies, government documents, news,
events, named recommendations. Training data is stale for these.

DOCS — the user is asking about Ongiini itself: pricing, privacy, terms,
how it works, hardware, who runs it, language coverage.

NONE — general knowledge (science, math, philosophy), generic how-to with
no local angle, emotional support, casual conversation.

Namibian places (Windhoek, Walvis Bay, Oshakati, Swakopmund) and institutions
(BIPA, NamRA, Bank of Namibia) imply Namibian context even without "Namibia".

Request: {user_text}

Reply: SEARCH, DOCS, or NONE.
"""


PROMPT_F_LEAN = """\
Classifier for Ongiini, AI helper for people in Namibia.

SEARCH — Namibian businesses, services, fees, prices, laws, policies, news,
events, named recommendations. Windhoek/Walvis Bay/Oshakati/BIPA/NamRA imply
Namibian context.
DOCS — about Ongiini itself (pricing, privacy, terms, how it works).
NONE — general knowledge, math, philosophy, generic how-to, casual chat,
emotion.

Request: {user_text}

Reply: SEARCH, DOCS, or NONE.
"""


PROMPT_G_TERSE = """\
Classifier for an AI helper for Namibia.

SEARCH — any Namibian-specific facts: businesses, fees, prices, laws, news,
recommendations. Windhoek/BIPA/NamRA imply Namibia.
DOCS — about the assistant itself.
NONE — everything else.

Request: {user_text}

Reply: SEARCH, DOCS, or NONE.
"""


PROMPTS = [
    ("A_LONG",     PROMPT_A_LONG),
    ("B_MEDIUM",   PROMPT_B_MEDIUM),
    ("C_SHORT",    PROMPT_C_SHORT),
    ("D_MICRO",    PROMPT_D_MICRO),
    ("E_TIGHT",    PROMPT_E_TIGHT),
    ("F_LEAN",     PROMPT_F_LEAN),
    ("G_TERSE",    PROMPT_G_TERSE),
]


# ────────────────────────────────────────────────────────────────────


async def classify(client: AsyncOpenAI, template: str, user_text: str) -> tuple[str, int, float]:
    """Send the classifier prompt. Return (verdict, prompt_tokens, latency_seconds).
    verdict in {SEARCH, DOCS, NONE, ?}."""
    t0 = time.monotonic()
    resp = await client.chat.completions.create(
        model=settings.vllm_model,
        messages=[{"role": "user", "content": template.format(user_text=user_text)}],
        temperature=0.0,
        max_tokens=5,
    )
    latency = time.monotonic() - t0
    raw = (resp.choices[0].message.content or "").strip().upper()
    verdict = "?"
    for tok in ("SEARCH", "DOCS", "NONE"):
        if tok in raw:
            verdict = tok
            break
    prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
    return verdict, prompt_tokens, latency


async def run_variant(client: AsyncOpenAI, name: str, template: str) -> dict:
    """Run all CASES through one prompt variant. Return summary stats."""
    results = []
    prompt_tokens_seen = 0
    total_latency = 0.0
    for c in CASES:
        verdict, p_toks, lat = await classify(client, template, c.text)
        ok = verdict == c.expected
        results.append((c, verdict, ok))
        prompt_tokens_seen = max(prompt_tokens_seen, p_toks)
        total_latency += lat

    n = len(results)
    correct = sum(1 for _, _, ok in results if ok)
    misses = [(c, v) for c, v, ok in results if not ok]
    return {
        "name": name,
        "correct": correct,
        "total": n,
        "accuracy": correct / n if n else 0,
        "prompt_tokens": prompt_tokens_seen,
        "avg_latency_ms": (total_latency / n * 1000) if n else 0,
        "misses": misses,
    }


async def main() -> None:
    client = AsyncOpenAI(base_url=settings.vllm_base_url, api_key="not-needed")

    print(f"Benchmark — {len(CASES)} cases x {len(PROMPTS)} prompt variants")
    print("-" * 70)

    summaries = []
    for name, template in PROMPTS:
        s = await run_variant(client, name, template)
        summaries.append(s)
        print(
            f"  {s['name']:9s} | acc {s['correct']:2d}/{s['total']:2d} "
            f"({s['accuracy']*100:5.1f}%) | "
            f"prompt={s['prompt_tokens']:3d} tok | "
            f"avg latency {s['avg_latency_ms']:5.0f} ms"
        )

    print()
    print("-" * 70)
    print("MISSES per variant:")
    for s in summaries:
        if s["misses"]:
            print(f"  {s['name']}:")
            for c, v in s["misses"]:
                print(f"    exp={c.expected:6s} got={v:6s}  [{c.category}] {c.text}")
        else:
            print(f"  {s['name']}: (none)")

    print()
    print("-" * 70)
    print("RECOMMENDATION: shortest variant that hits >=97% accuracy")
    best = None
    for s in summaries:
        if s["accuracy"] >= 0.97:
            if best is None or s["prompt_tokens"] < best["prompt_tokens"]:
                best = s
    if best:
        print(
            f"  -> {best['name']}: {best['accuracy']*100:.1f}% accuracy at "
            f"{best['prompt_tokens']} tokens"
        )
    else:
        print("  No variant cleared 97% — A_LONG is your floor.")


if __name__ == "__main__":
    asyncio.run(main())
