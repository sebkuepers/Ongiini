"""Held-out benchmark for the router classifier.

The dev set in router_eval.py was used to iterate on the prompt; running
against it again only confirms overfitting. This is a SEPARATE, fresh
test set written without looking at the dev cases, drawing from
"what would a real Namibian on WhatsApp actually ask?".

If the classifier generalizes well, accuracy here should be close to
what we saw on the dev set (97.5%). If it drops sharply, the prompt is
overfit to specific phrasings.

Same A_LONG prompt as production. No re-tuning allowed based on these
results — that defeats the purpose. We accept whatever number comes back.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass

sys.path.insert(0, "/app")

from openai import AsyncOpenAI
from ongiini.config import settings


@dataclass
class Case:
    text: str
    expected: str
    note: str  # human reasoning for the label


# ────────────────────────────────────────────────────────────────────
# Fresh held-out cases. Different shapes, different topics, different
# wordings from the dev set. Drawn from realistic-feeling production
# patterns: practical daily life, family/health, government services,
# education, business, sports, news, casual curiosity, meta-about-bot,
# multi-intent mixes. Some are deliberately tricky / ambiguous.
# ────────────────────────────────────────────────────────────────────

CASES: list[Case] = [
    # Practical daily life (SEARCH — current, local-anchored)
    Case("where's the cheapest petrol near Eros",                 "SEARCH", "specific Windhoek-suburb gas-price"),
    Case("any pharmacies open after 10pm in Klein Windhoek",      "SEARCH", "current local opening hours"),
    Case("internet cafe in Tsumeb",                               "SEARCH", "named local service in a Namibian town"),
    Case("TransNamib train schedule Windhoek to Walvis Bay",      "SEARCH", "current Namibian transport schedule"),
    Case("how late does Maerua Mall stay open on Sundays",        "SEARCH", "current local opening hours"),

    # Health / family (NONE — general advice; SEARCH if explicitly local)
    Case("my toddler has a high fever, what should I do",         "NONE",   "general health advice, no local lookup needed"),
    Case("is it safe to drink tap water in Windhoek",             "SEARCH", "local water quality varies, would benefit from a fresh source"),
    Case("foods to avoid when pregnant",                          "NONE",   "general medical knowledge"),

    # Government / procedural (SEARCH — fees, forms, deadlines update)
    Case("how do I get a tax clearance certificate",              "SEARCH", "procedural, Namibian (NamRA); fees/forms update"),
    Case("documents I need to renew my Namibian passport",        "SEARCH", "explicitly Namibian procedural"),
    Case("can a German citizen open a business in Namibia",       "SEARCH", "Namibian regulation, would benefit from current law"),

    # Education / schoolwork (NONE — pure)
    Case("explain how a transistor works",                        "NONE",   "pure science"),
    Case("difference between syntax and semantics",               "NONE",   "general linguistics"),
    Case("Pythagoras theorem with an example",                    "NONE",   "math"),

    # News / events / sports (SEARCH — current)
    Case("any Brave Warriors match this weekend",                 "SEARCH", "Namibian national team, current schedule"),
    Case("what happened in the Namibian parliament today",        "SEARCH", "current events"),
    Case("Etosha rainfall this season",                           "SEARCH", "current data on a specific place"),

    # Casual / emotional / conversational (NONE)
    Case("I'm so anxious about my driving test tomorrow",         "NONE",   "emotional support"),
    Case("tell me a fun fact about hippos",                       "NONE",   "general knowledge curiosity"),
    Case("do you ever sleep",                                     "NONE",   "casual chitchat with the bot"),

    # Meta / DOCS (about Ongiini itself)
    Case("is my chat with you private",                           "DOCS",   "asking about Ongiini's privacy"),
    Case("can you forget what I told you yesterday",              "DOCS",   "asking about Ongiini's memory/deletion"),
    Case("what AI model are you running on",                      "DOCS",   "asking about Ongiini hardware/stack"),
    Case("who pays for this service",                             "DOCS",   "asking about Ongiini operator/funding"),

    # Mixed intent / location + general (NONE)
    Case("I'm in Otjiwarongo and just want to chat",              "NONE",   "Namibian location but emotional/conversational, not factual"),
    Case("explain photosynthesis using mahangu as an example",    "NONE",   "general science with local flavour, no fresh facts needed"),

    # Tricky edge cases
    Case("how much is 100 dollars in Namibian rand",              "SEARCH", "current exchange rate; rate updates daily"),
    Case("best mathematics tutor in Otjiwarongo",                 "SEARCH", "local recommendation, specific people"),
    Case("what time is it in Lagos right now",                    "SEARCH", "current state of the world; arguably NONE since arithmetic, but search confirms DST/timezone reliably"),

    # AF freshness
    Case("waar kry ek 'n goeie braaivleis in Swakopmund",         "SEARCH", "Afrikaans, local recommendation"),
    Case("wat is 'n swart gat",                                   "NONE",   "Afrikaans, pure science (black hole)"),
]


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


async def classify(client: AsyncOpenAI, user_text: str) -> str:
    resp = await client.chat.completions.create(
        model=settings.vllm_model,
        messages=[{"role": "user", "content": PROMPT_A_LONG.format(user_text=user_text)}],
        temperature=0.0,
        max_tokens=5,
    )
    raw = (resp.choices[0].message.content or "").strip().upper()
    for tok in ("SEARCH", "DOCS", "NONE"):
        if tok in raw:
            return tok
    return "?"


async def main() -> None:
    client = AsyncOpenAI(base_url=settings.vllm_base_url, api_key="not-needed")

    print(f"HELD-OUT BENCHMARK — {len(CASES)} fresh cases, A_LONG prompt")
    print("-" * 72)

    results = []
    for c in CASES:
        verdict = await classify(client, c.text)
        ok = verdict == c.expected
        mark = "PASS" if ok else "FAIL"
        print(
            f"  {mark}  exp={c.expected:6s} got={verdict:6s}  | {c.text[:55]:55s} | {c.note}"
        )
        results.append((c, verdict, ok))

    n = len(results)
    correct = sum(1 for _, _, ok in results if ok)
    print()
    print("-" * 72)
    print(f"  ACCURACY: {correct}/{n}  ({100 * correct / n:.1f}%)")
    print(
        f"  (dev-set comparison: A_LONG hit 97.5% on the cases I designed; "
        f"this is the honest generalization number.)"
    )

    misses = [(c, v) for c, v, ok in results if not ok]
    if misses:
        print()
        print("MISSES — for honest assessment, not for prompt re-tuning:")
        for c, v in misses:
            print(f"  exp={c.expected:6s} got={v:6s}  {c.text}")
            print(f"    note: {c.note}")


if __name__ == "__main__":
    asyncio.run(main())
