"""Live eval of the PRODUCTION classifier: verdict accuracy + NONE depth.

router_eval_holdout.py measures an old three-word prompt (PROMPT_A_LONG)
that production no longer uses. This script runs the real
``GemmaClassifier`` (same prompt, JSON output, context rules) on:

  - the 41 held-out verdict cases from router_eval_holdout.py, and
  - depth cases for casual turns: NONE_SHALLOW (greetings, yes/no, quick
    facts, short follow-ups) vs NONE_DEEP (translations, CVs, letters,
    essays, step-by-step explanations, plans, lists of 5+).

The two depth errors are not equal. DEEP-expected → SHALLOW cuts a real
answer short (reply budget 320 vs 750 tokens): that is the rate that
matters, target < 10 %. SHALLOW-expected → DEEP only allows a longer
reply. Both are reported.

Read-only: contribute state is stubbed to "none", so no DB or mem0 read
happens; the only external call is to vLLM.

    docker cp ongiini/tests/router_depth_eval.py ongiini-webhook:/tmp/
    docker exec ongiini-webhook python3 /tmp/router_depth_eval.py
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter

sys.path.insert(0, "/app")

from owela import DEPTH_DEEP, DEPTH_SHALLOW, InboundMessage  # noqa: E402

from ongiini.config import settings  # noqa: E402
from ongiini.routers.gemma_classifier import GemmaClassifier  # noqa: E402
from ongiini.tests.router_eval_holdout import CASES as VERDICT_CASES  # noqa: E402

S, D = DEPTH_SHALLOW, DEPTH_DEEP

# (text, expected depth, history or None). All are NONE-verdict messages.
DEPTH_CASES: list[tuple[str, str, list[dict] | None]] = [
    # ---- SHALLOW: greetings, thanks, yes/no, quick facts, short follow-ups
    ("hi", S, None),
    ("good morning!", S, None),
    ("thank you so much", S, None),
    ("ok cool", S, None),
    ("what's the capital of Kenya", S, None),
    ("how many legs does a spider have", S, None),
    ("is a tomato a fruit", S, None),
    ("what does 'ubiquitous' mean", S, None),
    ("do you ever sleep", S, None),
    ("tell me a fun fact about hippos", S, None),
    ("baie dankie", S, None),
    ("hoe gaan dit met jou", S, None),
    ("wat is 7 keer 8", S, None),
    ("I'm bored", S, None),
    ("haha that's funny", S, None),
    ("yes please", S, [
        {"role": "user", "content": "what's a good name for a dog"},
        {"role": "assistant", "content": "How about Simba or Tula? Want more ideas?"},
    ]),
    ("which one is shorter?", S, [
        {"role": "user", "content": "give me two synonyms for happy"},
        {"role": "assistant", "content": "Joyful and glad. Want example sentences?"},
    ]),
    # ---- DEEP: translations, writing, step-by-step, plans, long lists
    ("translate this into Afrikaans: Dear Sir, I would like to apply for the "
     "position of receptionist advertised on your website. I have three years "
     "of experience and I am available immediately.", D, None),
    ("can you translate my message to my landlord into English, it's in Afrikaans: "
     "Goeie more meneer, die krag is al twee dae af in my kamer en ek kan nie kook "
     "nie. Kan u asseblief iemand stuur?", D, None),
    ("help me write a CV, I finished grade 12 and worked 1 year at a car wash", D, None),
    ("write a cover letter for a teller job at a bank", D, None),
    ("write me a short essay about the importance of water conservation", D, None),
    ("explain step by step how to solve 2x + 5 = 17", D, None),
    ("explain photosynthesis for my grade 10 test, with the equation", D, None),
    ("how do I calculate the area of a triangle, show me with an example", D, None),
    ("give me 10 interview questions for a sales job and how to answer them", D, None),
    ("make me a study timetable for my exams in 3 weeks: maths, biology, English", D, None),
    ("I want to start a small chicken business, give me a plan", D, None),
    ("help me write a speech for my sister's wedding", D, None),
    ("summarise the causes of World War 1 for my history assignment", D, None),
    ("skryf vir my 'n brief aan my skoolhoof om verlof te vra", D, None),
    ("verduidelik stap vir stap hoe fotosintese werk", D, None),
    ("help my met my CV asseblief, ek het matriek en werk by Pep", D, None),
    ("what should I pack for a 5 day trip, give me a full list", D, None),
    ("explain the difference between a virus and bacteria in detail", D, None),
    ("can you correct the grammar in my essay: " + "My name is Anna and I am live in "
     "Rundu. I likes to reading books and my dream are to became a nurse. "
     "Every day I goes to school with my brother.", D, None),
    ("now do the same for the second paragraph please", D, [
        {"role": "user", "content": "rewrite my essay's first paragraph so it sounds more formal: ..."},
        {"role": "assistant", "content": "Here is a more formal version of your first paragraph: ..."},
    ]),
    ("and in Afrikaans?", D, [
        {"role": "user", "content": "translate this letter to English: ..."},
        {"role": "assistant", "content": "Here is the English translation of your letter: Dear Mrs ..."},
    ]),
]


def _no_state(user_id: str):  # noqa: ARG001
    return {"pending_save": None, "awaiting_followup": False,
            "dialect": "unknown", "recently_declined": False}


def _msg(text, history=None):
    return InboundMessage(user_id="eval", msg_id="eval", text=text,
                          content_parts=[{"type": "text", "text": text}],
                          history=history or [])


async def main() -> None:
    clf = GemmaClassifier(base_url=settings.vllm_base_url, model_id=settings.vllm_model)
    clf._read_contribute_state = _no_state          # type: ignore[assignment]

    # ---- verdict accuracy on the held-out set
    ok = 0
    misses = []
    lat = []
    for c in VERDICT_CASES:
        r = await clf.classify(_msg(c.text))
        lat.append(r.tokens_out)
        got = r.verdict
        good = got == c.expected
        ok += good
        if not good:
            misses.append((c.expected, got, r.fallback_reason, c.text))
    print(f"VERDICT  {ok}/{len(VERDICT_CASES)} = {100 * ok / len(VERDICT_CASES):.1f}%")
    for exp, got, fb, text in misses:
        print(f"   exp={exp:6} got={got:18} fallback={fb}  {text}")

    # ---- depth on casual turns
    conf = Counter()
    depth_misses = []
    for text, exp, hist in DEPTH_CASES:
        r = await clf.classify(_msg(text, hist))
        got = r.depth if r.verdict == "NONE" else f"{r.verdict}"
        conf[(exp, got)] += 1
        if got != exp:
            depth_misses.append((exp, got, r.fallback_reason, text[:90]))
    n_deep = sum(1 for _, e, _ in DEPTH_CASES if e == D)
    n_shallow = len(DEPTH_CASES) - n_deep
    cut = conf[(D, S)]
    print(f"\nDEPTH  cases={len(DEPTH_CASES)} (deep {n_deep}, shallow {n_shallow})")
    print(f"   DEEP→SHALLOW (answer cut short): {cut}/{n_deep} = {100 * cut / n_deep:.1f}%   target < 10 %")
    print(f"   SHALLOW→DEEP (longer budget):    {conf[(S, D)]}/{n_shallow}")
    other = {k: v for k, v in conf.items() if k[1] not in (S, D)}
    if other:
        print(f"   routed away from NONE: {other}")
    exact = conf[(S, S)] + conf[(D, D)]
    print(f"   exact: {exact}/{len(DEPTH_CASES)} = {100 * exact / len(DEPTH_CASES):.1f}%")
    for exp, got, fb, text in depth_misses:
        print(f"   exp={exp:7} got={got:10} fallback={fb}  {text}")


if __name__ == "__main__":
    asyncio.run(main())
