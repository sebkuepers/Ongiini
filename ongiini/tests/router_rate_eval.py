"""Live eval: does the PRODUCTION classifier keep translating and checking apart?

RATE_INVITE (help CHECK existing translations → rating link) must not be
confused with CONTRIBUTE_INVITE (help TRANSLATE sentences in WhatsApp),
nor with DOCS (asking how rating works). Then re-runs the held-out
verdict set from router_depth_eval.py to catch regressions.

Targets (ongiini/CLAUDE.md): RATE_INVITE ≥ 85 %, existing ≥ 96 %, and no
translate↔check swap at all.

To evaluate unreleased code against the live vLLM, copy the package to
the data volume and point ONGIINI_CODE at it:

    rsync -a --exclude __pycache__ ongiini spark:~/dev/Ongiini/data/private/evalcode/
    docker exec -e ONGIINI_CODE=/data/private/evalcode ongiini-webhook \
        python3 /data/private/evalcode/ongiini/tests/router_rate_eval.py

Read-only: contribute state is stubbed, the only external call is vLLM.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, os.environ.get("ONGIINI_CODE", "/app"))

from owela import InboundMessage  # noqa: E402

from ongiini.config import settings  # noqa: E402
from ongiini.routers.gemma_classifier import GemmaClassifier  # noqa: E402

RATE, CONTRIB, DOCS = "RATE_INVITE", "CONTRIBUTE_INVITE", "DOCS"

CASES: list[tuple[str, str]] = [
    # checking existing translations → rating link
    ("I'd like to help check translations", RATE),          # the prefilled text from /rate
    ("can I help rate translations?", RATE),
    ("send me the link to check translations", RATE),
    ("I want to review Oshiwambo translations", RATE),
    ("my link for rating translations doesn't work anymore", RATE),
    ("I'm a native Oshikwanyama speaker, can I check your translations?", RATE),
    ("how can I help check if your translations are correct", RATE),
    ("I'd like to check some more translations", RATE),
    ("Hi, Rauna sent me. I want to help check translations", RATE),
    ("can I be a translation checker", RATE),
    # translating sentences themselves → contribute loop
    ("I want to help translate", CONTRIB),
    ("can I help translate into Oshiwambo?", CONTRIB),
    ("I'd like to translate some sentences for you", CONTRIB),
    ("I speak Oshindonga, I can help you with translating", CONTRIB),
    ("give me sentences to translate", CONTRIB),
    ("do you support Oshiwambo?", CONTRIB),
    # asking about it → docs
    ("how does the translation rating work?", DOCS),
    ("what is ongiini.ai/rate?", DOCS),
    ("how does the Oshiwambo translation project work?", DOCS),
]


def _no_state(user_id: str):  # noqa: ARG001
    return {"pending_save": None, "awaiting_followup": False,
            "dialect": "unknown", "recently_declined": False}


def _msg(text):
    return InboundMessage(user_id="eval", msg_id="eval", text=text,
                          content_parts=[{"type": "text", "text": text}], history=[])


async def main() -> None:
    clf = GemmaClassifier(base_url=settings.vllm_base_url, model_id=settings.vllm_model)
    clf._read_contribute_state = _no_state          # type: ignore[assignment]
    by = {RATE: [0, 0], CONTRIB: [0, 0], DOCS: [0, 0]}
    swaps = 0
    for text, exp in CASES:
        got = (await clf.classify(_msg(text))).verdict
        by[exp][0] += got == exp
        by[exp][1] += 1
        swaps += {exp, got} == {RATE, CONTRIB}
        if got != exp:
            print(f"   exp={exp:18} got={got:18} {text}")
    for k, (ok, n) in by.items():
        print(f"{k:18} {ok}/{n} = {100 * ok / n:.0f}%")
    print(f"translate↔check swaps: {swaps}   (target 0)")

    from ongiini.tests.router_eval_holdout import CASES as HOLDOUT  # noqa: E402
    ok = 0
    for c in HOLDOUT:
        got = (await clf.classify(_msg(c.text))).verdict
        ok += got == c.expected
        if got != c.expected:
            print(f"   holdout exp={c.expected:6} got={got:18} {c.text}")
    print(f"HOLDOUT  {ok}/{len(HOLDOUT)} = {100 * ok / len(HOLDOUT):.1f}%   target ≥ 96 %")


if __name__ == "__main__":
    asyncio.run(main())
