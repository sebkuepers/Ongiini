"""Planner — pre-act-loop decomposition for SEARCH_DEEP turns.

When the depth-aware classifier returns SEARCH_DEEP (multi-source
comparison, list of N items, "what's going on with X and Y"), running
the act loop immediately tends to produce one shallow search instead
of the structured multi-step research the question actually needs.
The planner step interrupts this: before the first model call, ask
Gemma to enumerate what's already known, what to look up, and the
order of searches.

The planner's output is injected as a system message by the
``OngiiniMemoryProvider`` when assembling the act-loop's first prompt.
The model then enters the act loop with a written plan in scope and
tends to follow it.

Prompt adapted from smolagents' ``initial_plan`` YAML, tightened for
the WhatsApp shape (output cap ~200 tokens, no markdown, plain text
suitable for prepending to a chat prompt).

Soft-fail contract: any timeout, parse failure, or other error
returns ``PlanStep(plan_text="")``. An empty plan_text means the
MemoryProvider skips the injection entirely — the act loop runs
exactly as it would without a planner. Never a hard failure.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from openai import AsyncOpenAI

from owela import InboundMessage, PlanStep, Policy, Step

log = logging.getLogger("ongiini.planner")


# Adapted from smolagents/src/smolagents/prompts/code_agent.yaml::initial_plan,
# narrowed for the WhatsApp / Namibia shape. The three sections
# (FACTS I ALREADY KNOW / FACTS TO LOOK UP / SEARCH PLAN) are the
# minimum decomposition that produces measurably better multi-step
# behaviour on Gemma 4. The ``PLAN_DONE`` sentinel mirrors smolagents'
# ``<end_plan>`` — gives the parser a reliable stop and protects
# against the model rambling into the act loop content.
_PLAN_PROMPT = """You're about to help a user in Namibia answer this question on WhatsApp:

  {question}

Before any search, write a 3-section plan in plain text. Keep the total
output under 200 tokens. No markdown, no preamble. Use this exact shape:

FACTS I ALREADY KNOW:
- 1-2 bullets of background you can answer confidently without searching
  (e.g. general geography, well-known institutions, language facts).
- Write "none" if it's genuinely all unknown to you.

FACTS TO LOOK UP:
- 1-3 bullets of specific things you'll need to search for (names of
  Namibian businesses / current prices / opening hours / current events).
- Each bullet should be one concrete sub-question.

SEARCH PLAN:
- 1-2 lines describing the search order. Mention whether to broaden or
  narrow the query if first results are thin.

End with the literal token PLAN_DONE.
"""


# Hard cap on plan output. The act loop will see this as a system
# message; bloating it eats prefix-cache headroom and slows the chat
# call. Empirically ~200 completion tokens is plenty for the structure
# above.
_PLAN_MAX_TOKENS = 220

# Latency budget. If Gemma can't produce a plan in this time, the
# planner returns an empty PlanStep and the executor proceeds without
# a plan — same shape as a v0 SEARCH_DEEP turn.
_TIMEOUT_S = 4.0


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

        try:
            resp = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self.model_id,
                    messages=[{
                        "role": "user",
                        "content": _PLAN_PROMPT.format(question=question),
                    }],
                    temperature=0.3,
                    max_tokens=self.max_tokens,
                    # stop=["PLAN_DONE"] would also work but some vLLM
                    # builds emit a partial last token when the stop
                    # fires — easier to strip the sentinel after.
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

        # Strip the sentinel and anything after it. The model occasionally
        # appends a stray paragraph after PLAN_DONE — ignore it.
        if "PLAN_DONE" in raw:
            raw = raw.split("PLAN_DONE", 1)[0].rstrip()

        step.plan_text = raw
        step.ended_at = time.monotonic()
        return step


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
