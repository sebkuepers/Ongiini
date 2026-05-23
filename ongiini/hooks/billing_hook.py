"""Owela ``Hook`` that records token usage to ``usage.log``.

The hook subscribes to step events and writes one line per BILLABLE
step into ``usage.log``. Billable steps:

  - ``RouterStep``                                  → ``kind="router"``
    (NOT counted toward the user's monthly cap; see usage.py)
  - ``ModelCallStep``, ``PlanStep``, ``CritiqueStep``, ``ReviseStep``
    → aggregated into one ``kind="chat"`` line per turn (sum, not
    per-call). The accounting now distinguishes "user-payload" tokens
    from "system-overhead" tokens — see the per-phase decision below.

v1.3.1 billing-fairness rules. The user's 1M monthly cap counts only
tokens that correspond to THEIR activity:

  - Planner input + output                          → BILLED (pure
    thinking about the user's question)
  - Compose call BEFORE any search ToolStep         → BILLED
  - Compose call AFTER a search ToolStep            → output only;
    input is system-added search context, not user's fault
  - Critique input + output                         → NOT BILLED
    (internal quality control — user didn't ask for it)
  - Revise input                                    → NOT BILLED
    (heavy input is QC overhead)
  - Revise output                                   → BILLED
    (the revised text replaces the draft and becomes the user-visible
    reply)
  - Synthesised ModelCallSteps                      → NOT BILLED
    (executor-driven, no real LLM call)

Raw vLLM cost (for internal auditing) is preserved in
``trace.jsonl`` (each ModelCallStep carries the unmodified token
counts).

Soft-fail: a broken billing log must never break a reply.
"""

from __future__ import annotations

import logging
from typing import Protocol

from owela import (
    CritiqueStep, ModelCallStep, PlanStep, ReviseStep, RouterStep, Step,
    ToolStep, TurnContext,
)


# Tool names whose ToolSteps flip the "search context is in the
# model's prompt now" flag. Subsequent ModelCallSteps see this
# content; their input tokens are system-overhead, not user payload.
_SEARCH_TOOL_NAMES = ("web_search", "fetch_url", "fetch_urls")

log = logging.getLogger("ongiini.hooks.billing")


class UsageRecorder(Protocol):
    """Module-level functions in webhook.ongiini.usage — injected so the hook
    can be unit-tested without touching the real usage.log."""
    def record(
        self,
        msisdn: str,
        tokens_in: int,
        tokens_out: int,
        used_search: bool,
        kind: str = "chat",
    ) -> None: ...


class BillingHook:
    """One usage.log line per model call. Designed to mirror the
    current respond()-loop billing behaviour:

      - aggregate (tokens_in, tokens_out) across all model calls in
        a turn = one "chat" line per turn (sum, not per-call)
      - one "router" line per router step (not counted toward the cap)

    Why aggregate per-turn rather than per-call: that's how the old
    code worked, and how the dashboard parses usage.log. Preserving
    line shape preserves analytics continuity.
    """

    def __init__(self, recorder: UsageRecorder) -> None:
        self._recorder = recorder

    async def on_turn_complete(self, steps: list[Step], ctx: TurnContext) -> None:
        msisdn = ctx.msg.user_id

        # Router cost is logged separately under kind="router" and is
        # NON-billable (excluded from the user's monthly cap inside
        # usage.summary_for).
        for s in steps:
            if isinstance(s, RouterStep) and (s.tokens_in or s.tokens_out):
                try:
                    self._recorder.record(
                        msisdn, s.tokens_in, s.tokens_out,
                        used_search=False, kind="router",
                    )
                except Exception as exc:                # noqa: BLE001 — soft-fail
                    log.warning("billing: router record failed: %s", exc)
                break   # one RouterStep per turn

        # v1.3.1 — billing-fairness accounting. Walk the step list once
        # with a ``search_ctx_active`` flag and apply per-phase rules
        # (see module docstring). The user only "pays" for tokens that
        # correspond to their actual activity; system-overhead (search
        # context, critique, revise input) is excluded.
        chat_in = 0
        chat_out = 0
        used_search = False
        search_ctx_active = False

        for s in steps:
            if isinstance(s, ToolStep):
                if s.tool_name in _SEARCH_TOOL_NAMES:
                    used_search = True
                    search_ctx_active = True
                continue

            billable_in, billable_out = self._billable_for_step(s, search_ctx_active)
            chat_in += billable_in
            chat_out += billable_out

        if chat_in or chat_out:
            try:
                self._recorder.record(
                    msisdn, chat_in, chat_out,
                    used_search=used_search, kind="chat",
                )
            except Exception as exc:                    # noqa: BLE001 — soft-fail
                log.warning("billing: chat record failed: %s", exc)

    @staticmethod
    def _billable_for_step(step: Step, search_ctx_active: bool) -> tuple[int, int]:
        """Per-phase billing decision. Returns (billable_in, billable_out)
        that count toward the user's monthly cap.

        Cached prompt tokens are subtracted from input regardless (they
        cost no real GPU work and the user shouldn't be charged for the
        prefix cache hits).
        """
        # Synthesised steps (multi-query fan-out + auto-followup) have
        # no LLM call at all — tokens are already 0 but be defensive.
        if isinstance(step, ModelCallStep) and step.synthesized:
            return 0, 0

        # Critique: internal quality control; user didn't ask. Free.
        if isinstance(step, CritiqueStep):
            return 0, 0

        # Revise: heavy input is QC overhead, but the output BECOMES
        # the user-visible reply (when revise succeeds), so output is
        # counted.
        if isinstance(step, ReviseStep):
            return 0, step.tokens_out

        # Compose ModelCallStep: bill input only if no search context
        # has been pumped into the prompt yet. After search, the bulk
        # of input is system-added tool results.
        if isinstance(step, ModelCallStep):
            billable_in = max(0, step.tokens_in - step.cached_tokens)
            if search_ctx_active:
                billable_in = 0
            return billable_in, step.tokens_out

        # PlanStep — full bill (pure thinking about the question;
        # never sees tool results in its prompt).
        if isinstance(step, PlanStep):
            billable_in = max(0, step.tokens_in - step.cached_tokens)
            return billable_in, step.tokens_out

        # Anything else (RouterStep is handled above; new step kinds
        # default to non-chat-aggregate behaviour).
        return 0, 0
