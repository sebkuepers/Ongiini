"""Owela ``Hook`` that records token usage to ``usage.log``.

The hook subscribes to step events and writes one line per BILLABLE
step into ``usage.log``. Billable steps:

  - ``RouterStep``                                  → ``kind="router"``
    (NOT counted toward the user's monthly cap; see usage.py)
  - ``ModelCallStep``, ``PlanStep``, ``CritiqueStep``, ``ReviseStep``
    → aggregated into one ``kind="chat"`` line per turn (sum, not
    per-call). All four count against the monthly cap because all
    four spend real vLLM tokens against the user's request.

Soft-fail: a broken billing log must never break a reply.
"""

from __future__ import annotations

import logging
from typing import Protocol

from owela import (
    CritiqueStep, ModelCallStep, PlanStep, ReviseStep, RouterStep, Step,
    TurnContext,
)


# Step types whose tokens roll up into the per-turn "chat" aggregate.
# Anti-trap: kept as a module-level tuple so adding a future phase
# (e.g. SummariseStep) is a one-line change in this file rather than a
# scattered hunt across hook code.
_CHAT_AGGREGATE_STEP_TYPES = (
    ModelCallStep,
    PlanStep,
    CritiqueStep,
    ReviseStep,
)

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

        # Aggregate cost across every step kind that spends real vLLM
        # tokens against this user's request — the act-loop model calls
        # AND the v1 planner / critique / revise phases. All roll into
        # one "chat" line per turn so the monthly cap stays honest no
        # matter which phases the policy enables.
        chat_in = 0
        chat_out = 0
        used_search = False
        for s in steps:
            if isinstance(s, _CHAT_AGGREGATE_STEP_TYPES):
                chat_in += s.tokens_in
                chat_out += s.tokens_out
            elif getattr(s, "tool_name", "") in ("web_search", "fetch_url", "fetch_urls"):
                used_search = True

        if chat_in or chat_out:
            try:
                self._recorder.record(
                    msisdn, chat_in, chat_out,
                    used_search=used_search, kind="chat",
                )
            except Exception as exc:                    # noqa: BLE001 — soft-fail
                log.warning("billing: chat record failed: %s", exc)
