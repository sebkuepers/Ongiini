"""State-gated classifier wrapper for the contribute family.

The bare `GemmaClassifier` is a stochastic LLM. It reliably emits the
right verdict for unambiguous inputs, but on ambiguous "looks like
non-English" text it over-fires CONTRIBUTE_SAVE — Afrikaans practice
("Ek drink water"), Oshindonga greetings outside a contribute flow,
even student multiple-choice answers ("Sunlight, option 1.B").

The fix is **not** more prompt engineering. The question "is this
message a translation contribution?" is fundamentally a state-machine
question: the contributor must be in `PENDING_TRANSLATION` (a
`pending_save` row was set by `contribute_next` within the last 30
minutes) for any reply to qualify as a save. No amount of textual
similarity to Oshiwambo can change that — the user wasn't asked.

This wrapper post-filters `GemmaClassifier`'s output. For every
CONTRIBUTE_* verdict, it consults fresh state and redirects to NONE
if the state contradicts the verdict. Non-contribute verdicts pass
through untouched.

Design:
- Wrapping (not modifying) keeps `GemmaClassifier` framework-clean and
  reusable. The state machine is an Ongiini concern, not a framework one.
- Gating happens AFTER `GemmaClassifier` returns, so the classifier's
  full attrs (confidence, reasoning, etc.) are preserved in the trace —
  including the original verdict (under `attrs["redirected_from"]`) so
  we can monitor false-positive rates without losing data.
- The actual state read uses `contributions.get_pending_save` and
  `contributions.is_awaiting_followup`, both of which apply their own
  TTLs. We don't second-guess those.
"""
from __future__ import annotations

import logging
from typing import Any

from owela import ClassifierResult, DEPTH_SHALLOW, InboundMessage, VERDICT_NONE

from .. import contributions
from .gemma_classifier import (
    VERDICT_CONTRIB_DECLINE,
    VERDICT_CONTRIB_DIALECT,
    VERDICT_CONTRIB_INVITE,
    VERDICT_CONTRIB_NEXT,
    VERDICT_CONTRIB_SAVE,
    VERDICT_CONTRIB_SKIP,
    VERDICT_CONTRIB_STATS,
)

log = logging.getLogger("ongiini.routers.state_gate")


# Verdicts that require contribute-flow state to be valid.
_GATED_VERDICTS = {
    VERDICT_CONTRIB_SAVE,
    VERDICT_CONTRIB_NEXT,
    VERDICT_CONTRIB_SKIP,
    VERDICT_CONTRIB_DECLINE,
}

# Verdicts that are state-independent (always pass through).
_PASS_THROUGH_CONTRIBUTE = {
    VERDICT_CONTRIB_INVITE,    # user volunteering — no prior state needed
    VERDICT_CONTRIB_DIALECT,   # setup-time — happens before pending_save exists
    VERDICT_CONTRIB_STATS,     # info query — orthogonal to flow state
}


def _verdict_allowed(verdict: str, state: dict) -> bool:
    """True if the contribute state permits this verdict to execute.
    A `state` dict has keys: pending (dict|None), awaiting_followup (bool).
    """
    pending = state["pending"]
    awaiting = state["awaiting_followup"]

    if verdict == VERDICT_CONTRIB_SAVE:
        # Save only allowed mid-task. pending is the proof that
        # contribute_next was called and a sentence is awaiting an answer.
        return pending is not None

    if verdict == VERDICT_CONTRIB_SKIP:
        # Skip only meaningful when a task has been served.
        return pending is not None

    if verdict == VERDICT_CONTRIB_NEXT:
        # "Another" only makes sense after a save (awaiting_followup) or
        # mid-task (pending — user could be asking for a different one
        # without explicitly skipping).
        return pending is not None or awaiting

    if verdict == VERDICT_CONTRIB_DECLINE:
        # "No thanks" only meaningful within an active contribute flow.
        return pending is not None or awaiting

    # Anything else slipping through is unexpected — refuse by default.
    return False


def _why_blocked(verdict: str, state: dict) -> str:
    """Short human-readable reason for the trace log."""
    pending = state["pending"]
    awaiting = state["awaiting_followup"]
    if pending is None and not awaiting:
        return "no_active_contribute_flow"
    if verdict == VERDICT_CONTRIB_SAVE and pending is None:
        return "save_without_pending"
    if verdict == VERDICT_CONTRIB_SKIP and pending is None:
        return "skip_without_pending"
    return "state_mismatch"


def _read_state(contributor_hash: str) -> dict:
    """Snapshot the contribute state used by the gate. Both reads apply
    their own TTLs (get_pending_save → 30 min; is_awaiting_followup → 30 min)
    so this function doesn't second-guess freshness."""
    return {
        "pending": contributions.get_pending_save(contributor_hash),
        "awaiting_followup": contributions.is_awaiting_followup(contributor_hash),
    }


class StateGatedClassifier:
    """Wraps a Classifier and gates CONTRIBUTE_* verdicts on fresh state.

    Implements the Owela `Classifier` protocol — the `Runtime` swaps the
    bare `GemmaClassifier` for this wrapper without other code changes.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def classify(self, msg: InboundMessage) -> ClassifierResult:
        result = await self._inner.classify(msg)

        verdict = result.verdict

        # Non-contribute verdicts pass through unchanged.
        if not verdict or not verdict.startswith("CONTRIBUTE_"):
            return result

        # State-independent contribute verdicts also pass through.
        if verdict in _PASS_THROUGH_CONTRIBUTE:
            return result

        if verdict not in _GATED_VERDICTS:
            # Unknown contribute-family verdict — pass through rather than
            # silently blocking. Keeps future verdicts forward-compatible.
            return result

        # Read fresh state at decision time.
        try:
            h = contributions.hash_msisdn(msg.user_id)
        except Exception as exc:                       # noqa: BLE001
            # Hash failure shouldn't break the agent. Pass through with
            # the original verdict — degraded, not broken.
            log.warning("hash_msisdn failed: %s — passing verdict %s through",
                        exc, verdict)
            return result

        state = _read_state(h)
        if _verdict_allowed(verdict, state):
            return result

        reason = _why_blocked(verdict, state)
        log.info(
            "state-gate redirect %s→NONE: user=%s reason=%s",
            verdict, h[:8], reason,
        )

        # Preserve the original verdict + classifier attrs in the redirected
        # result so the trace records what would have happened. Hooks
        # (BillingHook etc.) read tokens_in/out from the result — keep
        # those intact.
        attrs = dict(result.attrs or {})
        attrs["redirected_from"] = verdict
        attrs["redirect_reason"] = reason

        return ClassifierResult(
            verdict=VERDICT_NONE,
            depth=DEPTH_SHALLOW,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cached_tokens=getattr(result, "cached_tokens", 0),
            attrs=attrs,
        )
