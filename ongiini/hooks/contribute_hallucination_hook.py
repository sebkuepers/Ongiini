"""ContributeHallucinationGuardHook — recover bot-hallucinated tasks.

The intended contribute flow is:

    user "yes"  →  classifier CONTRIBUTE_NEXT  →  contribute_next tool
                   sets pending_save              returns task source_en
                   model formats the question     "how would you say this
                                                   in Oshindonga: '...'?"

Sometimes the model skips the tool and just *makes up* an English
sentence, satisfying the user's "yes" with hallucinated content. The
user then types a translation; pending_save is empty; the user's work
is lost (silently before the orphan path, silently again if the orphan
path is removed without this hook).

This hook runs at end-of-turn. If the bot's reply contains the
task-serving pattern AND no `contribute_next` / `contribute_set_dialect`
tool fired this turn, it retroactively creates a task row with
`category='hallucinated_recovery'` and sets `pending_save` for the user.
The *next* turn's reply lands normally through the state-gated
classifier and is saved against the recovered task.

Why a hook (not a pre-send mutation): Owela hooks are observe-only by
design. We don't change the reply that was already sent — we change
the state that determines the next turn's behaviour. The user sees
the bot's hallucinated question and answers it; we make sure the
answer is saved.

Side effects are limited to:
- one new row in `tasks` (category='hallucinated_recovery')
- one upsert into `contributors` (pending_save fields set)

Both writes are idempotent over the turn (if the hook ran twice the
second create would just produce another harmless row that the
state-gate ignores until used).
"""

from __future__ import annotations

import logging
import re

from owela import ReplyStep, Step, ToolStep, TurnContext

from .. import contributions

log = logging.getLogger("ongiini.hooks.contribute_hallucination")


# Pattern looks for "how would you say this in <DIALECT>: '<source>'".
# Captures the dialect (group 1) and the English source (group 2).
# Permissive about punctuation between dialect and the quoted sentence
# (colon / dash / question mark / spaces). Requires ≥8 chars in the
# quoted source to avoid matching the bot's quoted single words.
_TASK_PATTERN = re.compile(
    r'how\s+would\s+you\s+say\s+(?:this|that|it)\s+in\s+'
    r'(Oshindonga|Oshikwanyama)'
    r'\s*[:\-?—–]?\s*'
    r'["“‘’”]'
    r'([^"“‘’”]{8,500})'
    r'["“‘’”]',
    re.IGNORECASE,
)


# Tool names that legitimately set pending_save. If any of these fired
# this turn we know it's NOT a hallucination — the state was set
# correctly through the tool path.
_LEGITIMATE_PENDING_SETTERS = {
    "contribute_next",
    "contribute_set_dialect",   # chains to next_task internally
    "contribute_skip",           # also rotates pending
}


class ContributeHallucinationGuardHook:
    """Detect bot-hallucinated translation tasks and recover state."""

    async def on_turn_complete(
        self, steps: list[Step], ctx: TurnContext
    ) -> None:
        # Locate the reply step. Nothing sent → nothing to inspect.
        reply_step = next(
            (s for s in reversed(steps) if isinstance(s, ReplyStep)), None
        )
        if reply_step is None or not reply_step.sent:
            return

        reply_text = reply_step.attrs.get("reply_text", "") or ""
        match = _TASK_PATTERN.search(reply_text)
        if not match:
            return

        # If a legitimate tool already set pending_save this turn, the
        # bot's translation question is the *correct* follow-up from that
        # tool's source_en. Not a hallucination.
        for s in steps:
            if isinstance(s, ToolStep) and s.tool_name in _LEGITIMATE_PENDING_SETTERS:
                # Tool fired AND succeeded (error=None) → legitimate.
                if s.error is None:
                    return

        dialect = _normalise_dialect(match.group(1))
        source_en = match.group(2).strip()

        # We have a bot-hallucinated task. Recover state so the user's
        # next reply lands normally.
        try:
            h = contributions.hash_msisdn(ctx.msg.user_id)
        except Exception as exc:                       # noqa: BLE001
            log.warning("hallucination hook: hash_msisdn failed: %s", exc)
            return

        try:
            task_id = contributions.create_hallucinated_recovery_task(source_en)
            contributions.set_pending_save(h, task_id=task_id, dialect=dialect)
        except Exception as exc:                       # noqa: BLE001
            # Soft-fail per the Hook contract — never raise from a hook.
            log.exception(
                "hallucination recovery failed for user=%s: %s",
                h[:8] if isinstance(h, str) else "?", exc,
            )
            return

        log.warning(
            "HALLUCINATED TASK RECOVERED: user=%s dialect=%s task_id=%d "
            "source_en=%r — pending_save set retroactively. Next user "
            "reply will save against this task.",
            h[:8], dialect, task_id, source_en[:120],
        )


def _normalise_dialect(raw: str) -> str:
    """Normalise classifier-captured dialect string to canonical form."""
    cleaned = (raw or "").strip().lower()
    if cleaned.startswith("oshindonga"):
        return "Oshindonga"
    if cleaned.startswith("oshikwanyama"):
        return "Oshikwanyama"
    # Should be unreachable given the regex alternation, but be safe.
    raise ValueError(f"unrecognised dialect {raw!r}")
