"""Tool: contribute_translation — the model's interface to the
community Oshiwambo contribution loop.

Thin shim over ``ongiini.contributions`` (the sqlite module). The model
sees one tool with an ``action`` parameter; the implementation dispatches
to the right sqlite function and returns a JSON string the model can
quote naturally in its reply.

Why one tool with an action enum instead of five separate tools: the
contribution flow is a small state machine the model walks per
conversation (whoami → set_dialect? → next → save). Packing the steps
into a single tool keeps the function-call schema list short and the
model's prior weighting cleaner — only one tool to learn instead of
five competing surface names. Same pattern as the in-progress
``contribute`` skill documents.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from owela import ToolContext, tool

from .. import contributions

log = logging.getLogger("ongiini.tools.contribute")


_VALID_ACTIONS = {"whoami", "set_dialect", "next", "save", "stats", "decline"}


@tool(
    name="contribute_translation",
    description=(
        "Use ONLY when the user has agreed to contribute Oshiwambo "
        "translations to the open dataset, OR is mid-flow in the "
        "contribution loop. NEVER use as a side-effect of a normal "
        "translation request — those are answered conversationally. "
        "Six actions: 'whoami' returns the contributor's preferred "
        "dialect if known + whether they recently declined (so the "
        "skill can decide whether to invite at all), 'set_dialect' "
        "stores their dialect choice, 'next' fetches the next English "
        "sentence for them to translate, 'save' stores their "
        "translation, 'decline' records that they said no to the "
        "invitation (stops the bot from re-asking for 7 days), "
        "'stats' returns the total count collected so far (per-"
        "dialect breakdown included). See the 'contribute' skill "
        "for the full flow and phrasing."
    ),
    params={
        "action": "'whoami' | 'set_dialect' | 'next' | 'save' | 'stats'",
        "target_dialect": (
            "(required for 'set_dialect' and 'save') 'Oshindonga' or "
            "'Oshikwanyama'. Must match exactly — case-sensitive."
        ),
        "task_id": (
            "(required for 'save') the task id returned by a prior 'next' "
            "call. Pass it back verbatim — don't make one up."
        ),
        "translation": (
            "(required for 'save') the user's translation in their dialect, "
            "verbatim. Don't summarise or correct it — store what they said."
        ),
    },
)
async def contribute_translation(
    ctx: ToolContext,
    action: str,
    target_dialect: str = "",
    task_id: int = 0,
    translation: str = "",
) -> str:
    """Dispatch the action against the sqlite module. Returns a JSON
    string the model reads and paraphrases in its reply.

    JSON shape varies per action — keys are stable so the model can
    refer to them in templates / phrasing. Error responses always have
    an ``error`` key the model treats as a soft failure (the skill
    tells it to apologise warmly + offer to try again later).
    """
    if action not in _VALID_ACTIONS:
        return json.dumps({
            "error": f"unknown action {action!r}",
            "valid_actions": sorted(_VALID_ACTIONS),
        })

    # The contributor identity is always derived from the WhatsApp
    # msisdn (== ctx.user_id) via the salted hash. Never expose the
    # raw number to sqlite.
    try:
        contributor_hash = contributions.hash_msisdn(ctx.user_id)
    except RuntimeError as e:
        # CONTRIBUTIONS_HASH_SALT not set — deployment misconfig.
        # Soft-fail so the model doesn't crash the turn; ops will
        # see the warning in the logs.
        log.warning("contribute_translation aborted: %s", e)
        return json.dumps({"error": "contributions are temporarily unavailable"})

    try:
        if action == "whoami":
            return json.dumps({
                "status": contributions.whoami(contributor_hash),
                "recently_declined": contributions.recently_declined(contributor_hash),
                "total_contributions": contributions.contributor_total(contributor_hash),
            })

        if action == "decline":
            contributions.record_decline(contributor_hash)
            return json.dumps({"ok": True, "cooldown_days": contributions.DECLINE_COOLDOWN_DAYS})

        if action == "stats":
            return json.dumps(contributions.stats_summary())

        if action == "set_dialect":
            if not target_dialect:
                return json.dumps({"error": "target_dialect is required for set_dialect"})
            contributions.set_dialect(contributor_hash, target_dialect)
            return json.dumps({
                "ok": True,
                "preferred_dialect": target_dialect,
            })

        if action == "next":
            task = contributions.next_task(contributor_hash)
            if task is None:
                return json.dumps({
                    "task": None,
                    "message": "no more tasks available for this contributor",
                })
            return json.dumps({"task": task})

        if action == "save":
            if not target_dialect:
                return json.dumps({"error": "target_dialect is required for save"})
            if not task_id:
                return json.dumps({"error": "task_id is required for save"})
            if not translation or not translation.strip():
                return json.dumps({"error": "translation cannot be empty"})
            result = contributions.save_contribution(
                contributor_hash=contributor_hash,
                task_id=int(task_id),
                target_dialect=target_dialect,
                target_translation_raw=translation,
            )
            return json.dumps({"ok": True, **result})

    except ValueError as e:
        # Domain rejection (invalid dialect, missing task, empty after
        # PII-sanitise). These are user/model-driven errors — surface a
        # clean message instead of a stack trace.
        return json.dumps({"error": str(e)})
    except Exception as e:  # noqa: BLE001
        # Unexpected — log and soft-fail. The model gets a generic
        # error and the skill instructs it to apologise + offer to
        # retry later. Hooks-style soft-fail philosophy.
        log.exception("contribute_translation crashed on action=%s", action)
        return json.dumps({"error": "internal error storing contribution"})

    # Unreachable — every action above returns; this satisfies the
    # type checker.
    return json.dumps({"error": f"unhandled action {action!r}"})
