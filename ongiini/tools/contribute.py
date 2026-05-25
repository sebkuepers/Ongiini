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
        "Drives the community Oshiwambo translation-contribution loop. "
        "CALL THIS TOOL — never improvise the conversation. The user "
        "must never see an English source sentence you invented; only "
        "the source_en returned by a 'next' call is a real task in "
        "the database. Trigger → action mapping:\n"
        "  • User volunteers ('I want to help', 'can I contribute', "
        "'I'm a native speaker') OR uses a real multi-word Oshiwambo "
        "phrase OR asks about Oshiwambo support → call "
        "action='whoami' BEFORE composing your reply. The response "
        "tells you whether they're new, already declined, or have a "
        "known dialect, which determines the right next move.\n"
        "  • User just told you their dialect (any of 'Oshindonga', "
        "'Oshikwanyama', 'Ndonga', 'Kwanyama', 'Oshidonga') → call "
        "action='set_dialect' with the canonical name ('Oshindonga' "
        "or 'Oshikwanyama'), then in the same turn call action='next' "
        "and show the returned source_en VERBATIM.\n"
        "  • User said yes/sure/Tangi/Eewa to the invitation AND "
        "whoami returned 'known:Oshindonga' or 'known:Oshikwanyama' "
        "→ call action='next' immediately, show source_en VERBATIM.\n"
        "  • User just typed their translation (their message after "
        "you presented an English source sentence) → call "
        "action='save' with the task_id from your most recent 'next' "
        "response, their dialect, and their translation text "
        "VERBATIM. NEVER guess a task_id.\n"
        "  • User said no/not now/maybe later → call action='decline'.\n"
        "  • User asks how many translations collected → call "
        "action='stats'.\n"
        "See the 'contribute' skill for the full phrasing templates."
    ),
    params={
        "action": "'whoami' | 'set_dialect' | 'next' | 'save' | 'decline' | 'stats'",
        "target_dialect": (
            "(required for 'set_dialect' and 'save') 'Oshindonga' or "
            "'Oshikwanyama'. Must match exactly — case-sensitive. "
            "Normalise variants ('Ndonga' → 'Oshindonga', 'Kwanyama' "
            "→ 'Oshikwanyama', 'Oshidonga' → 'Oshindonga') before "
            "passing."
        ),
        "task_id": (
            "(required for 'save') the integer task id from the most "
            "recent 'next' tool response. Pass it back verbatim. If "
            "you don't have a real task_id from a prior 'next' call, "
            "do NOT call 'save' — call 'next' first."
        ),
        "translation": (
            "(required for 'save') the user's translation in their "
            "dialect, VERBATIM. Don't summarise, correct, or 'clean "
            "up' — store exactly what they wrote."
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
