"""Force-called opt-out tool for proactive-broadcast unsubscribes.

Wired in via classifier verdict ``OPT_OUT_BROADCAST`` → policy table
→ ``force_tool("opt_out_broadcast")``. The model never picks this
tool; it's called when the classifier detects an opt-out intent so
the user can't end up in a "you're still getting messages" loop
even if the model fails to compose the right reply.

Distinct from delete_my_data — opting out of broadcasts does NOT
wipe conversation data. The user can keep chatting normally; we
just won't proactively message them.
"""
from __future__ import annotations

import logging

from owela import ToolContext, tool

from ..broadcast import opt_outs as _opt_outs

log = logging.getLogger("ongiini.tools.opt_out")


@tool(
    name="opt_out_broadcast",
    description=(
        "FORCED by classifier verdict OPT_OUT_BROADCAST — the user just "
        "asked to stop receiving proactive update messages from Ongiini "
        "AI (STOP / unsubscribe / opt out). Records the opt-out in "
        "/data/broadcast_opt_outs.sqlite (salted hash, not the raw "
        "msisdn). Returns a short JSON status the model uses to "
        "compose a confirming reply. Idempotent — re-running on an "
        "already-opted-out user is a no-op."
    ),
    params={},
)
async def opt_out_broadcast(ctx: ToolContext) -> str:
    """Record the user as opted-out of broadcasts. Soft-fail with a
    JSON-shaped error string so the model can still apologise instead
    of crashing the turn."""
    import json

    try:
        newly_added = _opt_outs.record(ctx.user_id, source="stop_keyword")
    except RuntimeError as exc:
        # Hash salt missing → can't record. Don't crash; surface to
        # the model so it can apologise.
        log.warning("opt_out_broadcast hash failed: %s", exc)
        return json.dumps({
            "status": "error",
            "reason": "config_missing",
            "message_to_compose": (
                "Apologise briefly and ask the user to try again "
                "shortly — we're missing a config value."
            ),
        })
    except Exception as exc:                          # noqa: BLE001
        log.exception("opt_out_broadcast unexpected failure: %s", exc)
        return json.dumps({
            "status": "error",
            "reason": "unexpected",
            "message_to_compose": (
                "Apologise briefly and ask the user to try again later."
            ),
        })

    return json.dumps({
        "status": "ok",
        "newly_added": newly_added,
        "message_to_compose": (
            "Confirm warmly that they won't receive proactive update "
            "messages from Ongiini AI anymore. Mention that they can "
            "always keep chatting with you normally — only the "
            "outbound announcements are off. Two short sentences max, "
            "no emoji, no exclamation."
        ),
    })
