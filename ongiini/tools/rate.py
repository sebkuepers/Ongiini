"""rate_link — hand a WhatsApp user their personal link to ongiini.ai/rate.

Forced by the classifier verdict RATE_INVITE (the user wants to help
CHECK existing translations — not translate; that is the contribute_*
loop). Argument-less: the user is identified by the salted msisdn hash
(contributions.hash_msisdn), the number itself is never stored. See
ongiini/ratings.py::link_for_contributor.

The result carries the finished WhatsApp reply in ``reply`` and the policy
sends it as-is (Policy.reply_from_tool): no model call, so the link can't
be mangled, and the message stays tiny — mobile data is expensive in
Namibia. On a user's first-ever message the AI disclosure leads.
"""
from __future__ import annotations

import json
import logging

from owela import ToolContext, tool

from .. import contributions, ratings

log = logging.getLogger("ongiini.tools.rate")

DISCLOSURE = "Ongiini! I'm an AI assistant."
REPLIES = {
    "ok": "Great! Tap the link to get started:\n{url}",
    "returning": "Here's your new link (the old one no longer works):\n{url}",
    "whatsapp_only": "Checking translations works on WhatsApp — message Ongiini AI there.",
    "blocked": "Sorry, this isn't open to you. Happy to help with anything else!",
    "error": "Sorry, I couldn't create your link right now. Please try again later.",
}


def _with_reply(ctx: ToolContext, result: dict) -> str:
    key = "returning" if result.get("returning") else result["status"]
    reply = REPLIES[key].format(url=result.get("url", ""))
    if not any(m.get("role") == "assistant" for m in (ctx.msg.history or [])):
        reply = f"{DISCLOSURE}\n\n{reply}"
    return json.dumps({**result, "reply": reply})


@tool(
    name="rate_link",
    description=(
        "FORCED by classifier verdict RATE_INVITE — the user wants to help "
        "CHECK (rate) existing Oshiwambo translations on ongiini.ai/rate. "
        "Returns JSON: status ('ok' | 'whatsapp_only' | 'blocked' | 'error'), "
        "url (their personal link — quote it exactly), returning (bool: they "
        "had a link before; the old one no longer works), done (ratings so far), "
        "reply (the finished message, sent as-is)."
    ),
)
async def rate_link(ctx: ToolContext) -> str:
    if not ctx.user_id.isdigit():
        return _with_reply(ctx, {"status": "whatsapp_only"})
    try:
        h = contributions.hash_msisdn(ctx.user_id)
        con = ratings.connect()
        try:
            return _with_reply(ctx, ratings.link_for_contributor(con, h))
        finally:
            con.close()
    except Exception as e:                                  # noqa: BLE001 — soft-fail, the reply says so
        log.warning("rate_link failed: %s", type(e).__name__)
        return _with_reply(ctx, {"status": "error"})
