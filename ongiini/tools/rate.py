"""rate_link — hand a WhatsApp user their personal link to ongiini.ai/rate.

Forced by the classifier verdict RATE_INVITE (the user wants to help
CHECK existing translations — not translate; that is the contribute_*
loop). Argument-less: the user is identified by the salted msisdn hash
(contributions.hash_msisdn), the number itself is never stored. See
ongiini/ratings.py::link_for_contributor.
"""
from __future__ import annotations

import json
import logging

from owela import ToolContext, tool

from .. import contributions, ratings

log = logging.getLogger("ongiini.tools.rate")


@tool(
    name="rate_link",
    description=(
        "FORCED by classifier verdict RATE_INVITE — the user wants to help "
        "CHECK (rate) existing Oshiwambo translations on ongiini.ai/rate. "
        "Returns JSON: status ('ok' | 'whatsapp_only' | 'blocked' | 'error'), "
        "url (their personal link — quote it exactly), returning (bool: they "
        "had a link before; the old one no longer works), done (ratings so far)."
    ),
)
async def rate_link(ctx: ToolContext) -> str:
    if not ctx.user_id.isdigit():
        return json.dumps({"status": "whatsapp_only"})
    try:
        h = contributions.hash_msisdn(ctx.user_id)
        con = ratings.connect()
        try:
            return json.dumps(ratings.link_for_contributor(con, h))
        finally:
            con.close()
    except Exception as e:                                  # noqa: BLE001 — soft-fail, the reply says so
        log.warning("rate_link failed: %s", type(e).__name__)
        return json.dumps({"status": "error"})
