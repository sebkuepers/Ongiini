"""Send a single proactive WhatsApp template to one user.

Three things happen for every recipient, in this exact order:

  1. SHORT-TERM MEMORY WRITE — append the rendered template body as
     a synthetic assistant turn so the AI has context when the user
     replies. Sebastian flagged this: a broadcast that bypasses
     memory leaves the bot replying "Hi! What can I help with?" to
     a user who just got a feature announcement. The memory write
     prevents that.
  2. META API SEND — POST a `type: template` payload via
     `whatsapp.send_template`. Distinct payload shape from regular
     session text sends.
  3. PER-RECIPIENT RESULT — return a BroadcastResult that the CLI
     script logs (success / failure with reason).

We intentionally write to memory BEFORE sending. If the Meta call
fails, the memory has an orphan assistant turn the user will never
see referenced — but they also never received the broadcast, so the
next time they message us the AI's "I told them X" assumption is
benign. The opposite order (send first, write memory after) creates
the worse failure mode: user got the broadcast, replied, AI has no
context.

This module knows nothing about RECIPIENT ENUMERATION or
THROTTLING — those live in scripts/broadcast.py. It does one
broadcast for one msisdn at a time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from ..config import settings
from ..memory import short_term as memory
from ..whatsapp import send_template

log = logging.getLogger("ongiini.broadcast.sender")


@dataclass(frozen=True)
class BroadcastResult:
    """Outcome of broadcasting to one recipient."""
    msisdn: str
    ok: bool
    skipped_reason: str | None = None     # set when ok=False, send_attempted=False
    error: str | None = None              # set when ok=False, send_attempted=True
    meta_message_id: str | None = None    # set when ok=True
    memory_written: bool = False


def render_template_body(body_text: str) -> str:
    """The exact text we append to per-user memory. Mirrors what
    the user sees on WhatsApp from the approved template. Kept as a
    pure function so unit tests can assert on it.

    Template (`ongiini_announcement`):

        Update from Ongiini AI:

        {{1}}

        Tap below to learn more.

    Trailing line is required by Meta (they reject templates that end
    on a variable) — also doubles as a pointer to the Learn-more URL
    button beneath the body.
    """
    return f"Update from Ongiini AI:\n\n{body_text}\n\nTap below to learn more."


async def broadcast_to(
    msisdn: str,
    body_text: str,
    url_suffix: str = "",
    *,
    dry_run: bool = False,
) -> BroadcastResult:
    """Send one proactive broadcast to one user.

    Args:
        msisdn: Recipient phone number.
        body_text: The {{1}} value — the actual announcement copy.
        url_suffix: The {{2}} value — appended to https://ongiini.ai/
            in the "Learn more" button. Empty string → homepage.
        dry_run: If True, log what would happen and return a
            BroadcastResult marked as skipped. No memory write, no
            Meta API call.
    """
    if dry_run:
        log.info("[dry-run] would broadcast to %s (%d chars)", msisdn, len(body_text))
        return BroadcastResult(
            msisdn=msisdn,
            ok=False,
            skipped_reason="dry_run",
            memory_written=False,
        )

    # 1. Memory write — under the same per-user lock the agent uses,
    #    so an inbound message from this user mid-broadcast can't race.
    rendered = render_template_body(body_text)
    memory_written = False
    async with memory.lock_for(msisdn):
        try:
            memory.append_synthetic_assistant_turn(msisdn, rendered)
            memory_written = True
        except Exception:                       # noqa: BLE001
            log.exception("memory write failed for %s — proceeding to send anyway", msisdn)

    # 2. Meta API send. Build the params from settings + caller args.
    try:
        resp = await send_template(
            to=msisdn,
            template_name=settings.whatsapp_template_announcement_name,
            language_code=settings.whatsapp_template_announcement_language,
            body_params=[body_text],
            button_url_param=url_suffix,
        )
    except httpx.HTTPStatusError as exc:
        # 4xx: permanent failure. Log status + small text snippet so
        # the operator can triage (template not approved, recipient
        # blocked us, etc.). Don't log the body — we never log content.
        return BroadcastResult(
            msisdn=msisdn,
            ok=False,
            error=f"http_{exc.response.status_code}: {exc.response.text[:200]}",
            memory_written=memory_written,
        )
    except Exception as exc:                    # noqa: BLE001
        return BroadcastResult(
            msisdn=msisdn,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            memory_written=memory_written,
        )

    # 3. Meta returns {"messages": [{"id": "wamid..."}]} on success.
    msg_id = None
    try:
        msg_id = resp.get("messages", [{}])[0].get("id")
    except (AttributeError, IndexError, TypeError):
        pass

    return BroadcastResult(
        msisdn=msisdn,
        ok=True,
        meta_message_id=msg_id,
        memory_written=memory_written,
    )
