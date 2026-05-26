"""ongiini.broadcast.register_template — submit / inspect / delete the
proactive-broadcast WhatsApp template via Meta's Graph API.

Endpoint reference:
  POST   /v21.0/{WABA_ID}/message_templates       — submit for approval
  GET    /v21.0/{WABA_ID}/message_templates       — list status
  DELETE /v21.0/{WABA_ID}/message_templates       — remove by name

Requires:
  WHATSAPP_TOKEN                  — already in env
  WHATSAPP_BUSINESS_ACCOUNT_ID    — paste from Meta Business Manager
                                    → WhatsApp Manager → API setup
                                    (NOT the same as WHATSAPP_PHONE_ID)

Templates land in PENDING and auto-flip to APPROVED (1–2h typically)
or REJECTED. Use the `list` subcommand to poll.

USAGE
-----

    # 1. One-time: paste WABA ID from Meta Business Manager into .env
    #    WHATSAPP_BUSINESS_ACCOUNT_ID=10000xxxxxxxx

    # 2. Submit
    python -m ongiini.broadcast.register_template submit

    # 3. Poll status
    python -m ongiini.broadcast.register_template list

    # 4. (Rarely) remove if you need to resubmit after a rejection
    python -m ongiini.broadcast.register_template delete
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import httpx

from ..config import settings

log = logging.getLogger("ongiini.broadcast.register_template")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
)


GRAPH_URL = "https://graph.facebook.com/v21.0"

# Sample value Meta uses to validate the {{1}} body placeholder.
# Mirrors the kind of text we'd actually send.
_SAMPLE_BODY = (
    "We just added voice notes - send me a voice message any time "
    "and I will listen and reply in writing."
)
# Sample URL suffix for the dynamic {{1}} in the URL button.
# Empty suffix would resolve to the homepage which Meta sometimes
# flags during review; using 'contribute/' makes the full link more
# obviously legitimate.
_SAMPLE_URL_SUFFIX = "contribute/"


def _waba_id() -> str:
    """The WhatsApp Business Account ID. Read from env, fail fast if
    missing — without it the Graph API call has no valid endpoint."""
    waba = os.environ.get("WHATSAPP_BUSINESS_ACCOUNT_ID", "").strip()
    if not waba:
        raise SystemExit(
            "WHATSAPP_BUSINESS_ACCOUNT_ID is not set.\n\n"
            "Get it from Meta Business Manager:\n"
            "  WhatsApp Manager -> top right gear icon -> "
            "'WhatsApp Business Account ID'\n"
            "Then add to .env:\n"
            "  WHATSAPP_BUSINESS_ACCOUNT_ID=<the id>\n"
            "and re-run."
        )
    return waba


def _token() -> str:
    tok = settings.whatsapp_token
    if not tok:
        raise SystemExit("WHATSAPP_TOKEN is not set in .env.")
    return tok


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_token()}",
        "Content-Type": "application/json",
    }


# ── Template payload ───────────────────────────────────────────────


def _template_payload() -> dict:
    """The exact payload Meta gets for the `ongiini_announcement`
    template. Mirrors what sender.broadcast_to fills in at send time.

    NOTE: WhatsApp template variables in the BODY and BUTTON URL are
    numbered INDEPENDENTLY. The body has {{1}} (the announcement
    text); the URL button has its OWN {{1}} (the URL suffix). They
    don't share a namespace.
    """
    return {
        "name": settings.whatsapp_template_announcement_name,
        "language": settings.whatsapp_template_announcement_language,
        "category": "MARKETING",
        "components": [
            {
                "type": "BODY",
                "text": "Update from Ongiini AI:\n\n{{1}}",
                "example": {"body_text": [[_SAMPLE_BODY]]},
            },
            {
                "type": "BUTTONS",
                "buttons": [
                    {
                        "type": "URL",
                        "text": "Learn more",
                        "url": "https://ongiini.ai/{{1}}",
                        "example": [f"https://ongiini.ai/{_SAMPLE_URL_SUFFIX}"],
                    }
                ],
            },
        ],
    }


# ── Subcommands ────────────────────────────────────────────────────


def cmd_show() -> int:
    """Print the payload that would be submitted. Read-only, useful
    for review before firing the actual submit."""
    print(json.dumps(_template_payload(), indent=2, ensure_ascii=False))
    return 0


def cmd_submit() -> int:
    waba = _waba_id()
    payload = _template_payload()
    log.info("submitting template %r to WABA %s …", payload["name"], waba)
    r = httpx.post(
        f"{GRAPH_URL}/{waba}/message_templates",
        headers=_headers(),
        json=payload,
        timeout=20,
    )
    try:
        body = r.json()
    except Exception:                      # noqa: BLE001
        body = {"raw": r.text}
    print(json.dumps(body, indent=2, ensure_ascii=False))
    if r.status_code >= 400:
        log.error("submission failed (%s)", r.status_code)
        return 1
    log.info(
        "submitted. status will flip to APPROVED or REJECTED in 1-2h. "
        "Poll with: python -m ongiini.broadcast.register_template list"
    )
    return 0


def cmd_list() -> int:
    """List all templates on the WABA + their statuses."""
    waba = _waba_id()
    name = settings.whatsapp_template_announcement_name
    r = httpx.get(
        f"{GRAPH_URL}/{waba}/message_templates",
        headers=_headers(),
        params={"limit": 50, "fields": "name,language,status,category,rejected_reason"},
        timeout=15,
    )
    if r.status_code >= 400:
        print(r.text)
        return 1
    data = r.json().get("data", [])
    if not data:
        print("(no templates on this WABA yet)")
        return 0
    # Highlight ours
    for t in data:
        marker = " <-- ours" if t.get("name") == name else ""
        print(
            f"  {t.get('name'):<32} {t.get('language'):<6} "
            f"{t.get('category'):<14} {t.get('status'):<10}"
            f"{('  rejected: ' + t.get('rejected_reason', '')) if t.get('status') == 'REJECTED' else ''}"
            f"{marker}"
        )
    return 0


def cmd_delete() -> int:
    """Delete the template by name. Useful only after a REJECTED state
    when you want to resubmit a corrected version."""
    waba = _waba_id()
    name = settings.whatsapp_template_announcement_name
    log.info("DELETING template %r from WABA %s", name, waba)
    r = httpx.delete(
        f"{GRAPH_URL}/{waba}/message_templates",
        headers=_headers(),
        params={"name": name},
        timeout=15,
    )
    print(r.status_code, r.text)
    return 0 if r.status_code < 400 else 1


# ── Entrypoint ────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Register / inspect the proactive-broadcast WhatsApp template"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show", help="Print the payload (does nothing else)")
    sub.add_parser("submit", help="POST the template to Meta for approval")
    sub.add_parser("list", help="List templates + statuses on the WABA")
    sub.add_parser("delete", help="Delete the template (use after REJECTED)")
    args = p.parse_args(argv)
    if args.cmd == "show":
        return cmd_show()
    if args.cmd == "submit":
        return cmd_submit()
    if args.cmd == "list":
        return cmd_list()
    if args.cmd == "delete":
        return cmd_delete()
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
