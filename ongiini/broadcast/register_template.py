"""ongiini.broadcast.register_template — submit / inspect / delete
WhatsApp message templates via Meta's Graph API.

Manages a small registry of templates (`TEMPLATES` below). Each
template has its own name, category (MARKETING vs UTILITY),
language, components, and example values used during Meta's review.

Endpoint reference:
  POST   /v21.0/{WABA_ID}/message_templates       — submit for approval
  GET    /v21.0/{WABA_ID}/message_templates       — list status
  DELETE /v21.0/{WABA_ID}/message_templates       — remove by name

Requires:
  WHATSAPP_TOKEN                  — already in env
  WHATSAPP_BUSINESS_ACCOUNT_ID    — paste from Meta Business Manager

Templates land in PENDING and auto-flip to APPROVED (typically 1-2h)
or REJECTED. Use the `list` subcommand to poll.

USAGE
-----

    # Show available templates
    python -m ongiini.broadcast.register_template templates

    # Show the payload for a specific template (read-only review)
    python -m ongiini.broadcast.register_template show ongiini_announcement
    python -m ongiini.broadcast.register_template show ongiini_smart_followup
    python -m ongiini.broadcast.register_template show ongiini_token_status

    # Submit a template for Meta approval
    python -m ongiini.broadcast.register_template submit ongiini_smart_followup

    # Poll status across ALL templates on the WABA
    python -m ongiini.broadcast.register_template list

    # Remove (after REJECTED, before resubmitting a corrected version)
    python -m ongiini.broadcast.register_template delete ongiini_smart_followup
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


# ── Template registry ─────────────────────────────────────────────
#
# Each entry IS the full payload Meta sees (minus `name`, which is the
# dict key, and `language`, which we keep separate so we can submit
# the same template content in multiple languages later).
#
# Conventions:
#   - Body NEVER ends on a variable — Meta rejects those.
#   - Two variables max per template (review complexity grows fast).
#   - {{1}}, {{2}} are positional in the body; URL buttons get their
#     OWN {{1}} namespace.
#   - `examples` provide realistic sample values for Meta's review.
# ─────────────────────────────────────────────────────────────────

TEMPLATES: dict[str, dict] = {
    # ── A: MARKETING — proactive announcements ────────────────────
    "ongiini_announcement": {
        "category": "MARKETING",
        "language": "en",
        "components": [
            {
                "type": "BODY",
                "text": (
                    "Update from Ongiini AI:\n\n"
                    "{{1}}\n\n"
                    "Tap below to learn more."
                ),
                "example": {
                    "body_text": [[
                        "We just added voice notes - send me a voice "
                        "message any time and I will listen and reply "
                        "in writing."
                    ]],
                },
            },
            {
                "type": "BUTTONS",
                "buttons": [
                    {
                        "type": "URL",
                        "text": "Learn more",
                        "url": "https://ongiini.ai/{{1}}",
                        "example": ["https://ongiini.ai/contribute/"],
                    }
                ],
            },
        ],
    },

    # ── B: UTILITY — smart same-thread follow-up ──────────────────
    # Anchored on a specific past user action (the topic they asked
    # about). Sent 18-30h after the user's last activity, only once
    # per session, only if they didn't already come back.
    "ongiini_smart_followup": {
        "category": "UTILITY",
        "language": "en",
        "components": [
            {
                "type": "HEADER",
                "format": "TEXT",
                "text": "Following up from Ongiini AI",
            },
            {
                "type": "BODY",
                "text": (
                    "Yesterday you asked Ongiini AI about {{1}}.\n\n"
                    "{{2}}\n\n"
                    "Reply any time to keep going, or say \"stop\" "
                    "to skip these check-ins."
                ),
                "example": {
                    "body_text": [[
                        "your CV for the cashier position at Shoprite",
                        "Want to prep for the interview? I can run "
                        "you through some questions a Namibian "
                        "hiring manager would actually ask.",
                    ]],
                },
            },
            {
                "type": "BUTTONS",
                "buttons": [
                    {"type": "QUICK_REPLY", "text": "Yes, let's continue"},
                    {"type": "QUICK_REPLY", "text": "Not today"},
                ],
            },
        ],
    },

    # ── C: UTILITY — weekly account-status + usage suggestion ─────
    # Sent weekly. {{1}} is the remaining-token count; {{2}} is a
    # short suggestion for what to use the tokens on (generated per
    # user from their recent topic history).
    "ongiini_token_status": {
        "category": "UTILITY",
        "language": "en",
        "components": [
            {
                "type": "HEADER",
                "format": "TEXT",
                "text": "Your Ongiini AI usage this week",
            },
            {
                "type": "BODY",
                "text": (
                    "You have {{1}} free tokens left this month. "
                    "Your allowance refreshes on the 1st.\n\n"
                    "Idea for what to do with them: {{2}}\n\n"
                    "Reply any time, or just ignore this message — "
                    "no pressure."
                ),
                "example": {
                    "body_text": [[
                        "752,000",
                        "practise English conversation with me for 10 "
                        "minutes — I can give you feedback on your "
                        "phrasing",
                    ]],
                },
            },
            {
                "type": "BUTTONS",
                "buttons": [
                    {"type": "QUICK_REPLY", "text": "Yes, let's do that"},
                    {"type": "QUICK_REPLY", "text": "Not now"},
                ],
            },
        ],
    },
}


# ── Helpers ───────────────────────────────────────────────────────


def _waba_id() -> str:
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


def _payload_for(name: str) -> dict:
    """Build the exact submission payload Meta gets for `name`."""
    if name not in TEMPLATES:
        raise SystemExit(
            f"Unknown template {name!r}.\n"
            f"Available: {', '.join(sorted(TEMPLATES))}"
        )
    spec = TEMPLATES[name]
    return {
        "name": name,
        "language": spec["language"],
        "category": spec["category"],
        "components": spec["components"],
    }


def _our_template_names() -> set[str]:
    return set(TEMPLATES.keys())


# ── Subcommands ────────────────────────────────────────────────────


def cmd_templates() -> int:
    """List the templates this registry knows about."""
    print(f"{'name':<28}{'category':<12}{'language':<10}")
    for name, spec in sorted(TEMPLATES.items()):
        print(f"  {name:<26}{spec['category']:<12}{spec['language']:<10}")
    return 0


def cmd_show(name: str) -> int:
    payload = _payload_for(name)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_submit(name: str) -> int:
    waba = _waba_id()
    payload = _payload_for(name)
    log.info(
        "submitting template %r (%s) to WABA %s …",
        name, payload["category"], waba,
    )
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
    """List all templates on the WABA. Highlights the ones in our
    registry so it's easy to see which are 'ours' vs. anything else."""
    waba = _waba_id()
    ours = _our_template_names()
    r = httpx.get(
        f"{GRAPH_URL}/{waba}/message_templates",
        headers=_headers(),
        params={"limit": 100, "fields": "name,language,status,category,rejected_reason"},
        timeout=15,
    )
    if r.status_code >= 400:
        print(r.text)
        return 1
    data = r.json().get("data", [])
    if not data:
        print("(no templates on this WABA yet)")
        return 0
    for t in data:
        marker = " <-- ours" if t.get("name") in ours else ""
        rejected = ""
        if t.get("status") == "REJECTED":
            rejected = f"  rejected: {t.get('rejected_reason', '')}"
        print(
            f"  {t.get('name'):<32} {t.get('language'):<6} "
            f"{t.get('category'):<14} {t.get('status'):<10}{rejected}{marker}"
        )
    return 0


def cmd_delete(name: str) -> int:
    waba = _waba_id()
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
        description="Register / inspect WhatsApp message templates",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("templates", help="List templates the registry knows about")
    s_show = sub.add_parser("show", help="Print the submission payload (read-only)")
    s_show.add_argument("name", help="Template name (see `templates`)")
    s_sub = sub.add_parser("submit", help="POST the template to Meta for approval")
    s_sub.add_argument("name", help="Template name (see `templates`)")
    sub.add_parser("list", help="List templates + statuses on the WABA")
    s_del = sub.add_parser("delete", help="Delete a template (use after REJECTED)")
    s_del.add_argument("name", help="Template name to delete")
    args = p.parse_args(argv)
    if args.cmd == "templates":
        return cmd_templates()
    if args.cmd == "show":
        return cmd_show(args.name)
    if args.cmd == "submit":
        return cmd_submit(args.name)
    if args.cmd == "list":
        return cmd_list()
    if args.cmd == "delete":
        return cmd_delete(args.name)
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
