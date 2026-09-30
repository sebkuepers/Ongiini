"""Operator alerts over WhatsApp.

Free-text WhatsApp messages only reach a recipient who wrote to the bot
in the last 24 hours, so alerts go out as the pre-approved UTILITY
template ``ongiini_ops_alert`` (see ``broadcast/register_template.py``).
Until Meta has approved it — or if the template send fails for any
other reason — the alert falls back to plain text, which still arrives
inside an open 24h window.

Used by ``hooks/health_alert_hook.py`` and ``scripts/daily_health.sh``.
"""

from __future__ import annotations

import logging
import re

from .whatsapp import send_template, send_text

log = logging.getLogger("ongiini.ops_alert")

TEMPLATE_NAME = "ongiini_ops_alert"
TEMPLATE_LANGUAGE = "en"
# Meta rejects template parameters with newlines, tabs or 4+ spaces and
# caps their length; keep well inside it.
_MAX_PARAM_CHARS = 900


def template_param(text: str) -> str:
    """Flatten ``text`` into one template-safe line."""
    flat = re.sub(r"\s*\n\s*", "; ", text.strip())
    flat = re.sub(r"[\t ]{2,}", " ", flat.replace("\t", " "))
    if len(flat) > _MAX_PARAM_CHARS:
        flat = flat[: _MAX_PARAM_CHARS - 1].rstrip() + "…"
    return flat or "(no details)"


async def send_ops_alert(to: str, text: str) -> str:
    """Send ``text`` to the operator. Returns "template" or "text" for the
    path that was used. Raises only if both paths fail."""
    try:
        await send_template(to, TEMPLATE_NAME, TEMPLATE_LANGUAGE, [template_param(text)])
        return "template"
    except Exception as exc:                        # noqa: BLE001 — fall back to free text
        log.warning("ops alert template send failed (%s); falling back to text", exc)
    await send_text(to, text)
    return "text"
