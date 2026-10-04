#!/usr/bin/env bash
# Send a pipeline message to the operator's WhatsApp via the running webhook
# container (ongiini.ops_alert path: utility template, text fallback).
# Meta accepts a template and only reports failure later through the status
# webhook (e.g. 131042 unpaid invoice, 2026-10-04 — four alerts were lost
# while the API call looked fine). So we wait for the delivery status in
# /data/delivery_status.log and fall back to plain text (free, but only inside
# the 24 h window after the operator last wrote to the bot).
# Soft-fails: a lost message never stops a run.
#   bash deploy/train/notify.sh "B50k fertig: Ndonga 41.2"
cd "$HOME/dev/Ongiini" || exit 0
TO="$(grep '^ONGIINI_OPERATOR_MSISDN=' .env | cut -d= -f2-)"
[ -n "$TO" ] || { echo "notify: no ONGIINI_OPERATOR_MSISDN"; exit 0; }
timeout 150 docker exec -i -e TO="$TO" -e MSG="Spark-Training: $1" ongiini-webhook python3 - <<'PY' || echo "notify: delivery failed"
import asyncio, json, os, sys, time
sys.path.insert(0, "/app")
from ongiini.ops_alert import TEMPLATE_LANGUAGE, TEMPLATE_NAME, template_param
from ongiini.whatsapp import send_template, send_text

LOG = "/data/delivery_status.log"


def status_of(msg_id: str, wait_s: int = 60) -> str:
    """Last delivery status Meta reported for msg_id ("failed:<code>", "sent", ...)."""
    seen, end = "unknown", time.time() + wait_s
    while time.time() < end:
        try:
            with open(LOG, "rb") as f:
                f.seek(max(0, os.path.getsize(LOG) - 200_000))
                for line in f.read().decode("utf-8", "ignore").splitlines():
                    if msg_id in line:
                        r = json.loads(line)
                        seen = r["status"] + (f":{r['errors'][0]['code']}" if r.get("errors") else "")
        except (OSError, ValueError, KeyError, IndexError):
            pass
        if seen.startswith(("delivered", "read", "failed")):
            return seen
        time.sleep(5)
    return seen


async def main() -> str:
    to, text = os.environ["TO"], os.environ["MSG"]
    try:
        resp = await send_template(to, TEMPLATE_NAME, TEMPLATE_LANGUAGE, [template_param(text)])
        st = status_of(resp["messages"][0]["id"])
        if not st.startswith("failed"):
            return f"template {st}"
        first = f"template {st}"
    except Exception as exc:  # noqa: BLE001 — fall back to free text
        first = f"template error {exc!r}"[:120]
    await send_text(to, text)
    return f"{first}; text sent (arrives only inside the 24 h window)"

print("notify:", asyncio.run(main()))
PY
exit 0
