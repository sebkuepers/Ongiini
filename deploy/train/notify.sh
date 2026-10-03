#!/usr/bin/env bash
# Send a pipeline message to the operator's WhatsApp via the running webhook
# container (same path as scripts/daily_health.sh: ongiini.ops_alert, utility
# template with text fallback). Soft-fails: a lost message never stops a run.
#   bash deploy/train/notify.sh "B50k fertig: Ndonga 41.2"
cd "$HOME/dev/Ongiini" || exit 0
TO="$(grep '^ONGIINI_OPERATOR_MSISDN=' .env | cut -d= -f2-)"
[ -n "$TO" ] || { echo "notify: no ONGIINI_OPERATOR_MSISDN"; exit 0; }
docker exec -i -e TO="$TO" -e MSG="Spark-Training: $1" ongiini-webhook python3 - <<'PY' || echo "notify: delivery failed"
import asyncio, os, sys
sys.path.insert(0, "/app")
from ongiini.ops_alert import send_ops_alert
print("notify:", asyncio.run(send_ops_alert(os.environ["TO"], os.environ["MSG"])))
PY
exit 0
