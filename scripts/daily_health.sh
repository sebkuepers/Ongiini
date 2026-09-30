#!/usr/bin/env bash
# Daily capability check for the Spark host (cron).
#
# Runs `trace_query.py health` over the last 24 h of trace.jsonl. On a
# breach (exit 1) it writes the report to the system journal and, when
# ONGIINI_OPERATOR_MSISDN is set, sends the breach list to that WhatsApp
# number through the running webhook container.
#
# Install (crontab -e on the Spark, as the repo owner):
#   15 7 * * * bash ~/dev/Ongiini/scripts/daily_health.sh
# The operator number is read from ONGIINI_OPERATOR_MSISDN in the
# environment or, if unset, from the repo's .env.
#
# Why: the Tavily 402 outage of Aug–Sep 2026 failed every web search for
# five weeks before anyone noticed. This check fails on day one.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE="${ONGIINI_TRACE_PATH:-$REPO_DIR/data/trace.jsonl}"
CONTAINER="${ONGIINI_WEBHOOK_CONTAINER:-ongiini-webhook}"
if [ -z "${ONGIINI_OPERATOR_MSISDN:-}" ] && [ -f "$REPO_DIR/.env" ]; then
  ONGIINI_OPERATOR_MSISDN="$(grep '^ONGIINI_OPERATOR_MSISDN=' "$REPO_DIR/.env" | cut -d= -f2-)"
fi

report="$(python3 "$REPO_DIR/scripts/trace_query.py" health --window=24h --path "$TRACE")"
status=$?

if [ "$status" -eq 0 ]; then
  logger -t ongiini-health "ok"
  exit 0
fi

logger -t ongiini-health "BREACH: $(printf '%s' "$report" | python3 -c 'import json,sys; print("; ".join(json.load(sys.stdin).get("breaches", [])))')"

if [ -n "${ONGIINI_OPERATOR_MSISDN:-}" ]; then
  breaches="$(printf '%s' "$report" | python3 -c 'import json,sys; print("\n".join("- " + b for b in json.load(sys.stdin).get("breaches", [])))')"
  docker exec -i -e TO="$ONGIINI_OPERATOR_MSISDN" "$CONTAINER" python3 - <<PY || logger -t ongiini-health "alert delivery failed"
import asyncio, os, sys
sys.path.insert(0, "/app")
from ongiini.ops_alert import send_ops_alert
text = "daily health check found problems (last 24 h):\n" + """$breaches"""
print(asyncio.run(send_ops_alert(os.environ["TO"], text)))
PY
fi
exit 1
