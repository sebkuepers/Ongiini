#!/usr/bin/env bash
# Alert when the pipeline dies or hangs, so a failure is noticed in minutes,
# not the next morning. Started by pipeline_bt_curve.sh, ends with it.
#   bash deploy/train/watchdog.sh <pipeline pid> <log file>
set -u
PID=$1; LOG=$2; STALL_MIN=45; warned=0
while true; do
  sleep 120
  if ! kill -0 "$PID" 2>/dev/null; then
    grep -q "pipeline done" "$LOG" || grep -q "stopping" "$LOG" \
      || bash deploy/train/notify.sh "Pipeline-Prozess ist ohne Abschlussmeldung beendet. Letzte Zeile: $(tail -c 300 "$LOG" | tr '\r' '\n' | grep -v '^$' | tail -1)"
    exit 0
  fi
  age=$(( ( $(date +%s) - $(stat -c %Y "$LOG") ) / 60 ))
  if [ "$age" -ge "$STALL_MIN" ] && [ "$warned" = 0 ]; then
    bash deploy/train/notify.sh "Kein Fortschritt seit $age Minuten. Letzte Zeile: $(tail -c 300 "$LOG" | tr '\r' '\n' | grep -v '^$' | tail -1)"
    warned=1
  elif [ "$age" -lt "$STALL_MIN" ]; then
    warned=0
  fi
done
