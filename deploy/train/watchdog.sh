#!/usr/bin/env bash
# Alert when the pipeline dies or hangs, so a failure is noticed in minutes,
# not the next morning. Started by the pipeline scripts, ends with them.
#   bash deploy/train/watchdog.sh <pipeline pid> <log file>
# Progress = the log OR any GPU monitor file (data/private/experiments/gpu/*.csv,
# written every 10 s while a training container runs) changed recently: long
# CPT stages print little to the log, which caused false "no progress" alerts
# every 46 min on 2026-10-04/05. "Ended cleanly" is judged only on log lines
# written after the watchdog started (older runs' "stopping" lines are ignored).
set -u
PID=$1; LOG=$2; warned=0
STALL_MIN=${STALL_MIN:-45}; SLEEP_S=${WATCHDOG_SLEEP:-120}   # overridable for tests
NOTIFY=${NOTIFY_CMD:-"bash deploy/train/notify.sh"}
GPU_DIR="$(dirname "$LOG")/gpu"
START=$(wc -c < "$LOG" 2>/dev/null || echo 0)
newest_age_min() {
  local newest=0 t f
  for f in "$LOG" "$GPU_DIR"/*.csv; do
    [ -e "$f" ] || continue
    t=$(stat -c %Y "$f"); [ "$t" -gt "$newest" ] && newest=$t
  done
  echo $(( ( $(date +%s) - newest ) / 60 ))
}
while true; do
  sleep "$SLEEP_S"
  if ! kill -0 "$PID" 2>/dev/null; then
    tail -c +"$((START + 1))" "$LOG" | grep -aqE "pipeline done|stopping" \
      || $NOTIFY "Pipeline-Prozess ist ohne Abschlussmeldung beendet. Letzte Zeile: $(tail -c 300 "$LOG" | tr '\r' '\n' | grep -v '^$' | tail -1)"
    exit 0
  fi
  age=$(newest_age_min)
  if [ "$age" -ge "$STALL_MIN" ] && [ "$warned" = 0 ]; then
    $NOTIFY "Kein Fortschritt seit $age Minuten (weder Log noch GPU-Monitor). Letzte Zeile: $(tail -c 300 "$LOG" | tr '\r' '\n' | grep -v '^$' | tail -1)"
    warned=1
  elif [ "$age" -lt "$STALL_MIN" ]; then
    warned=0
  fi
done
