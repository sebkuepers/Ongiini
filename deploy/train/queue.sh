#!/usr/bin/env bash
# Experiment queue for the Spark (2026-10-11): the GPU must never sit idle while experiments
# are planned. data/private/experiments/queue.txt holds one shell command per line; this script
# runs them in order, waits before each start until no ongiini-(train|eval|ret) container runs,
# reports start / end / failure by WhatsApp, continues with the next line after a failure, and
# warns when a job has had no GPU container for IDLE_MIN minutes (a pipeline waiting on
# something) or the queue ran empty. Lines starting with "#" are skipped. A line is moved to
# queue.done (with exit code and duration) when it finishes, so new lines can be appended to
# queue.txt at any time without touching anything running. Runs from a private copy, so a
# git pull cannot change it mid-run.
#   setsid nohup bash deploy/train/queue.sh > data/private/experiments/queue.log 2>&1 < /dev/null &
set -u
if [ -z "${QUEUE_COPY:-}" ]; then
  cp "$0" "/tmp/queue.$$.sh"
  QUEUE_COPY=1 exec bash "/tmp/queue.$$.sh" "$@"
fi
cd "${ONGIINI_ROOT:-$HOME/dev/Ongiini}"
Q=${QUEUE_FILE:-data/private/experiments/queue.txt}
DONE=${Q%.txt}.done
JOBLOG=${Q%.txt}.jobs.log
IDLE_MIN=${IDLE_MIN:-30}
POLL=${QUEUE_POLL:-60}
PATTERN=${QUEUE_PATTERN:-'^ongiini-(train|eval|ret)-'}
NOTIFY=${NOTIFY_CMD:-"bash deploy/train/notify.sh"}
log() { echo "$(date '+%F %T') $*"; }
notify() { $NOTIFY "$*" | tail -1; }
gpu_busy() { docker ps --format '{{.Names}}' | grep -qE "$PATTERN"; }
next_line() { grep -vE '^\s*(#|$)' "$Q" 2>/dev/null | head -1; }
pop_line() {  # remove the first occurrence of "$1" from the queue
  python3 - "$Q" "$1" <<'PY'
import sys
q, line = sys.argv[1], sys.argv[2]
rows = open(q).read().split("\n")
for i, r in enumerate(rows):
    if r == line:
        del rows[i]; break
open(q, "w").write("\n".join(rows))
PY
}
log "queue start ($Q, idle warning after $IDLE_MIN min)"
empty_warned=0
while true; do
  line=$(next_line)
  if [ -z "$line" ]; then
    if [ "$empty_warned" = 0 ]; then
      log "queue empty — waiting for new lines"
      notify "Warteschlange leer: keine weiteren Experimente eingereiht. Die GPU steht still, bis etwas angehängt wird."
      empty_warned=1
    fi
    sleep "$POLL"; continue
  fi
  empty_warned=0
  if gpu_busy; then sleep "$POLL"; continue; fi
  pop_line "$line"
  name=${line:0:100}
  log "START: $line"
  notify "Warteschlange: starte '$name'"
  t0=$(date +%s)
  bash -c "$line" >> "$JOBLOG" 2>&1 &
  job=$!
  last_busy=$(date +%s); idle_warned=0
  while kill -0 "$job" 2>/dev/null; do
    if gpu_busy; then
      last_busy=$(date +%s); idle_warned=0
    elif [ "$idle_warned" = 0 ] && [ $(( ( $(date +%s) - last_busy ) / 60 )) -ge "$IDLE_MIN" ]; then
      log "idle: '$name' has had no GPU container for $IDLE_MIN min"
      notify "Warteschlange: '$name' läuft, aber seit $IDLE_MIN min ohne GPU-Job (wartet oder hängt). Bitte prüfen."
      idle_warned=1
    fi
    sleep "$POLL"
  done
  wait "$job"; rc=$?
  mins=$(( ( $(date +%s) - t0 ) / 60 ))
  echo "$(date '+%F %T') rc=$rc ${mins}min $line" >> "$DONE"
  if [ "$rc" = 0 ]; then
    log "DONE ($mins min): $line"
    notify "Warteschlange: '$name' fertig nach $mins min. Weiter mit dem nächsten Experiment."
  else
    log "FAILED rc=$rc ($mins min): $line"
    notify "Warteschlange: '$name' FEHLGESCHLAGEN (rc=$rc, $mins min). Weiter mit dem nächsten Experiment."
  fi
done
