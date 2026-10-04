#!/usr/bin/env bash
# Planned hand-over between pipelines: wait until the running pipeline's log
# shows <trigger> (e.g. "train B200k", the stage we drop), stop that pipeline
# cleanly (its log gets a "stopping" line so its watchdog stays quiet), make
# sure no GPU job is left, then start the next pipeline. Run detached:
#   setsid nohup bash deploy/train/switch_after.sh <pid> <log> <trigger regex> <next log> <next cmd...> \
#     > data/private/experiments/switch.log 2>&1 < /dev/null &
set -u
cd "$HOME/dev/Ongiini"
PID=$1 LOG=$2 TRIGGER=$3 NEXT_LOG=$4; shift 4
log() { echo "$(date '+%F %T') $*"; }
log "waiting for /$TRIGGER/ in $LOG (pipeline pid $PID)"
until grep -aqE "$TRIGGER" "$LOG"; do
  if ! kill -0 "$PID" 2>/dev/null; then
    log "pipeline $PID ended before the trigger — not starting the next one"
    bash deploy/train/notify.sh "Übergabe abgebrochen: die laufende Pipeline endete vor '$TRIGGER'. Nichts Neues gestartet." | tail -1
    exit 1
  fi
  sleep 5
done
log "trigger seen — stopping pipeline $PID"
echo "$(date '+%F %T') planned hand-over to the next pipeline ($TRIGGER skipped) — stopping" >> "$LOG"
# setsid made the pipeline its own process group: this ends it, its watchdog
# and monitor loops; a container it already started is killed separately.
kill -- -"$PID" 2>/dev/null || kill "$PID" 2>/dev/null
for _ in $(seq 1 12); do
  c=$(docker ps --format '{{.Names}}' | grep -E '^ongiini-(train|eval|ret)-' | tr '\n' ' ')
  [ -n "$c" ] && docker kill $c >/dev/null 2>&1
  sleep 5
done
c=$(docker ps --format '{{.Names}}' | grep -E '^ongiini-(train|eval|ret)-' | tr '\n' ' ')
if [ -n "$c" ] || kill -0 "$PID" 2>/dev/null; then
  log "could not stop cleanly (containers: '$c') — not starting the next one"
  bash deploy/train/notify.sh "Übergabe abgebrochen: alte Pipeline ließ sich nicht sauber stoppen ($c)." | tail -1
  exit 1
fi
log "stopped; starting: $*"
setsid nohup "$@" > "$NEXT_LOG" 2>&1 < /dev/null &
sleep 30
log "next pipeline log:"; head -5 "$NEXT_LOG"
