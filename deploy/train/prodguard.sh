#!/usr/bin/env bash
# Production guard: protects the Ongiini vLLM from GPU jobs next to it.
# Every 30 s a tiny completion request. After 3 failures in a row (no answer
# within 20 s) all training / eval / retention containers are killed and the
# operator is alerted; if vLLM still does not answer 5 minutes later it is
# restarted with deploy/spark/restart-vllm-with-mm-flags.sh. On 2026-10-05
# vLLM stalled for 47 minutes next to a training + bulk generation and only a
# manual restart brought Ongiini back. Runs as systemd unit ongiini-prodguard.
# Overridable for tests: PROBE_URL, INTERVAL_S, FAILS, RESTART_AFTER_S,
# PATTERN, NOTIFY_CMD, RESTART_CMD.
set -u
cd /home/nexus/dev/Ongiini
PROBE_URL=${PROBE_URL:-http://localhost:8124/v1/chat/completions}
INTERVAL_S=${INTERVAL_S:-30}; FAILS=${FAILS:-3}; RESTART_AFTER_S=${RESTART_AFTER_S:-300}
PATTERN=${PATTERN:-'^ongiini-(train|eval|ret)-'}
NOTIFY=${NOTIFY_CMD:-"sudo -u nexus bash deploy/train/notify.sh"}
RESTART=${RESTART_CMD:-"sudo -u nexus bash deploy/spark/restart-vllm-with-mm-flags.sh"}
log() { echo "$(date '+%F %T') $*"; }
probe() {
  curl -sf -m 20 -o /dev/null "$PROBE_URL" -H 'content-type: application/json' \
    -d '{"model":"gemma-4-26b","messages":[{"role":"user","content":"Say ok."}],"max_tokens":4}'
}
fails=0; tripped_at=0; restarts=(); capped=0
while true; do
  if probe; then
    [ "$fails" -ge "$FAILS" ] && { log "vLLM answers again"; timeout 180 $NOTIFY "Production antwortet wieder." & }
    fails=0; tripped_at=0; capped=0
  else
    fails=$((fails + 1)); log "probe failed ($fails)"
    if [ "$fails" -eq "$FAILS" ]; then
      victims=$(docker ps --format '{{.Names}}' | grep -E "$PATTERN" | tr '\n' ' ')
      [ -n "$victims" ] && docker kill $victims >/dev/null 2>&1
      tripped_at=$(date +%s)
      log "production not answering — killed GPU jobs: ${victims:-none}"
      timeout 180 $NOTIFY "Production (vLLM) antwortet nicht. GPU-Jobs gestoppt: ${victims:-keine}. Neustart von vLLM in $((RESTART_AFTER_S / 60)) min, falls es dann noch hängt." &
    elif [ "$tripped_at" -gt 0 ] && [ $(( $(date +%s) - tripped_at )) -ge "$RESTART_AFTER_S" ]; then
      now=$(date +%s); recent=0
      for t in "${restarts[@]}"; do [ $((now - t)) -lt 3600 ] && recent=$((recent + 1)); done
      if [ "$recent" -ge 2 ]; then  # never loop: at most 2 automatic restarts per hour
        log "still not answering, 2 restarts in the last hour — alert only"
        [ "$capped" = 0 ] && { timeout 180 $NOTIFY "vLLM hängt trotz 2 Neustarts in der letzten Stunde. Bitte manuell prüfen." & }
        capped=1
        tripped_at=$now
        sleep "$INTERVAL_S"; continue
      fi
      restarts+=("$now")
      log "still not answering — restarting vLLM"
      timeout 180 $NOTIFY "vLLM hängt weiter — automatischer Neustart (ca. 4 min)." &
      # /run, not /tmp: with protected_regular root may not write a /tmp file owned by
      # another user (the first automatic restart on 2026-10-06 failed that way).
      $RESTART > /run/ongiini-prodguard-vllm-restart.log 2>&1
      log "restart finished: $(tail -1 /run/ongiini-prodguard-vllm-restart.log)"
      tripped_at=$(date +%s)  # give it another full window before the next restart
    fi
  fi
  sleep "$INTERVAL_S"
done
