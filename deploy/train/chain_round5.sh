#!/usr/bin/env bash
# Round 5 after round 4, laptop-independent and strictly sequential:
#   wait until pipeline_goal4.sh has ended → only if its run ended with "pipeline done"
#   (2026-10-09: it did, with T4 diverged; round 5 now starts with T4b at lr 1e-4):
#   pipeline_goal5.sh as TINY (tiny model, plumbing) → only if that passes: the real run.
#   setsid nohup bash deploy/train/chain_round5.sh > data/private/experiments/chain_round5.log 2>&1 < /dev/null &
set -u
cd "$HOME/dev/Ongiini"
log() { echo "$(date '+%F %T') $*"; }
notify() { bash deploy/train/notify.sh "$*" | tail -1; }
G4=data/private/experiments/pipeline_goal4.log
log "waiting for pipeline_goal4.sh to end"
while pgrep -f "bash deploy/train/pipeline_goal4.sh" >/dev/null; do sleep 60; done
# judge only the last run (after the 11:09 restart)
if ! sed -n '/restart after fixes/,$p' "$G4" | grep -q "pipeline done"; then
  log "round 4 did not finish — round 5 not started"
  notify "Runde 5 nicht gestartet: Runde 4 endete ohne Abschluss (siehe pipeline_goal4.log)."
  exit 1
fi
log "TINY run of round 5"
rm -rf data/private/experiments/tinytest data/private/lora/tinytest data/private/corpus/tinytest
TINY=1 bash deploy/train/pipeline_goal5.sh > data/private/experiments/tiny_goal5.log 2>&1 < /dev/null
if ! grep -q "pipeline done" data/private/experiments/tiny_goal5.log; then
  log "TINY round 5 failed — real run not started"
  notify "Runde 5: Vorabtest (Mini-Modell) fehlgeschlagen, echter Lauf nicht gestartet. Letzte Zeile: $(grep -v '^$' data/private/experiments/tiny_goal5.log | tail -1 | cut -c1-200)"
  exit 1
fi
rm -rf data/private/experiments/tinytest data/private/lora/tinytest data/private/corpus/tinytest
log "real run of round 5"
bash deploy/train/pipeline_goal5.sh >> data/private/experiments/pipeline_goal5.log 2>&1 < /dev/null
log "chain done"
