#!/usr/bin/env bash
# Round 6 after round 5, laptop-independent and strictly sequential:
#   wait until pipeline_goal5.sh has ended → only if its last run ended with "pipeline done":
#   git pull (brings train_lora's divergence retry and 4-bit evaluation, held back while
#   round 5 ran) → pipeline_goal6.sh as TINY (incl. a forced divergence) → only if that
#   passes: the real run (H12 vs H31 probe).
#   setsid nohup bash deploy/train/chain_round6.sh > data/private/experiments/chain_round6.log 2>&1 < /dev/null &
set -u
cd "$HOME/dev/Ongiini"
log() { echo "$(date '+%F %T') $*"; }
notify() { bash deploy/train/notify.sh "$*" | tail -1; }
G5=data/private/experiments/pipeline_goal5.log
log "waiting for pipeline_goal5.sh to end"
while pgrep -f "bash deploy/train/pipeline_goal5.sh" >/dev/null; do sleep 60; done
if ! awk '/goal pipeline 5 start \(TINY=0\)/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$G5" | grep -q "pipeline done"; then
  log "round 5 did not finish — round 6 not started"
  notify "Runde 6 nicht gestartet: Runde 5 endete ohne Abschluss (siehe pipeline_goal5.log)."
  exit 1
fi
git checkout HEAD -- deploy/train/pipeline_goal5.sh  # taken from origin by hand on 2026-10-09; pull brings it again
git pull -q || { log "git pull failed"; notify "Runde 6 nicht gestartet: git pull auf dem Spark schlug fehl."; exit 1; }
log "at $(git log --oneline -1)"
log "TINY run of round 6"
rm -rf data/private/experiments/tinytest data/private/lora/tinytest data/private/corpus/tinytest
TINY=1 bash deploy/train/pipeline_goal6.sh > data/private/experiments/tiny_goal6.log 2>&1 < /dev/null
if ! grep -q "pipeline done" data/private/experiments/tiny_goal6.log; then
  log "TINY round 6 failed — real run not started"
  notify "Runde 6: Vorabtest (Mini-Modell) fehlgeschlagen, echter Lauf nicht gestartet. Letzte Zeile: $(grep -v '^$' data/private/experiments/tiny_goal6.log | tail -1 | cut -c1-200)"
  exit 1
fi
rm -rf data/private/experiments/tinytest data/private/lora/tinytest data/private/corpus/tinytest
log "real run of round 6"
bash deploy/train/pipeline_goal6.sh >> data/private/experiments/pipeline_goal6.log 2>&1 < /dev/null
log "chain done"
