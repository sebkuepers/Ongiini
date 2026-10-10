#!/usr/bin/env bash
# Rounds 6 and 7 after round 5, laptop-independent and strictly sequential:
#   wait until pipeline_goal5.sh has ended → only if its last run ended with "pipeline done":
#   git pull (brings code held back while round 5 ran: train_lora's divergence retry, 4-bit
#   evaluation, train_cpt's logged block order) → for each round: TINY first (tiny models;
#   round 7's TINY also forces a divergence), the real run only if TINY passes.
#   round 6 = pipeline_goal6.sh: C2 (CPT over all 21.2M tokens) + identical T4 SFT (C2-T4)
#   round 7 = pipeline_goal7.sh: size probe H12 vs H31 (Gemma 4 31B, 4-bit)
#   setsid nohup bash deploy/train/chain_round6.sh > data/private/experiments/chain_round6.log 2>&1 < /dev/null &
set -u
# bash reads a script while running it and the git pull below replaces this file:
# run from a private copy
if [ -z "${CHAIN_COPY:-}" ]; then
  cp "$0" "/tmp/chain_round6.$$.sh"
  CHAIN_COPY=1 exec bash "/tmp/chain_round6.$$.sh" "$@"
fi
cd "$HOME/dev/Ongiini"
log() { echo "$(date '+%F %T') $*"; }
notify() { bash deploy/train/notify.sh "$*" | tail -1; }
G5=data/private/experiments/pipeline_goal5.log
log "waiting for pipeline_goal5.sh to end"
while pgrep -f "bash deploy/train/pipeline_goal5.sh" >/dev/null; do sleep 60; done
if ! awk '/goal pipeline 5 start \(TINY=0\)/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$G5" | grep -q "pipeline done"; then
  log "round 5 did not finish — rounds 6/7 not started"
  notify "Runde 6 nicht gestartet: Runde 5 endete ohne Abschluss (siehe pipeline_goal5.log)."
  exit 1
fi
# pipeline_goal5.sh and this file were taken from origin by hand while round 5 ran (staged,
# identical to origin), so the fast-forward accepts them
git pull -q || { log "git pull failed"; notify "Runde 6 nicht gestartet: git pull auf dem Spark schlug fehl."; exit 1; }
log "at $(git log --oneline -1)"
for round in 6 7; do
  log "TINY run of round $round"
  rm -rf data/private/experiments/tinytest data/private/lora/tinytest data/private/corpus/tinytest
  TINY=1 bash "deploy/train/pipeline_goal$round.sh" > "data/private/experiments/tiny_goal$round.log" 2>&1 < /dev/null
  if ! grep -q "pipeline done" "data/private/experiments/tiny_goal$round.log"; then
    log "TINY round $round failed — real run not started"
    notify "Runde $round: Vorabtest (Mini-Modell) fehlgeschlagen, echter Lauf nicht gestartet. Letzte Zeile: $(grep -v '^$' "data/private/experiments/tiny_goal$round.log" | tail -1 | cut -c1-200)"
    exit 1
  fi
  rm -rf data/private/experiments/tinytest data/private/lora/tinytest data/private/corpus/tinytest
  log "real run of round $round"
  bash "deploy/train/pipeline_goal$round.sh" >> "data/private/experiments/pipeline_goal$round.log" 2>&1 < /dev/null
  grep -q "pipeline done" <(awk "/goal pipeline $round start \\(TINY=0\\)/{buf=\"\"} {buf=buf \$0 \"\\n\"} END{printf \"%s\", buf}" \
    "data/private/experiments/pipeline_goal$round.log") || { log "round $round did not finish — stopping the chain"; exit 1; }
done
log "chain done"
