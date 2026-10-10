#!/usr/bin/env bash
# One queue job = TINY rehearsal, then the real run (2026-10-11).
#   bash deploy/train/run_pipeline.sh deploy/train/pipeline_goal7.sh [NAME]
# TINY=1 runs the pipeline end to end with the tiny model into tiny_<NAME>.log; only if that
# ends with "pipeline done" does the real run start, appended to pipeline_<NAME>.log. Exit 0
# only if the real run's last start ended with "pipeline done". Extra environment (e.g. the
# variant variables of pipeline_sft_variant.sh) is passed through.
set -u
cd "${ONGIINI_ROOT:-$HOME/dev/Ongiini}"
P=$1; NAME=${2:-$(basename "$P" .sh | sed 's/^pipeline_//')}
EXP=data/private/experiments
log() { echo "$(date '+%F %T') [$NAME] $*"; }
notify() { bash deploy/train/notify.sh "$*" | tail -1; }
rm -rf "$EXP/tinytest" data/private/lora/tinytest data/private/corpus/tinytest
log "TINY run"
TINY=1 bash "$P" > "$EXP/tiny_$NAME.log" 2>&1 < /dev/null
if ! grep -q "pipeline done" "$EXP/tiny_$NAME.log"; then
  log "TINY failed — real run not started"
  notify "$NAME: Vorabtest (Mini-Modell) fehlgeschlagen, echter Lauf nicht gestartet. Letzte Zeile: $(grep -v '^$' "$EXP/tiny_$NAME.log" | tail -1 | cut -c1-200)"
  exit 1
fi
rm -rf "$EXP/tinytest" data/private/lora/tinytest data/private/corpus/tinytest
log "real run"
echo "$(date '+%F %T') === run_pipeline $NAME ===" >> "$EXP/pipeline_$NAME.log"
bash "$P" >> "$EXP/pipeline_$NAME.log" 2>&1 < /dev/null
if awk '/=== run_pipeline /{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$EXP/pipeline_$NAME.log" | grep -q "pipeline done"; then
  log "real run done"; exit 0
fi
log "real run did not finish"; exit 1
