#!/usr/bin/env bash
# One queue job = the real run only, no TINY rehearsal (use when the rehearsal already passed
# today and only a later stage is left — e.g. goal7 on 2026-10-11 after two passed rehearsals).
#   SKIP_PREFLIGHT_MEM=1 bash deploy/train/run_real.sh deploy/train/pipeline_goal7.sh goal7
set -u
cd "${ONGIINI_ROOT:-$HOME/dev/Ongiini}"
P=$1; NAME=${2:-$(basename "$P" .sh | sed 's/^pipeline_//')}
EXP=data/private/experiments
echo "$(date '+%F %T') === run_pipeline $NAME === (run_real: no rehearsal)" >> "$EXP/pipeline_$NAME.log"
bash "$P" >> "$EXP/pipeline_$NAME.log" 2>&1 < /dev/null
awk '/=== run_pipeline /{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$EXP/pipeline_$NAME.log" | grep -q "pipeline done"
