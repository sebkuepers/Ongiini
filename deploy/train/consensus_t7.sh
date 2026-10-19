#!/usr/bin/env bash
# Consensus distillation data for variant T7 (2026-10-19): six finished 12B adapters translate the
# English chat sentences of scripts/consensus_distill.py through one eval vLLM (multi-LoRA), then
# the per-sentence consensus is kept. GPU job: run it from the queue, between trainings.
#   bash deploy/train/consensus_t7.sh   → data/private/corpus/consensus_v1/consensus.jsonl
set -euo pipefail
cd "${ONGIINI_ROOT:-$HOME/dev/Ongiini}"
L=data/private/lora
log() { echo "$(date '+%F %T') $*"; }
notify() { bash deploy/train/notify.sh "$*" | tail -1; }
MODELS=(c1t4 cbrt4 t5 r64rep t4b c1rt4)
OUT=data/private/corpus/consensus_v1
P=(docker run --rm --name ongiini-eval-consensus --network host --user 1000:1000 -e HOME=/tmp -e USER=nexus -e LOGNAME=nexus
   -e PYTHONUNBUFFERED=1 -v "$PWD:/work" -w /work ongiini-evalsuite:latest python3 scripts/consensus_distill.py)
[ -s "$OUT/sources.jsonl" ] || "${P[@]}" sources --n 30000
log "consensus: serving ${MODELS[*]}"
EVAL_GPU_MEM_UTIL=0.45 EVAL_MAX_SEQS=48 bash deploy/eval/serve_12b_lora.sh c1t4=$L/C1-T4_12b cbrt4=$L/CBr-T4_12b \
  t5=$L/T5_12b r64rep=$L/B10kR-r64rep_12b t4b=$L/T4b_12b c1rt4=$L/C1r-T4_12b
trap 'docker rm -f ongiini-eval-vllm12b >/dev/null 2>&1' EXIT
"${P[@]}" generate --models "${MODELS[@]}" --concurrency 48 2>&1 | grep -vE "HTTP Request|Warning|INFO"
docker rm -f ongiini-eval-vllm12b >/dev/null 2>&1
res=$("${P[@]}" select --models "${MODELS[@]}" 2>&1 | tail -1)
log "consensus: $res"
[ -s "$OUT/consensus.jsonl" ] || { notify "Konsens-Daten T7 fehlgeschlagen"; exit 1; }
notify "Konsens-Daten T7 fertig: $res"
