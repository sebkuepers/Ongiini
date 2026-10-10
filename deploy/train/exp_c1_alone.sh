#!/usr/bin/env bash
# A1 (2026-10-11): where does C1-T4's forgetting come from? Measure the CPT adapter C1 ALONE
# (no SFT) with the same harness as every candidate: compat suite (IFEval, No Robots pairwise,
# Ongiini scenarios, GSM8K, safety, images, Oshindonga chat), tool suite, retention suite.
# If C1 alone keeps IFEval ≈ base, the loss comes from the SFT moving the r64 adapter
# (→ merge C1, train a fresh r32); if C1 alone is already worse, it is the CPT itself
# (→ replay / gentler CPT). ~1.5 h, no training.
#   bash deploy/train/exp_c1_alone.sh
set -u
cd "${ONGIINI_ROOT:-$HOME/dev/Ongiini}"
log() { echo "$(date '+%F %T') $*"; }
notify() { bash deploy/train/notify.sh "$*" | tail -1; }
EXP=data/private/experiments; ADAPTER=data/private/lora/C1_cpt_12b; TAG=C1cpt
others=$(docker ps --format '{{.Names}}' | grep -E '^ongiini-(train|eval|ret)-' | tr '\n' ' ')
[ -z "$others" ] || { log "GPU jobs running: $others — stop"; exit 1; }
crun() { local n=$1; shift
  docker run --rm --name "$n" --network host --user 1000:1000 -e HOME=/tmp -e USER=nexus -e LOGNAME=nexus \
    -e PYTHONUNBUFFERED=1 -v "$PWD:/work" -w /work ongiini-evalsuite:latest "$@"; }
if [ ! -f "data/private/compat/report_${TAG}_vs_base.json" ]; then
  EVAL_GPU_MEM_UTIL=0.45 EVAL_MAX_SEQS=32 bash deploy/eval/serve_12b_lora.sh "c1cpt=$ADAPTER" || { log "vLLM failed"; exit 1; }
  ok=1
  crun ongiini-eval-c1alone-compat python3 scripts/compat_suite.py generate --model c1cpt --label "$TAG" --concurrency 32 2>&1 \
    | grep -vE "HTTP Request|Warning" | tail -2 || ok=0
  crun ongiini-eval-c1alone-tools python3 scripts/tool_suite.py generate --model c1cpt --label "$TAG" --concurrency 16 2>&1 \
    | grep -vE "HTTP Request|Warning" | tail -2 || ok=0
  docker logs --tail 100 ongiini-eval-vllm12b > "$EXP/c1alone_vllm.log" 2>&1
  docker rm -f ongiini-eval-vllm12b >/dev/null 2>&1
  [ "$ok" = 1 ] || { log "generation had errors"; notify "A1 (C1 allein): Generierung mit Fehlern — abgebrochen."; exit 1; }
  crun ongiini-eval-c1alone-score python3 scripts/compat_suite.py score --labels base "$TAG" 2>&1 | grep -vE "HTTP Request|Warning" | tail -3
  crun ongiini-eval-c1alone-tscore python3 scripts/tool_suite.py score --labels base "$TAG" > /dev/null 2>&1
fi
# retention suite (HF, GPU) incl. the English judge
if [ ! -f "$EXP/retention/gemma-4-12b-$TAG.json" ]; then
  docker run --rm --name "ongiini-ret-$TAG" --gpus all --ipc host --shm-size 16g --user 1000:1000 -e HOME=/tmp -e HF_HOME=/tmp/hf \
    -e USER=nexus -e LOGNAME=nexus -e PYTHONUNBUFFERED=1 -e ONGIINI_EXP_DIR="$EXP" \
    -e OPENROUTER_API_KEY="$(grep '^OPENROUTER_API_KEY=' .env | cut -d= -f2-)" \
    -v "$HOME/models/gemma-4-12b-it-bf16:/models/gemma-4-12b-it-bf16:ro" -v "$PWD:/work" -w /work ongiini-train:latest \
    python3 scripts/retention_suite.py --model /models/gemma-4-12b-it-bf16 --adapter "$ADAPTER" --label "gemma-4-12b-$TAG" 2>&1 | tail -1
  docker run --rm --name ongiini-judge-c1 --user 1000:1000 -e HOME=/tmp -e ONGIINI_EXP_DIR="$EXP" \
    -e OPENROUTER_API_KEY="$(grep '^OPENROUTER_API_KEY=' .env | cut -d= -f2-)" -v "$PWD:/work" -w /work ongiini-train:latest \
    python3 scripts/retention_judge.py gemma-4-12b-base "gemma-4-12b-$TAG" 2>&1 | tail -1
fi
msg=$(python3 - <<PY
import json
from math import comb
r = json.load(open("data/private/compat/report_${TAG}_vs_base.json")); b, c = r["base"], r["$TAG"]; p = r["pairwise_${TAG}_vs_base"]
t = json.load(open("data/private/compat/tools_report_${TAG}_vs_base.json"))
try:
    ret = json.load(open("$EXP/retention/gemma-4-12b-$TAG.json")); j = ret.get("english_pairwise_vs_base", {})
    rs = f"Retention: GSM8K {ret.get('gsm8k')}, Anweisungen {ret.get('instructions_all')}, Tools {ret.get('tools_right_tool')}, Englisch-Richter {j.get('adapter_wins')}:{j.get('base_wins')}."
except Exception as e:
    rs = f"Retention fehlt ({e!r})."
print(f"IFEval {b['ifeval_prompt_strict']} -> {c['ifeval_prompt_strict']}, GSM8K {b['gsm_acc']} -> {c['gsm_acc']}, offen (Basis:C1) {p['open'].get('base')}:{p['open'].get('cand')}, "
      f"Tool-Suite {t['base']['all']} -> {t['$TAG']['all']} (Doku-Tool {t['$TAG'].get('lookup_ongiini_docs')}), OW-Antworten {c['owchat_answers_in_oshiwambo']}. {rs}")
PY
)
log "A1 result: $msg"
notify "A1, CPT-Adapter C1 allein (ohne SFT) vs Gemma 4: $msg"
