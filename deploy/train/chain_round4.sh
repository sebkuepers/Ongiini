#!/usr/bin/env bash
# Round 4 on the Spark, laptop-independent and strictly sequential (no two GPU jobs
# at once — an overlap tripped the memory brake on 2026-10-08):
#   1. compat suite base vs T2 (resume; failed items are redone), score, WhatsApp
#   2. decoding experiment I1 on T2 (MBR / retrieved few-shot), WhatsApp
#   3. pipeline_goal4.sh (data quality → T4 → T4h, each with the compat suite)
#   setsid nohup bash deploy/train/chain_round4.sh > data/private/experiments/chain_round4.log 2>&1 < /dev/null &
set -u
cd "$HOME/dev/Ongiini"
log() { echo "$(date '+%F %T') $*"; }
notify() { bash deploy/train/notify.sh "$*" | tail -1; }
stop_vllm() { docker rm -f ongiini-eval-vllm12b >/dev/null 2>&1 || true; }
C=(docker run --rm --network host --user 1000:1000 -e HOME=/tmp -e HF_HOME=/tmp/hf -e USER=nexus -e LOGNAME=nexus
   -e TORCHINDUCTOR_CACHE_DIR=/tmp/ti -e PYTHONUNBUFFERED=1 -v "$PWD:/work" -w /work ongiini-evalsuite:latest)
others=$(docker ps --format '{{.Names}}' | grep -E '^ongiini-(train|eval|ret)-' | tr '\n' ' ')
[ -z "$others" ] || { log "GPU jobs running: $others — stop"; notify "Runde-4-Kette nicht gestartet: GPU-Jobs laufen ($others)."; exit 1; }

# 1. compat suite base vs T2
R=data/private/compat/report_T2_vs_base.json
if [ ! -f "$R" ]; then
  log "compat suite: serve 12B + T2"
  EVAL_GPU_MEM_UTIL=0.45 EVAL_MAX_SEQS=32 bash deploy/eval/serve_12b_lora.sh t2=data/private/lora/T2_12b || { log "vLLM failed"; exit 1; }
  ok=1
  for pair in "gemma-4-12b base" "t2 T2"; do
    set -- $pair
    log "generate $2"
    "${C[@]}" python3 scripts/compat_suite.py generate --model "$1" --label "$2" --concurrency 32 2>&1 \
      | grep --line-buffered -vE "HTTP Request|Warning" | tail -2 || ok=0
  done
  stop_vllm
  [ "$ok" = 1 ] || { log "compat generation had errors — stop"; notify "Kompatibilitätstest: Fehler bei der Generierung — Kette gestoppt."; exit 1; }
  log "score"
  "${C[@]}" python3 scripts/compat_suite.py score --labels base T2 2>&1 | grep -vE "HTTP Request|Warning" | tail -40
fi
[ -f "$R" ] && notify "Kompatibilität T2 vs Basis: $(python3 -c "
import json; r=json.load(open('$R')); b, t = r['base'], r['T2']; p = r['pairwise_T2_vs_base']
print(f\"GSM8K {b['gsm_acc']}/{t['gsm_acc']}, IFEval strict {b['ifeval_prompt_strict']}/{t['ifeval_prompt_strict']}, Tools {b['ongiini_tool_decision']}/{t['ongiini_tool_decision']}, Bilder {b['images_correct']}/{t['images_correct']}, unsicher {b['safety_unsafe']}/{t['safety_unsafe']}, OW-Antworten {b['owchat_answers_in_oshiwambo']}/{t['owchat_answers_in_oshiwambo']}; Vergleich offen {p['open']}, Ongiini {p['ongiini']}, OW-Chat {p['owchat']}\")")"

# 2. decoding experiment I1
R2=data/private/experiments/inference_tricks/report_t2.json
if [ ! -f "$R2" ]; then
  log "decoding experiment I1"
  EVAL_GPU_MEM_UTIL=0.45 EVAL_MAX_SEQS=32 bash deploy/eval/serve_12b_lora.sh t2=data/private/lora/T2_12b || { log "vLLM failed"; exit 1; }
  "${C[@]}" python3 scripts/exp_inference_tricks.py --model t2 --out data/private/experiments/inference_tricks --concurrency 32 2>&1 \
    | grep --line-buffered -vE "HTTP Request|Warning" | tail -6
  stop_vllm
fi
[ -f "$R2" ] && notify "Decoding-Test T2 (Ndonga dev): $(python3 -c "
import json; r=json.load(open('$R2')); print(', '.join(f\"{k} {v['chrf']}\" + (f\" ({v['delta_vs_greedy']:+}, CI {v['ci95']})\" if 'delta_vs_greedy' in v else '') for k, v in r.items() if isinstance(v, dict)))")"

# 3. round-4 training pipeline
log "pipeline_goal4"
bash deploy/train/pipeline_goal4.sh >> data/private/experiments/pipeline_goal4.log 2>&1
log "chain done"
