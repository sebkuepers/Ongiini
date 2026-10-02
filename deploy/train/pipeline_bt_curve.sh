#!/usr/bin/env bash
# Paper 3, experiment B: back-translation learning curve, run unattended on the
# Spark (independent of the laptop). Start detached:
#   setsid nohup bash deploy/train/pipeline_bt_curve.sh > data/private/experiments/pipeline.log 2>&1 &
# Steps: finish back-translation (200k) → wait for the running B10k training →
# benchmark B10k → build 50k / 200k sets → train + benchmark B50k → selection
# experiment B50kG (grammar) / B50kV (vocabulary) → B200k.
# After each benchmark the retention suite (scripts/retention_suite.py) checks
# general abilities against the base model, with a blind LLM judge on English
# answers (scripts/retention_judge.py).
# Every training run is logged by monitor.sh (GPU, temperature, throttling,
# production latency). Results: data/private/experiments/lora/scores.json.
set -u
cd "$HOME/dev/Ongiini"
IMG=ongiini-train:latest
MODEL_DIR="$HOME/models/gemma-4-12b-it-bf16"
MODEL=/models/gemma-4-12b-it-bf16
D=data/private/corpus/osheng_v1
EXP=data/private/experiments
mkdir -p "$EXP/lora" "$EXP/gpu" "$EXP/retention" data/private/lora
log() { echo "$(date '+%F %T') $*"; }
run() {  # run <name> <gpu:yes|no> <cmd...>
  local name=$1 gpu=$2; shift 2
  local g=(); [ "$gpu" = yes ] && g=(--gpus all)
  docker run --rm --name "$name" "${g[@]}" --ipc host --shm-size 16g --user 1000:1000 \
    -e HOME=/tmp -e HF_HOME=/tmp/hf -e PYTHONUNBUFFERED=1 \
    -e OPENROUTER_API_KEY="$(grep '^OPENROUTER_API_KEY=' .env | cut -d= -f2-)" \
    -v "$MODEL_DIR:$MODEL:ro" -v "$PWD:/work" -w /work "$IMG" "$@"
}
retention() {  # retention <label> [adapter dir]
  local label=$1 adapter=${2:-}
  [ -f "$EXP/retention/$label.json" ] && return 0
  log "retention $label"
  if [ -n "$adapter" ]; then
    run "ongiini-ret-$label" yes python3 scripts/retention_suite.py --model "$MODEL" --adapter "$adapter" --label "$label" \
      | grep -v "^Loading" | tail -2
    run ongiini-judge no python3 scripts/retention_judge.py gemma-4-12b-base "$label" | tail -1
  else
    run "ongiini-ret-$label" yes python3 scripts/retention_suite.py --model "$MODEL" --label "$label" | grep -v "^Loading" | tail -2
  fi
}

train_and_eval() {  # train_and_eval <tag> <train file> <epochs>
  local tag=$1 train=$2 epochs=$3
  log "train $tag ($(wc -l < "$train") examples, $epochs epoch(s))"
  ( sleep 30; bash deploy/train/monitor.sh "ongiini-train-$tag" "$EXP/gpu/$tag.csv" ) &
  run "ongiini-train-$tag" yes python3 scripts/train_lora.py --model "$MODEL" --bf16-base \
    --train "$train" --val "$D/sft_A_parallel_ndo_val.jsonl" --out "data/private/lora/${tag}_12b_r16" \
    --epochs "$epochs" --batch 16 --accum 1 --group-by-length || { log "training $tag FAILED"; return 1; }
  log "benchmark $tag"
  run "ongiini-eval-$tag" yes python3 scripts/eval_lora_generate.py --model "$MODEL" \
    --adapter "data/private/lora/${tag}_12b_r16" --label "gemma-4-12b-$tag" --out "$EXP/lora" --batch 16 \
    || { log "benchmark $tag FAILED"; return 1; }
  run "ongiini-score" no python3 scripts/score_lora_runs.py --base gemma-4-12b-base \
    gemma-4-12b-A gemma-4-12b-B10k gemma-4-12b-B50k gemma-4-12b-B50kG gemma-4-12b-B50kV gemma-4-12b-B200k
  retention "gemma-4-12b-$tag" "data/private/lora/${tag}_12b_r16"
}

log "pipeline start"
log "back-translation to 200k (resumes)"
run ongiini-bt no python3 scripts/backtranslate_corpus.py --n 200000 \
  --dictionary data/private/corpus/dictionary_v1/entries.jsonl --out "$D/bt_ndo_dict.jsonl" --concurrency 64 \
  2>&1 | grep -v "HTTP Request" | tail -3
log "back-translated rows: $(wc -l < "$D/bt_ndo_dict.jsonl")"

log "waiting for B10k training"
while docker ps -q -f name=^ongiini-train-B10k$ | grep -q .; do sleep 60; done
if [ ! -f "$EXP/lora/gemma-4-12b-B10k_oshikwanyama.jsonl" ]; then
  log "benchmark B10k"
  run ongiini-eval-B10k yes python3 scripts/eval_lora_generate.py --model "$MODEL" \
    --adapter data/private/lora/B10k_12b_r16 --label gemma-4-12b-B10k --out "$EXP/lora" --batch 16
fi
run ongiini-score no python3 scripts/score_lora_runs.py --base gemma-4-12b-base gemma-4-12b-A gemma-4-12b-B10k
retention gemma-4-12b-base
retention gemma-4-12b-A data/private/lora/A_parallel_ndo_12b_r16
retention gemma-4-12b-B10k data/private/lora/B10k_12b_r16

log "build 50k / 200k sets"
run ongiini-build no python3 scripts/build_sft_bt.py --n 50000 200000
train_and_eval B50k "$D/sft_B50k_train.jsonl" 1

# Selection experiment: same size as B50k, chosen by grammar (constructions the
# newspaper corpus lacks) or by vocabulary coverage — runs before B200k.
log "waiting for grammar labels"
while docker ps -q -f name=^ongiini-label$ | grep -q .; do sleep 60; done
log "labels: $(wc -l < "$D/bt_labels.jsonl")"
run ongiini-build no python3 scripts/build_sft_selected.py --strategy grammar --n 50000 | tail -20
run ongiini-build no python3 scripts/build_sft_selected.py --strategy vocab --n 50000 | tail -6
train_and_eval B50kG "$D/sft_B50kG_train.jsonl" 1
train_and_eval B50kV "$D/sft_B50kV_train.jsonl" 1

train_and_eval B200k "$D/sft_B200k_train.jsonl" 1
log "pipeline done"
