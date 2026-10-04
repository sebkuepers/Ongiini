#!/usr/bin/env bash
# Paper 3, experiment B: back-translation learning curve, run unattended on the
# Spark (independent of the laptop). Start detached:
#   setsid nohup bash deploy/train/pipeline_bt_curve.sh > data/private/experiments/pipeline.log 2>&1 < /dev/null &
#
# Safeguards (after a day lost on 2026-10-03 to errors nobody saw for hours):
# 1. Preflight: every stage runs once in miniature with the exact flags
#    (20 training steps, 5 benchmark sentences, quick retention, 2 judge calls)
#    before anything long starts — API and format errors surface in ~15 min.
# 2. Plausibility checks after every stage (scripts/check_stage.py); a failed
#    check stops the pipeline.
# 3. WhatsApp messages to the operator (notify.sh) at each stage and on any
#    failure; watchdog.sh alerts when the pipeline dies or the log stalls.
# Every stage skips work that is already done, so a restart resumes.
#
# Order: retention base, A, B10k (rerun with the fixed suite) → B50k →
# selection B50kG / B50kV → B200k → B10kR (B10k with the corrected chat format,
# measures what the trl format cost). Results: data/private/experiments/lora/scores.json,
# retention/, GPU logs gpu/<tag>.csv.
set -u -o pipefail
cd "$HOME/dev/Ongiini"
IMG=ongiini-train:latest
MODEL_DIR="$HOME/models/gemma-4-12b-it-bf16"
MODEL=/models/gemma-4-12b-it-bf16
D=data/private/corpus/osheng_v1
EXP=data/private/experiments
LOG=$EXP/pipeline.log
LABELS="gemma-4-12b-A gemma-4-12b-B10k gemma-4-12b-B50k gemma-4-12b-B50kG gemma-4-12b-B50kV gemma-4-12b-B200k gemma-4-12b-B10kR"
mkdir -p "$EXP/lora" "$EXP/gpu" "$EXP/retention" data/private/lora
log() { echo "$(date '+%F %T') $*"; }
notify() { bash deploy/train/notify.sh "$*" | tail -1; }
fail() { log "$* — stopping"; notify "FEHLER, Pipeline gestoppt: $*"; exit 1; }
run() {  # run <name> <gpu:yes|no> <cmd...>
  local name=$1 gpu=$2; shift 2
  local g=(); [ "$gpu" = yes ] && g=(--gpus all)
  docker run --rm --name "$name" "${g[@]}" --ipc host --shm-size 16g --user 1000:1000 \
    -e HOME=/tmp -e HF_HOME=/tmp/hf -e PYTHONUNBUFFERED=1 \
    -e USER=nexus -e LOGNAME=nexus -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor -e TRITON_CACHE_DIR=/tmp/triton \
    -e OPENROUTER_API_KEY="$(grep '^OPENROUTER_API_KEY=' .env | cut -d= -f2-)" \
    -v "$MODEL_DIR:$MODEL:ro" -v "$PWD:/work" -w /work "$IMG" "$@"
}
check() { local out; out=$(python3 scripts/check_stage.py "$@") || fail "$out"; log "$out"; }

BATCH="--batch 16 --accum 1 --group-by-length"
train() {  # train <tag> <train file> <epochs> <chat format> [limit]; batching from $BATCH
  local tag=$1 file=$2 epochs=$3 fmt=$4 limit=${5:-0}
  run "ongiini-train-$tag" yes python3 scripts/train_lora.py --model "$MODEL" --bf16-base \
    --train "$file" --val "$D/sft_A_parallel_ndo_val.jsonl" --out "data/private/lora/${tag}_12b_r16" \
    --epochs "$epochs" $BATCH --chat-format "$fmt" --limit "$limit"
}
retention() {  # retention <label> [adapter dir]
  local label=$1 adapter=${2:-}
  [ -f "$EXP/retention/$label.json" ] && return 0
  log "retention $label"
  local a=(); [ -n "$adapter" ] && a=(--adapter "$adapter")
  run "ongiini-ret-$label" yes python3 scripts/retention_suite.py --model "$MODEL" "${a[@]}" --label "$label" \
    2>&1 | grep -v "^Loading" | tail -2
  [ -f "$EXP/retention/$label.json" ] || fail "retention $label wrote no result"
  if [ -n "$adapter" ]; then
    run ongiini-judge no python3 scripts/retention_judge.py gemma-4-12b-base "$label" 2>&1 | tail -1
    check retention "$label" --judged
  else
    check retention "$label"
  fi
}
summary() {  # result message for WhatsApp, with the earlier runs for comparison
  python3 - "$1" <<'PY'
import json, sys
label = sys.argv[1]
S = json.load(open("data/private/experiments/lora/scores.json"))
s = S[label]
def nd(l):
    return S[l]["oshindonga_dev"]["chrf++"] if l in S else None
others = ", ".join(f"{l.replace('gemma-4-12b-', '')} {nd(l)}" for l in S if l != label and nd(l) is not None)
lines = [f"Ndonga {s['oshindonga_dev']['chrf++']} (blind {s['oshindonga_blind']['chrf++']}, "
         f"+{s['oshindonga_dev']['delta_vs_base']} vs Basis). Zum Vergleich: {others}.",
         f"Kwanyama {s['oshikwanyama_dev']['chrf++']}; Drift Kwanyama-Ausgaben vs Ndonga-Referenz "
         f"{s.get('drift', {}).get('kua_request_vs_ndo_ref', '?')}."]
try:
    r = json.load(open(f"data/private/experiments/retention/{label}.json"))
    j = r.get("english_pairwise_vs_base", {})
    lines.append(f"Allgemein: Englisch-Perplexität {r['en_perplexity']}, GSM8K {r['gsm8k']} %, "
                 f"Vokabeltest {r['vocab_en_to_ndo']} %, Claude-Vergleich Englisch {j.get('adapter_wins')}:"
                 f"{j.get('base_wins')} ({j.get('ties')} unentschieden).")
except (OSError, KeyError) as exc:
    lines.append(f"Retention fehlt ({exc!r}).")
print(" ".join(lines))
PY
}

stage() {  # stage <tag> <train file> <epochs> <chat format>
  local tag=$1 file=$2 epochs=$3 fmt=$4 out="data/private/lora/${1}_12b_r16"
  if [ ! -f "$out/run.json" ]; then
    log "train $tag ($(wc -l < "$file") examples, $epochs epoch(s), format $fmt)"
    notify "Training $tag startet ($(wc -l < "$file") Beispiele)."
    ( sleep 30; bash deploy/train/monitor.sh "ongiini-train-$tag" "$EXP/gpu/$tag.csv" ) &
    train "$tag" "$file" "$epochs" "$fmt" || fail "training $tag"
  fi
  check train "$out"
  log "benchmark $tag"
  run "ongiini-eval-$tag" yes python3 scripts/eval_lora_generate.py --model "$MODEL" \
    --adapter "$out" --label "gemma-4-12b-$tag" --out "$EXP/lora" --batch 16 2>&1 | grep -v "^Loading" | tail -1 \
    || fail "benchmark $tag"
  run ongiini-score no python3 scripts/score_lora_runs.py --base gemma-4-12b-base $LABELS | tail -8
  check bench "gemma-4-12b-$tag"
  retention "gemma-4-12b-$tag" "$out"
  notify "$tag fertig: $(summary "gemma-4-12b-$tag")"
}

preflight() {
  log "preflight"
  local P=$EXP/preflight
  rm -rf "$P" data/private/lora/preflight_* "$EXP/retention/preflight"*
  for fmt in trl rendered; do
    log "preflight training ($fmt)"
    train "preflight_$fmt" "$D/sft_B50k_train.jsonl" 1 "$fmt" 320 2>&1 | tr '\r' '\n' \
      | grep -E '^loss on|^input:|^\{"eval_loss|Error' | cut -c1-400
    [ -f "data/private/lora/preflight_${fmt}_12b_r16/run.json" ] || fail "preflight training ($fmt)"
  done
  run ongiini-eval-preflight yes python3 scripts/eval_lora_generate.py --model "$MODEL" \
    --adapter data/private/lora/preflight_trl_12b_r16 --label preflight --out "$P" --limit 5 2>&1 | tail -1
  [ "$(cat "$P"/preflight_*.jsonl 2>/dev/null | wc -l)" = 10 ] || fail "preflight benchmark"
  run ongiini-ret-preflight yes python3 scripts/retention_suite.py --model "$MODEL" \
    --adapter data/private/lora/preflight_trl_12b_r16 --label preflight --quick 2>&1 | tail -1
  run ongiini-judge no python3 scripts/retention_judge.py gemma-4-12b-base preflight --limit 2 2>&1 | tail -1
  check retention preflight --judged --preflight
  rm -rf "$P" data/private/lora/preflight_* "$EXP/retention/preflight"*
  log "preflight ok"
}

log "pipeline start"
# One GPU job at a time next to production: a leftover training container
# plus a new one exhausted memory on 2026-10-03 and froze the Spark (and the
# WhatsApp webhook) for half an hour.
others=$(docker ps --format '{{.Names}}' | grep -E '^ongiini-(train|eval|ret)-' | tr '\n' ' ')
[ -z "$others" ] || fail "other GPU jobs still running: $others"
avail=$(free -g | awk '/^Speicher:|^Mem:/{print $7}')
[ "$avail" -ge 50 ] || fail "only ${avail} GB memory available (training needs ~50)"
( bash deploy/train/watchdog.sh $$ "$LOG" ) &
notify "Pipeline startet: Vorabtest aller Stufen (~20 min), danach B50k."
preflight
notify "Vorabtest bestanden. Lange Läufe starten."

# Retention v1 (before 2026-10-03 evening) had no <bos> for the perplexity
# and too few tokens for the instruction checks: archived, rerun.
if [ -f "$EXP/retention/gemma-4-12b-base.json" ] && [ ! -d "$EXP/retention/v1_nobos" ]; then
  mkdir -p "$EXP/retention/v1_nobos" && mv "$EXP/retention/"gemma-4-12b-*.json "$EXP/retention/v1_nobos/"
fi
retention gemma-4-12b-base
retention gemma-4-12b-A data/private/lora/A_parallel_ndo_12b_r16
retention gemma-4-12b-B10k data/private/lora/B10k_12b_r16

stage B50k "$D/sft_B50k_train.jsonl" 1 trl

# Selection experiment: same size as B50k, chosen by grammar (constructions the
# newspaper corpus lacks) or by vocabulary coverage.
[ -f "$D/sft_B50kG_train.jsonl" ] || run ongiini-build no python3 scripts/build_sft_selected.py --strategy grammar --n 50000 | tail -20
[ -f "$D/sft_B50kV_train.jsonl" ] || run ongiini-build no python3 scripts/build_sft_selected.py --strategy vocab --n 50000 | tail -6
stage B50kG "$D/sft_B50kG_train.jsonl" 1 trl
stage B50kV "$D/sft_B50kV_train.jsonl" 1 trl

stage B200k "$D/sft_B200k_train.jsonl" 1 trl

# B10k again with the corrected chat format; everything else as B10k
# (2 epochs, batch 8 x 2, no length grouping) so only the format differs.
BATCH="--batch 8 --accum 2"
stage B10kR "$D/sft_B10k_train.jsonl" 2 rendered

log "pipeline done"
notify "Pipeline komplett fertig (B50k, B50kG, B50kV, B200k, B10kR)."
