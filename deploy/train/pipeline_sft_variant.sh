#!/usr/bin/env bash
# SFT variant pipeline (2026-10-11/12): one parametrised pipeline for every "T4-recipe" experiment,
# on the 12B or (since the size probe H31 − H12 = +4.9 chrF) the 31B. From the environment:
#   VARIANT_TAG        adapter/label tag, e.g. T4b1 (label gemma-4-<model>-<TAG>)
#   VARIANT_EXTRA      train_lora.py args, e.g. "--targets all --rank 32 --alpha 32 --lr 1e-4 --max-len 1280"
#   VARIANT_INIT       optional: adapter dir to continue from (--init-adapter); TINY uses a stand-in
#   VARIANT_EPOCHS     default 1
#   VARIANT_BUILD_ARGS optional: build_sft_t4.py args (e.g. "--bt-n 60000 --vocab-n 3000") to build
#                      $OUTD/sft_<TAG>_train.jsonl with replay v1-v4 + tool replay; default = the T4 mix
#   VARIANT_MODEL_DIR  optional: a different base model directory (variant M: Gemma 4 with the CPT
#                      adapter merged in, scripts/merge_adapter.py); used for training, HF evals
#                      and the eval vLLM; comparisons stay against gemma-4-12b-base
#   VARIANT_MODEL      12b (default) or 31b: the 31B trains and is evaluated in 4-bit, micro-batch
#                      2 x accum 8 (preflight 2026-10-11: 66 GB free while training), results in
#                      experiments/r31; compat/tool/decoding stages need a 31B eval vLLM and are
#                      skipped for now (benchmark + retention run).
# Stages: memory preflight, SFT (retry on divergence), benchmark, retention, compat + tool suite
# (eval vLLM, one restart on errors), goal check, decoding test (MBR).
# Shared helpers copied from pipeline_goal7.sh (round history in the earlier pipelines' headers).
#   setsid nohup bash deploy/train/pipeline_sft_variant.sh >> data/private/experiments/pipeline_sft_variant.log 2>&1 < /dev/null &
# TINY=1 runs it end-to-end with the tiny model.
set -u -o pipefail
cd "${ONGIINI_ROOT:-$HOME/dev/Ongiini}"
IMG=ongiini-train:latest
SUF=12b
BF16_BASE="--bf16-base"; FOUR_BIT=""  # 12B defaults until use_model(); set -u killed the TINY retention without them (2026-10-10)
MODEL=/models/gemma-4-12b-it-bf16
D=data/private/corpus/osheng_v1
TINY=${TINY:-0}
# Memory preflight threshold. Calibrated 2026-10-11 against every logged run: the preflight on
# the 320 longest examples predicted the real run's minimum available memory within +-1 GB
# (T1 41/42, T2 40/40, T4b 44/44, C1-T4 44/44, T4b1 44/44, C1 34/33), so the old 25 GB rule
# (brake 12 GB + 13 GB guess) was over-conservative and cost the first H31 run (min 17 GB).
PREFLIGHT_MIN_GB=${PREFLIGHT_MIN_GB:-15}
LOAD_MIN_GB=${LOAD_MIN_GB:-14}  # the loading transient only has to stay clear of the 12 GB brake
if [ "$TINY" = 1 ]; then
  MODEL_DIR="$PWD/data/private/models-tiny/gemma4-tiny"
  EXP=data/private/experiments/tinytest
  LORA=data/private/lora/tinytest
  SFT_LIMIT=32; EVAL_LIMIT=""; RET_QUICK="--quick"; JUDGE_LIMIT="--limit 1"
  REPLAY_ARGS="--dolly 8 --gsm 4 --max-new 16 --batch 4 --keep-length"
  REPLAY3_ARGS="--dolly 8 --gsm 4 --tools 4 --max-new 16 --batch 4 --keep-length"
  SCORE_ARGS="--limit 400"; T4_ARGS="--bt-n 100 --vocab-n 50"; COMPAT=0; SCORE_ADAPTER=""
  OUTD=data/private/corpus/tinytest; rm -rf "$OUTD"; mkdir -p "$OUTD"
  TOOL_ARGS="-n 12 --batch 4 --max-new 12 --keep-length"; SFT_BATCH_T4="--batch 8 --accum 2"
  CPT_ARGS="--limit-docs 300 --block 256 --max-steps 4 --save-steps 2 --batch 2 --accum 1"
  PRE_CPT_ARGS="--limit-docs 300 --block 256 --max-steps 2 --save-steps 2 --batch 2 --accum 1"
else
  MODEL_DIR="$HOME/models/gemma-4-12b-it-bf16"
  EXP=data/private/experiments
  LORA=data/private/lora
  SFT_LIMIT=0; EVAL_LIMIT=""; RET_QUICK=""; JUDGE_LIMIT=""
  REPLAY_ARGS="--dolly 1500 --gsm 500 --max-new 384 --max-tokens 640 --batch 32"
  REPLAY3_ARGS="--dolly 3500 --gsm 800 --tools 400 --max-new 384 --max-tokens 640 --batch 32"
  SCORE_ARGS=""; T4_ARGS="--bt-n 30000 --vocab-n 3000"; COMPAT=1; SCORE_ADAPTER=data/private/lora/A_parallel_ndo_12b_r16
  OUTD=$D
  TOOL_ARGS="-n 2000 --batch 16 --max-new 320 --max-tokens 1200"  # real-scale test: ~7 s per row at 384
  # tool rows are up to 1200 tokens: half the batch keeps the padded tokens per step
  # (and the 262k-vocabulary logits) where T2's 8 x 640 were
  SFT_BATCH_T4="--batch 4 --accum 4 --group-by-length"
  CPT_ARGS="${CPT_ARGS:-}"  # e.g. "--max-steps 650" for half an epoch (2026-10-04: 58.7 s/step, full epoch 21 h)
  PRE_CPT_ARGS="--max-steps 20 --save-steps 1000"
fi
LABELS="gemma-4-12b-A gemma-4-12b-B10k gemma-4-12b-B50k gemma-4-12b-B50kG gemma-4-12b-B10kR gemma-4-12b-C1-B10kR gemma-4-12b-B10kR-r64 gemma-4-12b-C2r gemma-4-12b-B10kR-r64rep gemma-4-12b-G64t gemma-4-12b-G64trep gemma-4-12b-G16t2 gemma-4-12b-T1 gemma-4-12b-T2 gemma-4-12b-T3 gemma-4-12b-T4 gemma-4-12b-T4h gemma-4-12b-T4b gemma-4-12b-T4bh gemma-4-12b-C1-T4"
mkdir -p "$EXP/lora" "$EXP/gpu" "$EXP/retention" "$LORA"
log() { echo "$(date '+%F %T') $*"; }
notify() { if [ "$TINY" = 1 ]; then echo "notify (tiny, not sent): $*"; else bash deploy/train/notify.sh "$*" | tail -1; fi; }
fail() { log "$* — stopping"; notify "FEHLER, Ziel-Pipeline gestoppt: $*"; exit 1; }
run() {  # run <name> <gpu:yes|no> <cmd...>
  local name=$1 gpu=$2; shift 2
  local g=(); [ "$gpu" = yes ] && g=(--gpus all)
  # every GPU stage gets a monitor: the watchdog counts gpu/*.csv as progress
  [ "$gpu" = yes ] && ( sleep 20; bash deploy/train/monitor.sh "$name" "$EXP/gpu/${name#ongiini-}.csv" ) > /dev/null 2>&1 &
  docker run --rm --name "$name" "${g[@]}" --ipc host --shm-size 16g --user 1000:1000 \
    -e HOME=/tmp -e HF_HOME=/tmp/hf -e PYTHONUNBUFFERED=1 -e ONGIINI_EXP_DIR="$EXP" \
    -e DIVERGENCE_LIMIT="${DIVERGENCE_LIMIT:-4}" -e DIVERGENCE_AFTER="${DIVERGENCE_AFTER:-50}" \
    -e USER=nexus -e LOGNAME=nexus -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor -e TRITON_CACHE_DIR=/tmp/triton \
    -e OPENROUTER_API_KEY="$(grep '^OPENROUTER_API_KEY=' .env | cut -d= -f2-)" \
    -v "$MODEL_DIR:$MODEL:ro" -v "$PWD:/work" -w /work "$IMG" "$@"
}
check() {
  local out
  if out=$(ONGIINI_EXP_DIR="$EXP" python3 scripts/check_stage.py "$@"); then log "$out"
  elif [ "$TINY" = 1 ]; then log "TINY (not fatal): $out"
  else fail "$out"; fi
}
memwatch() {  # memwatch <container> <out file>: lowest MemAvailable (GB) while the container runs;
  # <out file>.train = lowest once the GPU is busy (>= 50 %), i.e. during training steps. The 31B
  # preflight (2026-10-11) bottomed at 18 GB while LOADING the checkpoint (RSS 57 GB) and then
  # trained with 66 GB free: the load is a transient, the training minimum is what matters.
  local c=$1 f=$2 m=999 mt=999 a g started=0
  sleep 5
  while docker ps -q -f "name=^$c\$" | grep -q .; do
    a=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo); [ "$a" -lt "$m" ] && m=$a
    g=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [ "${g:-0}" -ge 50 ] 2>/dev/null && started=1
    if [ "$started" = 1 ] && [ "$a" -lt "$mt" ]; then mt=$a; fi
    echo "$m" > "$f"; echo "$mt" > "$f.train"; sleep 3
  done
}

sft() {  # sft <tag> <train file> <epochs> <extra train_lora args...>
  local tag=$1 file=$2 epochs=$3; shift 3
  run "ongiini-train-${tag//\//-}" yes python3 scripts/train_lora.py --model "$MODEL" $BF16_BASE \
    --train "$file" --val "$D/sft_A_parallel_ndo_val.jsonl" --out "$LORA/${tag}_${SUF}" \
    --epochs "$epochs" ${SFT_BATCH:---batch 8 --accum 2} --chat-format "${SFT_FORMAT:-rendered}" --limit "$SFT_LIMIT" "$@"
}
retention() {  # retention <label> [adapter dir]
  local label=$1 adapter=${2:-}
  [ -f "$EXP/retention/$label.json" ] && return 0
  log "retention $label"
  local a=(); [ -n "$adapter" ] && a=(--adapter "$adapter")
  run "ongiini-ret-$label" yes python3 scripts/retention_suite.py --model "$MODEL" "${a[@]}" --label "$label" $RET_QUICK $FOUR_BIT \
    2>&1 | grep -v "^Loading" | tail -2
  [ -f "$EXP/retention/$label.json" ] || fail "retention $label wrote no result"
  if [ -n "$adapter" ]; then
    run ongiini-judge no python3 scripts/retention_judge.py "gemma-4-${SUF}-base" "$label" $JUDGE_LIMIT 2>&1 | tail -1
    check retention "$label" --judged
  else
    check retention "$label"
  fi
}
summary() {  # result message: this run next to the comparison runs
  ONGIINI_EXP_DIR="$EXP" python3 - "$1" <<'PY'
import json, os, sys
exp = os.environ["ONGIINI_EXP_DIR"]
label = sys.argv[1]
S = json.load(open(f"{exp}/lora/scores.json"))
s = S[label]
def nd(l):
    return S[l]["oshindonga_dev"]["chrf++"] if l in S else None
others = ", ".join(f"{l.split('-', 3)[-1]} {nd(l)}" for l in S if l != label and nd(l) is not None)
lines = [f"Ndonga {s['oshindonga_dev']['chrf++']} (blind {s['oshindonga_blind']['chrf++']}, "
         f"+{s['oshindonga_dev']['delta_vs_base']} vs Basis). Zum Vergleich: {others}.",
         f"Kwanyama {s['oshikwanyama_dev']['chrf++']}; Drift Kwanyama-Ausgaben vs Ndonga-Referenz "
         f"{s.get('drift', {}).get('kua_request_vs_ndo_ref', '?')}."]
try:
    r = json.load(open(f"{exp}/retention/{label}.json"))
    j = r.get("english_pairwise_vs_base", {})
    lines.append(f"Allgemein: Englisch-Perplexität {r['en_perplexity']}, GSM8K {r['gsm8k']} %, "
                 f"Vokabeltest {r['vocab_en_to_ndo']} %, Claude-Vergleich Englisch {j.get('adapter_wins')}:"
                 f"{j.get('base_wins')} ({j.get('ties')} unentschieden).")
except (OSError, KeyError) as exc:
    lines.append(f"Retention fehlt ({exc!r}).")
print(" ".join(lines))
PY
}
evaluate() {  # evaluate <tag> <adapter dir>: benchmark, scores, checks, retention, message
  local tag=$1 out=$2
  log "benchmark $tag"
  run "ongiini-eval-$tag" yes python3 scripts/eval_lora_generate.py --model "$MODEL" \
    --adapter "$out" --label "gemma-4-${SUF}-$tag" --out "$EXP/lora" --batch 16 $EVAL_LIMIT $FOUR_BIT 2>&1 | grep -v "^Loading" | tail -1 \
    || fail "benchmark $tag"
  run ongiini-score no python3 scripts/score_lora_runs.py --base "gemma-4-${SUF}-base" $LABELS 2>&1 | tail -10
  check bench "gemma-4-${SUF}-$tag"
  retention "gemma-4-${SUF}-$tag" "$out"
  notify "$tag fertig: $(summary "gemma-4-${SUF}-$tag" 2>&1)"
}
sft_stage() {  # sft_stage <tag> <train file> <epochs> <extra train_lora args...>
  local tag=$1 file=$2 epochs=$3; shift 3
  local out="$LORA/${tag}_${SUF}"
  if [ ! -f "$out/run.json" ]; then
    log "train $tag ($(wc -l < "$file") examples, $epochs epochs, $*)"
    notify "Training $tag startet."
    ( sleep 30; bash deploy/train/monitor.sh "ongiini-train-$tag" "$EXP/gpu/$tag.csv" ) &
    local attempt
    for attempt in 1 2 3; do
      sft "$tag" "$file" "$epochs" "$@" 2>&1 | tr '\r' '\n' \
        | grep --line-buffered -E '^loss on|^input:|^\{"eval_loss|^resuming|DIVERGED|Error|Traceback' | cut -c1-300 \
        | tee "$EXP/train_$tag.last" || true
      [ -f "$out/run.json" ] && break
      if grep -q DIVERGED "$EXP/train_$tag.last"; then  # goal pipeline: try the next candidate
        log "training $tag diverged — next candidate"
        notify "$tag ist divergiert (Loss explodiert) — weiter mit dem nächsten Kandidaten."
        return 1
      fi
      [ "$attempt" = 3 ] && fail "training $tag (interrupted 3 times)"
      # killed from outside (production guard, brake): resume once production answers
      log "training $tag interrupted — waiting for production, then resuming (attempt $((attempt + 1)))"
      notify "Training $tag unterbrochen (z. B. Production-Schutz). Setze nach Erholung von vLLM am letzten Checkpoint fort."
      local w
      for w in $(seq 1 60); do
        [ "$TINY" = 1 ] && break
        curl -sf -m 20 -o /dev/null localhost:8124/v1/chat/completions -H 'content-type: application/json' \
          -d '{"model":"gemma-4-26b","messages":[{"role":"user","content":"Say ok."}],"max_tokens":4}' && break
        sleep 30
      done
    done
    [ -f "$out/run.json" ] || fail "training $tag"
  fi
  check train "$out"
  evaluate "$tag" "$out"
}
cpt_stage() {
  local out="$LORA/C1_cpt_12b"
  if [ ! -f "$out/run.json" ]; then
    log "CPT C1 (all Oshiwambo text, LoRA r64 attention + MLP)"
    notify "CPT startet (ganzer Oshiwambo-Korpus, ca. 21 Mio. Tokens)."
    ( sleep 30; bash deploy/train/monitor.sh ongiini-train-C1 "$EXP/gpu/C1.csv" ) &
    run ongiini-train-C1 yes python3 scripts/train_cpt.py --model "$MODEL" --out "$out" $CPT_ARGS 2>&1 \
      | stdbuf -oL tr '\r' '\n' | grep --line-buffered -E '^blocks|^articles|^trainable|^\{.loss|^\{.eval_loss|^resuming|^\{"model|Error|Traceback' \
      | cut -c1-300 || true
    [ -f "$out/run.json" ] || fail "CPT C1"
  fi
  check cpt "$out"
  notify "CPT fertig: Oshiwambo-Perplexität (zurückgehaltene Artikel) $(python3 -c "import json;r=json.load(open('$out/run.json'));print(round(r['held_out_ppl_before'] or 0,1),'->',round(r['held_out_ppl_after'],1))"), Dauer $(python3 -c "import json;print(json.load(open('$out/run.json'))['minutes'])") min."
}

preflight_sft() {  # before B10kR: SFT in the rendered format, benchmark, retention, judge
  log "preflight SFT"
  local P=$EXP/preflight PL=$LORA/preflight
  rm -rf "$P" "$PL" "$EXP/retention/preflight"*
  mkdir -p "$P" "$PL"
  sft preflight/sft-r16 "$D/sft_B10k_train.jsonl" 1 --limit 320 2>&1 | tr '\r' '\n' \
    | grep -E '^loss on|^input:|^\{"eval_loss|Error|Traceback' | cut -c1-300 || true
  [ -f "$PL/sft-r16_12b/run.json" ] || fail "preflight SFT"
  run ongiini-eval-preflight yes python3 scripts/eval_lora_generate.py --model "$MODEL" \
    --adapter "$PL/sft-r16_12b" --label preflight --out "$P" --limit 5 2>&1 | tail -1
  [ "$(cat "$P"/preflight_*.jsonl 2>/dev/null | wc -l)" = 10 ] || fail "preflight benchmark"
  run ongiini-ret-preflight yes python3 scripts/retention_suite.py --model "$MODEL" \
    --adapter "$PL/sft-r16_12b" --label preflight --quick 2>&1 | tail -1
  run ongiini-judge no python3 scripts/retention_judge.py gemma-4-12b-base preflight --limit 2 2>&1 | tail -1
  check retention preflight --judged --preflight
  rm -rf "$P" "$PL" "$EXP/retention/preflight"*
  log "preflight SFT ok"
}

preflight_cpt() {  # before CPT: memory and speed at full size, then SFT continuing the CPT adapter
  log "preflight CPT"
  local P=$EXP/preflight PL=$LORA/preflight
  rm -rf "$P" "$PL"
  mkdir -p "$P" "$PL"
  memwatch ongiini-train-pre-cpt "$P/mem_cpt" &
  local t0; t0=$(date +%s)
  run ongiini-train-pre-cpt yes python3 scripts/train_cpt.py --model "$MODEL" --out "$PL/cpt" $PRE_CPT_ARGS 2>&1 \
    | tr '\r' '\n' | grep -E '^blocks|^articles|^trainable|^\{"model|Error|Traceback' | cut -c1-300 || true
  [ -f "$PL/cpt/run.json" ] || fail "preflight CPT"
  local mem eta; mem=$(cat "$P/mem_cpt" 2>/dev/null || echo 0)
  eta=$(python3 -c "import json;r=json.load(open('$PL/cpt/run.json'));print(round(r['sec_per_step']*r['steps_per_epoch']/3600,1))")
  log "preflight CPT: min available memory ${mem} GB, $(( $(date +%s) - t0 )) s incl. data prep and load, projected CPT epoch ${eta} h"
  local memt; memt=$(cat "$P/mem_cpt.train" 2>/dev/null || echo 999); [ "$memt" = 999 ] && memt=$mem
  log "preflight CPT: min available during training steps ${memt} GB"
  if [ "$TINY" != 1 ] && { [ "$memt" -lt "$PREFLIGHT_MIN_GB" ] || [ "$mem" -lt "$LOAD_MIN_GB" ]; }; then
    fail "CPT needs too much memory: min ${memt} GB during training (need >= $PREFLIGHT_MIN_GB), ${mem} GB while loading (need >= $LOAD_MIN_GB)"; fi
  notify "CPT-Vorabtest: min. ${mem} GB Speicher frei, CPT-Dauer hochgerechnet ${eta} h."
  log "preflight SFT on top of CPT"
  memwatch ongiini-train-pre-sft "$P/mem_sft" &
  run ongiini-train-pre-sft yes python3 scripts/train_lora.py --model "$MODEL" --bf16-base \
    --train "$D/sft_B10k_train.jsonl" --val "$D/sft_A_parallel_ndo_val.jsonl" --out "$PL/sft" \
    --epochs 1 --batch 8 --accum 2 --chat-format rendered --limit 320 --init-adapter "$PL/cpt" 2>&1 \
    | tr '\r' '\n' | grep -E '^loss on|^input:|^\{"eval_loss|Error|Traceback' | cut -c1-300 || true
  [ -f "$PL/sft/run.json" ] || fail "preflight SFT on CPT adapter"
  log "preflight SFT on CPT: min available memory $(cat "$P/mem_sft" 2>/dev/null) GB"
  grep -q '"r": 64' "$PL/sft/adapter_config.json" || fail "preflight SFT adapter is not the r64 CPT adapter"
  grep -q 'gate_proj' "$PL/sft/adapter_config.json" || fail "preflight SFT adapter lacks the MLP targets"
  rm -rf "$P" "$PL"
  log "preflight CPT ok"
}

: "${VARIANT_TAG:?VARIANT_TAG missing}" "${VARIANT_EXTRA:?VARIANT_EXTRA missing}"
log "sft variant $VARIANT_TAG start (TINY=$TINY, model ${VARIANT_MODEL:-12b}): extra=[$VARIANT_EXTRA] init=[${VARIANT_INIT:-}] build=[${VARIANT_BUILD_ARGS:-}]"
if [ "$TINY" != 1 ]; then
  others=$(docker ps --format '{{.Names}}' | grep -E '^ongiini-(train|eval|ret)-' | tr '\n' ' ')
  [ -z "$others" ] || fail "other GPU jobs still running: $others"
  avail=$(free -g | awk '/^Speicher:|^Mem:/{print $7}')
  [ "$avail" -ge 50 ] || fail "only ${avail} GB memory available (training needs ~50)"
  ( bash deploy/train/watchdog.sh $$ "$EXP/pipeline_sft_variant.log" ) &
else
  # tiny run: the base model's outputs and retention, which real runs already have
  run ongiini-eval-tinybase yes python3 scripts/eval_lora_generate.py --model "$MODEL" \
    --label gemma-4-12b-base --out "$EXP/lora" $EVAL_LIMIT 2>&1 | tail -1
  retention gemma-4-12b-base
fi
REPLAY=data/private/corpus/replay_v1/replay.jsonl
REPLAY2=data/private/corpus/replay_v2/replay.jsonl
REPLAY3=data/private/corpus/replay_v3/replay.jsonl
REPLAY4=data/private/corpus/replay_v4/replay.jsonl
TOOLR=data/private/corpus/replay_tools_v1/replay.jsonl
if [ "$TINY" = 1 ]; then
  REPLAY3=$OUTD/replay_v3.jsonl REPLAY4=$OUTD/replay_v4.jsonl TOOLR=$OUTD/replay_tools.jsonl
fi
[ -s "$REPLAY2" ] || fail "replay v2 missing: $REPLAY2"
preflight_mem() {  # preflight_mem <tag> <file> <train_lora args...>: 20 steps on the 320 longest examples
  local tag=$1 file=$2; shift 2
  [ -f "$LORA/${tag}_${SUF}/run.json" ] && return 0
  [ "${SKIP_PREFLIGHT_MEM:-0}" = 1 ] && { log "preflight memory $tag skipped (SKIP_PREFLIGHT_MEM=1, numbers known)"; return 0; }
  log "preflight memory $tag"
  local P=$EXP/preflight PL=$LORA/preflight
  rm -rf "$P" "$PL"; mkdir -p "$P" "$PL"
  python3 -c "
import json
rows = [l for l in open('$file') if l.strip()]
def size(r):
    return len(r['prompt']) + len(r['completion']) if 'prompt' in r else sum(len(m['content']) for m in r['messages'])
rows.sort(key=lambda l: -size(json.loads(l)))
open('$P/longest.jsonl', 'w').writelines(rows[:320])"
  memwatch "ongiini-train-preflight-mem" "$P/mem" &
  sft preflight/mem "$P/longest.jsonl" 1 --limit 320 "$@" 2>&1 | tr '\r' '\n' | grep -E 'DIVERGED|Error|Traceback' | cut -c1-200 || true
  [ -f "$PL/mem_${SUF}/run.json" ] || fail "preflight memory $tag"
  local mem memt; mem=$(cat "$P/mem" 2>/dev/null || echo 0); memt=$(cat "$P/mem.train" 2>/dev/null || echo 999)
  [ "$memt" = 999 ] && memt=$mem
  log "preflight memory $tag: min available ${mem} GB (during training steps ${memt} GB)"
  if [ "$TINY" != 1 ] && { [ "$memt" -lt "$PREFLIGHT_MIN_GB" ] || [ "$mem" -lt "$LOAD_MIN_GB" ]; }; then
    fail "$tag needs too much memory: min ${memt} GB during training (need >= $PREFLIGHT_MIN_GB), ${mem} GB while loading (need >= $LOAD_MIN_GB)"; fi
  rm -rf "$P" "$PL"
}
goal_check() {  # goal_check <tag>: report against the round-1 criteria (no stop: round 4 aims at > 50)
  local out
  if out=$(ONGIINI_EXP_DIR="$EXP" python3 scripts/check_stage.py goal "gemma-4-${SUF}-$1"); then
    log "round-1 goal criteria met by $1"
    notify "$1: alle Kriterien erfüllt (chrF >= 42, kein Vergessen). $(summary "gemma-4-${SUF}-$1" 2>&1)"
  else
    log "round-1 goal criteria not met by $1: $out"
    notify "$1 verfehlt Kriterien: ${out#*FAILED: }. $(summary "gemma-4-${SUF}-$1" 2>&1)"
  fi
}

replay_gen() {  # replay_gen <out> <seed> <exclude files...>
  local out=$1 seed=$2; shift 2
  [ -s "$out" ] && return 0
  log "replay $(basename "$(dirname "$out")"): base 12B answers new Dolly/GSM8K prompts + tool calls"
  run "ongiini-eval-replay-$seed" yes python3 scripts/build_replay_set.py --out "$out" --backend hf --tokenizer "$MODEL" \
    --exclude "$@" --seed "$seed" $REPLAY3_ARGS 2>&1 | grep --line-buffered -E '^prompts|^answered [0-9]*000/|^\{"kept|Error|Traceback' || true
  [ -s "$out" ] || fail "replay $out missing"
}
compat_stage() {  # compat_stage <tag>: compatibility suite base vs candidate through the eval vLLM
  [ "$COMPAT" = 1 ] || { log "compat suite skipped (tiny)"; return 0; }
  local tag=$1 lab=$1 rep="$EXP/../compat/report_$1_vs_base.json"
  [ -f "data/private/compat/report_${tag}_vs_base.json" ] && return 0
  log "compat suite $tag"
  EVAL_GPU_MEM_UTIL=0.45 EVAL_MAX_SEQS=32 bash deploy/eval/serve_12b_lora.sh "$(echo "$tag" | tr 'A-Z' 'a-z')=$LORA/${tag}_12b" \
    || fail "eval vLLM for $tag"
  local C=(docker run --rm --name "ongiini-eval-compat-$tag" --network host --user 1000:1000 -e HOME=/tmp -e USER=nexus -e LOGNAME=nexus
           -e TORCHINDUCTOR_CACHE_DIR=/tmp/ti -e PYTHONUNBUFFERED=1 -v "$PWD:/work" -w /work ongiini-evalsuite:latest)
  local gen_ok attempt lc; lc=$(echo "$tag" | tr 'A-Z' 'a-z')
  # 2026-10-09: the eval vLLM died during T4b's tool suite (255 connection errors, log lost
  # with the container); the rerun on a fresh server was clean. Both generators resume and
  # redo failed requests, so on errors: keep the dead server's log, restart, retry once.
  for attempt in 1 2; do
    gen_ok=1
    "${C[@]}" python3 scripts/compat_suite.py generate --model "$lc" --label "$tag" --concurrency 32 2>&1 \
      | grep --line-buffered -vE "HTTP Request|Warning" | tail -3 || gen_ok=0
    # tool suite: 86 scenarios x 3 samples with the production prompt and tools (base once)
    [ -s data/private/compat/tools_base.jsonl ] || "${C[@]}" python3 scripts/tool_suite.py generate --model gemma-4-12b --label base \
      --concurrency 16 2>&1 | grep -vE "HTTP Request|Warning" | tail -2 || gen_ok=0
    "${C[@]}" python3 scripts/tool_suite.py generate --model "$lc" --label "$tag" --concurrency 16 2>&1 \
      | grep -vE "HTTP Request|Warning" | tail -2 || gen_ok=0
    [ "$gen_ok" = 1 ] && break
    docker logs --tail 200 ongiini-eval-vllm12b > "$EXP/compat_vllm_${tag}_attempt$attempt.log" 2>&1
    [ "$attempt" = 2 ] && break
    log "compat/tool generation for $tag had errors — restarting the eval vLLM once (log kept)"
    EVAL_GPU_MEM_UTIL=0.45 EVAL_MAX_SEQS=32 bash deploy/eval/serve_12b_lora.sh "$lc=$LORA/${tag}_12b" \
      || fail "eval vLLM restart for $tag"
  done
  docker rm -f ongiini-eval-vllm12b >/dev/null 2>&1
  [ "$gen_ok" = 1 ] || fail "compat generation for $tag had errors"
  "${C[@]}" python3 scripts/compat_suite.py score --labels base "$tag" 2>&1 | grep -vE "HTTP Request|Warning" | tail -40
  [ -f "data/private/compat/report_${tag}_vs_base.json" ] || fail "compat report $tag missing"
  "${C[@]}" python3 scripts/tool_suite.py score --labels base "$tag" > /dev/null 2>&1 || fail "tool suite score $tag"
  notify "Kompatibilität $tag vs Basis: $(python3 -c "
import json; r=json.load(open('data/private/compat/report_${tag}_vs_base.json')); b, t = r['base'], r['$tag']; p = r['pairwise_${tag}_vs_base']
T = json.load(open('data/private/compat/tools_report_${tag}_vs_base.json')); tb, tt, tp = T['base'], T['$tag'], T['paired']
print(f\"Tool-Test {tb['all']}/{tt['all']} % (Websuche {tb.get('web_search')}/{tt.get('web_search')}, Seite lesen {tb.get('fetch_url')}/{tt.get('fetch_url')}; schlechter {tp['cand_worse']}, besser {tp['cand_better']}, p {tp['sign_test_p']}), GSM8K {b['gsm_acc']}/{t['gsm_acc']}, IFEval {b['ifeval_prompt_strict']}/{t['ifeval_prompt_strict']}, Tools {b['ongiini_tool_decision']}/{t['ongiini_tool_decision']}, Bilder {b['images_correct']}/{t['images_correct']}, unsicher {b['safety_unsafe']}/{t['safety_unsafe']}, OW-Antworten {b['owchat_answers_in_oshiwambo']}/{t['owchat_answers_in_oshiwambo']}; Vergleich offen {p['open']}, OW-Chat {p['owchat']}\")")"
}





# restored 2026-10-13 (lost in ee9842d, so the 12B branch called an undefined function); now
# with the ow→en understanding test, as in pipeline_cpt_variant.sh
decode_test() {  # decode_test <tag>: MBR / retrieved few-shot through the eval vLLM, Ndonga dev
  [ "$COMPAT" = 1 ] || { log "decoding test skipped (tiny)"; return 0; }
  local tag=$1 lc; lc=$(echo "$1" | tr 'A-Z' 'a-z')
  local R="$EXP/inference_tricks/report_$lc.json"
  [ -f "$R" ] && return 0
  [ -f "$LORA/${tag}_12b/run.json" ] || { log "decoding test $tag: no adapter"; return 0; }
  log "decoding test $tag"
  # T4b is served next to the candidate as the ow→en comparison (once; files are reused)
  local extra=(); [ -f "$LORA/T4b_12b/adapter_config.json" ] && extra=("t4b=$LORA/T4b_12b")
  EVAL_GPU_MEM_UTIL=0.45 EVAL_MAX_SEQS=32 bash deploy/eval/serve_12b_lora.sh "$lc=$LORA/${tag}_12b" "${extra[@]}" || fail "eval vLLM for $tag"
  # client container: host network (vLLM on 127.0.0.1:8200), evalsuite image (sacrebleu, openai)
  docker run --rm --name "ongiini-eval-decode-$lc" --network host --user 1000:1000 -e HOME=/tmp -e USER=nexus -e LOGNAME=nexus \
    -e PYTHONUNBUFFERED=1 -v "$PWD:/work" -w /work ongiini-evalsuite:latest \
    python3 scripts/exp_inference_tricks.py --model "$lc" --out "$EXP/inference_tricks" --concurrency 32 2>&1 \
    | grep --line-buffered -vE "HTTP Request|Warning" | tail -6
  # understanding (2026-10-13, after the experimental ratings): human Oshiwambo refs → English
  local O=(docker run --rm --name "ongiini-eval-owen-$lc" --network host --user 1000:1000 -e HOME=/tmp -e USER=nexus -e LOGNAME=nexus
           -e PYTHONUNBUFFERED=1 -v "$PWD:/work" -w /work ongiini-evalsuite:latest python3 scripts/eval_ow_en.py)
  local owl=("$tag") m
  for m in base12b:gemma-4-12b T4b:t4b "$tag:$lc"; do
    [ "${m%%:*}" = T4b ] && { [ ${#extra[@]} -gt 0 ] || continue; owl=(T4b "$tag"); }
    [ -s "$EXP/ow_en/${m%%:*}_oshikwanyama.jsonl" ] && [ "${m%%:*}" != "$tag" ] && continue
    "${O[@]}" generate --model "${m#*:}" --label "${m%%:*}" 2>&1 | grep -vE "HTTP Request|Warning|INFO" | tail -2
  done
  docker rm -f ongiini-eval-vllm12b >/dev/null 2>&1
  local owen; owen=$("${O[@]}" score --base base12b --labels "${owl[@]}" 2>&1 | grep -vE "HTTP Request|Warning|INFO" | tail -3)
  log "ow→en (understanding): $owen"
  notify "Verstehen OW→EN $tag: $(echo "$owen" | tail -1)"
  [ -f "$R" ] || fail "decoding report $tag missing"
  notify "Decoding-Test $tag (Ndonga dev): $(python3 -c "
import json; r=json.load(open('$R')); print(', '.join(f\"{k} {v['chrf']}\" + (f\" ({v['delta_vs_greedy']:+}, CI {v['ci95']})\" if 'delta_vs_greedy' in v else '') for k, v in r.items() if isinstance(v, dict)))")"
}

use_model() {  # use_model 12b|31b: model, 4-bit flags, experiment dir, label set
  SUF=$1
  if [ "$TINY" = 1 ]; then
    MODEL_DIR="$PWD/data/private/models-tiny/gemma4-tiny"
    [ "$SUF" = 31b ] && MODEL_DIR="$PWD/data/private/models-tiny/gemma4-31b-tiny"
  else
    MODEL_DIR="$HOME/models/gemma-4-${SUF}-it-bf16"
  fi
  MODEL=/models/gemma-4-${SUF}-it-bf16
  if [ "$SUF" = 31b ]; then
    BF16_BASE=""; FOUR_BIT="--four-bit"; EXP=$EXP0/r31
    # 31B in 4-bit at batch 4 x 1280 tokens left only 17 GB (preflight 2026-10-11, need >= 25):
    # half the micro-batch, same effective batch of 16
    SFT_BATCH="--batch 2 --accum 8 --group-by-length"
    LABELS="gemma-4-31b-H31 gemma-4-31b-${VARIANT_TAG}"  # own scores.json in r31
    mkdir -p "$EXP/lora" "$EXP/retention"
    [ -e "$EXP/gpu" ] || ln -s ../gpu "$EXP/gpu"  # the watchdog watches $EXP0/gpu
  else
    BF16_BASE="--bf16-base"; FOUR_BIT=""; EXP=$EXP0
    LABELS="$LABELS0 gemma-4-12b-${VARIANT_TAG}"  # score_lora_runs rewrites scores.json: keep every 12B run
  fi
}

EXP0=$EXP LABELS0=$LABELS
notify "Variante $VARIANT_TAG (${VARIANT_MODEL:-12b}) startet: $VARIANT_EXTRA${VARIANT_INIT:+ (weiter von $(basename "$VARIANT_INIT"))}${VARIANT_BUILD_ARGS:+ (Mix: $VARIANT_BUILD_ARGS)}. Danach alle Tests."
T4F="$OUTD/sft_T4_train.jsonl"
SC="$OUTD/bt_scores_A.jsonl"
INIT=${VARIANT_INIT:-}
BUILD_ARGS=${VARIANT_BUILD_ARGS:-}
if [ "$TINY" = 1 ]; then  # tiny stand-ins: tool replay, T4 mix, an init adapter
  run ongiini-eval-toolreplay yes python3 scripts/build_tool_replay.py --out "$TOOLR" --model "$MODEL" $TOOL_ARGS 2>&1 | tail -1
  run ongiini-build no python3 scripts/build_sft_t4.py --out "$T4F" --bt-n 0 --vocab-n 50 --replay "$REPLAY" \
    --tool-replay "$TOOLR" 2>&1 | tail -1
  # small stand-ins; a Kwanyama file in the real args is kept, so its rows are exercised too
  kua=$(echo "$BUILD_ARGS" | grep -oE -- '--kua-bt [^ ]+' || true)
  [ -n "$BUILD_ARGS" ] && BUILD_ARGS="--bt-n 0 --vocab-n 50${kua:+ $kua --kua-n 20}"
  if [ -n "$INIT" ]; then
    SFT_FORMAT=rendered SFT_BATCH="--batch 8 --accum 2" sft tinyinit "$T4F" 1 --targets all --rank 32 --alpha 32 2>&1 | grep -E "Error|Traceback" || true
    [ -f "$LORA/tinyinit_12b/run.json" ] || fail "tiny init adapter"
    INIT="$LORA/tinyinit_12b"
  fi
fi
use_model "${VARIANT_MODEL:-12b}"
if [ "$TINY" != 1 ] && [ -n "${VARIANT_MODEL_DIR:-}" ]; then
  [ -f "$VARIANT_MODEL_DIR/config.json" ] || fail "VARIANT_MODEL_DIR has no config.json: $VARIANT_MODEL_DIR"
  MODEL_DIR=$(cd "$VARIANT_MODEL_DIR" && pwd); MODEL="/models/$(basename "$MODEL_DIR")"
  export EVAL_MODEL_DIR=$MODEL_DIR
  log "base model for this variant: $MODEL_DIR"
fi
# optional CPT stage first (2026-10-13, CPT on the 31B): VARIANT_CPT_ARGS = train_cpt.py args; the SFT
# then continues that adapter (as C1-T4 did). Memory preflight: 20 CPT steps at full size.
if [ -n "${VARIANT_CPT_ARGS:-}" ]; then
  CPTD="$LORA/${VARIANT_TAG}_cpt_${SUF}"
  ca=($VARIANT_CPT_ARGS $FOUR_BIT); [ "$TINY" = 1 ] && ca+=($CPT_ARGS)
  if [ ! -f "$CPTD/run.json" ]; then
    if [ "$TINY" != 1 ] && [ "${SKIP_PREFLIGHT_MEM:-0}" != 1 ]; then
      log "preflight memory CPT $VARIANT_TAG"
      rm -rf "$LORA/preflight_cpt" "$EXP/cptpre_mem"*
      memwatch "ongiini-train-${VARIANT_TAG}cptpre" "$EXP/cptpre_mem" & mwp=$!
      run "ongiini-train-${VARIANT_TAG}cptpre" yes python3 scripts/train_cpt.py --model "$MODEL" --out "$LORA/preflight_cpt" \
        "${ca[@]}" --max-steps 20 --save-steps 1000 2>&1 | tr '\r' '\n' | grep -E 'Error|Traceback' | cut -c1-200 || true
      wait "$mwp"   # only the watcher: the pipeline's watchdog also runs in the background
      cm=$(cat "$EXP/cptpre_mem" 2>/dev/null || echo 0); cmt=$(cat "$EXP/cptpre_mem.train" 2>/dev/null || echo 999)
      [ "$cmt" = 999 ] && cmt=$cm
      log "preflight memory CPT $VARIANT_TAG: min available ${cm} GB (during training steps ${cmt} GB)"
      { [ "$cmt" -lt "$PREFLIGHT_MIN_GB" ] || [ "$cm" -lt "$LOAD_MIN_GB" ]; } \
        && fail "CPT $VARIANT_TAG needs too much memory: ${cmt} GB during training (need >= $PREFLIGHT_MIN_GB), ${cm} GB loading"
      rm -rf "$LORA/preflight_cpt"
    fi
    log "CPT $VARIANT_TAG: ${ca[*]}"
    notify "CPT $VARIANT_TAG ($SUF) startet: ${ca[*]}"
    run "ongiini-train-${VARIANT_TAG}cpt" yes python3 scripts/train_cpt.py --model "$MODEL" --out "$CPTD" "${ca[@]}" 2>&1 \
      | stdbuf -oL tr '\r' '\n' | grep --line-buffered -E '^blocks|^articles|^replay blocks|^training sequence|^trainable|^kept step|^english held-out|^\{.eval_loss|^resuming|^\{"model|Error|Traceback' \
      | cut -c1-300 || true
    [ -f "$CPTD/run.json" ] || fail "CPT $VARIANT_TAG"
  fi
  check cpt "$CPTD"
  notify "CPT $VARIANT_TAG fertig: $(python3 -c "
import json, math
r = json.load(open('$CPTD/run.json'))
eb, ea = r.get('english_held_out_before'), r.get('english_held_out_after')
en = f'Englisch-Held-out {round(math.exp(eb), 2)} -> {round(math.exp(ea), 2)}' if eb is not None and ea is not None else 'kein Englisch-Held-out'
print(f\"Oshiwambo-Perplexität {round(r['held_out_ppl_before'] or 0, 1)} -> {round(r['held_out_ppl_after'], 2)}, {en}, {r['minutes']} min\")")"
  INIT=$CPTD
fi
TRAIN=$T4F
if [ -n "$BUILD_ARGS" ]; then
  TRAIN="$OUTD/sft_${VARIANT_TAG}_train.jsonl"
  [ -s "$TRAIN" ] || run ongiini-build no python3 scripts/build_sft_t4.py --out "$TRAIN" $BUILD_ARGS --bt-scores "$SC" \
    --replay "$REPLAY" "$REPLAY2" "$REPLAY3" "$REPLAY4" --tool-replay "$TOOLR" || fail "build mix for $VARIANT_TAG"
fi
[ -s "$TRAIN" ] || fail "training file missing: $TRAIN"
log "training file $TRAIN ($(wc -l < "$TRAIN") rows)"
SFT_FORMAT=rendered; [ "$SUF" = 12b ] && SFT_BATCH=$SFT_BATCH_T4
extra=($VARIANT_EXTRA)
[ -n "$INIT" ] && extra+=(--init-adapter "$INIT")
preflight_mem "$VARIANT_TAG" "$TRAIN" "${extra[@]}"
if sft_stage "$VARIANT_TAG" "$TRAIN" "${VARIANT_EPOCHS:-1}" "${extra[@]}"; then
  if [ "$SUF" = 12b ]; then
    compat_stage "$VARIANT_TAG"
    goal_check "$VARIANT_TAG" || true
    decode_test "$VARIANT_TAG"
  elif [ "$COMPAT" = 1 ]; then
    # 31B: the live-check suite on an fp8 eval vLLM (same serving as experimental; base31b answers reused)
    log "31B tests $VARIANT_TAG (compat, tools, greedy vs MBR, ow→en)"
    EVAL_MAX_SEQS=16 EXP_MODEL=31b bash deploy/experimental/serve_model.sh "$LORA/${VARIANT_TAG}_31b" || fail "31B eval vLLM for $VARIANT_TAG"
    CHECK_CONCURRENCY=16 bash deploy/experimental/check_live.sh 31b "$VARIANT_TAG" 2>&1 | grep -E "vs base31b:|ow→en:|does not serve" | cut -c1-600
    docker rm -f ongiini-eval-experimental >/dev/null 2>&1
    [ -f "data/private/compat/report_${VARIANT_TAG}_vs_base31b.json" ] || fail "31B compat report $VARIANT_TAG missing"
  else
    log "31B tests skipped (tiny)"
  fi
fi
log "pipeline done"
notify "Variante $VARIANT_TAG fertig. Ergebnisse im Register."
