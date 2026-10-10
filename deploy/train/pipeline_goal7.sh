#!/usr/bin/env bash
# Round 7 (prepared 2026-10-09 as round 6; moved behind C2-T4 on 2026-10-10): does a bigger
# base model buy the last points to 50?
# Probe before any long run: the T4h mix (human pairs LAC + Viljoen examples, replay,
# 800 native tool rows; ~11k rows) trained from the base on Gemma 4 12B (H12) and on
# Gemma 4 31B (H31, dense, Apache-2.0; base zero-shot 20.8 vs 12B 14.8 on Ndonga dev),
# same recipe (r32 attn+MLP, alpha 32, lr 1e-4, 1 epoch, rendered). 31B trains and is
# evaluated in 4-bit (bf16 = 62.5 GB does not fit next to production); its base is
# evaluated in 4-bit too, so deltas are fair. 31B results live in experiments/r31.
# TINY also forces a divergence once to exercise train_lora's automatic retry.
# Shared helpers copied from pipeline_goal5.sh, parametrised by the model (SUF=12b|31b).
#   setsid nohup bash deploy/train/pipeline_goal7.sh >> data/private/experiments/pipeline_goal7.log 2>&1 < /dev/null &
# TINY=1 runs it end-to-end with the tiny model.
set -u -o pipefail
cd "${ONGIINI_ROOT:-$HOME/dev/Ongiini}"
IMG=ongiini-train:latest
SUF=12b
BF16_BASE="--bf16-base"; FOUR_BIT=""  # 12B defaults until use_model(); set -u killed the TINY retention without them (2026-10-10)
MODEL=/models/gemma-4-12b-it-bf16
D=data/private/corpus/osheng_v1
TINY=${TINY:-0}
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
memwatch() {  # memwatch <container> <out file>: lowest MemAvailable (GB) while the container runs
  local c=$1 f=$2 m=999 a
  sleep 5
  while docker ps -q -f "name=^$c\$" | grep -q .; do
    a=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo); [ "$a" -lt "$m" ] && m=$a
    echo "$m" > "$f"; sleep 3
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
  if [ "$TINY" != 1 ] && [ "$mem" -lt 25 ]; then fail "CPT needs too much memory: min ${mem} GB available (need >= 25)"; fi
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

log "goal pipeline 7 start (TINY=$TINY)"
if [ "$TINY" != 1 ]; then
  others=$(docker ps --format '{{.Names}}' | grep -E '^ongiini-(train|eval|ret)-' | tr '\n' ' ')
  [ -z "$others" ] || fail "other GPU jobs still running: $others"
  avail=$(free -g | awk '/^Speicher:|^Mem:/{print $7}')
  [ "$avail" -ge 50 ] || fail "only ${avail} GB memory available (training needs ~50)"
  [ -f "$HOME/models/gemma-4-31b-it-bf16/model.safetensors.index.json" ] || fail "Gemma 4 31B missing in ~/models"
  ( bash deploy/train/watchdog.sh $$ "$EXP/pipeline_goal7.log" ) &
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
  local mem; mem=$(cat "$P/mem" 2>/dev/null || echo 0)
  log "preflight memory $tag: min available ${mem} GB"
  if [ "$TINY" != 1 ] && [ "$mem" -lt 25 ]; then fail "$tag needs too much memory: min ${mem} GB available (need >= 25)"; fi
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
    LABELS="gemma-4-31b-H31"  # own scores.json in r31
    mkdir -p "$EXP/lora" "$EXP/retention"
    [ -e "$EXP/gpu" ] || ln -s ../gpu "$EXP/gpu"  # the watchdog watches $EXP0/gpu
  else
    BF16_BASE="--bf16-base"; FOUR_BIT=""; EXP=$EXP0
    LABELS="$LABELS0 gemma-4-12b-H12"  # score_lora_runs rewrites scores.json: keep every 12B run
  fi
}
EXP0=$EXP LABELS0=$LABELS

notify "Runde 7 startet: Größen-Test. Gleiche Daten (menschliche Übersetzungen + Replay + Tools) auf Gemma 4 12B (H12) und 31B (H31, 4-bit)."
T4H="$OUTD/sft_T4h_train.jsonl"
if [ "$TINY" = 1 ]; then  # tiny stand-ins
  [ -d data/private/models-tiny/gemma4-31b-tiny ] || docker run --rm --user 1000:1000 -e HOME=/tmp -e USER=nexus \
    -e LOGNAME=nexus -e CUDA_VISIBLE_DEVICES= -v "$HOME/models/gemma-4-31b-it-bf16:/models/gemma-4-31b-it-bf16:ro" \
    -v "$PWD:/work" -w /work "$IMG" python3 scripts/make_tiny_model.py /models/gemma-4-31b-it-bf16 \
    data/private/models-tiny/gemma4-31b-tiny 2>&1 | tail -1
  run ongiini-eval-toolreplay yes python3 scripts/build_tool_replay.py --out "$TOOLR" --model "$MODEL" $TOOL_ARGS 2>&1 | tail -1
  run ongiini-build no python3 scripts/build_sft_t4.py --out "$T4H" --bt-n 0 --vocab-n 0 --human-repeat 1 --replay "$REPLAY" \
    --tool-replay "$TOOLR" --tool-n 800 2>&1 | tail -1
  # the automatic retry after a divergence (train_lora.py): force a trip, expect two kept runs and a failed stage
  log "TINY: forced divergence"
  # 40 rows at batch 2 = 20 steps, so a loss gets logged (every 10 steps) and trips the guard
  DIVERGENCE_LIMIT=0 DIVERGENCE_AFTER=0 SFT_LIMIT=40 SFT_BATCH="--batch 2 --accum 1" BF16_BASE="--bf16-base" \
    sft divtest "$T4H" 1 --lr 2e-4 2>&1 | grep -E "^DIVERGED|^RETRY" | head -4
  [ ! -f "$LORA/divtest_12b/run.json" ] && ls -d "$LORA"/divtest_12b.diverged-lr0.0002 "$LORA"/divtest_12b.diverged-lr0.0001 \
    >/dev/null || fail "automatic divergence retry did not behave"
  log "TINY: forced divergence ok (retried at half lr, then failed as expected)"
fi
[ -s "$T4H" ] || fail "T4h mix missing: $T4H (round 5)"
SFT_FORMAT=rendered; SFT_BATCH=$SFT_BATCH_T4
extra=(--targets all --rank 32 --alpha 32 --lr 1e-4 --max-len 1280)
# 1. H12: the probe mix on 12B (same harness as every 12B run)
use_model 12b
sft_stage H12 "$T4H" 1 "${extra[@]}" || log "H12 failed"
# 2. H31: the same on 31B in 4-bit, with its own 4-bit base
use_model 31b
if [ "$TINY" = 1 ] || [ ! -f "$EXP/lora/gemma-4-31b-base_oshindonga.jsonl" ]; then
  log "benchmark 31B base (4-bit)"
  run ongiini-eval-31base yes python3 scripts/eval_lora_generate.py --model "$MODEL" --label gemma-4-31b-base \
    --out "$EXP/lora" --batch 16 $EVAL_LIMIT $FOUR_BIT 2>&1 | tail -1
fi
retention gemma-4-31b-base
preflight_mem H31 "$T4H" "${extra[@]}"
sft_stage H31 "$T4H" 1 "${extra[@]}" || log "H31 failed"
# 3. compare
msg=$(python3 - <<PY
import json
def get(p, l):
    try:
        s = json.load(open(p))[l]; return s["oshindonga_dev"]["chrf++"], s["oshindonga_blind"]["chrf++"]
    except (OSError, KeyError):
        return None
h12 = get("$EXP0/lora/scores.json", "gemma-4-12b-H12"); h31 = get("$EXP0/r31/lora/scores.json", "gemma-4-31b-H31")
b31 = get("$EXP0/r31/lora/scores.json", "gemma-4-31b-base") or (None, None)
print(f"H12 {h12}, H31 {h31} (dev, blind); 31B-Basis 4-bit dev {b31[0]}."
      + (f" Unterschied 31B-12B: {round(h31[0] - h12[0], 1)} Punkte (dev)." if h12 and h31 else ""))
PY
)
log "round 6 probe: $msg"
notify "Runde 7 (Größen-Test) fertig: $msg"
log "pipeline done"
