#!/usr/bin/env bash
# Goal pipeline (2026-10-07): find a Gemma 4 12B adapter with Ndonga chrF++ >= 42
# (dev and blind) AND no catastrophic forgetting (scripts/check_stage.py goal).
# Runs on the Spark without the laptop; stops at the first candidate that passes.
# Evidence: B50kG (trl chat format, r16 attention) kept every general ability
# but reached 39.5; the r64 attention+MLP adapters in the rendered format reached
# 45.8-46.2 but broke outside the training template (any LoRA scale).
# Round 4 (2026-10-08): goal chrF > 50 while keeping Gemma 4's abilities. T2 reached
# 42.1 / 44.2 with every ability intact (compat suite + retention). T4 = data quality:
# back-translations selected by adapter-A loss (human-data model, scripts/score_bt_pairs.py),
# tagged BT, human pairs (LAC/ASSAR + Viljoen dictionary examples) x3, dictionary hints in
# 20 % of prompts, dictionary vocabulary tasks, replay v1-v4 (~15 %); then T4h, a short
# final stage on human translations only (+ replay). Each candidate: benchmark,
# retention, goal check and the compatibility suite (base vs candidate, via vLLM 0.31).
#   setsid nohup bash deploy/train/pipeline_goal4.sh >> data/private/experiments/pipeline_goal4.log 2>&1 < /dev/null &
# TINY=1 runs it end-to-end with the tiny model.
set -u -o pipefail
cd "${ONGIINI_ROOT:-$HOME/dev/Ongiini}"
IMG=ongiini-train:latest
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
  CPT_ARGS="${CPT_ARGS:-}"  # e.g. "--max-steps 650" for half an epoch (2026-10-04: 58.7 s/step, full epoch 21 h)
  PRE_CPT_ARGS="--max-steps 20 --save-steps 1000"
fi
LABELS="gemma-4-12b-A gemma-4-12b-B10k gemma-4-12b-B50k gemma-4-12b-B50kG gemma-4-12b-B10kR gemma-4-12b-C1-B10kR gemma-4-12b-B10kR-r64 gemma-4-12b-C2r gemma-4-12b-B10kR-r64rep gemma-4-12b-G64t gemma-4-12b-G64trep gemma-4-12b-G16t2 gemma-4-12b-T1 gemma-4-12b-T2 gemma-4-12b-T3 gemma-4-12b-T4 gemma-4-12b-T4h"
mkdir -p "$EXP/lora" "$EXP/gpu" "$EXP/retention" "$LORA"
log() { echo "$(date '+%F %T') $*"; }
notify() { if [ "$TINY" = 1 ]; then echo "notify (tiny, not sent): $*"; else bash deploy/train/notify.sh "$*" | tail -1; fi; }
fail() { log "$* — stopping"; notify "FEHLER, Ziel-Pipeline gestoppt: $*"; exit 1; }
run() {  # run <name> <gpu:yes|no> <cmd...>
  local name=$1 gpu=$2; shift 2
  local g=(); [ "$gpu" = yes ] && g=(--gpus all)
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
  run "ongiini-train-${tag//\//-}" yes python3 scripts/train_lora.py --model "$MODEL" --bf16-base \
    --train "$file" --val "$D/sft_A_parallel_ndo_val.jsonl" --out "$LORA/${tag}_12b" \
    --epochs "$epochs" ${SFT_BATCH:---batch 8 --accum 2} --chat-format "${SFT_FORMAT:-rendered}" --limit "$SFT_LIMIT" "$@"
}
retention() {  # retention <label> [adapter dir]
  local label=$1 adapter=${2:-}
  [ -f "$EXP/retention/$label.json" ] && return 0
  log "retention $label"
  local a=(); [ -n "$adapter" ] && a=(--adapter "$adapter")
  run "ongiini-ret-$label" yes python3 scripts/retention_suite.py --model "$MODEL" "${a[@]}" --label "$label" $RET_QUICK \
    2>&1 | grep -v "^Loading" | tail -2
  [ -f "$EXP/retention/$label.json" ] || fail "retention $label wrote no result"
  if [ -n "$adapter" ]; then
    run ongiini-judge no python3 scripts/retention_judge.py gemma-4-12b-base "$label" $JUDGE_LIMIT 2>&1 | tail -1
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
others = ", ".join(f"{l.replace('gemma-4-12b-', '')} {nd(l)}" for l in S if l != label and nd(l) is not None)
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
    --adapter "$out" --label "gemma-4-12b-$tag" --out "$EXP/lora" --batch 16 $EVAL_LIMIT 2>&1 | grep -v "^Loading" | tail -1 \
    || fail "benchmark $tag"
  run ongiini-score no python3 scripts/score_lora_runs.py --base gemma-4-12b-base $LABELS 2>&1 | tail -10
  check bench "gemma-4-12b-$tag"
  retention "gemma-4-12b-$tag" "$out"
  notify "$tag fertig: $(summary "gemma-4-12b-$tag" 2>&1)"
}
sft_stage() {  # sft_stage <tag> <train file> <epochs> <extra train_lora args...>
  local tag=$1 file=$2 epochs=$3; shift 3
  local out="$LORA/${tag}_12b"
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

log "goal pipeline 4 start (TINY=$TINY)"
if [ "$TINY" != 1 ]; then
  others=$(docker ps --format '{{.Names}}' | grep -E '^ongiini-(train|eval|ret)-' | tr '\n' ' ')
  [ -z "$others" ] || fail "other GPU jobs still running: $others"
  avail=$(free -g | awk '/^Speicher:|^Mem:/{print $7}')
  [ "$avail" -ge 50 ] || fail "only ${avail} GB memory available (training needs ~50)"
  ( bash deploy/train/watchdog.sh $$ "$EXP/pipeline_goal4.log" ) &
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
[ -s "$REPLAY2" ] || fail "replay v2 missing: $REPLAY2"
preflight_mem() {  # preflight_mem <tag> <file> <train_lora args...>: 20 steps on the 320 longest examples
  local tag=$1 file=$2; shift 2
  [ -f "$LORA/${tag}_12b/run.json" ] && return 0
  log "preflight memory $tag"
  local P=$EXP/preflight PL=$LORA/preflight
  rm -rf "$P" "$PL"; mkdir -p "$P" "$PL"
  python3 -c "
import json
rows = [l for l in open('$file') if l.strip()]
rows.sort(key=lambda l: -sum(len(m['content']) for m in json.loads(l)['messages']))
open('$P/longest.jsonl', 'w').writelines(rows[:320])"
  memwatch "ongiini-train-preflight-mem" "$P/mem" &
  sft preflight/mem "$P/longest.jsonl" 1 --limit 320 "$@" 2>&1 | tr '\r' '\n' | grep -E 'DIVERGED|Error|Traceback' | cut -c1-200 || true
  [ -f "$PL/mem_12b/run.json" ] || fail "preflight memory $tag"
  local mem; mem=$(cat "$P/mem" 2>/dev/null || echo 0)
  log "preflight memory $tag: min available ${mem} GB"
  if [ "$TINY" != 1 ] && [ "$mem" -lt 25 ]; then fail "$tag needs too much memory: min ${mem} GB available (need >= 25)"; fi
  rm -rf "$P" "$PL"
}
goal_check() {  # goal_check <tag>: report against the round-1 criteria (no stop: round 4 aims at > 50)
  local out
  if out=$(ONGIINI_EXP_DIR="$EXP" python3 scripts/check_stage.py goal "gemma-4-12b-$1"); then
    log "round-1 goal criteria met by $1"
    notify "$1: alle Kriterien erfüllt (chrF >= 42, kein Vergessen). $(summary "gemma-4-12b-$1" 2>&1)"
  else
    log "round-1 goal criteria not met by $1: $out"
    notify "$1 verfehlt Kriterien: ${out#*FAILED: }. $(summary "gemma-4-12b-$1" 2>&1)"
  fi
}

replay_gen() {  # replay_gen <out> <seed> <exclude files...>
  local out=$1 seed=$2; shift 2
  [ -s "$out" ] && return 0
  log "replay $(basename "$(dirname "$out")"): base 12B answers new Dolly/GSM8K prompts + tool calls"
  run "ongiini-eval-replay-$seed" yes python3 scripts/build_replay_set.py --out "$out" --backend hf --tokenizer "$MODEL" \
    --exclude "$@" --seed "$seed" $REPLAY3_ARGS 2>&1 | grep --line-buffered -E '^prompts|^\{"kept|Error|Traceback' || true
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
  "${C[@]}" python3 scripts/compat_suite.py generate --model "$(echo "$tag" | tr 'A-Z' 'a-z')" --label "$tag" --concurrency 32 2>&1 \
    | grep --line-buffered -vE "HTTP Request|Warning" | tail -3
  docker rm -f ongiini-eval-vllm12b >/dev/null 2>&1
  "${C[@]}" python3 scripts/compat_suite.py score --labels base "$tag" 2>&1 | grep -vE "HTTP Request|Warning" | tail -40
  [ -f "data/private/compat/report_${tag}_vs_base.json" ] || fail "compat report $tag missing"
  notify "Kompatibilität $tag vs Basis: $(python3 -c "
import json; r=json.load(open('data/private/compat/report_${tag}_vs_base.json')); b, t = r['base'], r['$tag']; p = r['pairwise_${tag}_vs_base']
print(f\"GSM8K {b['gsm_acc']}/{t['gsm_acc']}, IFEval {b['ifeval_prompt_strict']}/{t['ifeval_prompt_strict']}, Tools {b['ongiini_tool_decision']}/{t['ongiini_tool_decision']}, Bilder {b['images_correct']}/{t['images_correct']}, unsicher {b['safety_unsafe']}/{t['safety_unsafe']}, OW-Antworten {b['owchat_answers_in_oshiwambo']}/{t['owchat_answers_in_oshiwambo']}; Vergleich offen {p['open']}, OW-Chat {p['owchat']}\")")"
}

notify "Ziel-Pipeline 4 startet: Datenqualität (gefilterte Rückübersetzungen, menschliche Daten x3, Wörterbuch) -> T4, danach T4h (Feinschliff auf menschlichen Übersetzungen). Je mit Kompatibilitätstest."
# 1. quality scores for the 200k back-translations (adapter A: human pairs only)
SC="$D/bt_scores_A.jsonl"
if [ ! -s "$SC" ] || [ "$TINY" = 1 ]; then
  log "score back-translations with adapter A"
  run ongiini-eval-btscore yes python3 scripts/score_bt_pairs.py --model "$MODEL" --adapter "$SCORE_ADAPTER" \
    --pairs "$D/bt_ndo_dict.jsonl" --out "$SC" $SCORE_ARGS 2>&1 | grep --line-buffered -E '^scored [0-9]*00/|Error|Traceback' || true
  [ -s "$SC" ] || fail "BT scores missing"
fi
# 2. more replay so general tasks stay ~15 % of the bigger mix
replay_gen "$REPLAY3" 37 "$REPLAY" "$REPLAY2"
replay_gen "$REPLAY4" 41 "$REPLAY" "$REPLAY2" "$REPLAY3"
# 3. T4
T4F="$D/sft_T4_train.jsonl"
[ -s "$T4F" ] || python3 scripts/build_sft_t4.py --out "$T4F" $T4_ARGS --replay "$REPLAY" "$REPLAY2" "$REPLAY3" "$REPLAY4" \
  || fail "build T4"
preflight_sft
SFT_FORMAT=rendered; SFT_BATCH="--batch 8 --accum 2 --group-by-length"
extra=(--targets all --rank 32 --alpha 32 --lr 2e-4 --max-len 640)
[ "$TINY" = 1 ] && SFT_BATCH="--batch 8 --accum 2"
preflight_mem T4 "$T4F" "${extra[@]}"
if sft_stage T4 "$T4F" 1 "${extra[@]}"; then
  compat_stage T4
  goal_check T4 || true
  # 4. T4h: short final stage on human translations only (+ a little replay), lower LR
  T4H="$D/sft_T4h_train.jsonl"
  [ -s "$T4H" ] || python3 scripts/build_sft_t4.py --out "$T4H" --bt-n 0 --vocab-n 0 --human-repeat 1 \
    --replay "$REPLAY" || fail "build T4h"
  sft_stage T4h "$T4H" 1 --init-adapter "$LORA/T4_12b" --lr 5e-5 --max-len 640 && { compat_stage T4h; goal_check T4h || true; }
fi
log "pipeline done"
notify "Ziel-Pipeline 4 fertig. Ergebnisse im Register."
