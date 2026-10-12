#!/usr/bin/env bash
# Compatibility, tool and decoding tests for a 31B adapter (2026-10-12). The 31B in bf16 is
# 62.5 GB, so the eval vLLM loads it with fp8 quantisation at load time (~32 GB) and serves the
# adapter next to it; the comparison baseline is the 31B instruct model itself ("base31",
# generated once). Runs as a queue job when the GPU is free (never next to training):
#   bash deploy/eval/compat_31b.sh T4b data/private/lora/T4b_31b
# Outputs: data/private/compat/out_base31.jsonl, out_<TAG>31.jsonl, report_<TAG>31_vs_base31.json,
# tools_*.jsonl, tools_report_*.json; decoding in experiments/r31/inference_tricks/.
set -u
cd "${ONGIINI_ROOT:-$HOME/dev/Ongiini}"
TAG=$1; ADAPTER=$2; LC=$(echo "$TAG" | tr 'A-Z' 'a-z')31; LABEL="${TAG}31"
EXP=data/private/experiments
log() { echo "$(date '+%F %T') $*"; }
notify() { bash deploy/train/notify.sh "$*" | tail -1; }
others=$(docker ps --format '{{.Names}}' | grep -E '^ongiini-(train|eval|ret)-' | tr '\n' ' ')
[ -z "$others" ] || { log "GPU jobs running: $others — stop"; exit 1; }
[ -f "$ADAPTER/adapter_config.json" ] || { log "no adapter in $ADAPTER"; exit 1; }
crun() { local n=$1; shift
  docker run --rm --name "$n" --network host --user 1000:1000 -e HOME=/tmp -e USER=nexus -e LOGNAME=nexus \
    -e PYTHONUNBUFFERED=1 -v "$PWD:/work" -w /work ongiini-evalsuite:latest "$@"; }
serve() {
  EVAL_MODEL_DIR="$HOME/models/gemma-4-31b-it-bf16" EVAL_SERVED_NAME=gemma-4-31b EVAL_QUANT=fp8 \
  EVAL_GPU_MEM_UTIL=${EVAL_GPU_MEM_UTIL:-0.5} EVAL_MAX_SEQS=${EVAL_MAX_SEQS:-16} \
    bash deploy/eval/serve_12b_lora.sh "$LC=$ADAPTER"
}
log "serve 31B (fp8) + $ADAPTER"
if ! serve; then
  docker logs --tail 80 ongiini-eval-vllm12b > "$EXP/compat31_vllm_fail.log" 2>&1
  docker rm -f ongiini-eval-vllm12b >/dev/null 2>&1
  notify "31B-Kompatibilitätstest: der eval-vLLM (fp8) startete nicht — Log in compat31_vllm_fail.log."
  exit 1
fi
log "serving; memory: $(free -g | awk '/^Speicher:|^Mem:/{print $7}') GB available"
ok=1
for attempt in 1 2; do
  ok=1
  [ -s data/private/compat/out_base31.jsonl ] && [ "$(grep -c . data/private/compat/out_base31.jsonl)" -ge 1318 ] \
    || crun ongiini-eval-compat31-base python3 scripts/compat_suite.py generate --model gemma-4-31b --label base31 --concurrency 16 2>&1 \
       | grep -vE "HTTP Request|Warning" | tail -2 || ok=0
  crun "ongiini-eval-compat31-$LC" python3 scripts/compat_suite.py generate --model "$LC" --label "$LABEL" --concurrency 16 2>&1 \
    | grep -vE "HTTP Request|Warning" | tail -2 || ok=0
  [ -s data/private/compat/tools_base31.jsonl ] \
    || crun ongiini-eval-tools31-base python3 scripts/tool_suite.py generate --model gemma-4-31b --label base31 --concurrency 8 2>&1 \
       | grep -vE "HTTP Request|Warning" | tail -2 || ok=0
  crun "ongiini-eval-tools31-$LC" python3 scripts/tool_suite.py generate --model "$LC" --label "$LABEL" --concurrency 8 2>&1 \
    | grep -vE "HTTP Request|Warning" | tail -2 || ok=0
  [ "$ok" = 1 ] && break
  docker logs --tail 200 ongiini-eval-vllm12b > "$EXP/compat31_vllm_${LABEL}_attempt$attempt.log" 2>&1
  [ "$attempt" = 2 ] && break
  log "generation had errors — restarting the eval vLLM once"
  serve || { ok=0; break; }
done
if [ "$ok" = 1 ]; then
  [ -f "$EXP/r31/inference_tricks/report_$LC.json" ] \
    || crun "ongiini-eval-decode-$LC" python3 scripts/exp_inference_tricks.py --model "$LC" --out "$EXP/r31/inference_tricks" \
         --concurrency 16 2>&1 | grep -vE "HTTP Request|Warning" | tail -5
fi
docker logs --tail 100 ongiini-eval-vllm12b > "$EXP/compat31_vllm_${LABEL}.log" 2>&1
docker rm -f ongiini-eval-vllm12b >/dev/null 2>&1
[ "$ok" = 1 ] || { notify "31B-Kompatibilitätstest $LABEL: Generierung mit Fehlern — abgebrochen."; exit 1; }
crun "ongiini-eval-score31-$LC" python3 scripts/compat_suite.py score --labels base31 "$LABEL" 2>&1 | grep -vE "HTTP Request|Warning" | tail -3
crun "ongiini-eval-tscore31-$LC" python3 scripts/tool_suite.py score --labels base31 "$LABEL" > /dev/null 2>&1
msg=$(python3 - <<PY
import json
r = json.load(open("data/private/compat/report_${LABEL}_vs_base31.json")); b, t = r["base31"], r["$LABEL"]; p = r["pairwise_${LABEL}_vs_base31"]
T = json.load(open("data/private/compat/tools_report_${LABEL}_vs_base31.json")); tb, tt, tp = T["base31"], T["$LABEL"], T["paired"]
try:
    d = json.load(open("$EXP/r31/inference_tricks/report_$LC.json")); dec = f"Decoding: greedy {d['greedy']['chrf']}, MBR {d['mbr']['chrf']} ({d['mbr']['delta_vs_greedy']:+}, CI {d['mbr']['ci95']})."
except Exception:
    dec = "Decoding fehlt."
print(f"Tool-Test {tb['all']}/{tt['all']} % (p {tp['sign_test_p']}), GSM8K {b['gsm_acc']}/{t['gsm_acc']}, IFEval {b['ifeval_prompt_strict']}/{t['ifeval_prompt_strict']}, "
      f"Tools {b['ongiini_tool_decision']}/{t['ongiini_tool_decision']}, Bilder {b['images_correct']}/{t['images_correct']}, unsicher {b['safety_unsafe']}/{t['safety_unsafe']}, "
      f"OW-Antworten {b['owchat_answers_in_oshiwambo']}/{t['owchat_answers_in_oshiwambo']}; offen {p['open']}, OW-Chat {p['owchat']}. {dec}")
PY
)
log "$LABEL vs 31B base: $msg"
notify "Kompatibilität $LABEL (31B) vs 31B-Basis: $msg"
