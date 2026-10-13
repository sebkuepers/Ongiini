#!/usr/bin/env bash
# "Is it still Gemma 4?" for the model that is live on experimental.ongiini.ai (2026-10-13):
# the compat suite (GSM8K, IFEval, No Robots pairwise, Ongiini scenarios, safety, images, owchat),
# the tool suite and the decoding test (greedy vs MBR), run against the live vLLM instead of a
# separate eval server — no GPU job of its own, low concurrency so visitors are not slowed down.
# The baseline is the plain base model, served by the same vLLM next to the adapter.
#   setsid nohup bash deploy/experimental/check_live.sh 31b T4b31 > data/private/experiments/check_live.log 2>&1 < /dev/null &
# Outputs as compat_31b.sh: data/private/compat/{out,report,tools}_*; decoding in
# data/private/experiments/r31/inference_tricks/{report_,}<LABEL>*; understanding (ow→en) in
# data/private/experiments/ow_en/. Also the 31B test stage of pipeline_sft_variant.sh
# (CHECK_CONCURRENCY=16, no visitors then).
set -u
cd "${ONGIINI_ROOT:-$HOME/dev/Ongiini}"
SIZE=$1 LABEL=$2 BASE=base$1 BASEMODEL=gemma-4-$1
C=${CHECK_CONCURRENCY:-4}
EXP=data/private/experiments
DEC=$EXP/r${SIZE%b}/inference_tricks   # r31, like the other 31B results
log() { echo "$(date '+%F %T') $*"; }
notify() { bash deploy/train/notify.sh "$*" | tail -1; }
BRIDGE=$(ip -4 -o addr show docker0 | awk '{print $4}' | cut -d/ -f1)
SERVER=http://$BRIDGE:8201/v1
curl -sf -m 10 "$SERVER/models" | grep -q "$BASEMODEL" || { log "live server $SERVER does not serve $BASEMODEL"; exit 1; }
crun() { local n=$1; shift
  docker run --rm --name "ongiini-livecheck-$n" --network host --user 1000:1000 -e HOME=/tmp -e USER=nexus -e LOGNAME=nexus \
    -e PYTHONUNBUFFERED=1 -e ONGIINI_EVAL_SERVER="$SERVER" -v "$PWD:/work" -w /work ongiini-evalsuite:latest "$@" 2>&1 \
    | grep -vE "HTTP Request|Warning" | tail -3; }
log "compat suite $BASE ($BASEMODEL) and $LABEL (ongiini-experimental) against $SERVER, concurrency $C"
[ "$(grep -c . data/private/compat/out_$BASE.jsonl 2>/dev/null || echo 0)" -ge 1318 ] \
  || crun compat-base python3 scripts/compat_suite.py generate --model "$BASEMODEL" --label "$BASE" --concurrency "$C"
crun compat-adapter python3 scripts/compat_suite.py generate --model ongiini-experimental --label "$LABEL" --concurrency "$C"
log "tool suite"
[ -s data/private/compat/tools_$BASE.jsonl ] \
  || crun tools-base python3 scripts/tool_suite.py generate --model "$BASEMODEL" --label "$BASE" --concurrency "$C"
crun tools-adapter python3 scripts/tool_suite.py generate --model ongiini-experimental --label "$LABEL" --concurrency "$C"
log "decoding test (greedy vs MBR)"
if [ ! -f "$DEC/report_$LABEL.json" ]; then
  crun decode python3 scripts/exp_inference_tricks.py --model ongiini-experimental --server "$SERVER" \
    --out "$DEC" --conditions greedy mbr --concurrency "$C"
  # the served name is the same for every adapter: file the results under the label
  for f in "$DEC"/*ongiini-experimental*; do [ -e "$f" ] && mv "$f" "${f/ongiini-experimental/$LABEL}"; done
fi
log "understanding (ow→en)"
[ -s "$EXP/ow_en/${BASE}_oshikwanyama.jsonl" ] \
  || crun owen-base python3 scripts/eval_ow_en.py generate --model "$BASEMODEL" --label "$BASE" --server "$SERVER" --concurrency "$C"
crun owen python3 scripts/eval_ow_en.py generate --model ongiini-experimental --label "$LABEL" --server "$SERVER" --concurrency "$C"
owl=("$LABEL"); [ "$LABEL" != T4b31 ] && [ -s "$EXP/ow_en/T4b31_oshikwanyama.jsonl" ] && owl=(T4b31 "$LABEL")
owen=$(crun owen-score python3 scripts/eval_ow_en.py score --base "$BASE" --labels "${owl[@]}" | tail -1)
log "scoring"
crun score python3 scripts/compat_suite.py score --labels "$BASE" "$LABEL"
crun tscore python3 scripts/tool_suite.py score --labels "$BASE" "$LABEL"
msg=$(python3 - <<PY
import json, glob
r = json.load(open("data/private/compat/report_${LABEL}_vs_${BASE}.json")); b, t = r["$BASE"], r["$LABEL"]; p = r["pairwise_${LABEL}_vs_${BASE}"]
T = json.load(open("data/private/compat/tools_report_${LABEL}_vs_${BASE}.json")); tb, tt, tp = T["$BASE"], T["$LABEL"], T["paired"]
try:
    d = json.load(open("$DEC/report_$LABEL.json"))
    dec = f"Decoding: greedy {d['greedy']['chrf']}, MBR {d['mbr']['chrf']} ({d['mbr']['delta_vs_greedy']:+}, CI {d['mbr']['ci95']})."
except Exception as e:
    dec = f"Decoding fehlt ({e})."
print(f"Tool-Test {tb['all']}/{tt['all']} % (p {tp['sign_test_p']}), GSM8K {b['gsm_acc']}/{t['gsm_acc']}, IFEval {b['ifeval_prompt_strict']}/{t['ifeval_prompt_strict']}, "
      f"Tools {b['ongiini_tool_decision']}/{t['ongiini_tool_decision']}, Bilder {b['images_correct']}/{t['images_correct']}, unsicher {b['safety_unsafe']}/{t['safety_unsafe']}, "
      f"OW-Antworten {b['owchat_answers_in_oshiwambo']}/{t['owchat_answers_in_oshiwambo']}; offen {p['open']}, OW-Chat {p['owchat']}. {dec}")
PY
)
log "$LABEL vs $BASE: $msg"
log "ow→en: $owen"
notify "Live-Check $LABEL vs Gemma 4 $SIZE: $msg Verstehen OW→EN: $owen"
