#!/usr/bin/env bash
# Round 6a (2026-10-10), the "Regler-Test": C1-T4 (CPT + T4) gained +1.7 chrF over T4b but lost
# instruction following (IFEval strict 88.0 -> 82.8, p 0.002) and open-task quality. Scale the
# C1-T4 adapter (lora_B x s; s = 1 trained, 0 = base) and measure, through one eval vLLM:
# Ndonga dev chrF (greedy, benchmark prompt), IFEval strict (541), tool suite (86 x 3) —
# T4b in the same harness as the reference. No training. Waits for round 5 to finish.
#   setsid nohup bash deploy/train/exp_lora_scale.sh > data/private/experiments/lora_scale.log 2>&1 < /dev/null &
set -u
cd "$HOME/dev/Ongiini"
log() { echo "$(date '+%F %T') $*"; }
notify() { bash deploy/train/notify.sh "$*" | tail -1; }
OUT=data/private/experiments/lora_scale; SC=data/private/lora/scaled
SCALES="0.5 0.7 0.85"
log "waiting for pipeline_goal5.sh to end"
while pgrep -f "bash deploy/train/pipeline_goal5.sh" >/dev/null; do sleep 60; done
others=$(docker ps --format '{{.Names}}' | grep -E '^ongiini-(train|eval|ret)-' | tr '\n' ' ')
[ -z "$others" ] || { log "GPU jobs running: $others — stop"; notify "Regler-Test nicht gestartet: GPU-Jobs laufen ($others)."; exit 1; }
mkdir -p "$OUT" "$SC"
C=(docker run --rm --network host --user 1000:1000 -e HOME=/tmp -e USER=nexus -e LOGNAME=nexus -e PYTHONUNBUFFERED=1
   -v "$PWD:/work" -w /work ongiini-evalsuite:latest)
crun() {  # crun <container name> <cmd...>: like C, with a name (it must come before the image)
  local n=$1; shift
  docker run --rm --name "$n" --network host --user 1000:1000 -e HOME=/tmp -e USER=nexus -e LOGNAME=nexus \
    -e PYTHONUNBUFFERED=1 -v "$PWD:/work" -w /work ongiini-evalsuite:latest "$@"
}
mods=("t4b=data/private/lora/T4b_12b" "c1t4=data/private/lora/C1-T4_12b")
for s in $SCALES; do
  d="$SC/C1-T4_s$s"
  [ -f "$d/adapter_model.safetensors" ] || "${C[@]}" python3 scripts/scale_adapter.py data/private/lora/C1-T4_12b "$s" "$d" \
    || { notify "Regler-Test: Skalieren auf $s fehlgeschlagen."; exit 1; }
  mods+=("c1t4s${s/./}=$d")
done
log "serve: ${mods[*]}"
EVAL_GPU_MEM_UTIL=0.45 EVAL_MAX_SEQS=32 bash deploy/eval/serve_12b_lora.sh "${mods[@]}" \
  || { notify "Regler-Test: eval vLLM startete nicht."; exit 1; }
for spec in "${mods[@]}"; do
  m=${spec%%=*}
  log "measure $m"
  [ -f "$OUT/report_$m.json" ] || crun "ongiini-eval-scale-chrf-$m" python3 scripts/exp_inference_tricks.py \
    --model "$m" --conditions greedy --out "$OUT" --concurrency 32 2>&1 | grep -vE "HTTP Request|Warning" | tail -1
  crun "ongiini-eval-scale-ifeval-$m" python3 scripts/compat_suite.py generate --model "$m" --label "scale_$m" \
    --sections ifeval --concurrency 32 2>&1 | grep -vE "HTTP Request|Warning" | tail -1
  crun "ongiini-eval-scale-tools-$m" python3 scripts/tool_suite.py generate --model "$m" --label "scale_$m" \
    --concurrency 16 2>&1 | grep -vE "HTTP Request|Warning" | tail -1
done
docker logs --tail 200 ongiini-eval-vllm12b > "$OUT/vllm.log" 2>&1
docker rm -f ongiini-eval-vllm12b >/dev/null 2>&1
for spec in "${mods[@]}"; do m=${spec%%=*}
  [ -f "$OUT/report_$m.json" ] && [ -s "data/private/compat/out_scale_$m.jsonl" ] && [ -s "data/private/compat/tools_scale_$m.jsonl" ] \
    || { log "measurement for $m missing — stop"; notify "Regler-Test: Messung für $m fehlt — abgebrochen (Log: lora_scale.log)."; exit 1; }
done
log "score"
"${C[@]}" python3 - "${mods[@]}" > "$OUT/summary.txt" 2>&1 <<'PY'
import json, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/third_party")
import compat_suite as C, tool_suite as T
items = {json.loads(l)["id"]: json.loads(l) for l in open(C.ITEMS)}
its = {i["id"]: i for i in T.items()}
rows = []
for spec in sys.argv[1:]:
    m = spec.split("=")[0]
    try:
        chrf = json.load(open(f"data/private/experiments/lora_scale/report_{m}.json"))["greedy"]["chrf"]
    except (OSError, KeyError):
        chrf = None
    o = {json.loads(l)["id"]: json.loads(l) for l in open(f"data/private/compat/out_scale_{m}.jsonl")}
    ife = [C._ifeval(it, o[i]["content"])[0] for i, it in items.items() if it["section"] == "ifeval" and i in o]
    errs = sum(str(r.get("finish", "")).startswith("error") for r in o.values())
    tr = [json.loads(l) for l in open(f"data/private/compat/tools_scale_{m}.jsonl")]
    tool = sum(T.correct(r, its[r["id"]]["ok"]) for r in tr) / len(tr)
    docs = [T.correct(r, its[r["id"]]["ok"]) for r in tr if its[r["id"]]["ok"][:1] == ["lookup_ongiini_docs"]]
    rows.append((m, chrf, round(100 * sum(ife) / len(ife), 1), len(ife), errs, round(100 * tool, 1),
                 round(100 * sum(docs) / max(1, len(docs)), 1)))
print("model | Ndonga dev chrF (greedy) | IFEval strict % (n, errors) | tools % | docs-lookup %")
for r in rows:
    print(f"{r[0]} | {r[1]} | {r[2]} ({r[3]}, {r[4]}) | {r[5]} | {r[6]}")
PY
cat "$OUT/summary.txt"
notify "Regler-Test fertig (Ndonga chrF | IFEval | Tools | Doku-Tool): $(tail -n +2 "$OUT/summary.txt" | tr '\n' ';' | cut -c1-700)"
log "done"
