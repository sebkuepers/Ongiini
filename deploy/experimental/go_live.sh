#!/usr/bin/env bash
# Put an adapter live on experimental.ongiini.ai, laptop-independent (2026-10-13). Waits until
# no training / eval / retention container runs (the model needs that memory), serves the model,
# rebuilds the page, starts backend + nginx, runs a smoke test through the public chat endpoint
# and reports by WhatsApp. Pause the experiment queue first (comment out data/private/experiments/
# queue.txt), otherwise its next job may start in between.
#   setsid nohup bash deploy/experimental/go_live.sh 31b data/private/lora/T4b_31b "Ongiini Ombuto" \
#     "Gemma 4 31B + Oshiwambo training T4b31 (2026-10-13)" > data/private/experiments/go_live.log 2>&1 < /dev/null &
# The name + note also go to data-experimental/feedback/model.txt, so every rating records the model.
set -u
cd "${ONGIINI_ROOT:-$HOME/dev/Ongiini}"
MODEL=$1 ADAPTER=$2 NAME=$3 NOTE=$4
log() { echo "$(date '+%F %T') $*"; }
notify() { bash deploy/train/notify.sh "$*" | tail -1; }
fail() { log "$* — stopping"; notify "experimental.ongiini.ai: $* — nicht live geschaltet."; exit 1; }
[ -f "$ADAPTER/adapter_config.json" ] || fail "kein Adapter in $ADAPTER"
log "waiting until no ongiini-(train|eval|ret) container runs"
while docker ps --format '{{.Names}}' | grep -E '^ongiini-(train|eval|ret)-' | grep -vq '^ongiini-eval-experimental$'; do
  sleep 60
done
log "GPU free; memory available: $(free -g | awk '/^Speicher:|^Mem:/{print $7}') GB"
EXP_MODEL=$MODEL bash deploy/experimental/serve_model.sh "$ADAPTER" || {
  docker logs --tail 80 ongiini-eval-experimental > data/private/experiments/go_live_vllm_fail.log 2>&1
  docker rm -f ongiini-eval-experimental >/dev/null 2>&1
  fail "das Modell startete nicht (Log in go_live_vllm_fail.log)"
}
log "model served; memory available: $(free -g | awk '/^Speicher:|^Mem:/{print $7}') GB"
python3 scripts/build_experimental_chat.py --model-name "$NAME" --model-note "$NOTE" || fail "Seite ließ sich nicht bauen"
mkdir -p data-experimental/feedback && echo "$NAME — $NOTE (adapter $ADAPTER)" > data-experimental/feedback/model.txt
docker compose --profile experimental up -d --build experimental experimental-web || fail "docker compose up schlug fehl"
sleep 20
SID=00000000-0000-4000-8000-00000000c0de  # a valid UUID v4, as the chat API requires
reply=""
for attempt in 1 2 3; do
  reply=$(curl -s -m 300 -X POST http://127.0.0.1:8447/v1/chat -H 'Content-Type: application/json' \
    -d '{"session_id":"'$SID'","text":"Translate into Oshindonga: Good morning, how did you sleep?"}')
  echo "$reply" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('reply') or d.get('text') or d.get('messages')) else 1)" 2>/dev/null && break
  reply=""; sleep 30
done
# the rating endpoint must answer through nginx: a reply that does not exist -> the API's own 404 text
fb=$(curl -s -m 30 -X POST http://127.0.0.1:8447/v1/chat/feedback -H 'Content-Type: application/json' \
  -d '{"session_id":"'$SID'","turn":999,"rating":"correct"}')
curl -s -m 30 -X POST http://127.0.0.1:8447/v1/chat/clear -H 'Content-Type: application/json' -d '{"session_id":"'$SID'"}' >/dev/null
[ -n "$reply" ] || fail "Rauchtest ohne Antwort (Backend-Log: docker logs ongiini-experimental)"
echo "$fb" | grep -q "no longer in this session" || fail "Bewertungs-Endpunkt nicht erreichbar: $fb"
page=$(curl -s -o /dev/null -w '%{http_code}' -m 30 https://experimental.ongiini.ai/)
log "smoke test ok; public page HTTP $page"
notify "experimental.ongiini.ai ist live mit $NAME ($NOTE). Rauchtest ok, Seite HTTP $page. Die Experiment-Warteschlange ist pausiert."
