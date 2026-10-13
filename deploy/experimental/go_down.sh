#!/usr/bin/env bash
# Take experimental.ongiini.ai offline while a new model trains (2026-10-13): the chat stack
# stops and a static "we are training a new model" page (deploy/experimental/maintenance/)
# takes over port 8447, so the tunnel keeps answering. A running check_live.sh may finish
# first — the vLLM server stays up until it exits, then it is removed and the GPU is free
# for the experiment queue. go_live.sh replaces the page again.
#   setsid nohup bash deploy/experimental/go_down.sh > data/private/experiments/go_down.log 2>&1 < /dev/null &
set -u
cd "${ONGIINI_ROOT:-$HOME/dev/Ongiini}"
log() { echo "$(date '+%F %T') $*"; }
notify() { bash deploy/train/notify.sh "$*" | tail -1; }
docker compose --profile experimental stop experimental experimental-web
docker compose --profile experimental rm -f experimental experimental-web >/dev/null
docker rm -f ongiini-experimental-maintenance >/dev/null 2>&1
docker run -d --name ongiini-experimental-maintenance --restart unless-stopped -p 127.0.0.1:8447:8080 \
  -v "$PWD/deploy/experimental/maintenance/index.html:/usr/share/nginx/html/index.html:ro" \
  -v "$PWD/deploy/experimental/maintenance/nginx.conf:/etc/nginx/conf.d/default.conf:ro" nginx:1.27-alpine >/dev/null
sleep 3
log "maintenance page: HTTP $(curl -s -o /dev/null -w '%{http_code}' -m 30 https://experimental.ongiini.ai/)"
while pgrep -f 'check_live.sh' >/dev/null; do sleep 60; done
docker rm -f ongiini-eval-experimental >/dev/null 2>&1
log "vLLM removed, GPU free"
notify "experimental.ongiini.ai zeigt die Wartungsseite, das Modell ist entladen. Die GPU ist frei für die Warteschlange."
