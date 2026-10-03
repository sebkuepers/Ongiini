#!/usr/bin/env bash
# Memory emergency brake for GPU jobs next to production (Spark, unified memory).
# Every 2 s: if MemAvailable drops below THRESHOLD_GB, kill every training /
# eval / retention container at once. Production (vLLM, webhook) is never
# touched. On the Spark, running out of memory does not fail cleanly — the
# kernel swaps out sshd, Tailscale, cloudflared and the webhook and the box
# freezes for hours (2026-10-03). Runs as systemd unit ongiini-membrake.
#   THRESHOLD_GB=12 bash deploy/train/membrake.sh
set -u
THRESHOLD_GB=${THRESHOLD_GB:-12}
PATTERN='^ongiini-(train|eval|ret)-'
REPO=/home/nexus/dev/Ongiini
while true; do
  avail_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  if [ "$avail_kb" -lt $((THRESHOLD_GB * 1024 * 1024)) ]; then
    victims=$(docker ps --format '{{.Names}}' | grep -E "$PATTERN" | tr '\n' ' ')
    if [ -n "$victims" ]; then
      docker kill $victims >/dev/null 2>&1
      msg="Notbremse: nur $((avail_kb / 1024 / 1024)) GB Speicher frei, GPU-Jobs gestoppt: $victims"
      echo "$(date '+%F %T') $msg"
      # notify in the background with a timeout: never block the brake.
      ( cd "$REPO" && timeout 60 sudo -u nexus bash deploy/train/notify.sh "$msg" ) &
      sleep 20
    fi
  fi
  sleep 2
done
