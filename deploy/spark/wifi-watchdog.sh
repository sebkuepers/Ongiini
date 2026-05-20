#!/bin/bash
# /usr/local/bin/wifi-watchdog.sh
#
# Pings the default gateway every minute. If unreachable for 3 consecutive
# checks, bounces the wifi interface. If bouncing doesn't help after 2
# attempts, restarts NetworkManager.
#
# Designed for the failure mode where the wifi AP loses power, comes back,
# but the client card stays in a stuck "associated" state without traffic
# flowing. NetworkManager doesn't notice on its own.

set -uo pipefail

IFACE="${WIFI_IFACE:-wlP9s9}"
FAIL_THRESHOLD=3
BOUNCE_FAIL_THRESHOLD=2
SLEEP_INTERVAL=60
REASSOC_WAIT=25

fail=0
bounces_failed=0

logger -t wifi-watchdog "starting (iface=$IFACE threshold=${FAIL_THRESHOLD}x${SLEEP_INTERVAL}s)"

while true; do
  GW=$(ip route show default 2>/dev/null | awk '/default/ {print $3; exit}')
  if [[ -n "${GW:-}" ]] && ping -c 1 -W 2 "$GW" >/dev/null 2>&1; then
    if [[ $fail -gt 0 ]]; then
      logger -t wifi-watchdog "gateway reachable again after $fail failure(s)"
    fi
    fail=0
    bounces_failed=0
  else
    fail=$((fail + 1))
    logger -t wifi-watchdog "gateway=${GW:-NONE} unreachable ($fail/$FAIL_THRESHOLD)"
    if [[ $fail -ge $FAIL_THRESHOLD ]]; then
      logger -t wifi-watchdog "bouncing $IFACE"
      ip link set "$IFACE" down || true
      sleep 3
      ip link set "$IFACE" up || true
      sleep $REASSOC_WAIT
      fail=0
      bounces_failed=$((bounces_failed + 1))
      if [[ $bounces_failed -ge $BOUNCE_FAIL_THRESHOLD ]]; then
        logger -t wifi-watchdog "bounce ineffective ${BOUNCE_FAIL_THRESHOLD}x — restarting NetworkManager"
        systemctl restart NetworkManager || true
        sleep 30
        bounces_failed=0
      fi
    fi
  fi
  sleep $SLEEP_INTERVAL
done
