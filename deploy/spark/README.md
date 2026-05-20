# Spark host-side bits

Things that run *outside* the Docker compose stack — on the Spark itself.
These tighten the system's behaviour around well-understood failure
modes specific to running this on a single computer over a household
WiFi connection.

## `wifi-watchdog`

Bash script + systemd unit that pings the default gateway every 60 s.
If unreachable for 3 consecutive checks, bounces `wlP9s9`. If bouncing
doesn't restore connectivity twice in a row, restarts NetworkManager
as the next escalation.

Designed for the failure mode where the WiFi access point loses power,
comes back, and the client card sits in a stuck "associated but no
traffic" state. NetworkManager doesn't notice on its own.

### Install

```sh
# From the repo root, on the Spark:
sudo install -m 755 deploy/spark/wifi-watchdog.sh /usr/local/bin/wifi-watchdog.sh
sudo install -m 644 deploy/spark/wifi-watchdog.service /etc/systemd/system/wifi-watchdog.service
sudo systemctl daemon-reload
sudo systemctl enable --now wifi-watchdog.service
```

### Verify

```sh
sudo systemctl status wifi-watchdog
sudo journalctl -u wifi-watchdog -f          # live log
sudo journalctl -t wifi-watchdog -n 50       # tag-based, last 50 events
```

### Tunables

Environment variables read at start:
- `WIFI_IFACE` — interface name to watch (default `wlP9s9`).

In-script constants:
- `FAIL_THRESHOLD` (3) — consecutive ping failures before bouncing.
- `BOUNCE_FAIL_THRESHOLD` (2) — bounces in a row before NetworkManager restart.
- `SLEEP_INTERVAL` (60 s) — poll cadence.
- `REASSOC_WAIT` (25 s) — wait after `ip link up` before considering reassoc done.

## Future bits to put here

- Container memory caps (cgroups) for vLLM
- Kernel panic auto-reboot sysctl (when desired)
- Telegram heartbeat / alert daemon
- vLLM model-swap helper (gemma4 ↔ qwen3 ↔ nemotron)
