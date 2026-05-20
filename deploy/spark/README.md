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

## `restart-vllm-with-mm-flags.sh`

Stop + recreate the `gemma4-vllm` container with the multimodal-required
flags layered on top of the base vLLM command. Run after any change to
the vLLM image, the model directory, or the multimodal config. The
restart takes ~3-4 minutes for the cold model load; the script polls
`/v1/models` and exits non-zero if vLLM doesn't come back.

The flags it adds over the base command:
- `--chat-template /vllm-workspace/examples/tool_chat_template_gemma4.jinja`
  → fixes vLLM #41452 (tools + image_url in one call).
- `--limit-mm-per-prompt '{"image":4,"audio":0}'`
  → required to enable image profiling; disables the unused audio tower.
- `--mm-processor-kwargs '{"max_soft_tokens":280}'`
  → explicit vision token budget for reproducibility.
- `--hf-overrides '{"vision_config":{"torch_dtype":"bfloat16"}}'`
  → forces vision_tower to bf16. NVFP4 doesn't cover it; without this
    flag vLLM's loader silently re-casts to fp16 (vLLM #40290).
- `--max-num-seqs 16`
  → tames concurrent multimodal KV-cache allocations on the Spark.
- `--async-scheduling`
  → recommended by the vLLM Gemma 4 recipe for MoE multimodal.

### Run

```sh
bash deploy/spark/restart-vllm-with-mm-flags.sh
```

There's a ~3-minute service interruption while the model reloads —
the webhook receives 5xx on `/v1/chat/completions` calls during that
window and Meta will redeliver inbound messages after the model is
back.

## Future bits to put here

- Container memory caps (cgroups) for vLLM
- Kernel panic auto-reboot sysctl (when desired)
- Telegram heartbeat / alert daemon
- vLLM model-swap helper (gemma4 ↔ qwen3 ↔ nemotron)
