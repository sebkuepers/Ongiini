# Ongiini

> "Ongiini" — "How are you?" in Oshiwambo.

A free WhatsApp AI assistant for people in Namibia. English & Afrikaans today,
Oshiwambo via a translation layer coming soon. Powered by
**Gemma 4 26B A4B** (NVFP4) running locally on an NVIDIA DGX Spark, exposed via
a Cloudflare Tunnel.

Pilot currently runs in Germany; the goal is to move the hardware to Namibia
once the service is sustainable.

## Architecture

```
WhatsApp Cloud API ──▶ api.ongiini.ai/whatsapp
                              │  (Cloudflare Tunnel)
                              ▼
                       127.0.0.1:8445  ──▶  FastAPI webhook (Docker)
                                              │
                                              ├─▶ vLLM Gemma 4 (host:8124)
                                              ├─▶ Tavily (web search)
                                              └─▶ /data (memory + log)

Browser ──▶ ongiini.ai / www.ongiini.ai ──▶  127.0.0.1:18789  ──▶  nginx (Docker)
                              ▲
                              │  (Cloudflare Tunnel)
                              │
```

Three processes on the Spark:
1. `gemma4-vllm` — Docker container (`vllm/vllm-openai:gemma4-0505-arm64-cu130`).
2. `ongiini-webhook` + `ongiini-website` — Docker compose stack in this repo.
3. `cloudflared` — systemd service exposing the two containers at
   `ongiini.ai` / `www.ongiini.ai` / `api.ongiini.ai`.

- `webhook/` — FastAPI service. Receives WhatsApp messages, filters by country
  code / whitelist, calls vLLM with tool-calling for `web_search`, replies,
  persists last 10 messages per user as JSON.
- `website/` — single-page static site, vanilla HTML+CSS+JS, ~14 KB gzipped.
- `data/` — JSON memory files + `usage.log` (mounted as a volume).

## DGX Spark host setup

### Gemma 4 vLLM

The vLLM container runs **on the host**, not under compose, so it can use the
full unified GB10 memory pool.

```sh
# 1. Download the model (~16.5 GB)
hf download bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4 \
  --local-dir ~/models/gemma-4-26b-a4b-nvfp4

# 2. Run vLLM (validated on the Spark — boots in ~3-4 min cold)
docker run -d \
  --name gemma4-vllm \
  --restart unless-stopped \
  --gpus all --ipc host --shm-size 64gb \
  -p 8124:8000 \
  -v ~/models/gemma-4-26b-a4b-nvfp4:/models/gemma4 \
  vllm/vllm-openai:gemma4-0505-arm64-cu130 \
  --model /models/gemma4 \
  --served-model-name gemma-4-26b \
  --host 0.0.0.0 --port 8000 \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.70 \
  --moe-backend marlin \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice --tool-call-parser gemma4
```

Notes on the flags:
- The `gemma4-0505-arm64-cu130` image already includes the Gemma4 model
  loader fixes, so **no** `gemma4_patched.py` overlay is needed.
- `--max-num-batched-tokens 8192` is required because Gemma 4 is multimodal
  and the default (2048) is smaller than a single image token block (2496).
- `--gpu-memory-utilization 0.70` leaves headroom for the rest of the host.
- The community `pythonic` parser does NOT match this build's output format;
  use `gemma4` (the model emits `<|tool_call>call:func{key:<|"|>val<|"|>}<tool_call|>`).

Quick check:
```sh
curl -s http://127.0.0.1:8124/v1/models | jq
```

### Ongiini compose stack

```sh
cd ~/dev/Ongiini
cp .env.example .env
# Edit .env with WhatsApp + Tavily creds
docker compose up -d --build
```

The WhatsApp link is hard-coded in `website/index.html` (currently
`wa.me/4915888635886`, the pilot number). To change it, edit the
`wa.me/...` `href` values directly in the HTML — the number is no
longer displayed on the page, only the CTA buttons that open WhatsApp.

### Cloudflare Tunnel

Public DNS for `ongiini.ai` lives on Cloudflare. A tunnel named
`ongiini-spark` (UUID `6ff62805-71c9-4d78-acf2-b76095b87310`) runs as a
systemd service on the Spark and routes:

| Hostname | Backend |
|---|---|
| `ongiini.ai` | `http://localhost:18789` (website) |
| `www.ongiini.ai` | `http://localhost:18789` (via CNAME chain) |
| `api.ongiini.ai` | `http://localhost:8445` (webhook) |

Config: `/etc/cloudflared/config.yml` + credentials at
`/etc/cloudflared/<UUID>.json`. Both are root-owned. To inspect:

```sh
sudo systemctl status cloudflared
sudo journalctl -u cloudflared -f
sudo cat /etc/cloudflared/config.yml
```

To restart after a config change:
```sh
sudo systemctl restart cloudflared
```

### Meta WhatsApp Cloud API config

In the Meta WhatsApp Business dashboard:

- Callback URL: `https://api.ongiini.ai/whatsapp`
- Verify token: whatever you set as `WHATSAPP_VERIFY_TOKEN` in `.env`

The webhook returns 403 until both the verify token in `.env` and the value
configured in Meta match.

## Endpoints

- `GET  /whatsapp` (webhook container) — Meta verification handshake.
- `POST /whatsapp` (webhook container) — incoming WhatsApp messages.
- `GET  /health` (webhook container) — liveness check.
- `GET  /` (website container) — static landing page.

Internally the webhook only listens on `/whatsapp`. Cloudflare Tunnel routes
the entire `api.ongiini.ai` host into port 8445 with the path preserved.

## Filter behaviour

- `+264…` numbers → processed normally.
- Other numbers → friendly auto-reply (English) explaining Namibia-only scope.
- Numbers in `WHITELIST` (comma-separated, no `+`) → bypass the country check.

## Data we keep

Per-user JSON at `/data/{msisdn}.json` — last 10 user+assistant turns.
Per-message line in `/data/usage.log` — token counts + timestamp, no content.

```
2026-05-20T14:32:11 | 264811234567 | tokens_in=342 tokens_out=187 | search=yes
```

Free-tier cap is **1 million tokens per user per month**, resetting on the
1st (cap enforcement not yet implemented in code — Phase 2).

## Out of scope for Phase 1

Voice messages (Whisper), image understanding, Oshiwambo via a translation
layer, hard rate-limit enforcement, Redis-backed memory, async queue.
