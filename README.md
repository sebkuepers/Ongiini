# Ongiini

> "Ongiini" — "How are you?" in Oshiwambo.

A free WhatsApp AI assistant for Namibia. English & Afrikaans, powered by
**Gemma 4 26B A4B** (NVFP4) on a local NVIDIA DGX Spark, exposed via
Tailscale Funnel.

## Architecture (Phase 1)

```
WhatsApp Cloud API ──┐
                     │ /webhooks/whatsapp
                     ▼
        Tailscale Funnel  ──▶  127.0.0.1:8445  ──▶  FastAPI webhook (Docker)
                                                      │
                                                      ├─▶ vLLM Gemma 4 (host:8124)
                                                      ├─▶ Tavily (web search)
                                                      └─▶ /data (memory + log)

Browser  ──▶  Tailscale Serve  ──▶  127.0.0.1:18789  ──▶  static website (Docker)
```

Two containers, no in-repo nginx — Tailscale serve/funnel provides the public
HTTPS surface and path routing.

- `webhook/` — FastAPI service. Receives WhatsApp messages, filters by country
  code / whitelist, calls vLLM with tool-calling for `web_search`, replies,
  persists last 10 messages per user as JSON.
- `website/` — single-page static site with a "Chat on WhatsApp" button & QR.
- `data/` — JSON memory files + `usage.log` (mounted as a volume).

## DGX Spark host setup

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
  0.85 from the ai-muninn blog crashes here because background services
  (k3s, openshell, Docker, signal-cli, Grafana, Prometheus) consume ~40 GB
  of unified memory before vLLM starts.
- The community `pythonic` parser does NOT match this build's output format;
  use `gemma4` (the model emits `<|tool_call>call:func{key:<|"|>val<|"|>}<tool_call|>`).

Quick check:
```sh
curl -s http://127.0.0.1:8124/v1/models | jq
```

## Ongiini stack setup

```sh
cd ~/dev/Ongiini
cp .env.example .env
# Edit .env with WhatsApp + Tavily creds + PUBLIC_WHATSAPP_NUMBER

# Bake the public number into the static site (one-off per number change)
sed -i.bak "s/__WA_NUMBER__/$(grep '^PUBLIC_WHATSAPP_NUMBER=' .env | cut -d= -f2)/" website/index.html

docker compose up -d --build
```

## Tailscale exposure

The Spark already has these routes provisioned via `tailscale serve`:

```
https://spark-dccf.tailac3921.ts.net/             → 127.0.0.1:18789  (website)
https://spark-dccf.tailac3921.ts.net/webhooks/    → 127.0.0.1:8445   (webhook)
```

For Meta's WhatsApp Cloud API to reach the webhook over the public internet,
enable Funnel on the webhook path:

```sh
tailscale funnel --bg --set-path=/webhooks/ http://127.0.0.1:8445
```

Then in Meta's WhatsApp Business app dashboard:

- Callback URL: `https://spark-dccf.tailac3921.ts.net/webhooks/whatsapp`
- Verify token: same as `WHATSAPP_VERIFY_TOKEN` in `.env`

## Endpoints

- `GET  /whatsapp` — Meta verification handshake.
- `POST /whatsapp` — incoming WhatsApp messages.

The public URL Meta calls is `https://<tailnet>/webhooks/whatsapp` — Tailscale
serve strips the `/webhooks/` prefix before forwarding to the webhook
container, so internally the routes live under `/whatsapp`.
- `GET  /health` — liveness check.
- `GET  /` (website container) — static landing page.

## Filter behaviour

- `+264...` numbers → processed normally.
- Other numbers → friendly auto-reply (English) saying Ongiini is Namibia-only.
- Numbers in `WHITELIST` (comma-separated, no `+`) → bypass the country check.

## Memory

- Per-user JSON at `/data/{msisdn}.json`. Last 10 user+assistant turns kept.

## Usage log

`/data/usage.log`, one line per handled message:

```
2026-05-20T14:32:11 | 264811234567 | tokens_in=342 tokens_out=187 | search=yes
```

## Out of scope for Phase 1

Redis, async queues, Whisper, image input, rate limiting, dashboards — see
project briefing for the Phase 2 roadmap.
