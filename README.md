# Ongiini

> "Ongiini" — "How are you?" in Oshiwambo.

A free WhatsApp AI assistant for Namibia. English & Afrikaans, powered by
Gemma 4 on a local NVIDIA DGX Spark, served via Tailscale Funnel.

## Architecture (Phase 1)

```
WhatsApp Cloud API ──▶ Tailscale Funnel ──▶ nginx ──▶ FastAPI webhook
                                                 │
                                                 ├─▶ vLLM (host:8000)  ← Gemma 4
                                                 ├─▶ Tavily (web search)
                                                 └─▶ /data (memory + log)

                                                 ↘  nginx ──▶ static website
```

- `webhook/` — FastAPI service. Receives WhatsApp messages, filters by country
  code / whitelist, calls vLLM with tool-calling for `web_search`, replies,
  persists last 10 messages per user as JSON.
- `website/` — single-page static site with a "Chat on WhatsApp" button & QR.
- `nginx/` — reverse proxy that maps `/webhook` → FastAPI and `/` → website.
- `data/` — JSON memory files + `usage.log` (mounted as a volume).

vLLM runs **directly on the host**, not in Docker.

## Setup

1. **Copy env:**
   ```sh
   cp .env.example .env
   ```
   Fill in:
   - `WHATSAPP_TOKEN`, `WHATSAPP_PHONE_ID`, `WHATSAPP_VERIFY_TOKEN`
   - `TAVILY_API_KEY`
   - `VLLM_MODEL` if different from the default
   - `PUBLIC_WHATSAPP_NUMBER` (international format, no `+`) — used by the website

2. **Run vLLM on the host** (example):
   ```sh
   vllm serve google/gemma-3-27b-it --host 0.0.0.0 --port 8000 \
     --enable-auto-tool-choice --tool-call-parser hermes
   ```

3. **Bake the WhatsApp number into the static site** (one-time per number change):
   ```sh
   sed -i.bak "s/__WA_NUMBER__/$PUBLIC_WHATSAPP_NUMBER/" website/index.html
   ```

4. **Bring up the stack:**
   ```sh
   docker compose up -d
   ```

5. **Expose via Tailscale Funnel** (on the host):
   ```sh
   tailscale funnel 8088
   ```

6. **Point the Meta webhook** at `https://<your-tailnet>/webhook` using the
   `WHATSAPP_VERIFY_TOKEN` you set.

## Endpoints

- `GET  /webhook` — Meta verification handshake.
- `POST /webhook` — incoming WhatsApp messages.
- `GET  /health` — liveness check.
- `GET  /` — static website.

## Filter behaviour

- `+264...` numbers → processed normally.
- Other numbers → friendly auto-reply (English) saying Ongiini is Namibia-only.
- Numbers in `WHITELIST` (comma-separated, no `+`) → bypass the country check.

## Memory

- Per-user JSON at `/data/{msisdn}.json`. Last 10 user+assistant turns kept.
- No cleanup job; files grow only if a user is active. Trivial to reset.

## Usage log

`/data/usage.log`, one line per handled message:

```
2026-05-20T14:32:11 | 264811234567 | tokens_in=342 tokens_out=187 | search=yes
```

## Out of scope for Phase 1

Redis, async queues, Whisper, image input, rate limiting, dashboards — see
project briefing for the Phase 2 roadmap.
