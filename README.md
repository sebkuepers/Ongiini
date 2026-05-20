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
                                              │       ↳ text + image (multimodal)
                                              ├─▶ Tavily (web search)
                                              ├─▶ mem0 long-term semantic memory
                                              │       ↳ embedded qdrant on /data
                                              │       ↳ all-MiniLM-L6-v2 embedder (CPU)
                                              └─▶ /data (short-term JSON, qdrant, usage log)

Browser ──▶ ongiini.ai / www.ongiini.ai ──▶  127.0.0.1:18789  ──▶  nginx (Docker)
                              ▲
                              │  (Cloudflare Tunnel)
                              │
```

Three processes on the Spark:
1. `gemma4-vllm` — Docker container (`vllm/vllm-openai:gemma4-0505-arm64-cu130`).
   Serves Gemma 4 26B A4B (NVFP4) for text AND image input via the OpenAI-compatible
   chat-completions endpoint.
2. `ongiini-webhook` + `ongiini-website` — Docker compose stack in this repo.
3. `cloudflared` — systemd service exposing the two containers at
   `ongiini.ai` / `www.ongiini.ai` / `api.ongiini.ai`.

- `webhook/` — FastAPI service. Receives WhatsApp messages, filters by country
  code / whitelist, calls vLLM with tool-calling (`web_search`, `fetch_url`,
  `delete_my_data`, `whats_in_my_memory`, `my_token_usage`). Two-tier memory: a
  short-term JSON window per user (about 50 turns) and a mem0 long-term layer
  that extracts typed facts (`[PROFILE]`, `[PREFERENCE]`, `[SITUATION]`,
  `[COMMITMENT]`, `[QUOTE]`, `[EMOTION]`) across all chats and retrieves them
  by semantic relevance per turn.
- `website/` — single-page static site, vanilla HTML+CSS+JS, ~14 KB gzipped.
- `data/` — short-term JSON memory files + qdrant vector store + `usage.log`
  + mem0 history sqlite (mounted as a volume; everything stays on the Spark).

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
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.70 \
  --moe-backend marlin \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice --tool-call-parser gemma4 \
  --chat-template /vllm-workspace/examples/tool_chat_template_gemma4.jinja \
  --limit-mm-per-prompt '{"image":4,"audio":0}' \
  --mm-processor-kwargs '{"max_soft_tokens":280}' \
  --hf-overrides '{"vision_config":{"torch_dtype":"bfloat16"}}' \
  --async-scheduling
```

The multimodal flags (`--chat-template`, `--limit-mm-per-prompt`,
`--mm-processor-kwargs`, `--hf-overrides`) are required for the vision
pathway:
- The chat template fixes vLLM #41452 (mixing `tools=` with `image_url`
  in one call). Without it the prompt-replacement step desyncs.
- `--limit-mm-per-prompt` explicitly disables the audio tower (we don't
  use it) and caps image inputs at 4 per call.
- `--hf-overrides` forces the vision tower to bf16. The NVFP4 quant
  doesn't cover vision_tower, and vLLM's loader would otherwise re-cast
  to fp16 and overflow (vLLM #40290).

See `deploy/spark/restart-vllm-with-mm-flags.sh` for the exact command
documented as a runnable script, including the safety checks for the
~3 minute cold restart.

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

Three things per user, all on the Spark, nothing leaves the box:

1. **Short-term memory** at `/data/{msisdn}.json` — about the last 50 turns of
   user+assistant back-and-forth (capped at 100 entries on disk). Marathon chats
   that cross 70 entries fold their oldest turns into a single leading `system`
   message ("Earlier in this conversation: …") and keep the last 40 entries
   verbatim.
   Before any message is written, regex-scrubbed for obvious PII patterns
   (email, IBAN, credit card, 11-digit Namibian ID — replaced with
   `[REDACTED:kind]` placeholders).
2. **Long-term memory** via mem0 — typed facts about the user extracted across
   ALL conversations: `[PROFILE]` (location, role, family), `[PREFERENCE]`
   (language, style), `[SITUATION]` (ongoing projects), `[COMMITMENT]`
   (reminders, follow-ups), `[QUOTE]` (verbatim phrasing worth keeping), and
   `[EMOTION]` (recent state). Stored as 384-dim embeddings in an embedded
   qdrant on `/data/qdrant/`, retrieved by semantic similarity per turn.
   Facts canonicalised to English regardless of input language so cross-language
   recall works (user writes Afrikaans, retrieves on English query). Image
   turns produce facts via the assistant's own image description rather than
   passing the base64 bytes through mem0's extraction prompt.
3. **Token-count log** at `/data/usage.log` — per-message line with token
   counts + timestamp, **no message content ever**.

```
2026-05-20T14:32:11 | 264811234567 | tokens_in=342 tokens_out=187 | search=yes
```

Users can ask "what do you remember about me?" (any wording, English or
Afrikaans) and the `whats_in_my_memory` tool reads both tiers back grouped
by tag. "Delete my data" / "forget everything" / "vergeet alles" wipes both
tiers via `delete_my_data`. Token balance: "how many tokens have I used?"
fires `my_token_usage`, which aggregates the usage.log against the
1 million / user / month free-tier cap.

## Multimodal

Gemma 4 is multimodal — the webhook accepts WhatsApp image messages, fetches
the bytes from Meta's media API, normalises them (resize to multiples of 48
to satisfy Gemma 4's vision pooler grid, clamp to ≤896 per side, re-encode
as JPEG), and passes them as OpenAI-style multipart content to vLLM. The
model describes what it sees and the resulting [SITUATION] fact lands in
mem0 ("Shared photo of maize leaves with yellowing tips; worried about
crop health"). Image bytes themselves are NEVER stored — only the assistant's
textual description is persisted.

One production caveat baked into the code:
- **Image dimensions must be multiples of 48** before they reach vLLM
  (`main.py::_resize_for_gemma4`). Off-grid sizes crash the Gemma 4 vision
  pooler with `cudaErrorNotPermitted`. The webhook resizes every inbound
  image to 48-aligned bounds clamped to `[336×192, 896×896]` and
  re-encodes as JPEG before sending. Captions are PII-scrubbed up front
  so credentials in image text never reach mem0 or the on-disk placeholder.

Tools (`web_search`, `delete_my_data`, etc.) fire on image-bearing
calls too — the `--chat-template tool_chat_template_gemma4.jinja` flag
in the vLLM startup is the upstream fix for #41452. There's still a
caption-router in `main.py` that detects admin-intent captions
("delete my data", "wat onthou jy", "how many tokens") and processes
them as text-only as defence-in-depth, but the underlying vLLM
restriction it worked around is gone.

## Out of scope for Phase 1

Voice messages (Whisper), Oshiwambo via a translation layer, hard rate-limit
enforcement, Redis-backed memory, async queue.
