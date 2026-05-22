# Ongiini

> "Ongiini" — "How are you?" in Oshiwambo.

A free WhatsApp AI assistant for people in Namibia. English & Afrikaans today,
Oshiwambo via a translation layer coming soon. Powered by
**Gemma 4 26B A4B** (NVFP4) running locally on an NVIDIA DGX Spark, exposed via
a Cloudflare Tunnel.

Pilot currently runs in Germany; the goal is to move the hardware to Namibia
once the service is sustainable.

Ongiini is the first project of the **Common Intelligence Foundation** — a
non-profit foundation currently being formally established in Estonia. Until
that registration is complete, operations are carried out by Sebastian Küpers
in his personal capacity, on a non-profit basis. See
[common-intelligence.org](https://common-intelligence.org) for the manifest.

The foundation's website source lives under `foundation/` in this repo but is
**gitignored** (kept local-only until the foundation is formally registered).
It is deployed to Cloudflare Pages directly via `wrangler pages deploy`.

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
                                              ├─▶ qualitative-analysis loop
                                              │       ↳ LLM-emergent topic + WHO clustering
                                              │       ↳ /data/qualia.sqlite (label cache)
                                              │       ↳ /data/synthesis-*.json (cluster output)
                                              └─▶ /data (short-term JSON, qdrant, usage log)

Browser ──▶ ongiini.ai/statistics      ──┐
                                          ├─▶ Cloudflare Pages (static)
Browser ──▶ ongiini.ai (landing page)  ──┘       │
                                                  │  /api/stats Pages Function
                                                  ▼
                                          api.ongiini.ai/stats.json
                                                  │  (same Cloudflare Tunnel)
                                                  ▼
                                          FastAPI webhook /stats.json
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
  by semantic relevance per turn. Also exposes `GET /stats.json` for the
  transparency / `/statistics` page — see `webhook/app/stats/`.
- `website/` — Cloudflare Pages site with two surfaces:
  - `website/index.html` + subpages (`/privacy/`, `/terms/`, `/imprint/`,
    `/statistics/`). Vanilla HTML+CSS+JS, no build step.
  - `functions/api/stats.js` — Cloudflare Pages Function that proxies
    `/api/stats` to the DGX-hosted webhook's `/stats.json`. Keeps the page
    same-origin so the browser never hits a different hostname.
- `data/` — short-term JSON memory files + qdrant vector store + `usage.log`
  + mem0 history sqlite + transparency caches (`qualia.sqlite`,
  `synthesis-*.json`, optional `objections.txt`). Mounted as a volume;
  everything stays on the Spark.

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

- `GET  /whatsapp` (webhook) — Meta verification handshake.
- `POST /whatsapp` (webhook) — incoming WhatsApp messages.
- `GET  /health` (webhook) — liveness check.
- `GET  /status` (webhook) — public status indicator polled by the landing
  page footer.
- `GET  /stats.json` (webhook) — transparency-reporting payload. Aggregate
  data only, no per-user content. Read by the Cloudflare Pages Function
  at `/api/stats` and rendered on `/statistics/`. See
  [`docs/statistics.md`](docs/statistics.md) for the framework.
- `GET  /api/stats` (Pages Function) — same-origin proxy to the webhook's
  `/stats.json`, with `STATS_API_URL` Pages env var pointing at
  `https://api.ongiini.ai`.
- `GET  /` (Cloudflare Pages) — static landing page.
- `GET  /privacy/`, `/terms/`, `/imprint/`, `/statistics/` — static subpages.

The webhook's container only listens on `:8080` internally (`:8445` on the
host loopback). Cloudflare Tunnel routes the entire `api.ongiini.ai` host
into the webhook with the path preserved.

## Filter behaviour

- `+264…` numbers → processed normally.
- Other numbers → friendly auto-reply (English) explaining Namibia-only scope.
- Numbers in `WHITELIST` (comma-separated, no `+`) → bypass the country check.

## Data we keep

Per user, all on the Spark, nothing leaves the box:

1. **Short-term memory** at `/data/{msisdn}.json` — about the last 50 turns of
   user+assistant back-and-forth (capped at 100 entries on disk). Marathon chats
   that cross 70 entries fold their oldest turns into a single leading `system`
   message ("Earlier in this conversation: …") and keep the last 40 entries
   verbatim.
   Before any message is written, regex-scrubbed for obvious PII patterns
   (email, IBAN, credit card, 11-digit Namibian ID — replaced with
   `[REDACTED:kind]` placeholders). Image and voice messages are stored as
   text placeholders only: `[image attached] <caption>` for images,
   `[voice note] <transcript>` for voice (the audio bytes are discarded
   after Whisper transcription).
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
2026-05-20T14:32:11 | 264811234567 | tokens_in=342 tokens_out=187 | search=yes | kind=chat
```

Aggregate-only by-products of the transparency layer (no per-user content,
see `docs/statistics.md`):

- `/data/qualia.sqlite` — short-label cache from the qualitative-analysis
  loop. One row per (analysis, content-hash, version) → label. The LLM-
  produced labels pass through a regex sanitiser (`webhook/app/stats/safety.py`)
  that drops anything containing identifying patterns before storage.
- `/data/synthesis-{topics,roles,regions,languages,family,situations}.json`
  — cluster output written by the periodic synthesis pass. Used by
  `/stats.json` to surface emergent use-cases and demographic dimensions
  on `/statistics/`.
- `/data/objections.txt` — optional list of MSISDNs that have objected to
  research processing (Art. 21 GDPR). The aggregator excludes these at the
  source so their data contributes to no aggregate, current or future.
- `/data/trace.jsonl` — one JSON line per handled message, recording
  structural signals (token counts, tool calls by name, latencies, finish
  reasons). **No message content, no tool arguments verbatim** — only
  lengths and names.

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

## Voice notes

WhatsApp voice notes are transcribed on the Spark via `faster-whisper`
(CTranslate2 INT8 Whisper-large-v3-turbo) and routed through the same
text path as a typed message — same memory writes, same tools, same
EN/AF language redirect. The model runs on CPU so it doesn't compete
with Gemma 4 for GPU memory. Typical 30s voice note transcribes in 2-5s.

Audio bytes are NEVER persisted. Short-term memory and mem0 only ever
see the transcript text, which goes through the same PII scrub the text
path uses (email / IBAN / card / 11-digit ID → placeholders before
write).

Code: `webhook/app/audio.py` (transcribe wrapper),
`webhook/app/main.py::handle_audio_message` (download → transcribe →
delegate to `handle_message`).

Voice replies (TTS) are not yet supported — Ongiini answers voice notes
in text. That's the only intentional asymmetry.

## Transparency reporting (`/statistics`)

Hidden-but-online page at [`ongiini.ai/statistics`](https://ongiini.ai/statistics/)
(unlinked from the main nav, `<meta name="robots" content="noindex">`).
Renders **aggregate** statistics about service usage — never individual
conversations. Legally scaffolded by Privacy Policy Section 7 (Art. 6(1)(f)
GDPR legitimate interest, with Art. 89 GDPR / § 27 BDSG research privilege
when results are published as research).

What it shows:
- **Volume KPIs** — unique users, conversations, free tokens generated,
  web searches, voice notes, photos, with WoW delta tags.
- **Growth** — cumulative users + DAU sparklines.
- **Engagement** — retention curve (cohort-averaged), per-user engagement
  buckets, conversation-depth (median/p95/mean + histogram).
- **Time-of-week heatmap** — 7×24 grid in Africa/Windhoek time.
- **How people use Ongiini.ai** — emergent use-case donut + top-topics list,
  produced by an LLM-driven two-pass clustering loop running on the same
  computer as the chat service.
- **Who uses Ongiini.ai** — five emergent panels (roles, regions,
  languages, family situation, current life context) extracted from mem0
  facts.
- **How it performs** — median + p95 latency, tool-call rate, truncation
  rate.

The qualitative passes use **no fixed taxonomy**. The LLM reads each user
message, produces a short generic label (e.g. *"yellowing maize leaves"*,
*"grade 11 chemistry homework"*), and a periodic synthesis pass clusters
those labels into named themes the data itself suggests. Categories are
emergent, not pre-decided.

Privacy guardrails (defence in depth):
- Extraction prompts include an explicit anti-PII prefix forbidding names,
  places below country level, ages, dates, specific numbers, or any
  identifying detail.
- A regex sanitiser (`webhook/app/stats/safety.py::sanitise_label`) drops
  any label containing 4-digit numbers, capitalised possessives
  (`Joseph's`), known Namibian town names, anything the existing PII
  scrubber catches, or labels longer than 80 characters. Failed labels
  are logged but never stored.
- Bucket floor of 5 for user-demographic categories: any cluster
  represented by fewer than 5 users folds into "Other".
- Cohort retention only published when at least one cohort has 5+ users
  with the required follow-up window.
- Opt-out via `/data/objections.txt` — MSISDNs listed here are filtered
  at the source so their data contributes to nothing.

Full deep-dive: [`docs/statistics.md`](docs/statistics.md).

## Out of scope for Phase 1

Voice replies (TTS), Oshiwambo via a translation layer, hard rate-limit
enforcement, Redis-backed memory, async queue.

## AI literacy and EU AI Act compliance

Ongiini is classified as a **limited-risk AI system** (a chatbot) under
Regulation (EU) 2024/1689 (the AI Act) and is subject to the transparency
obligations of Article 50, which become formally applicable on 2 August 2026.
It is **not** a high-risk AI system under Annex III and engages in **none** of
the prohibited practices listed in Article 5.

The operator is aware of the system's capabilities and limitations:

- Outputs are probabilistic — the model can be wrong, especially on numbers,
  dates, recent events, region-specific facts, and anything requiring
  professional expertise.
- Known model failure modes: hallucinated citations, occasional reasoning
  errors, language-specific quality variance (Afrikaans is supported but the
  model is most accurate in English).
- The system prompt explicitly directs the model to defer to qualified humans
  on medical, legal, financial and safety-critical questions, and to use
  Tavily web search for time-sensitive or location-specific questions.
- Every reply to a new user begins with an explicit AI disclosure to satisfy
  Article 50(1). See `webhook/app/llm.py` → `SYSTEM_PROMPT` → "FIRST MESSAGE
  DISCLOSURE".
- Memory (short-term JSON window + mem0 long-term layer) is bounded and
  user-controllable via the `whats_in_my_memory` and `delete_my_data` tools,
  exposed as natural-language prompts ("what do you remember about me?" /
  "delete my data") and operating in English and Afrikaans.
- The underlying model (Gemma 4 26B) is a general-purpose AI model provided
  by Google DeepMind. Under Article 25 of the AI Act, the operator accepts
  the provider responsibilities for the integrated Ongiini chatbot system
  built on top of it.
