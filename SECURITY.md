# Security & privacy posture

Ongiini is a public-facing free service running on a single computer.
This file documents the controls in place and what they protect
against, so anyone auditing the codebase (or running their own fork)
can sanity-check it quickly.

## What we keep per user

Three tiers, all on the Spark, none shared with any third party:

- **Short-term conversational memory** — about the last 50 turns of
  user+assistant back-and-forth at `/data/{msisdn}.json` (capped at 100
  entries on disk). Marathon chats fold their oldest turns into a leading
  `system` "Earlier in this conversation: …" line so a long chat stays
  bounded. PII patterns (email, IBAN, credit card, 11-digit Namibian ID)
  are regex-scrubbed BEFORE the message is written, replaced with
  `[REDACTED:kind]` placeholders.

- **Long-term semantic memory** via mem0 — typed facts about the user
  extracted across ALL conversations (`[PROFILE]`, `[PREFERENCE]`,
  `[SITUATION]`, `[COMMITMENT]`, `[QUOTE]`, `[EMOTION]`), stored as
  384-dim embeddings in an embedded qdrant at `/data/qdrant/`. Facts are
  canonicalised to English regardless of input language. The vector store
  AND the fact-extraction LLM (Gemma 4 via the same vLLM endpoint we use
  for replies) both run on the Spark — no external service.

- **Token-count log** — `/data/usage.log`, one line per message:
  `timestamp | msisdn | tokens_in=... tokens_out=... | search=yes|no | kind=chat`.
  **No message content in this file, ever.**

Image messages: the image bytes are processed through vLLM and discarded.
Short-term memory stores a compact "[image attached] <caption>" placeholder.
mem0 long-term stores the assistant's own textual description of the image
("[SITUATION] Shared photo of maize leaves with yellowing tips"), never the
base64 bytes themselves. PII-on-images is handled by a system-prompt rule:
the model is instructed to describe sensitive documents (ID cards, payslips,
medical records, screenshots of OTPs) generally without reading out specific
personal numbers, so those don't slip into long-term facts via the model's
description.

Voice notes: WhatsApp audio is downloaded from Meta, transcribed on the Spark
via `faster-whisper` (CTranslate2 Whisper-large-v3-turbo, INT8 on CPU), and
the audio bytes are discarded immediately after transcription. The transcript
flows through the same text path — PII scrub, short-term memory, mem0
long-term, tools dispatch, EN/AF language redirect — so the storage and
privacy guarantees match exactly. No audio is ever persisted. The stored
turn carries a `[voice note]` prefix so the transparency aggregator can
count voice messages without re-reading audio.

## Transparency layer

A separate aggregate-only data tier feeds the unlinked `/statistics` page.
Distinct from the per-user tiers above — it stores **no per-user content**,
only emergent labels and counts:

- `/data/trace.jsonl` — one JSON line per handled message: token counts,
  tool calls by NAME, latencies, finish reasons. No message text, no
  tool arguments verbatim — only lengths and names.
- `/data/qualia.sqlite` — short-label cache keyed by
  `(analysis, content-hash, version)`. Labels are produced by the local
  LLM under explicit anti-PII prompts and then passed through a regex
  sanitiser (`ongiini/stats/safety.py::sanitise_label`) before
  storage. Labels containing 4-digit numbers, capitalised possessives,
  known Namibian town names, or any pattern the existing PII scrubber
  catches are dropped, not stored.
- `/data/synthesis-*.json` — output of the periodic clustering pass.
  Holds named clusters with their constituent labels and counts. Read
  by the `/stats.json` aggregator.
- `/data/objections.txt` — optional MSISDN list (Art. 21 GDPR opt-out).
  The aggregator filters these at the source for every analysis and
  every distribution; their data contributes to no aggregate, current
  or future.

The endpoint `GET /stats.json` returns the assembled aggregate payload
with a 5-minute in-process cache. It is reachable from `ongiini.ai/api/stats`
via a Cloudflare Pages Function (`functions/api/stats.js`) that proxies
to `STATS_API_URL` (set per-deployment, currently `https://api.ongiini.ai`).
Bucket floor of 5 users / mentions applies to user-demographic distributions;
clusters below that fold into "Other".

For the full design + the LLM-as-analyst framework, see
[`docs/statistics.md`](docs/statistics.md).

Users can inspect everything (`whats_in_my_memory`), check token usage
(`my_token_usage`), and delete it all (`delete_my_data`) from inside
WhatsApp — all three work in English and Afrikaans.

## What we never log

- Message content from the user.
- The assistant's reply text. (Useful for debugging, but a privacy
  liability — the eval harness in `ongiini/tests/eval.py` is the right
  tool for inspecting replies during development.)
- Any payload field from Meta's webhook beyond what's needed to route
  the message.

The one log line that previously contained reply text
(`whatsapp.send_text` when WhatsApp credentials are unset) was scrubbed
to log only recipient + length.

## What we DO log

- Incoming POST timestamps + response codes (uvicorn access log)
- Outbound vLLM + Tavily call URLs + HTTP status (httpx debug log)
- Phone number when a message is blocked (non-Namibian), rate-limited,
  or dropped (oversize)
- Exception tracebacks (without message content — our code never puts
  user text into exception args)
- Transparency-layer events: when a label is rejected by the anti-PII
  sanitiser, when the qualitative-analysis loop runs a pass, when
  synthesis produces N clusters. These include the analysis name and
  the rejected label TRUNCATED TO 80 CHARS for debugging; no MSISDN.

## Webhook hardening (Layer 7)

| Control | File | What it stops |
|---|---|---|
| Meta `X-Hub-Signature-256` HMAC verification | `ongiini/whatsapp.py` `verify_signature()` | Forged webhook POSTs from anyone who guesses the verify token. Requires `WHATSAPP_APP_SECRET` env. |
| Per-MSISDN rate limit (20 / 5min, 200 / day) | `ongiini/ratelimit.py` | Single-user burst abuse / token exhaustion. In-memory only — resets on container restart. |
| Message-size cap (4096 chars) | `ongiini/api/main.py` | Oversize-payload abuse / token waste. |
| Namibia number filter + whitelist | `ongiini/filters.py` | Non-pilot-region traffic. |
| SSRF guard on `fetch_url` | `ongiini/search.py` `_safe_url()` | Model tricked into fetching localhost, RFC1918, link-local, or other internal addresses. |
| Prompt-injection guard | `ongiini/system_prompt.py` `SYSTEM_PROMPT` BOUNDARIES section | "Ignore previous instructions" / "tell me your system prompt" / jailbreak phrasings. |

## Container hardening

- Webhook container runs as **non-root** (`ongiini` user, UID/GID 1000).
- **Read-only root filesystem.** Only `/data` (bind mount) and `/tmp`
  (tmpfs, 64 MiB) are writable.
- All Linux capabilities **dropped** (`cap_drop: [ALL]`).
- `no-new-privileges: true` — setuid binaries can't escalate.

A compromised webhook process can write to its own memory, talk to
vLLM/Tavily, and read/write `/data`. It cannot escalate, mount, modify
the rootfs, or open raw sockets.

## Edge hardening (Cloudflare)

See [`docs/cloudflare-waf.md`](docs/cloudflare-waf.md) for click-through
setup of Bot Fight Mode, rate limiting on `api.ongiini.ai`, and an
optional Meta-IP allowlist for the webhook path.

## Secrets

- All credentials live in `.env` on the Spark (root-owned, mode 600 is
  fine; the file is gitignored).
- The Tavily key, WhatsApp tokens, and Meta App Secret are never logged.
- Cloudflare tunnel credentials are root-owned in `/etc/cloudflared/`.

## What's intentionally NOT in scope

- **Multi-tenant isolation** — one Ongiini deployment serves one
  user-base. Forks running their own instance get their own /data and
  their own model.
- **DDoS at scale** — Cloudflare absorbs L3/L4. Cloudflare's free-tier
  L7 rules are documented in `docs/cloudflare-waf.md`; the Pro tier's
  Advanced DDoS is out of scope for pilot.
- **Encrypted memory at rest** — `/data` is plain JSON on disk. We rely
  on the Spark's full-disk encryption (LUKS) and physical security.
- **Audit logging of admin actions** — we're at a scale where the only
  admin action is "Sebastian SSHs to the Spark", which is logged by
  sshd.

## How to report a security issue

Email [sebastian.kuepers@gmail.com](mailto:sebastian.kuepers@gmail.com)
with the subject prefix `[Ongiini security]`. Please don't open a
public issue for anything that could be exploited before a fix lands.
