# Security & privacy posture

Ongiini is a public-facing free service running on a single computer.
This file documents the controls in place and what they protect
against, so anyone auditing the codebase (or running their own fork)
can sanity-check it quickly.

## What we keep per user

- The last 10 messages from each phone number, stored as
  `/data/{msisdn}.json` on the Spark. Used as short-term conversational
  memory.
- A token-count line per message in `/data/usage.log` —
  `timestamp | msisdn | tokens_in=... tokens_out=... | search=yes|no`.
  No message content in this file, ever.

## What we never log

- Message content from the user.
- The assistant's reply text. (Useful for debugging, but a privacy
  liability — the eval harness in `webhook/tests/eval.py` is the right
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

## Webhook hardening (Layer 7)

| Control | File | What it stops |
|---|---|---|
| Meta `X-Hub-Signature-256` HMAC verification | `webhook/app/whatsapp.py` `verify_signature()` | Forged webhook POSTs from anyone who guesses the verify token. Requires `WHATSAPP_APP_SECRET` env. |
| Per-MSISDN rate limit (20 / 5min, 200 / day) | `webhook/app/ratelimit.py` | Single-user burst abuse / token exhaustion. In-memory only — resets on container restart. |
| Message-size cap (4096 chars) | `webhook/app/main.py` | Oversize-payload abuse / token waste. |
| Namibia number filter + whitelist | `webhook/app/filters.py` | Non-pilot-region traffic. |
| SSRF guard on `fetch_url` | `webhook/app/search.py` `_safe_url()` | Model tricked into fetching localhost, RFC1918, link-local, or other internal addresses. |
| Prompt-injection guard | `webhook/app/llm.py` `SYSTEM_PROMPT` BOUNDARIES section | "Ignore previous instructions" / "tell me your system prompt" / jailbreak phrasings. |

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
