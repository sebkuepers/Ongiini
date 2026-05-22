# Ongiini repo — Claude Code anchor

You are working in the Ongiini monorepo: a free WhatsApp AI assistant
for people in Namibia, plus the chat-agent framework it's built on.

This file is a tiny repo-wide anchor. The detailed contributor guides
live closer to the code:

- [`owela/CLAUDE.md`](./owela/CLAUDE.md) — the framework
- [`ongiini/CLAUDE.md`](./ongiini/CLAUDE.md) — the application

Open whichever directory you're working in and read its CLAUDE.md
before changing code there.

---

## Map

```
owela/         Framework — pure library, no product specifics
ongiini/       Application — Gemma 4 + WhatsApp + mem0 wired into Owela
website/       Cloudflare Pages site (vanilla HTML)
functions/     Cloudflare Pages Functions (api/stats proxy)
foundation/    Foundation site (gitignored)
deploy/        DGX Spark host scripts (vLLM restart, wifi watchdog)
docs/          Architecture + ops markdown
scripts/       Repo scripts (product knowledge builder, pre-commit hook)
data/          Bind-mount target for runtime data (gitignored)
Dockerfile     Webhook container — repo root build context
docker-compose.yml   webhook + website services
```

Read the [root README](./README.md) for the operator manual,
deployment recipe, and AI Act compliance posture.

---

## Where to look for what

| I want to change… | …open this |
|---|---|
| the agent loop / executor | `owela/executor.py` |
| how a turn's shape is decided | `owela/policy.py` + `ongiini/runtime.py` (the table) |
| a Gemma 4 quirk (image dims, reasoning) | `ongiini/models/vllm_gemma.py` or `ongiini/api/main.py` |
| a WhatsApp behaviour (sending, formatting, dead URLs) | `ongiini/transports/whatsapp_transport.py` |
| a tool the model can call | `ongiini/tools/ongiini_tools.py` |
| the system prompt | `ongiini/system_prompt.py` |
| classifier prompt or depth output | `ongiini/routers/gemma_classifier.py` |
| short-term memory (per-user JSON history) | `ongiini/memory/short_term.py` |
| long-term mem0 facts | `ongiini/memory/long_term.py` |
| PII scrubbing | `ongiini/pii.py` |
| Token-usage accounting | `ongiini/usage.py` + `ongiini/hooks/billing_hook.py` |
| Tracing format | `ongiini/hooks/tracing_hook.py` |
| Statistics rendering | `ongiini/stats/` + `website/statistics/` |
| vLLM startup flags / Gemma command | `deploy/spark/` |
| website copy | `website/*.html` |
| product knowledge (lookup_ongiini_docs source) | `website/*.html` → regenerate via `scripts/build_product_knowledge.py` |

---

## Common commands

```sh
# Run unit tests (no live stack needed)
pytest owela/tests ongiini/tests

# Build + start the webhook container (locally or on Spark)
docker compose up -d --build webhook

# SSH to the Spark host
ssh spark-dccf.local

# Tail webhook logs in production
ssh spark-dccf.local "docker logs -f ongiini-webhook"

# Regenerate the product knowledge after editing website HTML
python3 scripts/build_product_knowledge.py

# Check product.md is in sync with website HTML (CI does this too)
python3 scripts/build_product_knowledge.py --check
```

---

## Conventions

- **Python 3.12** — match the container.
- **pytest** for tests, no other test framework.
- **Plain dataclasses** preferred over Pydantic inside `owela/`.
  Pydantic OK inside `ongiini/` if it simplifies a contract.
- **Soft-fail** on hooks, memory writes, and transport-side
  ancillary calls (mark_as_read, send_interstitial). Never raise
  from a Hook. Never raise from `transport.acknowledge`.
- **Never log message content** — `tracing.jsonl` and `usage.log`
  store only lengths, names, durations, counts. Reply text never
  hits a log.
- **PII scrub at write time** — the LLM sees raw text; disk + mem0
  see `[REDACTED:kind]` placeholders.
- **Image bytes never persisted** — only the assistant's textual
  description of what it saw.
- **Force-pushes to main are blocked** — for good reason. Use
  follow-up commits, not `--amend` after publish.

---

## Pointers

- [`owela/CLAUDE.md`](./owela/CLAUDE.md) — framework extension guide,
  the eight anti-trap principles.
- [`ongiini/CLAUDE.md`](./ongiini/CLAUDE.md) — application-side
  contributor guide.
- [`SECURITY.md`](./SECURITY.md) — privacy posture, container
  hardening, secret handling.
- [`docs/statistics.md`](./docs/statistics.md) — transparency
  reporting framework + LLM-as-analyst design.
- [`docs/webhook-resilience.md`](./docs/webhook-resilience.md) —
  reliability patterns (fire-and-forget, msg-id dedup, etc.).
