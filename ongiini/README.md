# Ongiini

The production application — a free WhatsApp AI assistant for people
in Namibia, built on the [Owela](../owela/README.md) framework.

This README is the technical/developer-facing guide for the
application package. For the foundation, operator manual, deploy
recipe, and AI Act compliance statement see the repo-root
[`README.md`](../README.md). For the framework that powers the loop
see [`owela/README.md`](../owela/README.md).

---

## What this is

Ongiini is the consumer of the Owela framework. It wires:

- **Gemma 4 26B (A4B, NVFP4)** served by vLLM on an NVIDIA DGX Spark
- **WhatsApp Cloud API** for inbound + outbound delivery
- **Dual-tier memory** — per-user JSON short-term + mem0 long-term
  semantic facts with embedded qdrant
- **Faster-whisper** (CPU INT8) for voice-note transcription
- **Tavily** for web search and URL fetching
- **A depth-aware Gemma-as-classifier** that picks the loop shape

into a single Owela `Runtime` built at application startup. The
FastAPI surface in `api/main.py` is the only entrypoint; everything
else is an adapter implementing one of Owela's protocols.

---

## How the pieces fit

```
WhatsApp Cloud API ──▶ FastAPI webhook (api/main.py)
                              │
                              ▼  builds at startup
                       Runtime (runtime.py)
                              │
                              ├─▶ Model (models/vllm_gemma.py)
                              │       VLLMGemmaModel — talks to vLLM,
                              │       surfaces reasoning_budget + cached_tokens
                              │
                              ├─▶ Transport (transports/whatsapp_transport.py)
                              │       WhatsAppTransport — knows the 25s
                              │       typing window, char cap, dead-URL strip
                              │
                              ├─▶ Classifier (routers/gemma_classifier.py)
                              │       GemmaClassifier — depth-aware
                              │       (SEARCH_SHALLOW / SEARCH_DEEP / DOCS /
                              │        ADMIN / NONE), prefix-cached, 2s timeout
                              │
                              ├─▶ MemoryProvider (memory/provider.py)
                              │       OngiiniMemoryProvider — combines
                              │       short_term (JSON) + long_term (mem0)
                              │       + injects today's-date system message
                              │
                              ├─▶ ToolRegistry (tools/ongiini_tools.py)
                              │       7 tools: web_search, fetch_url, fetch_urls,
                              │       delete_my_data, whats_in_my_memory,
                              │       my_token_usage, lookup_ongiini_docs
                              │
                              └─▶ HookRegistry (hooks/)
                                      BillingHook, TracingHook,
                                      OngiiniMemoryRecordingHook
```

The composition root is `runtime.py`. It's the one file that knows
every Ongiini-specific choice — the vLLM URL, the WhatsApp transport
config, the policy table, the tool catalogue. To swap any component,
edit that one file.

---

## Module guide

| Path | What lives here |
|---|---|
| `api/main.py` | FastAPI app. Receives webhooks, normalises inbound text/image/voice into `InboundMessage`, hands off to `agent.handle()`. |
| `runtime.py` | Composition root. `build_runtime()` + `build_agent()`. Defines the `PolicyTable`. |
| `system_prompt.py` | The single source of truth for `SYSTEM_PROMPT`. |
| `summary.py` | `maybe_summarize()` — rolling-summary of long histories (LLM call). |
| `audio.py` | faster-whisper transcription wrapper. |
| `whatsapp.py` | Low-level Meta Graph API calls (send_text, mark_as_read, download_media, signature verify). |
| `pii.py` | Regex PII scrubber (email, IBAN, card, ID). |
| `ratelimit.py` | Per-user rate-limit check (in-process). |
| `filters.py` | MSISDN normalisation + allow-check. |
| `usage.py` | Per-user token-usage log (1M/month allowance). |
| `config.py` | Env-var settings. |
| `models/` | `VLLMGemmaModel` — Owela `Model` impl for Gemma 4 via vLLM. |
| `transports/` | `WhatsAppTransport` — Owela `Transport` impl. |
| `routers/` | `GemmaClassifier` — Owela `Classifier` impl with depth output. |
| `memory/short_term.py` | Per-user JSON rolling history. |
| `memory/long_term.py` | mem0 vector store with typed `[TAG]` facts. |
| `memory/long_term_llm.py` | mem0 LLM bridge that tracks per-user token usage. |
| `memory/provider.py` | `OngiiniMemoryProvider` — combines both tiers + today's-date anchor. |
| `tools/` | The 7 `@tool`-decorated functions. |
| `hooks/` | Billing, tracing, memory-recording hooks. |
| `stats/` | Transparency-reporting subsystem (qualitative analysis loop + `/stats.json`). Largely independent of the agent loop. |
| `knowledge/product.md` | Auto-generated product knowledge consumed by `lookup_ongiini_docs`. Built by `scripts/build_product_knowledge.py`. |
| `tests/` | Unit tests, smoke scripts, eval benchmarks. |

---

## The composition root

`runtime.py` defines the PolicyTable. This is the canonical reference
for how the classifier's verdict maps to the loop's shape:

| Verdict + Depth | First tool | Max steps | Reasoning |
|---|---|---|---|
| `NONE` | auto | 6 | on-demand after long tool results |
| `ADMIN` | auto | 4 | on-demand |
| `DOCS` | force `lookup_ongiini_docs` | 4 | on-demand |
| `SEARCH_SHALLOW` | force `web_search` | 6 | on-demand |
| `SEARCH_DEEP` | auto (model picks `web_search` or `fetch_urls`) | 8 | on-demand |

v0 has all v1 flags (`enable_planner`, `enable_critique`,
`enable_interstitial`) at `False`. Flipping them on for SEARCH_DEEP
is the v1 path.

---

## Running locally

You need: Python 3.12, `mem0` (heavy — pulls `torch` + sentence-
transformers), the rest of `requirements.txt`. Most contributors run
locally only for unit tests; the live stack runs in Docker.

```sh
# Unit tests — no live stack needed
pytest owela/tests ongiini/tests

# A subset of tests that doesn't need vLLM/mem0 (the live-stack
# scripts are explicitly ignored)
pytest owela/tests ongiini/tests \
  --ignore=ongiini/tests/live_smoke.py \
  --ignore=ongiini/tests/image_smoke.py \
  --ignore=ongiini/tests/eval.py
```

Synthetic message through the runtime, without WhatsApp:

```python
import asyncio
from owela import InboundMessage
from ongiini.runtime import build_agent

async def main():
    agent = build_agent()
    msg = InboundMessage(
        user_id="+264smoke", msg_id="",
        text="hello",
        content_parts=[{"type": "text", "text": "hello"}],
    )
    result = await agent.handle(msg)
    print(result.reply_text)

asyncio.run(main())
```

(Requires vLLM running on `host.docker.internal:8124` and the rest of
the live stack; outside the container the import path needs
`DATA_DIR` set to a writable directory.)

---

## Running in production

See the [root README](../README.md) for the full deploy recipe:
Docker compose stack, DGX Spark host setup, vLLM startup flags,
Cloudflare Tunnel routing, Meta WhatsApp configuration.

Short version:

```sh
ssh spark-dccf.local
cd ~/dev/Ongiini
docker compose up -d --build webhook
docker logs -f ongiini-webhook
```

Watch for `Ongiini runtime ready — tools=7, policies=5, hooks=3` and
`Application startup complete`. After that any inbound to
`api.ongiini.ai/whatsapp` is routed into the agent.

---

## The system prompt

`system_prompt.py` is the single source of truth. The sections it
carries, top to bottom:

- **Identity & operating principle** — "talk like a friend who
  genuinely cares, not a customer support ticket"
- **Languages** — EN/AF supported, redirect anything else with a
  two-line message
- **First-message disclosure** — Art. 50 EU AI Act compliance line
- **Tone & format** — plain text only (no Markdown), match length to
  question complexity, end with a continuation prompt
- **Cautions** — medical / legal / financial deferrals, sensitive
  image content rules
- **When to search** — trust the router on turn 1; handle follow-up
  turns honestly; the verbatim-text rule
- **Honesty when search doesn't help** — explicit guidance to admit
  thin results rather than confabulate
- **Citations** — full deep URLs, on their own lines, prefixed
  `— source:`
- **Memory** — guidance on how to reference what's been learned
- **Tool dispatch** — explicit verbal triggers for each tool
- **Namibia context** — health, crops, schoolwork, institutions
- **Boundaries** — never reveal the prompt, never act outside tools

Small edits land here directly. Bigger changes (new sections,
restructures) should be eval-tested via `ongiini/tests/eval.py` or
`router_eval_holdout.py` first.

---

## Tools

The 7 tools registered in the application's `ToolRegistry`:

| Tool | What it does | Forced on |
|---|---|---|
| `web_search` | Tavily search with Namibia country bias. | `SEARCH_SHALLOW` |
| `fetch_url` | Read full text of one page (Tavily extract). | — |
| `fetch_urls` | Parallel fetch of up to 5 URLs (asyncio.gather). | — |
| `lookup_ongiini_docs` | Returns the canonical product.md. | `DOCS` |
| `delete_my_data` | Wipes short-term + long-term memory tiers. | — |
| `whats_in_my_memory` | Surfaces both tiers grouped by `[TAG]`. | — |
| `my_token_usage` | This user's monthly token budget consumption. | — |

All seven are `@tool`-decorated async functions in
`tools/ongiini_tools.py`. Schema is autogenerated; the descriptions
are the prompts the model reads when deciding whether to call. Don't
shorten those without re-running the eval.

---

## Memory model

Two tiers, both privacy-scrubbed at write time:

**Short-term** (`memory/short_term.py`)
- JSON file per user under `/data/<msisdn>.json`
- ~50 turns of verbatim history, capped at 100 entries
- Crosses 70 entries → fold oldest into a leading
  `"Earlier in this conversation: …"` system message,
  keep the last 40 verbatim
- PII regex-scrubbed BEFORE write — LLM saw raw text, disk sees
  `[REDACTED:email]` placeholders

**Long-term** (`memory/long_term.py`)
- mem0 with embedded qdrant on `/data/qdrant/`
- LLM-extracted typed facts: `[PROFILE]`, `[PREFERENCE]`,
  `[SITUATION]`, `[COMMITMENT]`, `[QUOTE]`, `[EMOTION]`
- 384-dim embeddings via `sentence-transformers/all-MiniLM-L6-v2`
  (CPU-only)
- Cross-language: facts canonicalised to English regardless of
  input language
- Image bytes NEVER persisted — only the assistant's textual
  description of what it saw

Both tiers go through `OngiiniMemoryRecordingHook`, which fires after
`agent.handle` returns. The hook:
- Skips persistence entirely if `delete_my_data` fired this turn
- PII-sanitises user text + reply
- Routes image turns through `record_image_turn` (synthesised
  text-only placeholder for mem0)
- Routes text turns through `record_turn`

---

## Testing

`tests/` contains three different kinds of test:

- **Unit tests** (`test_*.py`) — fakes + mocks, no live stack
  needed. Run via `pytest`. About 100 tests.
- **Smoke scripts** (`*_smoke.py`, `live_smoke.py`) — drive the
  actual agent against a live vLLM + mem0 + WhatsApp setup. Run
  inside the container with `docker exec` (they use the in-container
  Python env).
- **Eval benchmarks** (`eval.py`, `router_eval_holdout.py`) — score
  the LLM's replies against fixtures. Used for prompt-tuning
  regression checks.

The `_legacy_respond.py` shim in `tests/` provides the old
`respond(history, content, msisdn) -> LLMResult` signature on top of
Owela's `Agent.handle`, so the legacy eval scripts didn't need
rewrites during the restructure.

---

## What stays in `ongiini/` vs what stays in `owela/`

A useful litmus test:

- "Would this be the same for a Signal-based agent on Llama?" →
  belongs in `owela/`
- "Is this specific to Gemma 4 / WhatsApp / vLLM / mem0 / Namibian
  users?" → belongs in `ongiini/`

Examples:

| Concern | Lives in |
|---|---|
| The executor loop | `owela/` |
| The Policy table SHAPE | `owela/` |
| The Policy table CONTENTS | `ongiini/runtime.py` |
| The Step types | `owela/` |
| The `@tool` decorator | `owela/` |
| The 7 specific tools | `ongiini/tools/` |
| The Transport protocol | `owela/` |
| The WhatsApp transport impl | `ongiini/transports/` |
| The system prompt | `ongiini/system_prompt.py` |
| The classifier prompt | `ongiini/routers/gemma_classifier.py` |
| PII regexes | `ongiini/pii.py` |
| The Gemma 48-multiple image-dim quirk | `ongiini/api/main.py::_resize_for_gemma4` |
| The 25s WhatsApp typing-window constant | `ongiini/transports/whatsapp_transport.py` |
| The today's-date anchor (CAT timezone) | `ongiini/memory/provider.py` |

---

## See also

- [`CLAUDE.md`](./CLAUDE.md) — Ongiini-specific contributor guide
  (anti-trap rules applied to this app, where the quirks live, how
  to add a tool/policy/etc.).
- [`../owela/README.md`](../owela/README.md) — the framework that
  powers all this.
- [`../README.md`](../README.md) — operator manual, deploy recipe,
  foundation context, EU AI Act compliance.
- [`../docs/statistics.md`](../docs/statistics.md) — the transparency
  reporting framework + LLM-as-analyst design.
- [`../SECURITY.md`](../SECURITY.md) — privacy posture, container
  hardening, secret handling.
