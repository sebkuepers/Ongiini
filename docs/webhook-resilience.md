# Webhook resilience during outages — design sketch

**Status:** not built. This file describes what we'd do to make
incoming WhatsApp messages survive an extended Spark outage. For the
pilot we rely on Meta's built-in retry behaviour, which is good enough
for the typical wifi-drop / brief-restart cases.

## What happens today during an outage

When the Spark is unreachable, `api.ongiini.ai/whatsapp` fails. Meta's
behaviour:

- On 5xx or network failure, Meta retries the webhook delivery.
- The retry window is roughly **24 hours** with exponential backoff
  (~7 attempts spread across the window).
- After 24 h, the delivery is given up and the message is **lost** as
  far as our webhook is concerned. The user's WhatsApp client will
  show the message as delivered to the business number, but we never
  saw it.

For a household wifi outage (minutes to a few hours), Meta's retry
covers us — messages get delivered when the Spark comes back. For an
outage longer than ~24 h (hardware migration, extended trip, etc.)
some messages will be silently dropped.

## What the design would look like

Inject a small always-on receiver between Meta and the Spark.
Cloudflare Workers + Queues fit because:

- Workers run at every CF PoP, never offline.
- Queues persist messages for up to **4 days** by default.
- Both have generous free tiers (~10 M requests/month for Workers,
  1 M queue ops/month for Queues — well above our pilot scale).
- We're already on Cloudflare's edge for the tunnel + Pages, so this
  doesn't add a new vendor.

### Topology

```
Meta WhatsApp Cloud API
        │
        ▼
api.ongiini.ai
        │
   ┌────┴───────────────────────────────────────────────────┐
   │  Cloudflare Worker  (always up at the edge)            │
   │  1. Verify X-Hub-Signature-256 (App Secret in CF env)  │
   │  2. Push raw body + msisdn to Queue "wa-incoming"      │
   │  3. Return 200 OK to Meta immediately                  │
   └────┬───────────────────────────────────────────────────┘
        │
        ▼
   Queue "wa-incoming"  (persisted at the edge, ~4 day TTL)
        │
        ▼ (Spark consumer drains)
   ┌─────────────────────────────────────────────┐
   │  Spark webhook /whatsapp/queue-consume       │
   │  (called by Worker or polled by Spark)       │
   │  Replays the original POST shape to the      │
   │  existing handle_message logic.              │
   └─────────────────────────────────────────────┘
```

### Two consumer patterns

**Push (Cloudflare delivers to Spark)** — the Queue is configured with
the Spark as an HTTP consumer. When the Spark is reachable, Cloudflare
delivers messages with normal retry semantics. When the Spark is down,
messages accumulate in the queue and CF re-tries until consumed or the
4-day TTL elapses. Simpler to reason about, no new endpoint to expose.

**Pull (Spark polls)** — the Spark runs a small consumer process that
periodically calls Cloudflare's Queue HTTP API to fetch + ack
messages. More control on the Spark side (rate limit ingestion when
warming up after an outage), but more code to maintain.

Push is simpler and the default I'd pick.

### What changes on the Spark side

The webhook's existing `POST /whatsapp` handler stays. We add:

- A new endpoint, e.g. `POST /whatsapp/queue-consume`, that receives
  the Queue-delivered batch payload, unwraps each original Meta event,
  and calls `handle_message` for each.
- Same rate limit, same memory persistence, same model path.
- An idempotency check using Meta's `message_id` so a redelivered
  message doesn't get answered twice.

The original `POST /whatsapp` keeps working for normal pilot use; the
queued path is an alternate entry point. We'd switch Meta's callback
URL to the Worker once we're confident.

### Cost / complexity envelope

- 1 Worker, ~60 lines of TypeScript.
- 1 Queue (no config needed beyond name + TTL).
- ~50 lines of new Python on the webhook side (queue-consume + idempotency).
- Free tier sufficient for the projected pilot scale (10 K-100 K
  messages/month).

### When to build this

When **any** of the following becomes true:

- We're routinely seeing >12 h outages (e.g. hardware in transit to Namibia).
- We hit the Meta retry window and lose messages we wanted to keep.
- Real users report "I messaged Ongiini and never heard back" — meaning
  Meta dropped the delivery after retries exhausted.

Until then, the **status indicator on the page** + Meta's retry
window cover the common case. We're explicit with users that this is a
pilot running on a single computer.

## Out of scope for this design

- **Sending replies during an outage** — not possible. The reply
  requires the LLM. When the Spark is back, queued messages get
  answered with full context; the user just sees a delayed reply.
- **Multi-region failover** — would require running the model in two
  places, which contradicts the "one open computer in Namibia"
  identity of the project.
- **Email/SMS fallback** — out of scope; if WhatsApp is reachable, so
  is the queue.
