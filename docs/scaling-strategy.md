# Scaling & Reliability Strategy

**Status:** plan, not built. Draft 2026-05-24 in response to early growth
(11 → 22 → 107 users on three consecutive mornings).

## The situation

Ongiini's growth curve over the first three mornings:

| Morning | Users |
|---|---|
| Day 1 | 11 |
| Day 2 | 22 |
| Day 3 | 107 |

That's ~5× growth on the latest day. If the curve continues, day 4 lands
in the hundreds, day 5 in four-digit territory. Current architecture is
**single-host**: a DGX Spark hosts vLLM (Gemma 4 26B NVFP4), the FastAPI
webhook, mem0/qdrant, faster-whisper, nginx, the Cloudflare tunnel
terminus, and the monitoring sidecars. Hardware lives at home on
residential power + internet — no UPS, no cellular backup. Spare
capacity available: **2 Mac minis, M2/M3/M4, 16 GB+ RAM each.**

We need a strategy that:

1. **Prevents multi-hour outages** by isolating failure domains and
   ensuring graceful degradation when something is down.
2. **Buys runway** for 5–10× growth without panic engineering during a
   Sunday-morning surge.
3. **Stays minimal** — no Kubernetes, no service mesh, no premature
   multi-region. Two-host + warm spare is the ceiling for this plan.
4. **Honest about hard limits**: Gemma 4 26B NVFP4 only runs on the
   Spark's GPU. Mac minis can run the webhook, mem0/qdrant, whisper,
   and proxy requests to a remote Gemma 4 API — but they cannot replace
   the Spark for local inference.

## What we have today

The Spark runs **everything**: `gemma4-vllm` (8 GPUs, 70% VRAM),
`ongiini-webhook` (FastAPI, CPU only, bind-mounts `/data`),
`ongiini-website` (nginx), `ongiini-autoheal`, `ongiini-cadvisor`,
`ongiini-node-exporter`. State all on Spark local disk: `mem0/`, per-
msisdn `short_term/*.json`, `source_index/*.json`, unbounded
`trace.jsonl`, unbounded `usage.log`, `hf_cache/`, opt-in
`revise_eval/`.

**No backups. No log rotation. No graceful-degrade reply when vLLM is
down. No external metrics/alerts. Rate-limit state is in-memory
(resets on restart).** Webhook now has a healthcheck + Autoheal as of
2026-05-24. WiFi watchdog exists. Cloudflare Pages serves the static
site + a Function proxying `/api/stats`.

## The strategy in one line

**Spark becomes a pure inference appliance. Webhook + state move to
Mac Mini A. Mac Mini B is the warm spare AND the backup target. When
the Spark is unreachable, the webhook fails over to a remote Gemma 4
API so users keep getting real replies instead of "please retry".**

---

## Phase 0 — Survival hygiene (this week, ~1–2 days work)

Done BEFORE any architecture change. These are bugs masquerading as
features and they will bite at current growth.

### 0a. Log rotation
- `trace.jsonl` and `usage.log` grow unbounded. At 100 turns/day ×
  5 KB/turn ≈ 180 MB/year *per file*. At 1 000 turns/day, 1.8 GB/year.
  Will fill the Spark disk eventually.
- Add a logrotate config OR code-side rotation in `BillingHook` and
  `TracingHook` (rotate when file > N MB, keep K back).
- Recommend: logrotate in `deploy/spark/`, rotate daily, keep 14 days,
  compress old.

### 0b. Daily backup to Mac Mini B
- One `rsync` cron on Mac B pulling `/data/` from Spark every night.
  Retain 7 daily + 4 weekly snapshots.
- Critical: the 2026-05-24 mem0 warning (*"qdrant point count 9
  significantly below expected 101"*) shows we're ALREADY losing facts.
  Without backups one bad bug = total memory loss.
- Exclude `hf_cache/` (re-downloadable) and `revise_eval/*.json` if
  disk is tight (regenerable from new traffic).

### 0c. Graceful "overloaded / unavailable" reply
- Today: if vLLM hangs / fails, the webhook eventually returns nothing
  and the user sees silence. Meta retries the inbound webhook; we keep
  failing; users keep waiting.
- New: when the model adapter raises after retries OR exceeds a hard
  budget (e.g. 60 s), the transport sends *"I'm having trouble right
  now — please send your question again in a few minutes."* Tag it so
  it doesn't pollute conversation history.
- This is the **bridge** to Phase 2's API fallback. Once the fallback
  is wired, 0c only fires when BOTH the primary AND the fallback are
  unreachable (rare).

### 0d. Operator dashboard (minimal)
- cAdvisor + node-exporter are ALREADY running but nothing scrapes
  them. Add a tiny Prometheus + Grafana on Mac B (or on the operator's
  laptop). Scrape Spark's exporters across the LAN.
- One dashboard with: vLLM GPU utilisation, webhook RSS + thread count,
  request rate, response p50/p95, error rate.
- Without this, "the bot is slow today" is operator-anecdote, not
  signal.

### 0e. Per-msisdn idempotency on outbound sends
- Today: WhatsApp `send_text` retries 3× but has no request ID. If
  Meta's ACK is lost, the retry sends the message again — user sees
  duplicate.
- Add a deterministic request ID per `(msg_id, attempt)` plus a local
  "last-sent" cache to detect post-send duplicates.
- Small change, prevents user-visible weirdness under load.

**Phase 0 ships in commits, no architecture change.** ≤2 days work.

---

## Phase 1 — Split inference from everything else (next 1–2 weeks)

The big architectural shift. Goal:

- **Spark = vLLM only.** Plus autoheal, cAdvisor, node-exporter, WiFi
  watchdog. Nothing else.
- **Mac Mini A = webhook host.** Runs `ongiini-webhook`,
  `ongiini-website`, mem0/qdrant, faster-whisper, Cloudflare tunnel
  terminus. Owns all of `/data`.
- **Mac Mini B = warm spare + backup target.** Receives nightly `/data`
  rsync from A. Has the same Docker image pulled but not running. Can
  be promoted manually within ~5 minutes.

### Why this split

1. **Failure isolation.** A webhook wedge today is risky because
   anything touching `/data` (mem0 writes, hook persistence) can
   corrupt state the inference loop depends on. After split, webhook
   crashes don't touch the GPU process and vice versa.
2. **Spark GPU dedicated.** vLLM gets the full host. CPU contention
   with the webhook/whisper/mem0 disappears.
3. **Backup geography.** State leaves the Spark every night. Mem0 data-
   loss stops being a "manually restore from nothing" situation.
4. **Headroom for inference scaling.** Once Spark is pure inference, we
   can bump `--max-num-seqs` (currently 16), tune batching, experiment
   with smaller-context modes — without risking the webhook.

### What changes in code/config

| Area | Change |
|---|---|
| `docker-compose.yml` | Split into `compose.spark.yaml` (vLLM + autoheal + cadvisor + node-exporter) and `compose.mac.yaml` (webhook + website + tunnel). |
| `ongiini/config.py` | `VLLM_BASE_URL` already exists — point at Spark over LAN: `http://<spark-lan-ip>:8124/v1`. |
| `.env` on Mac A | Add `VLLM_BASE_URL`, copy `WHATSAPP_*`, `TAVILY_API_KEY`, `DATA_DIR=/data`. |
| Cloudflare tunnel | Re-point to Mac A's `127.0.0.1:8445`. Meta webhook URL unchanged. |
| `deploy/mac/` | **NEW**: bring-up script, README, launchd unit for `docker compose up -d`. |
| `deploy/spark/` | Trim to vLLM-only mode. Keep watchdogs. |
| `data/` migration | One-time rsync from Spark to Mac A, with webhook stopped on Spark first. Window: <5 min downtime. |

### Cutover plan

1. **Prep Mac A**: install Docker, pull repo, build image, configure
   `.env`, test `docker compose up` against an empty `/data`.
2. **LAN connectivity**: confirm Mac A can reach `<spark-ip>:8124` and
   back. WireGuard tunnel between hosts is optional but recommended.
3. **First sync**: `rsync` Spark `/data/` → Mac A `/data/` while Spark
   webhook is still serving. Most data syncs OK.
4. **Cutover window** (low-traffic, e.g. 03:00 CAT): stop webhook on
   Spark, do final delta `rsync`, start webhook on Mac A, flip
   Cloudflare tunnel to Mac A. Total: ~2 minutes.
5. **Verify**: send a test WhatsApp message; confirm reply arrives,
   `trace.jsonl` grows on Mac A, `mem0` writes work.
6. **Spark cleanup**: stop the webhook container on Spark, leave the
   image present as emergency-fallback. Spark now runs ONLY
   `gemma4-vllm` + monitoring.

### What the user sees

Nothing. Same WhatsApp number, same replies. Maybe ~50–100 ms of extra
latency per turn from the webhook→Spark LAN hop, well within noise.

### Risks + mitigations

- **Network blip between Mac A and Spark** → vLLM call fails. Phase 0c
  covers this, Phase 2's API fallback covers it better.
- **State drift during migration** → final `rsync --delete` during the
  cutover window with webhook stopped.
- **Mac A's tunnel auth drifts** → keep Spark's tunnel config as a
  backup; if Mac A fails to come up, flip back to Spark in <5 min.

---

## Phase 2 — Gemma 4 API fallback + warm-spare promotion

Trigger this phase when EITHER:
- A real Spark outage exceeds 10 minutes user-visible, OR
- Sustained ≥300 users/day.

### 2a. Gemma 4 hosted-API fallback (the key new piece)

**The insight:** if the Spark is down but Mac A is up, the webhook can
still call a hosted Gemma 4 API (Google AI Studio, Vertex AI, or
OpenRouter — all offer Gemma 4 endpoints). Same model family, OpenAI-
compatible API, fully external to our hardware. Users get a real reply
instead of "please retry".

**Implementation sketch:**

| Piece | Detail |
|---|---|
| `ongiini/models/vllm_gemma.py` (or new `gemma4_api_fallback.py`) | Add a secondary model client. On primary failure (timeout, connection refused, 5xx), retry once against the fallback `base_url` + `api_key`. |
| `ongiini/config.py` | New env vars: `GEMMA4_FALLBACK_BASE_URL`, `GEMMA4_FALLBACK_API_KEY`, `GEMMA4_FALLBACK_MODEL` (e.g. `google/gemma-4-26b-it` or whatever the provider calls it). Empty = feature off. |
| `OngiiniMemoryProvider` | No change — context assembly is the same. |
| Tools (Tavily, fetch_url) | No change — they were already external. |
| Reply post-processing | When fallback fires, prepend a small system notice: *"(I'm running on backup mode right now — replies may be slightly slower or simpler. Normal service will resume shortly.)"* So users understand any quality drop and know it's temporary. |
| `usage.log` | Tag fallback calls so they show up separately in cost/usage stats. |
| Tracing | `ModelCallStep.attrs["fallback_used"] = True` for audit + observability. |

**Cost considerations:**

| Provider | Notes |
|---|---|
| **Google AI Studio** | Free tier ~1 500 RPM on Gemma 4. Easy signup. Likely the right starting point. |
| **Google Vertex AI** | Paid, enterprise-grade SLA. Justified if outages persist or volumes grow. |
| **OpenRouter** | A few cents per million tokens. Provider-agnostic — handy if we want to A/B the fallback model. |

For our scale, free tier covers it. Even paid is trivial (<$1/month at
current volumes, since the fallback only fires during outages).

**Privacy + compliance trade-off** — this is the important one:

- The foundation's published stance is *"no US cloud provider sits in
  the pipeline."* Google AI Studio / Vertex AI are US-cloud-adjacent
  (Google routes through US regions for Gemma serving).
- **The fallback only fires when the local Spark is down** — i.e.,
  during outages. Normal traffic never touches it.
- This trade-off MUST be disclosed transparently:
  1. Add a section to `website/privacy.html` and the WhatsApp first-
     message disclosure: *"If our local hardware is unavailable, your
     message may be processed by Google's hosted Gemma API as a
     temporary fallback. Normal operation runs entirely on our hardware
     in Namibia."*
  2. Document the fallback in `docs/webhook-resilience.md`.
  3. Two defensible stances; pick deliberately:
     - **Mission-purity**: refuse the fallback. Accept multi-hour
       outages when Spark is down. Users get the "please retry"
       message.
     - **User-first**: accept the disclosed fallback. Outages drop
       from hours to ~zero user-visible impact.
  - The plan recommends **user-first with transparent disclosure**,
    because losing every user's session for hours is a worse violation
    of the mission than briefly routing through a US cloud with clear
    notice.

**Fallback chain:**

```
1. Try local vLLM on Spark              → succeed → reply
                                        ↓ fail
2. Try Gemma 4 hosted API (Google AI)   → succeed → reply WITH "backup mode" notice
                                        ↓ fail
3. Return Phase 0c's "please retry"     → user retries later
```

### 2b. Mac B as promotable spare
- Mac B already receives nightly `/data` rsync (Phase 0b). Add a
  15-min cron `rsync` to keep it within a quarter-hour of A.
- Document a single promote script: `deploy/mac/promote.sh` — stops
  services on A (if reachable), starts on B, flips Cloudflare tunnel.
  ~5 min RTO, ~15 min RPO.

### 2c. Automatic ingress failover (nice-to-have)
- A small Cloudflare Worker (free tier) sits in front of the Cloudflare
  tunnel. Pings `/health` on Mac A every 30 s. If A is unreachable for
  >2 minutes, swaps the tunnel target to B.
- Cheaper alternative: manual flip via the promote script. Worker is
  optional for the 300-users level.

---

## Phase 3 — When 2 Mac minis + API fallback aren't enough (deferred)

Document only — don't build. Signals to revisit:

- Sustained ≥1 000 users/day
- vLLM queue depth > N consistently
- Latency p95 climbs over 45 s
- API fallback bills meaningfully (>$50/month → outages are too frequent)

Then consider:
- **Bigger vLLM batch / second GPU host** if growth justifies hardware
  spend.
- **External rate-limit store** (Cloudflare D1 free tier) so multi-host
  webhooks share limits.
- **Sharded mem0** or move to a managed vector store — only if mem0
  actually earns its cost (see v1.7-eval work).
- **Geographic redundancy** (Mac mini at a different physical site) —
  only matters if a single-home outage is unacceptable.

---

## Critical files

| File | Change |
|---|---|
| `deploy/spark/logrotate.conf` | **NEW**: rotate `trace.jsonl`, `usage.log` daily, keep 14, compress |
| `deploy/mac/backup.sh` | **NEW**: nightly rsync from Spark `/data/` to Mac B with retention |
| `deploy/mac/README.md` | **NEW**: Mac bring-up runbook |
| `deploy/mac/compose.yaml` | **NEW**: Mac-side docker-compose (webhook + website + tunnel) |
| `deploy/mac/promote.sh` | **NEW** (Phase 2): promote Mac B from spare to primary |
| `deploy/spark/compose.yaml` | **MODIFY**: trim to vLLM + monitoring only |
| `docker-compose.yml` (root) | **REMOVE** or replace with dev-only single-host variant |
| `ongiini/models/vllm_gemma.py` | **MODIFY** (Phase 2a): primary→fallback chain |
| `ongiini/config.py` | **MODIFY** (Phase 2a): add `GEMMA4_FALLBACK_*` env vars |
| `ongiini/transports/whatsapp_transport.py` | Add "model unavailable" message helper + "backup mode" notice prefix |
| `ongiini/api/main.py` | Wire the unavailable-message path when both primary and fallback raise |
| `ongiini/whatsapp.py` | Add per-`(msg_id, attempt)` request ID for outbound sends |
| `website/privacy.html` + `ongiini/system_prompt.py` | Disclose the fallback in privacy policy + first-message text |
| `scripts/grafana/` or `monitoring/` | **NEW**: Grafana dashboard JSON |

**Owela framework**: untouched. All scaling work is in `ongiini/`
(small additions to model adapter + transport + main.py) and `deploy/`.

---

## Verification

### Phase 0
- Confirm `trace.jsonl` rotates after manually filling past threshold;
  old files compressed; beyond retention removed.
- Trigger a backup manually (`bash deploy/mac/backup.sh`); confirm Mac
  B has a fresh snapshot with expected file count.
- Kill vLLM on Spark; send a WhatsApp message; confirm user gets the
  "please retry" message within ~30 s, not silence.
- Open the Grafana dashboard; confirm vLLM GPU% + webhook RSS +
  request rate are live.

### Phase 1
- After cutover, send 10 mixed WhatsApp messages (casual,
  SEARCH_SHALLOW, SEARCH_DEEP, image, audio). All must produce correct
  replies.
- Confirm `trace.jsonl` on Mac A grows; Spark `/data` is now read-only
  / empty.
- Confirm `nvidia-smi` on Spark shows ONLY vLLM using GPU.
- Stop the webhook on Mac A; confirm Meta retries land in the queue (or
  fail gracefully). Restart Mac A webhook; queue drains.
- Pull the Spark network cable; confirm Mac A returns "please retry"
  rather than silence (still pre-Phase 2).

### Phase 2
- Set `GEMMA4_FALLBACK_*` env on Mac A and restart.
- Send a normal WhatsApp message; confirm primary (local Spark) handles
  it; trace shows `fallback_used=False`.
- Stop vLLM on Spark; send another message; confirm fallback fires,
  reply includes the "backup mode" notice, trace shows
  `fallback_used=True`, `usage.log` tags the call as `fallback`.
- Disable the fallback API key temporarily, kill Spark vLLM; confirm
  the "please retry" message fires (chain step 3).
- Promote Mac B manually; confirm full functionality.

---

## What I'm NOT proposing

- **Kubernetes / service mesh.** Two hosts. Docker Compose is right-
  sized.
- **Moving local inference off the Spark.** Gemma 4 26B NVFP4 needs the
  Spark GPU. Mac minis can call a hosted API but can't replace the
  local model for normal operation.
- **Cloud / AWS for primary path.** Violates the "no US cloud" stance.
  The API fallback is disclosed-and-bounded, not the primary path.
- **Synchronous replication.** Async nightly + 15-min rsync is enough.
- **UPS / cellular backup recommendations.** Hardware purchases aren't
  this plan's call, but noted: residential power is the real HA floor.

---

## Cost summary

| Phase | Work | Hardware | Recurring |
|---|---|---|---|
| Phase 0 | 1–2 days | $0 | $0 |
| Phase 1 | 3–5 days | $0 | $0 |
| Phase 2 | 3–5 days | $0 | ~$0–10/mo (API only fires during outages; free tier likely covers it) |
| Phase 3 | TBD | TBD | TBD |

Total time-to-failure-isolated-system: **~2 weeks**. Time-to-survivable-
Spark-outage with real replies during outage: **~3 weeks** including
Phase 2.

---

## Honest tradeoffs to accept

1. **Residential power is the unfixable risk floor.** All three hosts
   on the same circuit. A power outage >60 s is a user-visible outage
   regardless of architecture, UNLESS the API fallback chain extends
   to "ingress also runs from somewhere with independent power" — which
   Cloudflare Workers provide (if we put the Worker in front, ingress
   stays up even when the home circuit is dead; though the Worker
   doesn't have access to mem0 or short-term state, so it would have to
   reply "we're offline, your message is queued"). Out of scope for
   now.
2. **Inference capacity is fixed at one Spark + API fallback.** No
   amount of architecture work increases the tokens-per-second the GPU
   produces. Phase 1+2 reduce GPU contention from other services; if
   pure inference demand exceeds the GPU, we need more hardware OR
   admission control (rate limits) OR cheaper-per-turn responses (see
   the v1.7-eval architectural critique).
3. **Each phase is independently shippable.** Don't gate Phase 0 on
   Phase 1. Phase 0 alone substantially reduces outage risk.
4. **The API fallback is a calculated departure** from the "no US
   cloud" mission stance, scoped to outage windows only, with
   transparent user-facing disclosure. The alternative is multi-hour
   outages when the Spark is down. Mission-purity vs user-service is a
   real tradeoff and the plan picks user-service with honesty.
