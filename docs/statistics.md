# Transparency & analytics — `/statistics`

Ongiini publishes aggregate usage statistics at
[`ongiini.ai/statistics`](https://ongiini.ai/statistics/). The page is
**online but unlinked** from the main navigation while the dataset is
small — a `<meta name="robots" content="noindex">` keeps it out of
search indexes. The legal basis lives in the Privacy Policy
([Section 7 — Research, analytics & transparency reporting](https://ongiini.ai/privacy/)):
Art. 6(1)(f) GDPR legitimate interest, with Art. 89 GDPR / § 27 BDSG
(*Forschungsprivileg*) applying when results are published as research.

This doc explains the framework, the privacy posture, and how to
extend it.

---

## What's on the page

1. **Volume KPIs** — unique users, conversations, free tokens
   generated, web searches, voice notes, photos. Each tile carries an
   optional WoW delta (`+18% vs prior week`) when there's enough prior
   data to compare.
2. **Growth over time** — cumulative-users area chart, daily-active-
   users line.
3. **Engagement** — cohort retention sparkline (avg across qualifying
   cohorts), per-user message-count distribution, conversation depth
   (median + p95 + mean tiles plus a length histogram).
4. **Time-of-week heatmap** — 7×24 grid in Africa/Windhoek local time.
   Each cell's terracotta opacity = count / max.
5. **How people use Ongiini.ai** — emergent use-case donut (clustered)
   + top-topics list (raw extracted phrases, threshold ≥5).
6. **Who uses Ongiini.ai** — five separate emergent-clustering panels:
   roles, regions, languages, family situation, current life context.
7. **How it performs** — median + p95 latency, tool-call rate,
   truncation rate. Rendered against an `--ink` dark background so it
   visually separates ops signals from user-facing data.
8. **Methodology** — every page disclosure: what the data is, the two
   qualitative passes, the heavy skew, the privacy floor, the
   anti-leak guardrails, the right to object.

---

## Backend (Spark)

All code lives under [`webhook/app/stats/`](../webhook/app/stats/):

| File | Role |
|---|---|
| `taxonomy.py` | `ANALYSIS_VERSION` schema tag. Bumping it invalidates the qualia cache and forces a clean re-extraction under updated prompts. |
| `safety.py` | `ANTI_PII_PROMPT` extraction prompt prefix + `sanitise_label()` regex post-filter. Pure stdlib, importable without dragging mem0/openai. |
| `analyses.py` | The LLM-driven framework. Defines `Analysis` dataclass and six concrete analyses (one topic + five WHO). Background `run_forever()` loop runs extract + synthesize for each. |
| `analyses_io.py`, `synthesis_io.py` | Stdlib-only readers for the qualia cache + synthesis JSON files. Importable from the aggregator without the LLM dependency graph. |
| `aggregator.py` | Reads `usage.log`, `trace.jsonl`, per-user memory files, and the synthesis outputs. Assembles the `/stats.json` payload. |
| `cache.py` | In-process TTL cache (5 min) so repeated polling doesn't re-aggregate. |
| `api.py` | FastAPI router mounting `GET /stats.json`. |

Lifespan in `webhook/app/main.py` starts the analysis loop as an asyncio
task during startup and cancels it on shutdown.

### Data sources (already on disk for service operation)

| Source | Path | Format |
|---|---|---|
| Per-message usage | `/data/usage.log` | Pipe-delimited, UTC |
| Structural trace | `/data/trace.jsonl` | JSONL, no content — only lengths/names/latencies |
| Short-term memory | `/data/{msisdn}.json` | PII-scrubbed message arrays |
| Long-term memory | `/data/qdrant/` + `/data/mem0_history.db` | mem0 facts (typed) |

### Data sources (created by the transparency layer)

| Source | Path | Format |
|---|---|---|
| Extraction cache | `/data/qualia.sqlite` | `(analysis, content_hash, version, label, extracted_at)` |
| Synthesis output | `/data/synthesis-{topics,roles,regions,languages,family,situations}.json` | Cluster list with summaries + counts |
| Opt-outs | `/data/objections.txt` | Plain text, one MSISDN per line |

---

## The qualitative framework — LLM as analyst, not classifier

The design choice: **no fixed taxonomy**. Categories emerge from the
data. Concretely, two passes:

### Pass 1 — Extract (cheap, per item, incremental)

For each new user message (or each new user's PROFILE facts, depending
on the analysis), the local Gemma 4 produces a short generic label:

- *"yellowing maize leaves"* (from a message asking about crop disease)
- *"smallholder farmer"* (from a profile fact)
- *"matric student"* (from a profile fact)

Labels are cached in `qualia.sqlite` keyed by content hash + analysis
name + `ANALYSIS_VERSION`. Re-running is cheap: known hashes are
skipped, only new items get the LLM call.

Every label passes through `sanitise_label()` before storage. Labels
containing identifying patterns (towns, names, ages, dates, possessives,
anything the existing PII regex catches, anything >80 chars) are
dropped. Failed labels are logged but not stored.

### Pass 2 — Synthesize (periodic, batched)

Every ~10 minutes (or when 25+ new extractions have accumulated since
the last run), the LLM is shown the entire collection of cached labels
with their counts and asked to cluster them into a small set of named
themes. The model produces strict JSON:

```json
{
  "clusters": [
    {"label": "Agriculture", "summary": "...",
     "items": ["yellowing maize leaves", "irrigation guidance", ...]}
  ]
}
```

The aggregator matches items back to the cached labels
(case-insensitive, punctuation-tolerant) and sums their counts to get
the cluster size. Unmatched labels fall through to "Other".

Synthesis output is written atomically to `synthesis-{name}.json` and
served by the aggregator without further LLM calls.

### Adding a new analysis

Define a new `Analysis` dataclass in `analyses.py`:

```python
NEW_ANALYSIS = Analysis(
    name="dialect",                # → synthesis-dialect.json
    description="...",
    source=_iter_user_messages,    # or another source iterator
    extract_system=ANTI_PII_PROMPT + "Task: ...",
    synthesize_system="...",
    item_kind="message",           # or "user"
)
```

Append to `ALL_ANALYSES`. The background loop will pick it up on the
next pass. Surface it in `aggregator.py` by adding a
`_format_synthesis_block("dialect", ...)` call into the response dict.

No other code changes required. The framework handles extraction,
sanitisation, caching, and synthesis identically across analyses.

---

## Privacy posture

### What we publish

Aggregate-only:

- Counts and percentages
- Time-series of counts
- Clusters with their (sanitised) constituent labels
- WoW delta values

### What we never publish

- MSISDNs
- Raw message content
- Per-user breakdowns
- Any field with reasonable risk of identifying a specific person
- Categories represented by fewer than 5 users (per `stats_minimum_bucket`)

### Multi-layer anti-PII guardrails

The qualitative section is the highest-risk surface (the LLM is reading
content and producing labels). Two layers of defence:

**Layer 1 — extraction prompts.** Every analysis's `extract_system` is
prefixed with [`safety.py::ANTI_PII_PROMPT`](../webhook/app/stats/safety.py).
The prefix:

- Explicitly forbids: person names, places below country level, ages,
  dates, years, quantities, verbatim quotes.
- Defines a `REDACTED` sentinel: if the model can't produce a fully
  generic phrase, it outputs `REDACTED` and we drop the item.
- Includes BAD → GOOD examples so the rules land concretely.

The `regions` analysis gets a tailored override: the 14 Namibian
administrative regions (which each cover 100k+ people) are publishable;
towns are not. If a user mentions a town, the model is instructed to
generalise to the region.

**Layer 2 — regex post-filter.** Every LLM-produced label runs through
[`safety.py::sanitise_label()`](../webhook/app/stats/safety.py)
before storage. Drops labels containing:

- A 4-digit number (years, IDs)
- A capitalised possessive (`Joseph's`, strong proper-noun signal)
- Any pattern the existing PII scrubber catches (email, phone, IBAN,
  card, 11-digit Namibian ID)
- A known Namibian town/village name (substring match, lowercase)
- Anything longer than 80 chars (model failed to be concise)

Failed labels are logged (`topic sanitiser rejected label for
analysis=X: ...`) but never written to `qualia.sqlite`.

**Layer 3 — bucket floor.** At display time the aggregator applies
`settings.stats_minimum_bucket` (default 5). Any cluster or category
with fewer members folds into "Other". Cohort retention only publishes
averages of cohorts that individually meet the floor.

**Right to object (Art. 21 GDPR).** Users who write
`object to research processing` to the contact address in the Privacy
Policy with their phone number are added to `/data/objections.txt`.
The aggregator's `_load_objections()` is called at the START of every
compute pass; excluded MSISDNs contribute to no metric, time series, or
qualitative analysis, current or future.

---

## Routing

```
Browser → ongiini.ai/statistics/ (static page, Cloudflare Pages)
         │
         │ fetch('/api/stats')
         ▼
Cloudflare Pages Function (functions/api/stats.js)
         │
         │ forwards to env.STATS_API_URL + "/stats.json"
         ▼
Cloudflare Tunnel (api.ongiini.ai)
         │
         ▼
Spark webhook GET /stats.json
         │
         ├─ in-process TTL cache (5 min)
         └─ aggregator.compute()  ← reads /data
```

The Pages Function is same-origin from the browser's perspective —
no CORS dance. Two cache layers exist:

- **Webhook in-process** (5 min) — protects against repeated aggregation
  if the page is reloaded rapidly.
- **CDN edge** (5 min, set by the Function's `Cache-Control` and
  `cf.cacheTtl`) — protects the Spark from spikes if the page goes
  viral.

Both expire together; refresh latency is bounded by `max-age`.

See [`cloudflare-pages.md`](cloudflare-pages.md) for the Pages Function
setup (env var, directory layout pitfall) and
[`webhook-resilience.md`](webhook-resilience.md) for what happens when
the Spark is offline.

---

## Config knobs (`webhook/app/config.py`)

| Setting | Default | Effect |
|---|---|---|
| `stats_cache_ttl_seconds` | 300 | In-process aggregate cache TTL. |
| `topic_classify_interval_seconds` | 600 | How often the background loop wakes. |
| `stats_minimum_bucket` | 5 | Floor for user-demographic clusters. |

---

## Updating the analysis prompts

Editing an `extract_system` or `synthesize_system` in `analyses.py`
changes the semantics of stored labels. Always **bump
`ANALYSIS_VERSION`** in `taxonomy.py` when doing this. The cache is
keyed by version, so the loop will re-extract everything under the new
prompts. Old-version rows orphan harmlessly in `qualia.sqlite`; you can
purge them manually if you want at-rest hygiene:

```sql
DELETE FROM extractions WHERE version < <current>;
```

---

## Verification checklist

After any change to the transparency layer:

1. **Backend smoke:** synthetic `usage.log` + `trace.jsonl` fixtures →
   the response dict has the expected shape. See the integration
   pattern in past commits (`webhook/tests/` is the right home for a
   formal harness).
2. **Sanitiser tests:** every change to `safety.py` should re-verify
   the existing 21 test cases (Heinis, Oshakati, Joseph's grade 11
   exam, email/phone patterns, the REDACTED sentinel, length cap, all
   continue to drop; generic phrases continue to pass).
3. **Privacy spot-check:** `curl https://api.ongiini.ai/stats.json | grep -E '\\b[0-9]{6,}\\b'`
   should return nothing — no MSISDN-sized digit strings in the JSON.
4. **End-to-end:** load `https://ongiini.ai/statistics/` in a fresh
   browser. All sections render. Sample sizes display. The
   methodology block disclosures match the actual code (anti-PII,
   bucket floor, opt-out path).
