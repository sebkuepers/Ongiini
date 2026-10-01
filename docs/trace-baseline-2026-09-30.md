# Trace baseline — 2026-09-30

Aggregate numbers from `data/trace.jsonl` (33,347 turns, 2026-05-20 to
2026-09-30), taken before the Owela v2 foundation work. Counts, lengths and
timings only — no message content, no user identifiers. Use these as the
"before" column when judging the changes.

Caveat: `web_search` failed on every call from 2026-09-04 to 2026-09-30
(Tavily `402 Payment Required`). Search-turn quality numbers for that period
measure that outage, not the pipeline.

## Traffic

| | All time | Last 30 days |
|---|---|---|
| Turns | 33,347 | 1,912 |
| `none` policy share | 80 % | 90 % |
| `search_shallow` + `search_deep` | 9.5 % | 8.8 % |
| `docs` | 1.4 % | 0.8 % |
| `contribute_*` | 5.3 % | 0.3 % |
| Turns with an image | 8.6 % | 23 % |

## Latency (model calls only; tool time is not in `total_latency_ms`)

| Policy | p50 | p90 | p95 |
|---|---|---|---|
| `none` (last 30 days) | 17.2 s | 28.7 s | 32.6 s |
| `search_shallow` (all) | 34.6 s | 55.9 s | 60.8 s |
| `search_deep` (all) | 44.6 s | 65.6 s | 72.9 s |
| `docs` (all) | 19.4 s | 40.3 s | 48.0 s |
| Router / classifier | 1.4 s | — | 1.7–3.4 s |

- Turns over WhatsApp's 25 s typing window: 17 % all time, 25 % last 30 days.
- Decode speed on `none` turns: p10 18, p50 30, p90 36 tokens/s.

## Reply shape

- `none` reply length (last 30 days): p50 2,083 chars, p90 3,561, p95 4,104
  (the WhatsApp cap is 4,096).
- `tokens_in` p50 on `none` turns: 11,169; `cached_tokens` p50 15,680 (static
  prefix only — the history after the minute-precision date anchor is
  re-prefilled every turn).

## Compensation machinery

| Signal | Value |
|---|---|
| Critique REVISE rate, `search_shallow` | 80 % all time, 67 % last 30 days |
| Critique REVISE rate, `search_deep` | 87 % all time, 84 % last 30 days |
| Critique REVISE rate, `docs` | 52 % all time, 73 % last 30 days |
| Planner plans with 0 queries (`search_deep`) | 288 / 1,763 (16 %) |
| `finish_reason=length` on `search_deep` calls | 145 |
| `reasoning_leak_stripped > 0` on `search_deep` calls | 177 |
| `web_search` tool errors | 213 / 6,188 all time; 100 % from 2026-09-04 |

## Fields that could not be measured

`truncated` (always false), critique timeouts (never set),
`dead_urls_stripped` (never set), `truncated_thinking_blocked` (not traced),
router confidence / fallback (not traced), turns that crashed before the
reply (no trace line at all), wall-clock turn time (tool latency missing).

## Chat template swap (same day)

vLLM restarted 2026-09-30 ~09:36 UTC with the vLLM `main`
`tool_chat_template_gemma4.jinja`. Smoke checks before and after (plain chat,
forced tool call, tool round-trip, image + tools, thinking on/off, prefix
cache) all passed on both templates.
