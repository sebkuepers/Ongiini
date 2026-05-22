"""Aggregate transparency-reporting analytics for /statistics page.

See Privacy Policy Section 7 for the legal framing — this package
materialises the "Research, analytics & transparency reporting" promise
made there. Every aggregate published here is computed from data we
already store for the service (usage.log, trace.jsonl, per-user memory
files, mem0 facts); no new data collection is introduced. PII scrubbing
happens at the source, before any storage, and is not re-checked here.

Publication boundary (enforced in `aggregator.assemble`):
  - no MSISDNs
  - no raw message content
  - no per-user breakdowns
  - distribution buckets below settings.stats_minimum_bucket collapse
    into "Other"
  - users listed in /data/objections.txt are excluded entirely at the
    source (their data contributes to nothing)
"""
