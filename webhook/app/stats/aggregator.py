"""Aggregate transparency-reporting metrics from on-disk service data.

Source files (all under settings.data_dir):
  - usage.log     (pipe-delimited, one line per message turn)
  - trace.jsonl   (one JSON object per turn, structural signals only)
  - {msisdn}.json (per-user short-term memory — used only to count
                    unique users and message kinds; PII already scrubbed)

This module is intentionally I/O-only: it reads, it counts. No LLM
calls, no network, no writes. Topic + profession aggregation is added
in Phase 3 by reading separate caches that those modules maintain.

The output dict is the contract documented in the plan and consumed
verbatim by the Pages Function and the /statistics/ page.

Privacy guard: every category-distribution helper passes through
`_collapse_small_buckets`, which folds counts below
`settings.stats_minimum_bucket` into a single "Other" entry. The
top-level totals are not subject to this floor because they are
inherently non-identifying.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..config import settings
from .taxonomy import TAXONOMY_VERSION

# --- Time-zone helpers ------------------------------------------------------

UTC = timezone.utc
NAMIBIA_TZ = timezone(timedelta(hours=2), name="Africa/Windhoek")


# --- File paths -------------------------------------------------------------

def _usage_path() -> Path:
    return settings.data_dir / "usage.log"


def _trace_path() -> Path:
    return settings.data_dir / "trace.jsonl"


def _objections_path() -> Path:
    return settings.data_dir / "objections.txt"


def _memory_glob() -> Iterator[Path]:
    """Yield per-user short-term-memory files in the data dir.

    Pattern: digit-only filenames ending in .json. Filters out
    qdrant/, mem0_history.db, topic_cache.sqlite, etc.
    """
    for p in settings.data_dir.glob("*.json"):
        stem = p.stem
        if stem.isdigit():
            yield p


# --- Opt-out filter ---------------------------------------------------------

def _load_objections() -> frozenset[str]:
    """Return MSISDNs that have objected to research processing.

    Missing file is treated as empty (no opt-outs). One MSISDN per
    line, comments and blanks ignored.
    """
    path = _objections_path()
    if not path.exists():
        return frozenset()
    msisdns: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Normalise: strip + and any whitespace
        digits = re.sub(r"\D", "", line)
        if digits:
            msisdns.add(digits)
    return frozenset(msisdns)


# --- usage.log parsing ------------------------------------------------------

# Pre-v3 format (no kind=) and current format (kind=chat/memory/summary).
# We extract every field including kind; if kind is absent we default to
# "chat" (matches usage.summary_for() semantics).
_USAGE_RE = re.compile(
    r"^(?P<ts>\S+)\s\|\s(?P<msisdn>\S+)\s\|\s"
    r"tokens_in=(?P<tin>\d+)\stokens_out=(?P<tout>\d+)\s\|\s"
    r"search=(?P<search>yes|no)"
    r"(?:\s\|\skind=(?P<kind>[a-zA-Z_]+))?"
)


def _parse_usage_lines(excluded: frozenset[str]) -> Iterator[dict[str, Any]]:
    path = _usage_path()
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = _USAGE_RE.match(line)
            if not m:
                continue
            msisdn = m["msisdn"]
            if msisdn in excluded:
                continue
            yield {
                "ts": m["ts"],
                "msisdn": msisdn,
                "tokens_in": int(m["tin"]),
                "tokens_out": int(m["tout"]),
                "search": m["search"] == "yes",
                "kind": m["kind"] or "chat",
            }


# --- trace.jsonl parsing ----------------------------------------------------

def _parse_trace_lines(excluded: frozenset[str]) -> Iterator[dict[str, Any]]:
    path = _trace_path()
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msisdn = str(obj.get("msisdn", ""))
            if msisdn in excluded:
                continue
            yield obj


# --- Bucket-floor enforcement (privacy) -------------------------------------

def _collapse_small_buckets(
    rows: list[dict[str, Any]],
    count_key: str = "count",
    other_label: str = "Other",
) -> list[dict[str, Any]]:
    """Fold any entry with count < settings.stats_minimum_bucket into 'Other'.

    Input rows must already carry `label` and `count_key`. Returns a new
    list sorted descending by count. If the rolled-up 'Other' bucket
    itself ends up below the floor, it is dropped (no entry rather than
    a sub-floor 'Other').
    """
    floor = settings.stats_minimum_bucket
    keep: list[dict[str, Any]] = []
    other_total = 0
    for row in rows:
        c = int(row.get(count_key, 0))
        if c >= floor and row.get("label") != other_label:
            keep.append(dict(row))
        else:
            other_total += c
    if other_total >= floor:
        keep.append({"label": other_label, count_key: other_total})
    keep.sort(key=lambda r: int(r.get(count_key, 0)), reverse=True)
    return keep


def _with_pct(rows: list[dict[str, Any]], count_key: str = "count") -> list[dict[str, Any]]:
    total = sum(int(r.get(count_key, 0)) for r in rows)
    if total == 0:
        for r in rows:
            r["pct"] = 0.0
        return rows
    for r in rows:
        r["pct"] = round(int(r.get(count_key, 0)) / total * 100, 1)
    return rows


# --- Time helpers -----------------------------------------------------------

def _parse_ts(ts: str) -> datetime | None:
    """Parse the UTC timestamps both logs use.

    usage.log: 'YYYY-MM-DDTHH:MM:SS' (no tz suffix, but always UTC)
    trace.jsonl: same shape via isoformat(timespec='seconds')
    """
    try:
        # Tolerate both naive ('...:SS') and tz-aware ('...+00:00') forms.
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts[:-1]).replace(tzinfo=UTC)
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        return None


# --- Conversation grouping --------------------------------------------------

CONVERSATION_GAP = timedelta(minutes=30)


def _count_conversations(per_user_timestamps: dict[str, list[datetime]]) -> int:
    """A 'conversation' is a contiguous burst of messages from one user.

    Any gap of 30 minutes or more starts a new conversation. Counts
    across all users.
    """
    total = 0
    for ts_list in per_user_timestamps.values():
        if not ts_list:
            continue
        ts_list.sort()
        total += 1
        for prev, cur in zip(ts_list, ts_list[1:]):
            if cur - prev >= CONVERSATION_GAP:
                total += 1
    return total


# --- Synthesis-block reader (qualitative output) ----------------------------

def _format_synthesis_block(
    analysis_name: str,
    *,
    denominator: int,
    empty_status: str,
) -> dict[str, Any]:
    """Read /data/synthesis-{name}.json and shape it for the API response.

    Applies the same minimum-bucket-size floor as quantitative
    distributions: clusters with count < settings.stats_minimum_bucket
    are rolled into 'Other'.

    `denominator` is the universe size for the coverage metric:
      - for `topics`: total user messages
      - for any WHO dimension: total unique users
    """
    from .synthesis_io import load_synthesis

    payload = load_synthesis(analysis_name)
    if payload is None:
        return {
            "method": "LLM-emergent clustering of extracted labels",
            "coverage": 0.0,
            "n_observations": 0,
            "denominator": denominator,
            "categories": [],
            "status": empty_status,
        }

    clusters = payload.get("clusters") or []
    rows = [{"label": c.get("label", "?"), "count": int(c.get("count", 0))} for c in clusters]
    summary_by_label = {c.get("label", "?"): str(c.get("summary", "")) for c in clusters}
    rows = _collapse_small_buckets(rows)
    rows = _with_pct(rows)
    total_assigned = sum(int(r.get("count", 0)) for r in rows)
    # Cap at 1.0: synthesis from a previous larger dataset can in
    # principle have more observations than the current universe (e.g.
    # after opt-outs prune the universe but the cache hasn't refreshed).
    coverage = round(min(1.0, total_assigned / denominator), 3) if denominator else 0.0
    out_categories = [
        {
            "label": r["label"],
            "count": r["count"],
            "pct": r["pct"],
            "summary": summary_by_label.get(r["label"], ""),
        }
        for r in rows
    ]
    return {
        "method": payload.get("method", "LLM-emergent clustering"),
        "coverage": coverage,
        # n_observations is the number of items the synthesis actually
        # saw. For topics it's user-messages-with-an-extracted-label;
        # for a WHO dimension it's users-with-data-for-that-dimension.
        # Crucially: NOT the universe size — that's `denominator`.
        "n_observations": payload.get("total_items_analysed", 0),
        "denominator": denominator,
        "categories": out_categories,
        "generated_at": payload.get("generated_at"),
        "distinct_labels": payload.get("distinct_labels", 0),
    }


def _top_topics_block(top_n: int = 20) -> dict[str, Any]:
    """Return the most-mentioned raw topic extractions — one level down
    from the use-case clusters.

    Lets readers see specific user concerns ('yellowing maize leaves',
    'VAT registration') under the broader buckets ('Agriculture',
    'Government'). Read directly from the extraction cache without
    needing a synthesis.
    """
    try:
        from .analyses_io import load_label_counts_via_io
    except ImportError:
        # Fall back to lazy import of analyses (which DOES drag in
        # mem0); only used when analyses_io isn't yet present.
        from .analyses import load_label_counts_sync as load_label_counts_via_io  # type: ignore

    counts = load_label_counts_via_io("topics")
    if not counts:
        return {
            "n_distinct": 0,
            "n_shown": 0,
            "labels": [],
            "status": "Computing — first extraction pass not yet complete.",
        }
    # Top-topics has a LOWER threshold than the per-user bucket floor:
    # individual topic phrases don't identify a person the way roles or
    # locations might. We filter true singletons (count==1) which are
    # mostly noise (verbatim once-off phrasings the model produced).
    # Two-or-more keeps things interesting without privacy harm. Also
    # drop 'small talk' — it's a placeholder, not a real topic.
    rows = [
        {"label": lbl, "count": c}
        for lbl, c in counts.items()
        if c >= 2 and lbl.lower() != "small talk"
    ]
    rows.sort(key=lambda r: r["count"], reverse=True)
    shown = rows[:top_n]
    return {
        "n_distinct": len(counts),
        "n_shown": len(shown),
        "labels": shown,
    }


# --- Main aggregator --------------------------------------------------------

def _compute_sync() -> dict[str, Any]:
    """The actual aggregation. Called inside a thread by `compute()` so
    a long sweep doesn't block the event loop.
    """
    excluded = _load_objections()

    # ---- Pass 1: usage.log ----
    chat_lines: list[dict[str, Any]] = []
    all_lines: list[dict[str, Any]] = []
    for row in _parse_usage_lines(excluded):
        all_lines.append(row)
        if row["kind"] == "chat":
            chat_lines.append(row)

    unique_users: set[str] = {row["msisdn"] for row in chat_lines}
    msisdn_first_seen: dict[str, datetime] = {}
    chat_per_user_ts: dict[str, list[datetime]] = defaultdict(list)
    chat_per_day: dict[str, int] = defaultdict(int)
    tokens_per_day_in: dict[str, int] = defaultdict(int)
    tokens_per_day_out: dict[str, int] = defaultdict(int)
    new_users_per_day: dict[str, int] = defaultdict(int)
    dau: dict[str, set[str]] = defaultdict(set)
    hour_utc_counts: dict[int, int] = defaultdict(int)
    hour_local_counts: dict[int, int] = defaultdict(int)
    dow_counts: dict[int, int] = defaultdict(int)
    web_searches = 0

    for row in chat_lines:
        dt = _parse_ts(row["ts"])
        if dt is None:
            continue
        msisdn = row["msisdn"]
        day = dt.date().isoformat()
        chat_per_day[day] += 1
        tokens_per_day_in[day] += row["tokens_in"]
        tokens_per_day_out[day] += row["tokens_out"]
        dau[day].add(msisdn)
        chat_per_user_ts[msisdn].append(dt)
        first = msisdn_first_seen.get(msisdn)
        if first is None or dt < first:
            msisdn_first_seen[msisdn] = dt
        if row["search"]:
            web_searches += 1
        hour_utc_counts[dt.hour] += 1
        hour_local_counts[dt.astimezone(NAMIBIA_TZ).hour] += 1
        dow_counts[dt.weekday()] += 1  # Mon=0

    for msisdn, dt in msisdn_first_seen.items():
        new_users_per_day[dt.date().isoformat()] += 1

    # ---- Pass 2: trace.jsonl (richer kind, performance, url_fetches) ----
    voice_notes = 0
    photos = 0
    deletions = 0
    truncations = 0
    url_fetches = 0
    web_searches_trace = 0
    tool_call_turns = 0
    total_trace_turns = 0
    latencies: list[int] = []

    for tr in _parse_trace_lines(excluded):
        total_trace_turns += 1
        calls = tr.get("calls", []) or []
        any_tool_called = False
        for call in calls:
            for tc in call.get("tool_calls", []) or []:
                name = tc.get("name", "")
                any_tool_called = True
                if name == "web_search":
                    web_searches_trace += 1
                elif name == "fetch_url":
                    url_fetches += 1
        if any_tool_called:
            tool_call_turns += 1
        if tr.get("deleted_data"):
            deletions += 1
        if tr.get("truncated"):
            truncations += 1
        lat = tr.get("total_latency_ms")
        if isinstance(lat, (int, float)) and lat > 0:
            latencies.append(int(lat))

    # ---- Pass 3: per-user memory files (kinds count) ----
    # The memory files preserve the actual message stream including
    # role+content; we look at the user-role entries to infer how many
    # were audio (transcripts) vs image vs text. Audio messages are
    # stored as plain text after Whisper, so we cannot distinguish
    # them here without a marker — but image messages carry the literal
    # "[image attached]" placeholder we emit at write time (see
    # mem.add_image_turn). We use that as the only reliable counter for
    # photos. Audio counts default to 0 until we add a similar marker;
    # in the meantime the trace-based total works for the overall photo
    # count.
    image_marker = "[image attached]"
    voice_marker = "[voice note]"
    for fp in _memory_glob():
        msisdn = fp.stem
        if msisdn in excluded:
            continue
        try:
            arr = json.loads(fp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(arr, list):
            continue
        for entry in arr:
            if not isinstance(entry, dict):
                continue
            if entry.get("role") != "user":
                continue
            content = entry.get("content")
            if not isinstance(content, str):
                continue
            if image_marker in content:
                photos += 1
            elif voice_marker in content:
                voice_notes += 1

    # ---- Assemble period + timeseries ----
    parsed_timestamps = [
        dt for r in chat_lines if (dt := _parse_ts(r["ts"])) is not None
    ]
    if parsed_timestamps:
        period_from = min(parsed_timestamps).date().isoformat()
        period_to = max(parsed_timestamps).date().isoformat()
    else:
        period_from = None
        period_to = None

    # Per-day → set(msisdn) for both DAU and cumulative.
    per_day_users: dict[str, set[str]] = defaultdict(set)
    for row in chat_lines:
        dt = _parse_ts(row["ts"])
        if dt is None:
            continue
        per_day_users[dt.date().isoformat()].add(row["msisdn"])

    sorted_days = sorted(chat_per_day.keys())
    cumulative_unique: list[tuple[str, int]] = []
    running_set: set[str] = set()
    daily_active_users_series: list[tuple[str, int]] = []
    messages_per_day_series: list[tuple[str, int]] = []
    tokens_per_day_series: list[tuple[str, int]] = []
    new_users_per_day_series: list[tuple[str, int]] = []
    for day in sorted_days:
        running_set |= per_day_users.get(day, set())
        cumulative_unique.append((day, len(running_set)))
        daily_active_users_series.append((day, len(per_day_users.get(day, set()))))
        messages_per_day_series.append((day, chat_per_day[day]))
        tokens_per_day_series.append(
            (day, tokens_per_day_in[day] + tokens_per_day_out[day])
        )
        new_users_per_day_series.append((day, new_users_per_day.get(day, 0)))

    # ---- Distributions ----
    by_hour_utc = [{"hour": h, "count": hour_utc_counts.get(h, 0)} for h in range(24)]
    by_hour_local = [{"hour": h, "count": hour_local_counts.get(h, 0)} for h in range(24)]
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_day_of_week = [
        {"day": dow_labels[i], "count": dow_counts.get(i, 0)} for i in range(7)
    ]

    # By message kind: in chat lines we know everything is the user-facing
    # call; the image vs audio split comes from the memory-file scan above.
    # For Phase 1 we surface a coarse split: chat / memory / summary kinds
    # from usage.log (which is internally interesting) and a separate
    # "media" split (text vs image vs audio) — but until the audio marker
    # lands, audio shows 0 and the bulk shows as 'text'.
    kind_counts: dict[str, int] = defaultdict(int)
    for row in all_lines:
        kind_counts[row["kind"]] += 1
    by_kind_internal = [
        {"label": k, "count": v} for k, v in sorted(kind_counts.items())
    ]

    # Photo / audio / text shares (user-facing media split)
    total_user_msgs = len(chat_lines)
    text_msgs = max(0, total_user_msgs - photos - voice_notes)
    by_media = _with_pct(
        [
            {"label": "Text", "count": text_msgs},
            {"label": "Image", "count": photos},
            {"label": "Voice", "count": voice_notes},
        ]
    )

    # Performance
    if latencies:
        latencies.sort()
        median_lat = int(statistics.median(latencies))
        p95_idx = max(0, int(len(latencies) * 0.95) - 1)
        p95_lat = latencies[p95_idx]
    else:
        median_lat = 0
        p95_lat = 0

    perf = {
        "median_latency_ms": median_lat,
        "p95_latency_ms": p95_lat,
        "tool_call_rate": round(tool_call_turns / total_trace_turns, 4)
        if total_trace_turns
        else 0.0,
        "truncation_rate": round(truncations / total_trace_turns, 4)
        if total_trace_turns
        else 0.0,
    }

    # ---- Conversations count ----
    conversations = _count_conversations(chat_per_user_ts)

    # ---- Totals ----
    tokens_in_total = sum(r["tokens_in"] for r in all_lines)
    tokens_out_total = sum(r["tokens_out"] for r in all_lines)
    free_tokens_generated = sum(r["tokens_out"] for r in chat_lines)

    # Web searches: prefer trace's exact count if non-zero (more
    # accurate, counts individual tool invocations); fall back to the
    # per-message search=yes flag from usage.log.
    web_search_total = web_searches_trace if web_searches_trace else web_searches

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "cache_ttl_seconds": settings.stats_cache_ttl_seconds,
        "taxonomy_version": TAXONOMY_VERSION,
        "period": {"from": period_from, "to": period_to},
        "totals": {
            "unique_users": len(unique_users),
            "conversations": conversations,
            "messages_user": total_user_msgs,
            "messages_assistant": total_user_msgs,  # 1-to-1 with chat lines
            "tokens_in_total": tokens_in_total,
            "tokens_out_total": tokens_out_total,
            "free_tokens_generated": free_tokens_generated,
            "web_searches": web_search_total,
            "url_fetches": url_fetches,
            "voice_notes": voice_notes,
            "photos": photos,
            "deletions_invoked": deletions,
        },
        "timeseries": {
            "daily_active_users": [list(t) for t in daily_active_users_series],
            "new_users_per_day": [list(t) for t in new_users_per_day_series],
            "unique_users_cumulative": [list(t) for t in cumulative_unique],
            "messages_per_day": [list(t) for t in messages_per_day_series],
            "tokens_per_day": [list(t) for t in tokens_per_day_series],
        },
        "distributions": {
            "by_message_kind_internal": _collapse_small_buckets(by_kind_internal),
            "by_media": _collapse_small_buckets(by_media),
            "by_hour_of_day_utc": by_hour_utc,
            "by_hour_of_day_local": by_hour_local,
            "by_day_of_week": by_day_of_week,
            # by_language is added by Phase 3 once detection is wired
        },
        "topics": _format_synthesis_block(
            "topics",
            denominator=total_user_msgs,
            empty_status="Computing — first classification pass not yet complete.",
        ),
        "top_topics": _top_topics_block(top_n=20),
        "who": {
            "roles": _format_synthesis_block(
                "roles",
                denominator=len(unique_users),
                empty_status="Computing — extracting roles from profile facts.",
            ),
            "regions": _format_synthesis_block(
                "regions",
                denominator=len(unique_users),
                empty_status="Computing — extracting regions from profile facts.",
            ),
            "languages": _format_synthesis_block(
                "languages",
                denominator=len(unique_users),
                empty_status="Computing — extracting preferred languages.",
            ),
            "family": _format_synthesis_block(
                "family",
                denominator=len(unique_users),
                empty_status="Computing — extracting family / household context.",
            ),
            "situations": _format_synthesis_block(
                "situations",
                denominator=len(unique_users),
                empty_status="Computing — extracting current life situations.",
            ),
        },
        "performance": perf,
    }


async def compute() -> dict[str, Any]:
    """Async entrypoint — runs the synchronous aggregator off-thread."""
    return await asyncio.to_thread(_compute_sync)
