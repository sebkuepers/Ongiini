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
    """Yield per-user short-term-memory files for REAL Namibian users.

    Pattern: filename stem must be a valid Namibian msisdn (11 digits,
    starts with 264). Filters out qdrant/, mem0_history.db,
    topic_cache.sqlite, AND any synthetic / non-Namibian / operator-test
    memory files that may exist alongside.
    """
    for p in settings.data_dir.glob("*.json"):
        if _is_real_namibian_msisdn(p.stem):
            yield p


# --- Opt-out filter ---------------------------------------------------------

# Real Namibian phone numbers, post-normalisation, are exactly 12 digits:
# country-code 264 (3) + national number (9). Mobile prefixes after 264
# are 81 / 82 / 85 / 86 (Namibian operators) plus a 7-digit subscriber.
# The application's intake filter (ongiini/filters.py::is_allowed) already
# enforces this shape for INBOUND message routing, but usage.log +
# trace.jsonl also receive synthetic entries from eval / smoke-test runs
# (e.g. "+264baseline_medical_aid"), from the operator's own testing
# number (a German +49... number), and from a pre-prod test entry
# (99000000777777).
#
# The transparency surface (/stats.json + /statistics) is meant to
# reflect REAL Namibian users — not the operator, not synthetic eval
# traffic, not pre-prod test data. This validator gates every read of
# usage.log, trace.jsonl, and per-user memory files so non-Namibian
# msisdns never enter the aggregate.
_NAMIBIAN_MSISDN_RE = re.compile(r"^264\d{9}$")


def _is_real_namibian_msisdn(msisdn: str) -> bool:
    """True iff the string is a valid post-normalisation Namibian
    phone number (11 digits, country-code 264). Filters out:
      - synthetic eval entries (non-digit chars like '+', '_')
      - non-Namibian operator/test numbers (49..., 99...)
      - malformed log lines
    """
    return bool(_NAMIBIAN_MSISDN_RE.match(msisdn))


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
            if not _is_real_namibian_msisdn(msisdn):
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
            if not _is_real_namibian_msisdn(msisdn):
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


def _conversation_lengths(
    per_user_timestamps: dict[str, list[datetime]],
) -> list[int]:
    """Return a flat list of conversation lengths (turn counts) across all
    users. Same conversation-grouping rules as `_count_conversations`.

    A conversation's "length" is the number of user messages in that
    burst (one user turn = one entry in this count). Sorted output is
    NOT guaranteed.
    """
    lengths: list[int] = []
    for ts_list in per_user_timestamps.values():
        if not ts_list:
            continue
        ts_list.sort()
        run = 1
        for prev, cur in zip(ts_list, ts_list[1:]):
            if cur - prev >= CONVERSATION_GAP:
                lengths.append(run)
                run = 1
            else:
                run += 1
        lengths.append(run)
    return lengths


# --- Engagement helpers ----------------------------------------------------

# Buckets for per-user message-count distribution. Order matters for
# the output (it determines rendering order on the page). Tuples are
# (label, lower_bound, upper_bound_inclusive_or_None).
_ENGAGEMENT_BUCKETS: list[tuple[str, int, int | None]] = [
    ("1 message", 1, 1),
    ("2-5 messages", 2, 5),
    ("6-20 messages", 6, 20),
    ("21-100 messages", 21, 100),
    ("100+ messages", 101, None),
]


def _compute_engagement_distribution(
    chat_per_user_ts: dict[str, list[datetime]],
) -> list[dict[str, Any]]:
    """Bucket users by their total chat-message count. Returns a list
    of {label, count, pct} rows in display order.

    Privacy: applies the same minimum-bucket-size floor as other
    distributions. Buckets below the floor roll into nothing (we
    don't aggregate them into an 'Other' bucket here because the
    semantic of 'Other' would be ambiguous — a sub-floor 'heavy users'
    is meaningfully different from a sub-floor 'try-once' group).
    Empty buckets are simply omitted.
    """
    if not chat_per_user_ts:
        return []
    counts_per_bucket: dict[str, int] = {b[0]: 0 for b in _ENGAGEMENT_BUCKETS}
    for ts_list in chat_per_user_ts.values():
        n = len(ts_list)
        if n <= 0:
            continue
        for label, lo, hi in _ENGAGEMENT_BUCKETS:
            if n >= lo and (hi is None or n <= hi):
                counts_per_bucket[label] += 1
                break
    # No privacy floor on activity-level buckets: a bucket like
    # "100+ messages" is a usage-pattern label, not a personal trait.
    # Knowing "1 user sent 100+ messages" does not identify them
    # without other dimensions, and the total unique-users count is
    # already published. The floor exists to prevent surfacing rare
    # personal categories (helicopter pilot), not common activity
    # tiers.
    total_users = sum(counts_per_bucket.values())
    rows: list[dict[str, Any]] = []
    for label, _, _ in _ENGAGEMENT_BUCKETS:
        c = counts_per_bucket[label]
        if c <= 0:
            continue
        pct = round(c / total_users * 100, 1) if total_users else 0.0
        rows.append({"label": label, "count": c, "pct": pct})
    return rows


_CONVERSATION_DEPTH_BUCKETS: list[tuple[str, int, int | None]] = [
    ("1 turn", 1, 1),
    ("2-4 turns", 2, 4),
    ("5-10 turns", 5, 10),
    ("11-30 turns", 11, 30),
    ("30+ turns", 31, None),
]


def _compute_conversation_depth(lengths: list[int]) -> dict[str, Any]:
    """Compute conversation-depth statistics from a flat list of
    per-conversation turn counts.

    Returns median, mean, p95, and a bucketed histogram. Empty input
    yields zeroed stats and an empty histogram.
    """
    if not lengths:
        return {
            "n_conversations": 0,
            "median_turns": 0,
            "mean_turns": 0.0,
            "p95_turns": 0,
            "histogram": [],
        }
    sorted_l = sorted(lengths)
    n = len(sorted_l)
    median_v = int(statistics.median(sorted_l))
    mean_v = round(statistics.fmean(sorted_l), 2)
    p95_idx = max(0, int(round(n * 0.95)) - 1)
    p95_v = sorted_l[p95_idx]

    # Same reasoning as engagement distribution: conversation-length
    # buckets are activity tiers, not personal categories. No floor.
    bucket_counts: dict[str, int] = {b[0]: 0 for b in _CONVERSATION_DEPTH_BUCKETS}
    for v in lengths:
        for label, lo, hi in _CONVERSATION_DEPTH_BUCKETS:
            if v >= lo and (hi is None or v <= hi):
                bucket_counts[label] += 1
                break
    histogram = [
        {"label": label, "count": bucket_counts[label]}
        for label, _, _ in _CONVERSATION_DEPTH_BUCKETS
        if bucket_counts[label] > 0
    ]
    return {
        "n_conversations": n,
        "median_turns": median_v,
        "mean_turns": mean_v,
        "p95_turns": p95_v,
        "histogram": histogram,
    }


def _iso_week_key(dt: datetime) -> tuple[int, int]:
    """ISO year-week tuple for cohort grouping. Done in NAMIBIA_TZ so
    cohorts align with the local week boundaries users experience."""
    local = dt.astimezone(NAMIBIA_TZ)
    iso = local.isocalendar()
    return (iso.year, iso.week)


_RETENTION_DAY_OFFSETS = [1, 3, 7, 14, 30]
# Retention chart needs sturdier samples than the global privacy floor
# (5). Tiny early cohorts (e.g. 8 users) produce wild swings — 0/8 reads
# as 0% but is statistically meaningless. Only publish a retention point
# when at least one cohort of 20+ users has fully elapsed past the
# offset. The page hides the card and shows a "coming soon" placeholder
# until that threshold is met.
_RETENTION_MIN_COHORT = 20


def _compute_retention_curve(
    chat_per_user_ts: dict[str, list[datetime]],
) -> dict[str, Any]:
    """Day-based cumulative cohort retention.

    Group users by the DAY of their first chat (Africa/Windhoek time).
    For each offset N in {1, 3, 7, 14, 30}, look at cohorts whose
    target day (cohort_date + N) has ALREADY fully elapsed — i.e.
    target_date < today. Average the per-cohort cumulative-return
    rate: % of cohort users who came back AT LEAST ONCE on any day
    in [cohort_date + 1, target_date].

    Cumulative (window) rather than exact-day return — better matches
    how an episodic WhatsApp helper is actually used (people return
    when they need help, not every single day). The curve is
    monotonically non-decreasing.

    Crucially: offsets whose target day is in the future or is today
    are SKIPPED — not averaged in as 0% — so a 5-day-old service
    doesn't look like everyone churned when really we just don't have
    enough days of data yet.

    Returns:
        days: [0, 1, 3, ...]   x-axis labels (always starts at 0)
        retained_pct: [100, X, ...]   y-values (non-decreasing)
        n_cohorts: max cohorts averaged at any displayed offset
        min_cohort_size: the privacy floor applied
        max_day_measurable: the largest offset we could publish today
    """
    floor = _RETENTION_MIN_COHORT
    if not chat_per_user_ts:
        return {
            "days": [0],
            "retained_pct": [],
            "n_cohorts": 0,
            "min_cohort_size": floor,
            "max_day_measurable": 0,
        }

    today_local = datetime.now(NAMIBIA_TZ).date()

    # first-chat date → list of user "active dates" sets (one set per user)
    cohorts: dict[Any, list[set[Any]]] = defaultdict(list)
    for msisdn, ts_list in chat_per_user_ts.items():
        if not ts_list:
            continue
        sorted_ts = sorted(ts_list)
        first_date = sorted_ts[0].astimezone(NAMIBIA_TZ).date()
        active_dates = {t.astimezone(NAMIBIA_TZ).date() for t in sorted_ts}
        cohorts[first_date].append(active_dates)

    days_out: list[int] = [0]
    pct_out: list[float] = [100.0]
    n_cohorts_used = 0

    for offset in _RETENTION_DAY_OFFSETS:
        rates: list[float] = []
        for cohort_date, user_active_sets in cohorts.items():
            target_date = cohort_date + timedelta(days=offset)
            # Skip cohorts whose target day hasn't fully elapsed.
            # target_date < today means yesterday or earlier — fully
            # elapsed and safe to measure.
            if target_date >= today_local:
                continue
            cohort_size = len(user_active_sets)
            if cohort_size < floor:
                continue
            # Cumulative: user is "retained at day N" if they were
            # active on ANY day strictly after cohort_date up to and
            # including target_date.
            retained = 0
            for s in user_active_sets:
                if any(
                    (cohort_date + timedelta(days=k)) in s
                    for k in range(1, offset + 1)
                ):
                    retained += 1
            rates.append(retained / cohort_size * 100)
        if not rates:
            # No cohort old enough at this offset. Stop here — we
            # don't show this offset OR any beyond it.
            break
        days_out.append(offset)
        pct_out.append(round(sum(rates) / len(rates), 1))
        n_cohorts_used = max(n_cohorts_used, len(rates))

    return {
        "days": days_out,
        "retained_pct": pct_out,
        "n_cohorts": n_cohorts_used,
        "min_cohort_size": floor,
        "max_day_measurable": days_out[-1] if len(days_out) > 1 else 0,
    }


def _compute_heatmap_dow_hour(
    chat_lines: list[dict[str, Any]],
) -> list[dict[str, int]]:
    """7×24 matrix of chat-message counts by (day_of_week, hour) in
    Africa/Windhoek local time. Always returns all 168 cells (zeros
    where empty), in row-major Mon→Sun, 0→23 order.

    Critical: day-of-week here is the LOCAL day, not UTC. A message
    sent at UTC 23:00 on Sunday is Monday 01:00 in Namibia and counts
    under Monday — same logic as `by_hour_of_day_local`.
    """
    matrix: dict[tuple[int, int], int] = defaultdict(int)
    for row in chat_lines:
        dt = _parse_ts(row["ts"])
        if dt is None:
            continue
        local = dt.astimezone(NAMIBIA_TZ)
        matrix[(local.weekday(), local.hour)] += 1
    return [
        {"day": d, "hour": h, "count": matrix.get((d, h), 0)}
        for d in range(7)
        for h in range(24)
    ]


def _compute_deltas(
    chat_lines: list[dict[str, Any]],
    all_lines: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    photos: int,
    voice_notes: int,
) -> dict[str, Any]:
    """Compute KPI deltas: rolling 7-day "current" vs the 7 days
    immediately before that.

    Anchor on the latest observed timestamp rather than wall-clock
    `now()` — keeps deltas meaningful when traffic isn't real-time
    (e.g. when computing this from a snapshot or after a Spark outage).

    For each KPI, returns {current, prior, pct_change}. pct_change is
    omitted when the prior window has zero observations (avoids inf /
    misleadingly-large deltas off a tiny base).

    NOTE: photos and voice_notes can't be split per-window without
    timestamps on those events. For now we omit them from the delta
    block (the page tile will simply not show a delta sub-row).
    """
    timestamps = [t for r in chat_lines if (t := _parse_ts(r["ts"])) is not None]
    if not timestamps:
        return {}
    anchor = max(timestamps)
    current_start = anchor - timedelta(days=7)
    prior_start = anchor - timedelta(days=14)

    def _in_window(dt: datetime, start: datetime, end: datetime) -> bool:
        return start < dt <= end

    def _user_count_in(start: datetime, end: datetime) -> int:
        users: set[str] = set()
        for r in chat_lines:
            dt = _parse_ts(r["ts"])
            if dt is None:
                continue
            if _in_window(dt, start, end):
                users.add(r["msisdn"])
        return len(users)

    def _msg_count_in(start: datetime, end: datetime) -> int:
        return sum(
            1
            for r in chat_lines
            if (dt := _parse_ts(r["ts"])) is not None and _in_window(dt, start, end)
        )

    def _convs_in(start: datetime, end: datetime) -> int:
        per_user: dict[str, list[datetime]] = defaultdict(list)
        for r in chat_lines:
            dt = _parse_ts(r["ts"])
            if dt is None:
                continue
            if _in_window(dt, start, end):
                per_user[r["msisdn"]].append(dt)
        return _count_conversations(per_user)

    def _tokens_out_in(start: datetime, end: datetime) -> int:
        return sum(
            int(r["tokens_out"])
            for r in chat_lines
            if (dt := _parse_ts(r["ts"])) is not None and _in_window(dt, start, end)
        )

    def _trace_count_in(start: datetime, end: datetime, predicate) -> int:
        n = 0
        for tr in trace_rows:
            dt = _parse_ts(tr.get("ts", ""))
            if dt is None:
                continue
            if not _in_window(dt, start, end):
                continue
            if predicate(tr):
                n += 1
        return n

    def _web_search_in(start: datetime, end: datetime) -> int:
        def has_search(tr: dict[str, Any]) -> bool:
            for call in tr.get("calls", []) or []:
                for tc in call.get("tool_calls", []) or []:
                    if tc.get("name") == "web_search":
                        return True
            return False
        return _trace_count_in(start, end, has_search)

    def _delta(curr: int, prior: int) -> dict[str, Any]:
        out = {"current": curr, "prior": prior}
        if prior > 0:
            out["pct_change"] = round((curr - prior) / prior * 100, 1)
        return out

    return {
        "window_days": 7,
        "anchor": anchor.isoformat(),
        "unique_users": _delta(
            _user_count_in(current_start, anchor),
            _user_count_in(prior_start, current_start),
        ),
        "conversations": _delta(
            _convs_in(current_start, anchor),
            _convs_in(prior_start, current_start),
        ),
        "messages_user": _delta(
            _msg_count_in(current_start, anchor),
            _msg_count_in(prior_start, current_start),
        ),
        "free_tokens_generated": _delta(
            _tokens_out_in(current_start, anchor),
            _tokens_out_in(prior_start, current_start),
        ),
        "web_searches": _delta(
            _web_search_in(current_start, anchor),
            _web_search_in(prior_start, current_start),
        ),
    }


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


def _collapse_languages(block: dict[str, Any]) -> dict[str, Any]:
    """Collapse the WHO Languages panel to a fixed 4-bucket taxonomy
    readers actually ask about: English / Afrikaans / Oshiwambo / Other.

    The LLM-clustered output can return any number of labels
    (Oshindonga, Oshikwanyama, Khoekhoegowab, German, …); we map each
    cluster's label into one of the four canonical buckets, sum counts,
    re-compute pct, and return a block with the same schema so the
    frontend renderer stays unchanged.
    """
    cats = block.get("categories") or []
    if not cats:
        return block

    BUCKETS = ("English", "Afrikaans", "Oshiwambo", "Other")
    sums = {b: 0 for b in BUCKETS}

    def _bucket_for(label: str) -> str:
        s = (label or "").lower()
        if "afrikaan" in s:
            return "Afrikaans"
        if "english" in s:
            return "English"
        if any(k in s for k in (
            "oshiwambo", "oshindonga", "oshikwanyama",
            "ndonga", "kwanyama", "ovambo",
        )):
            return "Oshiwambo"
        return "Other"

    for c in cats:
        sums[_bucket_for(c.get("label", ""))] += int(c.get("count", 0) or 0)

    total = sum(sums.values()) or 1
    collapsed = [
        {"label": b, "count": sums[b], "pct": round(sums[b] / total * 100, 1)}
        for b in BUCKETS
        if sums[b] > 0
    ]
    collapsed.sort(key=lambda r: r["count"], reverse=True)

    return {
        **block,
        "categories": collapsed,
        "distinct_labels": len(collapsed),
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
    # Top-topics uses the same privacy floor as use-case clusters
    # (default 5). Even though the extraction layer aggressively
    # rejects PII before storage, we belt-and-braces: a topic phrase
    # only goes on the public page once at least N distinct messages
    # produced the same label. This makes accidental specificity-leak
    # (e.g. one user with an unusual life situation) statistically
    # unlikely to surface. Also drop 'small talk' — placeholder.
    floor = settings.stats_minimum_bucket
    rows = [
        {"label": lbl, "count": c}
        for lbl, c in counts.items()
        if c >= floor and lbl.lower() != "small talk"
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

    # Community-contribution counts live in their own sqlite (see
    # ongiini.contributions). Soft-fail if the module / file isn't
    # available — the rest of the payload still computes.
    try:
        from .. import contributions as _contrib
        contrib_stats = _contrib.stats_summary()
    except Exception:
        contrib_stats = {
            "total_contributions": 0,
            "by_dialect": {},
            "total_contributors": 0,
            "total_tasks": 0,
        }

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
        # IMPORTANT: weekday + hour are LOCAL (Africa/Windhoek). A
        # message sent at UTC 23:00 Sunday is Monday 01:00 in Namibia
        # and counts under Monday — not Sunday. Same convention as the
        # heatmap below.
        local_dt = dt.astimezone(NAMIBIA_TZ)
        hour_utc_counts[dt.hour] += 1
        hour_local_counts[local_dt.hour] += 1
        dow_counts[local_dt.weekday()] += 1  # Mon=0

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

    # Materialise trace rows so we can also use them for deltas.
    trace_rows: list[dict[str, Any]] = list(_parse_trace_lines(excluded))
    for tr in trace_rows:
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

    # ---- Conversations: total count + per-conversation length list (for depth) ----
    conversation_lengths = _conversation_lengths(chat_per_user_ts)
    conversations = len(conversation_lengths)

    # ---- Engagement helpers (retention / per-user / depth) ----
    engagement_block = {
        "retention_curve": _compute_retention_curve(chat_per_user_ts),
        "per_user_distribution": _compute_engagement_distribution(chat_per_user_ts),
        "conversation_depth": _compute_conversation_depth(conversation_lengths),
    }

    # ---- WoW deltas ----
    deltas_block = _compute_deltas(
        chat_lines, all_lines, trace_rows, photos, voice_notes
    )

    # ---- 7×24 heatmap (local time) ----
    heatmap_matrix = _compute_heatmap_dow_hour(chat_lines)

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
            "tokens_in_total": tokens_in_total,
            "tokens_out_total": tokens_out_total,
            "free_tokens_generated": free_tokens_generated,
            "web_searches": web_search_total,
            "url_fetches": url_fetches,
            "voice_notes": voice_notes,
            "photos": photos,
            "deletions_invoked": deletions,
            # Community-contribution loop (added 2026-05-25). Total
            # translations submitted, per-dialect breakdown, and the
            # number of unique native-speaker contributors. The
            # dataset is post-collection-pre-review at this stage;
            # numbers include unreviewed submissions.
            "contributions_total": contrib_stats["total_contributions"],
            "contributions_by_dialect": contrib_stats["by_dialect"],
            "contributors_count": contrib_stats["total_contributors"],
        },
        "totals_deltas": deltas_block,
        "engagement": engagement_block,
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
            # 7×24 matrix in Africa/Windhoek local time. Replaces the
            # two separate histograms above on the rendered page.
            "heatmap_dow_hour": heatmap_matrix,
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
            "languages": _collapse_languages(
                _format_synthesis_block(
                    "languages",
                    denominator=len(unique_users),
                    empty_status="Computing — extracting preferred languages.",
                )
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
