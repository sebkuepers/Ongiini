"""Analyse where the last N Ongiini users came from.

Run on Spark where the data lives:

    ssh spark-dccf.local 'python3 ~/dev/Ongiini/scripts/analyze_user_growth.py'

Common variations:

    # Last 100 users (default), text output
    python3 scripts/analyze_user_growth.py

    # Last 250 users, with a daily-arrival breakdown
    python3 scripts/analyze_user_growth.py --count 250 --daily

    # Just users who first arrived in the last 7 days
    python3 scripts/analyze_user_growth.py --window-days 7

    # JSON output (for piping into jq, Grafana annotations, etc.)
    python3 scripts/analyze_user_growth.py --format json

The script reads:
- /home/nexus/dev/Ongiini/data/trace.jsonl  (one row per turn, has msisdn + ts)
- /home/nexus/dev/Ongiini/data/<msisdn>.json (per-user short-term memory)

The per-user JSONs were already PII-scrubbed at write time (placeholders
like ``[REDACTED:email]`` are already inlined), so reading them is safe.
This script NEVER emits raw message bodies — only counts, buckets, and
classification flags. The only user identifier shown is the masked msisdn
(country code + last 3 digits).

Classification (first user message of each user):

- ``fb_ad_prefill`` — matches the literal Meta click-to-chat pre-fill
  text shipped with our Facebook ad ("Hello! Can I get more info on
  this?" and close variants). High confidence the user clicked the ad.
- ``bare_greeting`` — just "hi" / "hello" / "ongiini" / "good morning"
  with nothing else. AMBIGUOUS: could be an ad click whose user typed
  fresh, or a friend-referred organic arrival who's testing the bot.
- ``organic_specific`` — anything with a real question, named topic,
  or sentence-level content. High confidence the user already knows
  what they want from us = organic / referral / word-of-mouth.
- ``unknown`` — empty message or no history file readable.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Defaults ────────────────────────────────────────────────────────
# Production data path on Spark. Override with --data-dir for local
# development / replaying a /data snapshot.
DEFAULT_DATA = Path("/home/nexus/dev/Ongiini/data")
DEFAULT_COUNT = 100


# ── First-message classification patterns ─────────────────────────
# Pinned to the literal Meta click-to-chat WhatsApp ad template + a
# handful of natural typos/variants we've observed in production.
# These are HIGH CONFIDENCE FB-ad signals — kept narrow so we don't
# accidentally classify organic users into the ad bucket.
_FB_PATTERNS = [
    re.compile(r"\bcan i get more info(rmation)?\b", re.IGNORECASE),
    re.compile(r"\bmore info(rmation)? (on|about) this\b", re.IGNORECASE),
    re.compile(r"^hello[!.\s]+can i (get|have)\b", re.IGNORECASE),
    re.compile(r"^hi[,.!\s]+i'?d like to know more\b", re.IGNORECASE),
    re.compile(r"^hello[,.!\s]+how does this work\b", re.IGNORECASE),
    re.compile(r"^hi[,.!\s]+can you help me\??$", re.IGNORECASE),
]

# Bare greetings — "hi", "hello", "hey", "ongiini" alone or with one
# trailing punctuation/emoji. Could be ad-typed-fresh or organic — the
# script flags them as AMBIGUOUS and reports separately, so the
# organic/ad ratio isn't muddied by guesses.
_BARE_GREETING_RE = re.compile(
    r"^(hi|hello|hey|hola|halo|moro|howzit|ongiini|good morning|"
    r"good afternoon|good evening|wat se jy|moin)"
    r"[!.,?\s🇳🇦👋🏾👋🏿👋🏽👋🏼👋🏻👋]*$",
    re.IGNORECASE,
)


def classify_first_msg(text: str) -> str:
    """Return one of: fb_ad_prefill / bare_greeting / organic_specific
    / unknown. Order matters — FB pre-fills also start with 'Hello!' so
    we check them first, then bare greetings, then default to organic."""
    text = (text or "").strip()
    if not text:
        return "unknown"
    for pat in _FB_PATTERNS:
        if pat.search(text):
            return "fb_ad_prefill"
    if _BARE_GREETING_RE.match(text):
        return "bare_greeting"
    return "organic_specific"


# ── Data loading ────────────────────────────────────────────────────


def load_first_seen(trace_path: Path) -> dict[str, str]:
    """Walk trace.jsonl, return {msisdn: earliest_iso_ts} for Namibian
    numbers only. Older format rows still have ts + msisdn even when
    they don't have the newer router/policy fields, so this works
    across the full file lifetime."""
    first: dict[str, str] = {}
    with trace_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            msisdn = r.get("msisdn") or ""
            ts = r.get("ts") or ""
            if not msisdn.startswith("264") or not ts:
                continue
            if msisdn not in first or ts < first[msisdn]:
                first[msisdn] = ts
    return first


def first_user_message(history_path: Path) -> str:
    """Return the verbatim first user message from a per-user JSON
    file, or empty string if the file is missing/malformed/empty."""
    if not history_path.exists():
        return ""
    try:
        history = json.loads(history_path.read_text())
    except (json.JSONDecodeError, OSError):
        return ""
    for m in history:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content", "")
            return c if isinstance(c, str) else ""
    return ""


# ── Reporting ───────────────────────────────────────────────────────


def mask(msisdn: str) -> str:
    """Country code + last 3 digits, middle masked. Reversible-enough
    for the operator to recognise a number they know, but not for a
    casual reader to recover the full identifier."""
    if len(msisdn) < 7:
        return msisdn
    return f"{msisdn[:3]}***{msisdn[-3:]}"


def parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def build_report(data_dir: Path, count: int, window_days: int | None) -> dict:
    """Pure function — returns a dict ready to be rendered as text or JSON."""
    trace = data_dir / "trace.jsonl"
    if not trace.exists():
        return {"error": f"trace.jsonl not found at {trace}"}

    first_seen = load_first_seen(trace)
    if not first_seen:
        return {"error": "no Namibian users found in trace.jsonl"}

    # Window filter first, then take the newest N
    if window_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        windowed = {
            m: ts for m, ts in first_seen.items()
            if (p := parse_ts(ts)) and p >= cutoff
        }
    else:
        windowed = first_seen

    # Sort newest first, take top N
    cohort = sorted(windowed.items(), key=lambda x: x[1], reverse=True)[:count]

    if not cohort:
        return {"error": "no users in the requested window"}

    # Classify each user
    buckets: Counter = Counter()
    by_day: defaultdict[str, Counter] = defaultdict(Counter)
    per_user: list[dict] = []
    for msisdn, first_ts in cohort:
        first_msg = first_user_message(data_dir / f"{msisdn}.json")
        bucket = classify_first_msg(first_msg)
        buckets[bucket] += 1
        day = first_ts[:10]
        by_day[day][bucket] += 1
        per_user.append({
            "id_masked": mask(msisdn),
            "first_seen": first_ts[:19],
            "first_msg_len": len(first_msg),
            "bucket": bucket,
        })

    total = len(cohort)
    oldest = min(t for _, t in cohort)
    newest = max(t for _, t in cohort)
    fb = buckets.get("fb_ad_prefill", 0)
    bare = buckets.get("bare_greeting", 0)
    organic = buckets.get("organic_specific", 0)
    unknown = buckets.get("unknown", 0)
    organic_share_strict = organic / total if total else 0
    organic_share_optimistic = (organic + bare) / total if total else 0

    return {
        "cohort_size": total,
        "window_days": window_days,
        "first_seen_oldest": oldest,
        "first_seen_newest": newest,
        "buckets": {
            "fb_ad_prefill": fb,
            "bare_greeting": bare,
            "organic_specific": organic,
            "unknown": unknown,
        },
        "ratios": {
            "fb_ad_pct": round(100 * fb / total, 1),
            "bare_greeting_pct": round(100 * bare / total, 1),
            "organic_pct": round(100 * organic / total, 1),
            "unknown_pct": round(100 * unknown / total, 1),
            "organic_share_strict_pct": round(100 * organic_share_strict, 1),
            "organic_share_optimistic_pct": round(100 * organic_share_optimistic, 1),
        },
        "by_day": {
            day: dict(c) for day, c in sorted(by_day.items())
        },
        "per_user": per_user,
    }


def render_text(rep: dict, show_daily: bool = False, show_users: bool = False) -> str:
    if "error" in rep:
        return f"ERROR: {rep['error']}"
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("Ongiini — last-N-users source analysis")
    lines.append("=" * 70)
    lines.append(f"Cohort:    {rep['cohort_size']} users")
    if rep.get("window_days") is not None:
        lines.append(f"Window:    last {rep['window_days']} days")
    lines.append(f"Spans:     {rep['first_seen_oldest']}  →  {rep['first_seen_newest']}")
    lines.append("")
    b = rep["buckets"]
    r = rep["ratios"]
    lines.append("── First-message classification ─────────────────────────────")
    lines.append(f"  FB ad pre-fill (high confidence ad):   {b['fb_ad_prefill']:>4d}  ({r['fb_ad_pct']:>4.1f}%)")
    lines.append(f"  Bare greeting (ambiguous):             {b['bare_greeting']:>4d}  ({r['bare_greeting_pct']:>4.1f}%)")
    lines.append(f"  Organic / specific question:           {b['organic_specific']:>4d}  ({r['organic_pct']:>4.1f}%)")
    lines.append(f"  Unknown / unreadable:                  {b['unknown']:>4d}  ({r['unknown_pct']:>4.1f}%)")
    lines.append("")
    lines.append("── Organic share — two ways to read it ──────────────────────")
    lines.append(f"  Strict:     organic_specific / total = {r['organic_share_strict_pct']:>4.1f}%")
    lines.append(f"              (only counts users who arrived with a real question)")
    lines.append(f"  Optimistic: (organic + bare_greeting) / total = {r['organic_share_optimistic_pct']:>4.1f}%")
    lines.append(f"              (assumes all bare 'hi' arrivals are organic)")
    lines.append(f"  Truth is somewhere in between — bare greetings could be")
    lines.append(f"  ad-clickers who typed their own opener, or referred users.")

    if show_daily and rep.get("by_day"):
        lines.append("")
        lines.append("── Daily arrival breakdown ──────────────────────────────────")
        lines.append(f"  {'date':10s}  {'fb_ad':>5s}  {'bare':>5s}  {'organic':>7s}  {'unk':>3s}  {'total':>5s}  {'org%':>5s}")
        for day, c in rep["by_day"].items():
            fb_n = c.get("fb_ad_prefill", 0)
            bare_n = c.get("bare_greeting", 0)
            org_n = c.get("organic_specific", 0)
            unk_n = c.get("unknown", 0)
            tot = fb_n + bare_n + org_n + unk_n
            org_pct = 100 * org_n / tot if tot else 0
            lines.append(f"  {day}  {fb_n:>5d}  {bare_n:>5d}  {org_n:>7d}  {unk_n:>3d}  {tot:>5d}  {org_pct:>4.1f}%")

    if show_users:
        lines.append("")
        lines.append("── Per-user table (newest first) ────────────────────────────")
        lines.append(f"  {'id':12s}  {'first seen':19s}  {'len':>4s}  bucket")
        for u in rep["per_user"]:
            lines.append(f"  {u['id_masked']:12s}  {u['first_seen']:19s}  {u['first_msg_len']:>4d}  {u['bucket']}")

    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Analyse where the last N Ongiini users came from.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--count", type=int, default=DEFAULT_COUNT,
                   help=f"How many newest users to include (default {DEFAULT_COUNT})")
    p.add_argument("--window-days", type=int, default=None,
                   help="Only include users first-seen within last N days "
                        "(default: no window, all-time)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA,
                   help=f"Data directory (default {DEFAULT_DATA})")
    p.add_argument("--format", choices=("text", "json"), default="text",
                   help="Output format (default text)")
    p.add_argument("--daily", action="store_true",
                   help="Include daily arrival breakdown in text output")
    p.add_argument("--users", action="store_true",
                   help="Include per-user table (masked ids) in text output")
    args = p.parse_args(argv)

    rep = build_report(args.data_dir, args.count, args.window_days)
    if args.format == "json":
        json.dump(rep, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(render_text(rep, show_daily=args.daily, show_users=args.users))
    return 0 if "error" not in rep else 1


if __name__ == "__main__":
    sys.exit(main())
