#!/usr/bin/env python3
"""Analyze in-window proactive follow-up performance.

Joins the send pipeline (in_window_followup.log + snapshots) with
downstream chat activity (usage.log + per-user memory) to measure
who got pinged, who replied, and whether the experience was good.

Run inside the webhook container:

    docker exec -i ongiini-webhook python3 -m scripts.analyze_followups \
        --since 2026-05-26 --until 2026-05-27 \
        --classify-sentiment

Or with explicit snapshots (when the snapshot file naming doesn't
match the conventions we look for):

    docker exec -i ongiini-webhook python3 -m scripts.analyze_followups \
        --snapshot /data/in_window_clean.json \
        --snapshot /data/in_window_pending_afternoon_2026-05-27_sent.json

Inputs (auto-discovered in --data-dir, default /data):
  - in_window_followup.log          one JSON line per send attempt
  - in_window_pending*.json         all matching snapshot files with
                                    msisdn + topic + follow_up text
  - usage.log                       pipe-delimited turn events
                                    (router/chat/memory per turn)
  - <msisdn>.json                   per-user short-term memory

Outputs:
  - stdout: human-readable report (Markdown)
  - --report-json PATH: machine-readable
  - --report-md PATH: same as stdout but to a file
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import dataclasses
import datetime as dt
import glob
import hashlib
import json
import logging
import re
import statistics
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("analyze_followups")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

WINDHOEK = dt.timezone(dt.timedelta(hours=2))

# Pipe-delimited usage.log row:
#   2026-05-27T09:15:21 | 264812003072 | tokens_in=5 tokens_out=2 | search=no | kind=router
USAGE_LINE = re.compile(
    r"^(?P<ts>\S+)\s\|\s(?P<msi>\d+)\s\|\s.*?kind=(?P<kind>\w+)"
)


def h6(msisdn: str) -> str:
    return hashlib.sha256(msisdn.encode()).hexdigest()[:12]


# ── 1. Build the universe of sends ───────────────────────────────


@dataclasses.dataclass
class Send:
    """One proactive send."""
    msisdn: str                  # full msisdn (from snapshot)
    msisdn_hash6: str
    send_ts_utc: dt.datetime
    topic: str = ""
    follow_up: str = ""
    source_snapshot: str = ""    # which pending file the text came from
    batch_label: str = ""        # e.g. "morning_2026-05-26", "afternoon_2026-05-27"


def load_send_log(log_path: Path) -> list[dict]:
    """Read the per-send log into a list of dicts."""
    if not log_path.exists():
        return []
    out: list[dict] = []
    with log_path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out


def load_snapshots(data_dir: Path, explicit: list[Path] | None) -> dict[str, dict]:
    """Build hash → {msisdn, topic, follow_up, source_snapshot} from
    all available pending snapshots. Later snapshots win on conflict
    (i.e. an explicit --snapshot or a more recent file overrides)."""
    if explicit:
        files = sorted(explicit, key=lambda p: p.stat().st_mtime)
    else:
        # Pull every in_window_pending* and in_window_clean*. Sort by
        # mtime ascending so the newest writes win the last-write race.
        patterns = ["in_window_pending*.json", "in_window_clean*.json"]
        seen: set[Path] = set()
        files: list[Path] = []
        for pat in patterns:
            for p in sorted(data_dir.glob(pat)):
                if p not in seen:
                    seen.add(p)
                    files.append(p)
        files.sort(key=lambda p: p.stat().st_mtime)

    by_hash: dict[str, dict] = {}
    for f in files:
        try:
            rows = json.loads(f.read_text())
        except Exception as exc:                # noqa: BLE001
            log.warning("could not parse %s: %s", f, exc)
            continue
        if not isinstance(rows, list):
            continue
        for r in rows:
            m = r.get("msisdn")
            if not m:
                continue
            entry = {
                "msisdn": m,
                "topic": r.get("topic", ""),
                "follow_up": r.get("follow_up", ""),
                "status": r.get("status", ""),
                "skip_reason": r.get("skip_reason", ""),
                "source_snapshot": f.name,
            }
            by_hash[h6(m)] = entry
    log.info("snapshots scanned: %d files, %d unique users", len(files), len(by_hash))
    return by_hash


def build_sends(
    send_records: list[dict],
    snapshots: dict[str, dict],
    since: dt.datetime | None,
    until: dt.datetime | None,
) -> list[Send]:
    """Combine send records with snapshot text. Filter by time window."""
    sends: list[Send] = []
    skipped_no_snapshot = 0
    skipped_out_of_window = 0

    for rec in send_records:
        if rec.get("status") != "sent":
            continue
        h = rec.get("msisdn_hash6")
        ts_s = rec.get("ts")
        if not (h and ts_s):
            continue
        try:
            ts = dt.datetime.fromisoformat(ts_s.replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        if since and ts < since:
            skipped_out_of_window += 1
            continue
        if until and ts >= until:
            skipped_out_of_window += 1
            continue

        snap = snapshots.get(h)
        if not snap:
            skipped_no_snapshot += 1
            # We can still track the send, just without text/topic
            sends.append(Send(
                msisdn="(unknown)",
                msisdn_hash6=h,
                send_ts_utc=ts,
                batch_label=batch_label(ts),
            ))
            continue

        sends.append(Send(
            msisdn=snap["msisdn"],
            msisdn_hash6=h,
            send_ts_utc=ts,
            topic=snap.get("topic", ""),
            follow_up=snap.get("follow_up", ""),
            source_snapshot=snap.get("source_snapshot", ""),
            batch_label=batch_label(ts),
        ))

    log.info(
        "sends: %d in-window, %d out-of-window-filtered, %d no-snapshot-match",
        len(sends), skipped_out_of_window, skipped_no_snapshot,
    )
    return sends


def batch_label(ts_utc: dt.datetime) -> str:
    """Human-readable batch label based on Windhoek date and hour bucket."""
    local = ts_utc.astimezone(WINDHOEK)
    bucket = "morning" if local.hour < 12 else "afternoon" if local.hour < 18 else "evening"
    return f"{bucket}_{local.date().isoformat()}"


# ── 2. Reply detection from usage.log ────────────────────────────


def find_replies(
    usage_path: Path,
    sends: list[Send],
) -> dict[str, list[dt.datetime]]:
    """For each msisdn in sends, return the list of user-turn timestamps
    that happened AFTER our send. We use `kind=router` as the
    user-turn signal (router fires once per inbound message)."""
    if not usage_path.exists():
        log.warning("usage.log not found at %s", usage_path)
        return {}

    # Index sends by full msisdn (we need the full one to match usage.log,
    # which contains full numbers, not hashes).
    sends_by_msisdn: dict[str, list[dt.datetime]] = collections.defaultdict(list)
    for s in sends:
        if s.msisdn != "(unknown)":
            sends_by_msisdn[s.msisdn].append(s.send_ts_utc)

    earliest_send = min((s.send_ts_utc for s in sends), default=None)
    if earliest_send is None:
        return {}

    replies: dict[str, list[dt.datetime]] = collections.defaultdict(list)
    with usage_path.open() as f:
        for ln in f:
            m = USAGE_LINE.match(ln.strip())
            if not m or m.group("kind") != "router":
                continue
            msi = m.group("msi")
            if msi not in sends_by_msisdn:
                continue
            try:
                ts = dt.datetime.fromisoformat(m.group("ts")).replace(
                    tzinfo=dt.timezone.utc
                )
            except ValueError:
                continue
            # Only count turns AFTER this user's first send
            if ts > min(sends_by_msisdn[msi]):
                replies[msi].append(ts)
    log.info("found user-turns for %d msisdns post-send", len(replies))
    return replies


# ── 3. Conversation slice from per-user memory ───────────────────


def conversation_after(
    msisdn: str,
    after_ts: dt.datetime,
    data_dir: Path,
    max_turns: int = 8,
) -> list[dict]:
    """Return the conversation turns that landed AFTER the given send
    timestamp. Reads /data/<msisdn>.json (short-term memory)."""
    p = data_dir / f"{msisdn}.json"
    if not p.exists():
        return []
    try:
        turns = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(turns, list):
        return []
    # Short-term memory entries don't have timestamps reliably; we use
    # position relative to a synthetic-assistant turn whose content
    # matches the follow_up. For most cases we can just take the LAST
    # `max_turns` and trust that captures the post-followup arc.
    return turns[-max_turns:]


# ── 4. Sentiment classifier (optional, Gemma-backed) ─────────────


SENTIMENT_PROMPT = """\
You are classifying how a user reacted to a proactive WhatsApp
follow-up from Ongiini AI. You will be shown the message we sent
and the conversation that followed.

Output JSON only, no surrounding text:

{
  "outcome": "positive" | "neutral" | "declined" | "hijacked" | "negative" | "no_reply",
  "evidence": "<= 100 char quote from user supporting the outcome",
  "engagement_turns": <int — count of user turns AFTER the follow-up>
}

Outcomes:
  positive  — user took up the offered topic or related, on-topic engagement,
              productive back-and-forth
  neutral   — user replied briefly, conversation ended naturally without
              friction
  declined  — user politely declined (no thanks / not now / busy)
  hijacked  — bot drifted to unrelated task (e.g., translation flow when
              follow-up was about something else)
  negative  — user expressed confusion, frustration, or annoyance
  no_reply  — no meaningful user response after the follow-up
"""


async def classify_sentiment(
    client, model: str, follow_up: str, conv: list[dict]
) -> dict:
    """One Gemma call per replied conversation."""
    if not conv:
        return {"outcome": "no_reply", "evidence": "", "engagement_turns": 0}
    lines = []
    for m in conv:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        if len(content) > 500:
            content = content[:500] + " […]"
        lines.append(f"[{role}] {content}")
    conv_text = "\n\n".join(lines)
    user_prompt = (
        f"FOLLOW-UP WE SENT:\n\n{follow_up}\n\n"
        f"---\n\nCONVERSATION AFTER (oldest first):\n\n{conv_text}\n"
    )
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SENTIMENT_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
    except Exception as exc:                    # noqa: BLE001
        log.warning("sentiment call failed: %s", exc)
        return {"outcome": "error", "evidence": str(exc)[:120], "engagement_turns": 0}
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return {"outcome": "error", "evidence": "unparsable JSON", "engagement_turns": 0}


# ── 5. Aggregation + Reporting ───────────────────────────────────


def aggregate(
    sends: list[Send],
    replies: dict[str, list[dt.datetime]],
    sentiments: dict[str, dict] | None,
) -> dict[str, Any]:
    by_batch: dict[str, dict] = collections.defaultdict(lambda: {
        "sent": 0, "replied": 0, "reply_gaps_sec": [],
        "outcomes": collections.Counter(),
    })
    by_topic_category: dict[str, dict] = collections.defaultdict(lambda: {
        "sent": 0, "replied": 0,
    })
    all_gaps: list[float] = []
    overall_outcomes: collections.Counter = collections.Counter()

    for s in sends:
        bb = by_batch[s.batch_label]
        bb["sent"] += 1
        user_turns = sorted(replies.get(s.msisdn, []))
        if user_turns:
            first_reply = user_turns[0]
            gap = (first_reply - s.send_ts_utc).total_seconds()
            if gap > 0:
                bb["replied"] += 1
                bb["reply_gaps_sec"].append(gap)
                all_gaps.append(gap)
        # Topic category — coarse buckets so we get signal across small N
        cat = topic_category(s.topic)
        by_topic_category[cat]["sent"] += 1
        if user_turns:
            by_topic_category[cat]["replied"] += 1
        if sentiments:
            sent_obj = sentiments.get(s.msisdn, {})
            outcome = sent_obj.get("outcome", "no_reply")
            bb["outcomes"][outcome] += 1
            overall_outcomes[outcome] += 1

    return {
        "totals": {
            "sent": len(sends),
            "replied": sum(b["replied"] for b in by_batch.values()),
            "median_gap_min": (
                statistics.median(all_gaps) / 60 if all_gaps else 0.0
            ),
            "p90_gap_min": (
                (statistics.quantiles(all_gaps, n=10)[8] / 60)
                if len(all_gaps) >= 10 else 0.0
            ),
        },
        "by_batch": {
            k: {
                "sent": v["sent"],
                "replied": v["replied"],
                "reply_rate": (
                    v["replied"] / v["sent"] if v["sent"] else 0.0
                ),
                "median_gap_min": (
                    statistics.median(v["reply_gaps_sec"]) / 60
                    if v["reply_gaps_sec"] else 0.0
                ),
                "outcomes": dict(v["outcomes"]),
            }
            for k, v in sorted(by_batch.items())
        },
        "by_topic_category": {
            k: {
                "sent": v["sent"],
                "replied": v["replied"],
                "reply_rate": (
                    v["replied"] / v["sent"] if v["sent"] else 0.0
                ),
            }
            for k, v in sorted(
                by_topic_category.items(),
                key=lambda x: -x[1]["sent"],
            )
        },
        "overall_outcomes": dict(overall_outcomes) if sentiments else None,
    }


def topic_category(topic: str) -> str:
    """Group fine-grained topics into coarse buckets for cohort analysis."""
    t = topic.lower()
    if any(w in t for w in ("cv", "cover letter", "interview", "job", "application", "resume")):
        return "jobs"
    if any(w in t for w in ("exam", "study", "revision", "biology", "chemistry", "physics", "math", "nssco", "academic")):
        return "study"
    if any(w in t for w in ("oshikwanyama", "oshindonga", "oshiwambo", "translation", "translate", "dataset")):
        return "translation"
    if any(w in t for w in ("afrikaans", "english", "language", "phrase", "vocabulary")):
        return "language"
    if any(w in t for w in ("business", "shop", "market", "hostel", "brand", "client")):
        return "business"
    if any(w in t for w in ("lesson", "teacher", "ecd", "class", "rubric")):
        return "teaching"
    if any(w in t for w in ("research", "thesis", "proposal", "honours")):
        return "research"
    if any(w in t for w in ("savings", "budget", "bank", "fraud", "finance")):
        return "finance"
    if any(w in t for w in ("relationship", "trust", "connection", "talk")):
        return "personal"
    return "other"


def render_markdown(report: dict, examples: list[dict]) -> str:
    """Pretty-print the aggregate as Markdown."""
    out: list[str] = []
    t = report["totals"]
    out.append(f"# In-window follow-up analysis")
    out.append(f"\nGenerated at: {dt.datetime.now(WINDHOEK).isoformat(timespec='seconds')}\n")
    out.append("## Headline")
    rate = (t["replied"] / t["sent"] * 100) if t["sent"] else 0
    out.append(
        f"\n- **{t['sent']} sent · {t['replied']} replied · {rate:.1f}% reply rate**\n"
        f"- median reply gap: **{t['median_gap_min']:.1f} min** · "
        f"p90: {t['p90_gap_min']:.1f} min\n"
    )

    out.append("## By batch")
    out.append("\n| Batch | Sent | Replied | Rate | Median gap |")
    out.append("|---|---|---|---|---|")
    for label, b in report["by_batch"].items():
        out.append(
            f"| {label} | {b['sent']} | {b['replied']} | "
            f"{b['reply_rate']*100:.0f}% | {b['median_gap_min']:.1f} min |"
        )

    out.append("\n## By topic category")
    out.append("\n| Category | Sent | Replied | Rate |")
    out.append("|---|---|---|---|")
    for cat, b in report["by_topic_category"].items():
        out.append(
            f"| {cat} | {b['sent']} | {b['replied']} | {b['reply_rate']*100:.0f}% |"
        )

    if report["overall_outcomes"]:
        out.append("\n## Conversation outcomes (Gemma-classified)")
        out.append("\n| Outcome | Count | Share |")
        out.append("|---|---|---|")
        total = sum(report["overall_outcomes"].values())
        for outcome in ("positive", "neutral", "declined", "hijacked",
                        "negative", "no_reply", "error"):
            c = report["overall_outcomes"].get(outcome, 0)
            share = c / total * 100 if total else 0
            out.append(f"| {outcome} | {c} | {share:.0f}% |")

    if examples:
        out.append("\n## Per-outcome examples")
        for ex in examples:
            out.append(f"\n### {ex['outcome']} — ...{ex['msisdn'][-4:]} · {ex['topic']}")
            out.append(f"\n**Follow-up:** {ex['follow_up']}")
            out.append(f"\n**Evidence:** {ex['evidence']}")
            out.append(f"\n**Engagement turns:** {ex['engagement_turns']}")

    return "\n".join(out) + "\n"


# ── 6. CLI ───────────────────────────────────────────────────────


def parse_date(s: str) -> dt.datetime:
    """YYYY-MM-DD → midnight UTC."""
    d = dt.date.fromisoformat(s)
    return dt.datetime.combine(d, dt.time.min, dt.timezone.utc)


async def main_async(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    explicit_snaps = [Path(p) for p in args.snapshot] if args.snapshot else None
    since = parse_date(args.since) if args.since else None
    until = parse_date(args.until) + dt.timedelta(days=1) if args.until else None

    log.info("loading snapshots …")
    snaps = load_snapshots(data_dir, explicit_snaps)

    log.info("reading send log …")
    send_records = load_send_log(data_dir / "in_window_followup.log")
    log.info("send log entries: %d", len(send_records))

    sends = build_sends(send_records, snaps, since, until)
    if not sends:
        print("(no sends found in the requested window)", file=sys.stderr)
        return 1

    log.info("scanning usage.log for replies …")
    replies = find_replies(data_dir / "usage.log", sends)

    sentiments: dict[str, dict] = {}
    examples: list[dict] = []

    if args.classify_sentiment:
        log.info("classifying sentiment via Gemma (this hits vLLM, "
                 "expect ~1.5s per replied conv) …")
        try:
            # Import here so the script still runs without openai installed
            # for pure-aggregation mode.
            from openai import AsyncOpenAI
            from ongiini.config import settings
            client = AsyncOpenAI(
                base_url=settings.vllm_base_url, api_key="not-needed",
            )
            model = settings.vllm_model
        except Exception as exc:                # noqa: BLE001
            log.error("could not init Gemma client: %s — skipping sentiment", exc)
        else:
            per_outcome: dict[str, list[dict]] = collections.defaultdict(list)
            for i, s in enumerate(sends, start=1):
                user_turns = replies.get(s.msisdn, [])
                if not user_turns:
                    sentiments[s.msisdn] = {
                        "outcome": "no_reply", "evidence": "",
                        "engagement_turns": 0,
                    }
                    continue
                conv = conversation_after(
                    s.msisdn, s.send_ts_utc, data_dir, max_turns=10,
                )
                result = await classify_sentiment(
                    client, model, s.follow_up, conv,
                )
                sentiments[s.msisdn] = result
                # Stash for per-outcome examples
                if len(per_outcome[result.get("outcome", "")]) < 3:
                    per_outcome[result["outcome"]].append({
                        "msisdn": s.msisdn,
                        "topic": s.topic,
                        "follow_up": s.follow_up,
                        "outcome": result.get("outcome", ""),
                        "evidence": result.get("evidence", ""),
                        "engagement_turns": result.get("engagement_turns", 0),
                    })
                if i % 10 == 0:
                    log.info("classified %d / %d", i, len(sends))
                await asyncio.sleep(0.2)  # be nice to GPU
            # Flatten examples in a useful order
            for outcome in ("positive", "negative", "hijacked",
                            "declined", "neutral", "no_reply", "error"):
                examples.extend(per_outcome.get(outcome, []))

    report = aggregate(sends, replies, sentiments or None)

    md = render_markdown(report, examples)
    print(md)

    if args.report_md:
        Path(args.report_md).write_text(md)
        log.info("wrote markdown to %s", args.report_md)
    if args.report_json:
        Path(args.report_json).write_text(json.dumps({
            "report": report,
            "examples": examples,
        }, indent=2, default=str))
        log.info("wrote json to %s", args.report_json)

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Analyze in-window proactive follow-up performance",
    )
    p.add_argument("--data-dir", default="/data",
                   help="Where /data lives. Default: /data (in-container).")
    p.add_argument("--snapshot", action="append", default=None,
                   help="Path to a pending-snapshot file. Repeatable. "
                        "If omitted, auto-discovers in_window_pending*.json "
                        "and in_window_clean*.json.")
    p.add_argument("--since", type=str, default=None,
                   help="YYYY-MM-DD. Lower bound on send timestamp (UTC).")
    p.add_argument("--until", type=str, default=None,
                   help="YYYY-MM-DD. Upper bound (inclusive of that date).")
    p.add_argument("--classify-sentiment", action="store_true",
                   help="Run Gemma over each replied conversation to "
                        "classify outcome. Adds ~1.5s per replied user.")
    p.add_argument("--report-md", type=str, default=None,
                   help="Also write markdown report to this path.")
    p.add_argument("--report-json", type=str, default=None,
                   help="Also write structured JSON report to this path.")
    args = p.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
