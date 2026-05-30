"""In-window proactive follow-up — FREE.

For users whose last meaningful conversation today ended before a cutoff
(default 13:00 Africa/Windhoek), generate a context-aware follow-up via
Gemma and send it as a free-form text message within their open 24-hour
service window. No template, no Meta template fee.

Three-stage CLI so the human can sanity-check the LLM output before sends:

    1. python -m ongiini.broadcast.in_window_followup generate
       → identifies candidates, calls Gemma per user, writes batch to
         /data/in_window_pending.json
    2. python -m ongiini.broadcast.in_window_followup review
       → prints the batch for human review (msisdn hash + generated text)
    3. python -m ongiini.broadcast.in_window_followup send [--dry-run]
       → sends each pending follow-up via send_text, writes memory turn
         first, throttles, logs per-recipient outcome
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from openai import AsyncOpenAI

from . import opt_outs
from .sender import clear_contribute_state_for_proactive
from ..config import settings
from ..filters import InvalidMsisdn, is_allowed, normalize
from ..memory import long_term as mem
from ..memory import short_term as memory
from ..whatsapp import send_text

log = logging.getLogger("ongiini.broadcast.in_window")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
)

NAMIBIA_TZ = timezone(timedelta(hours=2), name="Africa/Windhoek")
PENDING_PATH = settings.data_dir / "in_window_pending.json"

LINE_RE = re.compile(
    r"^(?P<ts>\S+)\s*\|\s*(?P<msi>\d+)\s*\|\s*"
    r"tokens_in=(?P<tin>\d+)\s+tokens_out=(?P<tout>\d+)\s*\|\s*"
    r"search=(?P<s>yes|no)\s*(?:\|\s*kind=(?P<k>\w+))?"
)


# ── 1. Candidate identification ──────────────────────────────────


def _recent_followup_recipient_hashes(within_hours: float) -> set[str]:
    """Read in_window_followup.log, return sha256[:12] hashes of msisdns
    we successfully sent a follow-up to within the last `within_hours`.
    Used to avoid double-tapping the same user across consecutive batches.
    """
    log_path = settings.data_dir / "in_window_followup.log"
    if not log_path.exists() or within_hours <= 0:
        return set()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=within_hours)
    hashes: set[str] = set()
    with log_path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if rec.get("status") != "sent":
                continue
            ts_raw = rec.get("ts")
            h = rec.get("msisdn_hash6")
            if not (ts_raw and h):
                continue
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                hashes.add(h)
    return hashes


def find_candidates(
    *,
    cutoff_hour: int | None = 13,
    after_hour: int | None = None,
    target_date: date | None = None,
    min_lines: int = 5,
    min_tokens: int = 500,
    exclude_active_today: bool = False,
    exclude_recent_followup_hours: float = 0.0,
) -> list[dict]:
    """Users whose chat activity on `target_date` (Windhoek local) matches
    the time filter.

      - `cutoff_hour` (default 13): include users whose LAST message on
        `target_date` was at-or-before this hour. Original behaviour.
      - `after_hour` (overrides cutoff_hour when set): include users
        whose LAST message on `target_date` was AFTER this hour. Used
        for catching the evening cohort the morning-batch missed.
      - `target_date` defaults to today (Windhoek local). Set to a past
        date to backfill (e.g. running tomorrow morning for users active
        yesterday afternoon).
      - `exclude_active_today`: drop any candidate who has ANY chat
        activity AFTER end-of-target_date. The point is to not nudge a
        user who's already chatting with us today — they don't need a
        proactive ping.
      - `exclude_recent_followup_hours`: drop candidates we already sent
        an in-window follow-up to within the last N hours (parsed from
        in_window_followup.log). Prevents stacking nudges.

    Threshold filters (`min_lines`, `min_tokens`) and the country / opt-out
    filters apply as before.
    """
    if after_hour is not None and cutoff_hour is not None:
        # Caller passed both — interpret as "after_hour wins" (the newer
        # arg). Keep cutoff_hour as the implicit fallback when neither.
        pass

    now_local = datetime.now(NAMIBIA_TZ)
    target = target_date or now_local.date()

    # Track per-user: chat events on target_date AND count of any chat
    # events strictly AFTER end-of-target_date (used for the
    # exclude_active_today filter).
    end_of_target = datetime.combine(
        target, datetime.min.time(), NAMIBIA_TZ
    ) + timedelta(days=1)

    per_user: dict[str, dict] = {}
    with (settings.data_dir / "usage.log").open() as f:
        for ln in f:
            m = LINE_RE.match(ln.strip())
            if not m or (m.group("k") or "chat") != "chat":
                continue
            ts_local = (
                datetime.fromisoformat(m.group("ts"))
                .replace(tzinfo=timezone.utc)
                .astimezone(NAMIBIA_TZ)
            )
            msi = m.group("msi")
            u = per_user.setdefault(
                msi,
                {"ts_list": [], "tokens_out": 0, "lines": 0,
                 "activity_after_target": 0},
            )
            if ts_local.date() == target:
                u["ts_list"].append(ts_local)
                u["tokens_out"] += int(m.group("tout"))
                u["lines"] += 1
            elif ts_local >= end_of_target:
                u["activity_after_target"] += 1

    excluded_hashes = opt_outs.all_opted_out_hashes()
    recent_followup_hashes = _recent_followup_recipient_hashes(
        exclude_recent_followup_hours
    )
    candidates: list[dict] = []

    for msi, u in per_user.items():
        if not u["ts_list"]:
            continue
        if u["lines"] < min_lines:
            continue
        if u["tokens_out"] < min_tokens:
            continue
        last_ts = max(u["ts_list"])
        # Time filter
        if after_hour is not None:
            threshold = datetime.combine(
                target, datetime.min.time(), NAMIBIA_TZ
            ).replace(hour=after_hour)
            if last_ts <= threshold:
                continue
        elif cutoff_hour is not None:
            threshold = datetime.combine(
                target, datetime.min.time(), NAMIBIA_TZ
            ).replace(hour=cutoff_hour)
            if last_ts > threshold:
                continue
        # Exclude users already chatting after target_date
        if exclude_active_today and u["activity_after_target"] > 0:
            continue
        # Country allowlist
        try:
            if not is_allowed(msi):
                continue
        except InvalidMsisdn:
            continue
        # Opt-out + recent-followup filter (share hash function)
        try:
            from ..contributions import hash_msisdn
            h_full = hash_msisdn(msi)
        except RuntimeError:
            log.warning("hash salt missing — refusing to enumerate candidates")
            return []
        if h_full in excluded_hashes:
            continue
        # Recent-followup uses sha256[:12] of msisdn (matches in_window_followup.log)
        if recent_followup_hashes:
            h_short = hashlib.sha256(msi.encode()).hexdigest()[:12]
            if h_short in recent_followup_hashes:
                continue
        candidates.append({
            "msisdn": msi,
            "last_ts": last_ts.isoformat(),
            "lines": u["lines"],
            "tokens_out": u["tokens_out"],
        })

    candidates.sort(key=lambda c: c["last_ts"])
    return candidates


# ── 2. Generation via Gemma ──────────────────────────────────────


# v3 prompt — adjacency framework + default-to-generate. The v4
# attempt with bordered hard-rules was more restrictive (14/46 vs
# 41/60) without a clear quality win, so we're back on v3 which is
# the validated production version with the 40-44% reply rate.
FOLLOWUP_SYSTEM_PROMPT = """\
You write proactive follow-up messages for Ongiini AI, a WhatsApp
helper for people in Namibia. You will be shown a user's conversation
from earlier today.

YOUR GOAL: Be a thoughtful friend who noticed what they were working
on and has a useful idea for what to do NEXT — something adjacent
to what they did, that builds on it or takes them to a natural
next step in their journey. Be CREATIVE.

Adjacency can be many things. Think about:
- The NEXT STAGE of their journey: CV drafted → interview prep →
  first-day-at-work tips → asking for a raise after 6 months.
- A PRACTICAL APPLICATION of what they were learning: Afrikaans
  vocabulary → how to use it in a real situation (a shop, a
  meeting, a date).
- A DIFFERENT ANGLE on the same domain: specific exam topic →
  broader study skills, or a related topic in the syllabus.
- A COMPLEMENTARY skill or task: drafted a memo → tips for short
  professional emails → handling a difficult reply.
- An EXTENSION that deepens the work: lesson plan → assessment
  rubric → ideas for student engagement.
- A SIDE-BENEFIT they may not have considered: business idea
  brainstorm → how to register at BIPA → opening a business bank
  account.

THE ONE RULE: Don't propose to do the SAME task they just did.
They got that done — or got enough that they stopped. Don't say
"want to continue X" or "ready to finish Y". Propose something
DIFFERENT but useful given what we now know they care about.

The fact that the user didn't reply to the bot's last message is
NORMAL — people are busy, get pulled away, save the chat for later.
It does NOT mean don't reach out. It means reach out about
something new and adjacent, not the same question.

NEVER propose things we cannot actually do:
- Editing, generating, designing images, photos, posters, flyers,
  logos, videos
- Sending voice notes or audio
- Sending downloadable files (PDFs, Word docs, spreadsheets)
- Booking, buying, signing up, transacting on the user's behalf
- Writing or translating INTO Oshiwambo / Oshindonga / Oshikwanyama
  (we cannot generate these languages reliably)
- Real-time data the bot can't actually access (live match scores,
  current stock prices, today's weather without search)
- Anything we don't already do in normal text chat

NEVER propose contributing more translations to the Oshiwambo dataset.
Even when the user has been working on translations or has recently
contributed, the follow-up should suggest a DIFFERENT type of help
(school, work, health, life questions, anything practical). Translation
contribution is a separate ask we make via dedicated invitations — it
is not what this nudge is for.

SKIP ONLY for safety reasons (output {"skip": true, "reason": "..."}):
- Conversation involved suicide ideation, self-harm, severe distress
- User explicitly asked us to stop messaging them

DO NOT skip for any of these reasons:
- "I couldn't find a clear next step" — look DEEPER into the
  conversation. Every user has been working on SOMETHING — go past
  the last 5 turns. Earlier in the same conversation, or in the
  long-term facts, there is ALWAYS a real domain to anchor to
  (a CV they drafted, a school topic they wrestled with, a business
  idea they brainstormed, a tax question, a relationship situation,
  a health concern, a music project, …)
- "Recent turns are just goodbye / 'thanks' / 'hello?'" — those
  recent turns are noise. The signal is in the substantive work
  earlier in the conversation and in the long-term facts.
- "User said they were done with topic X" — fine, propose something
  DIFFERENT from X. Don't go back to X, find another domain from
  their history.
- "Conversation was just curiosity / no real task" — look at the
  long-term facts. There's usually a fact ([SITUATION], [PROFILE])
  that anchors a useful nudge even when the chat itself was light.

WORK ORDER:
1. Read the LONG-TERM FACTS first. They name the real domains: roles,
   goals, situations, struggles. That's where the signal lives.
2. Read the conversation, skipping past recent goodbyes / silence.
   Find the substantive turns — the CV review, the homework problem,
   the business idea, the health worry, the legal question. ANY of
   them is a valid anchor.
3. Pick the most consequential or recent substantive domain.
4. Propose a SPECIFIC next step in that domain — never generic
   ("ask me your favourite topic"), always concrete and named
   ("ready to draft the constitution section we discussed?",
   "want me to help with the next chapter of your accounting study
   guide?", "should we look at the cover letter to go with that
   CV?").

DEFAULT to generating. Skipping is reserved for safety only.

LENGTH: 1-2 sentences. Max 200 characters total.
LANGUAGE: match the user's (English or Afrikaans). Brief warmth
words like "tangi" or "kaume" are fine if they used them. NEVER
generate sentences in Oshiwambo / Oshindonga / Oshikwanyama.
TONE: a friend with an idea, not a marketer asking permission.
"Here's a thought:..." energy, not "are you ready to...". Be
specific, warm, brief.

If safe to follow up, output:
{
  "skip": false,
  "topic": "what they did this morning, 3-8 words",
  "follow_up": "the message"
}

If skipping:
{
  "skip": true,
  "reason": "short explanation"
}

Output JSON only — no surrounding text.
"""


def _build_user_context(msisdn: str, max_turns: int = 40) -> tuple[str, str]:
    """Return (short_term_conversation, long_term_facts) as plain text
    blocks the model can read directly.

    max_turns bumped from 12 → 40 on 2026-05-30 after a real test where
    the user's last few turns were "stop translating" / "hello?" and
    Gemma skipped, missing that EARLIER in the same conversation they'd
    been working on non-profit / tax planning — exactly the kind of
    domain we should build on. With only 12 turns of context Gemma was
    making the call on stale recent noise; the older substantive work
    is what the suggestion should anchor to.
    """
    # Short-term: last N turns from per-user memory JSON
    mem_data = memory.load(msisdn)
    if not mem_data:
        return ("", "")
    recent = mem_data[-max_turns:]
    lines = []
    for m in recent:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        # Cap per-turn length so the prompt stays bounded
        if len(content) > 800:
            content = content[:800] + " […truncated]"
        lines.append(f"[{role}] {content}")
    short_term = "\n\n".join(lines)

    # Long-term: mem0 facts
    try:
        facts = mem.list_all(msisdn)
        if facts:
            fact_lines = []
            for f in facts[:40]:    # was 20 — surface more domain facts
                text = (f.get("memory") or f.get("text") or "").strip()
                if text:
                    fact_lines.append(f"- {text}")
            long_term = "\n".join(fact_lines)
        else:
            long_term = "(none on file)"
    except Exception as exc:                   # noqa: BLE001
        log.warning("mem0 list_all failed for %s: %s", msisdn, exc)
        long_term = "(could not load — proceeding without)"

    return short_term, long_term


async def generate_for_user(
    client: AsyncOpenAI, msisdn: str
) -> dict | None:
    """Call Gemma to generate the follow-up for one user. Returns the
    parsed JSON dict (with skip/topic/follow_up) or None on failure."""
    short_term, long_term = _build_user_context(msisdn)
    if not short_term:
        log.info("%s: no short-term memory, skipping", msisdn[-6:])
        return {"skip": True, "reason": "no_memory"}

    user_prompt = (
        f"CONVERSATION (oldest first):\n\n{short_term}\n\n"
        f"---\n\nLONG-TERM FACTS about this user:\n\n{long_term}\n"
    )

    try:
        resp = await client.chat.completions.create(
            model=settings.vllm_model,
            messages=[
                {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
    except Exception as exc:                   # noqa: BLE001
        log.warning("Gemma call failed for %s: %s", msisdn[-6:], exc)
        return None

    raw = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("%s: unparseable JSON from Gemma: %r", msisdn[-6:], raw[:200])
        return None

    # Length-cap the follow-up text so a noisy generation can't blow past
    # what's reasonable for a WhatsApp nudge.
    if not parsed.get("skip"):
        text = (parsed.get("follow_up") or "").strip()
        if len(text) > 280:
            text = text[:280].rstrip() + "…"
            parsed["follow_up"] = text

    return parsed


async def generate_batch(
    *,
    cutoff_hour: int | None,
    after_hour: int | None,
    target_date: date | None,
    min_lines: int,
    min_tokens: int,
    exclude_active_today: bool,
    exclude_recent_followup_hours: float,
    only_msisdn: list[str] | None,
    rate_per_sec: float = 5.0,
) -> dict:
    """Identify candidates + generate per-user follow-ups, write batch
    to PENDING_PATH. Returns a summary dict."""
    if only_msisdn:
        candidates = [{"msisdn": normalize(m), "last_ts": "n/a", "lines": -1, "tokens_out": -1}
                      for m in only_msisdn]
    else:
        candidates = find_candidates(
            cutoff_hour=cutoff_hour,
            after_hour=after_hour,
            target_date=target_date,
            min_lines=min_lines,
            min_tokens=min_tokens,
            exclude_active_today=exclude_active_today,
            exclude_recent_followup_hours=exclude_recent_followup_hours,
        )
    log.info("identified %d candidates", len(candidates))

    if not candidates:
        PENDING_PATH.write_text("[]\n")
        return {"candidates": 0, "generated": 0, "skipped": 0, "failed": 0}

    client = AsyncOpenAI(base_url=settings.vllm_base_url, api_key="not-needed")
    interval = 1.0 / max(0.1, rate_per_sec)

    results: list[dict] = []
    n_gen = n_skip = n_fail = 0
    for i, c in enumerate(candidates, start=1):
        loop_start = time.monotonic()
        parsed = await generate_for_user(client, c["msisdn"])
        if parsed is None:
            n_fail += 1
            results.append({
                **c,
                "status": "generation_failed",
            })
        elif parsed.get("skip"):
            n_skip += 1
            results.append({
                **c,
                "status": "skipped",
                "skip_reason": parsed.get("reason", ""),
            })
        else:
            n_gen += 1
            results.append({
                **c,
                "status": "generated",
                "topic": parsed.get("topic", ""),
                "follow_up": parsed.get("follow_up", ""),
            })

        if i % 10 == 0 or i == len(candidates):
            log.info(
                "progress: %d/%d (generated=%d, skipped=%d, failed=%d)",
                i, len(candidates), n_gen, n_skip, n_fail,
            )

        # Throttle so we don't pin the GPU at exact moment users are chatting
        elapsed = time.monotonic() - loop_start
        sleep_for = interval - elapsed
        if sleep_for > 0 and i < len(candidates):
            await asyncio.sleep(sleep_for)

    PENDING_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    )
    log.info("wrote %d rows to %s", len(results), PENDING_PATH)
    return {
        "candidates": len(candidates),
        "generated": n_gen,
        "skipped": n_skip,
        "failed": n_fail,
    }


# ── 3. Review (read-only) ────────────────────────────────────────


def cmd_review() -> int:
    if not PENDING_PATH.exists():
        print("(no pending batch — run `generate` first)")
        return 1
    rows = json.loads(PENDING_PATH.read_text())
    if not rows:
        print("(pending batch is empty)")
        return 0
    n_gen = sum(1 for r in rows if r.get("status") == "generated")
    n_skip = sum(1 for r in rows if r.get("status") == "skipped")
    n_fail = sum(1 for r in rows if r.get("status") == "generation_failed")
    print(f"# Pending batch: {len(rows)} rows  ({n_gen} generated, {n_skip} skipped, {n_fail} failed)")
    print()
    for i, r in enumerate(rows, start=1):
        suf = r["msisdn"][-6:]
        status = r.get("status", "?")
        if status == "generated":
            print(f"[{i:3}] ✓ {suf} — topic: {r.get('topic', '')!r}")
            print(f"        → {r.get('follow_up', '')}")
        elif status == "skipped":
            print(f"[{i:3}] · {suf} — SKIPPED: {r.get('skip_reason', '')}")
        else:
            print(f"[{i:3}] ! {suf} — {status}")
    return 0


# ── 4. Send via free in-window text ──────────────────────────────


async def send_batch(*, dry_run: bool, rate_per_sec: float = 5.0) -> dict:
    """Read PENDING_PATH, send each `generated` row as a free in-window
    text. Writes a synthetic assistant turn into per-user memory FIRST
    so user replies land with context."""
    if not PENDING_PATH.exists():
        log.error("no pending batch — run `generate` first")
        return {"sent": 0, "failed": 0, "skipped": 0}
    rows = json.loads(PENDING_PATH.read_text())
    to_send = [r for r in rows if r.get("status") == "generated"]
    log.info(
        "%d sendable rows (of %d total). dry_run=%s",
        len(to_send), len(rows), dry_run,
    )
    if not to_send:
        return {"sent": 0, "failed": 0, "skipped": len(rows) - len(to_send)}

    interval = 1.0 / max(0.1, rate_per_sec)
    n_sent = n_fail = 0
    log_path = settings.data_dir / "in_window_followup.log"

    for i, r in enumerate(to_send, start=1):
        loop_start = time.monotonic()
        msisdn = r["msisdn"]
        body = r["follow_up"]
        result = {"msisdn_hash6": hashlib.sha256(msisdn.encode()).hexdigest()[:12],
                  "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "len": len(body)}
        if dry_run:
            log.info("[dry-run] would send to %s (%d chars)", msisdn[-6:], len(body))
            result["status"] = "dry_run"
        else:
            # 1. Memory write first (same invariant as broadcast_to)
            try:
                async with memory.lock_for(msisdn):
                    memory.append_synthetic_assistant_turn(msisdn, body)
            except Exception as exc:               # noqa: BLE001
                log.exception("memory write failed for %s — skipping send", msisdn[-6:])
                result["status"] = "memory_write_failed"
                result["error"] = f"{type(exc).__name__}: {exc}"
                n_fail += 1
                with log_path.open("a") as f:
                    f.write(json.dumps(result, separators=(",", ":")) + "\n")
                continue
            # Clear stale contribute pending-state so the classifier
            # doesn't misroute the user's reply through the contribute
            # force-tool. Best-effort.
            clear_contribute_state_for_proactive(msisdn)
            # 2. Send via free-form text (FREE in 24h window)
            try:
                await send_text(msisdn, body)
                result["status"] = "sent"
                n_sent += 1
            except Exception as exc:               # noqa: BLE001
                log.warning("send_text failed for %s: %s", msisdn[-6:], exc)
                result["status"] = "send_failed"
                result["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                n_fail += 1
        with log_path.open("a") as f:
            f.write(json.dumps(result, separators=(",", ":")) + "\n")
        if i % 10 == 0 or i == len(to_send):
            log.info("progress: %d/%d (sent=%d, failed=%d)", i, len(to_send), n_sent, n_fail)
        elapsed = time.monotonic() - loop_start
        sleep_for = interval - elapsed
        if sleep_for > 0 and i < len(to_send):
            await asyncio.sleep(sleep_for)

    return {"sent": n_sent, "failed": n_fail,
            "skipped": len(rows) - len(to_send)}


# ── CLI ──────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="In-window proactive follow-up — free same-day re-engagement"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Find candidates + generate follow-ups via Gemma")
    g.add_argument("--target-date", type=str, default=None,
                   help="YYYY-MM-DD (Windhoek local) — date users were active. "
                        "Default: today.")
    grp = g.add_mutually_exclusive_group()
    grp.add_argument("--cutoff-hour", type=int, default=None,
                     help="Include users whose LAST msg on target-date was "
                          "AT-OR-BEFORE this hour. Default 13 if --after-hour not set.")
    grp.add_argument("--after-hour", type=int, default=None,
                     help="Include users whose LAST msg on target-date was "
                          "AFTER this hour. Use to catch the evening cohort.")
    g.add_argument("--min-lines", type=int, default=5,
                   help="Min chat lines on target-date. Default 5.")
    g.add_argument("--min-tokens", type=int, default=500,
                   help="Min total tokens_out on target-date. Default 500.")
    g.add_argument("--exclude-active-today", action="store_true",
                   help="Drop candidates with chat activity AFTER end-of-target-date "
                        "(don't ping users already in conversation today).")
    g.add_argument("--exclude-recent-followup-hours", type=float, default=0.0,
                   help="Drop candidates we already sent a follow-up to within the "
                        "last N hours. 0 = no filter.")
    g.add_argument("--only-msisdn", action="append", default=None,
                   help="Restrict to one or more msisdns (smoke test). Repeatable.")
    g.add_argument("--rate-per-sec", type=float, default=5.0)

    sub.add_parser("review", help="Print the pending batch for human review")

    s = sub.add_parser("send", help="Send the pending batch via free in-window text")
    s.add_argument("--dry-run", action="store_true",
                   help="Log what would send, but do not actually send")
    s.add_argument("--rate-per-sec", type=float, default=5.0)

    args = p.parse_args(argv)
    if args.cmd == "generate":
        # Default to legacy behaviour (cutoff_hour=13) when neither flag set,
        # so existing invocations keep working unchanged.
        cutoff = args.cutoff_hour
        after = args.after_hour
        if cutoff is None and after is None:
            cutoff = 13
        target = None
        if args.target_date:
            try:
                target = date.fromisoformat(args.target_date)
            except ValueError:
                log.error("--target-date must be YYYY-MM-DD, got %r", args.target_date)
                return 2
        out = asyncio.run(generate_batch(
            cutoff_hour=cutoff,
            after_hour=after,
            target_date=target,
            min_lines=args.min_lines,
            min_tokens=args.min_tokens,
            exclude_active_today=args.exclude_active_today,
            exclude_recent_followup_hours=args.exclude_recent_followup_hours,
            only_msisdn=args.only_msisdn,
            rate_per_sec=args.rate_per_sec,
        ))
        log.info(
            "DONE: %d candidates → %d generated, %d skipped, %d failed",
            out["candidates"], out["generated"], out["skipped"], out["failed"],
        )
        log.info("review with: python -m ongiini.broadcast.in_window_followup review")
        return 0
    if args.cmd == "review":
        return cmd_review()
    if args.cmd == "send":
        out = asyncio.run(send_batch(
            dry_run=args.dry_run,
            rate_per_sec=args.rate_per_sec,
        ))
        log.info("DONE: sent=%d, failed=%d, skipped=%d",
                 out["sent"], out["failed"], out["skipped"])
        return 0 if out.get("failed", 0) == 0 else 1
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
