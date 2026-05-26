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
from datetime import datetime, timedelta, timezone
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


def find_candidates(
    *,
    cutoff_hour: int = 13,
    min_lines: int = 5,
    min_tokens: int = 500,
) -> list[dict]:
    """All users whose LAST chat-line today was at or before cutoff_hour
    (Africa/Windhoek local), with at least `min_lines` chat lines AND
    `min_tokens` total tokens_out generated. Opt-outs are excluded.
    """
    now_local = datetime.now(NAMIBIA_TZ)
    today = now_local.date()
    cutoff = datetime.combine(today, datetime.min.time(), NAMIBIA_TZ).replace(hour=cutoff_hour)

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
            if ts_local.date() != today:
                continue
            msi = m.group("msi")
            u = per_user.setdefault(
                msi, {"ts_list": [], "tokens_out": 0, "lines": 0}
            )
            u["ts_list"].append(ts_local)
            u["tokens_out"] += int(m.group("tout"))
            u["lines"] += 1

    excluded_hashes = opt_outs.all_opted_out_hashes()
    candidates: list[dict] = []

    for msi, u in per_user.items():
        if u["lines"] < min_lines:
            continue
        if u["tokens_out"] < min_tokens:
            continue
        last_ts = max(u["ts_list"])
        if last_ts > cutoff:
            continue
        # Country allowlist
        try:
            if not is_allowed(msi):
                continue
        except InvalidMsisdn:
            continue
        # Opt-out filter
        try:
            from ..contributions import hash_msisdn
            if hash_msisdn(msi) in excluded_hashes:
                continue
        except RuntimeError:
            # Hash salt missing; can't safely check opt-out. Refuse.
            log.warning("hash salt missing — refusing to enumerate candidates")
            return []
        candidates.append({
            "msisdn": msi,
            "last_ts": last_ts.isoformat(),
            "lines": u["lines"],
            "tokens_out": u["tokens_out"],
        })

    candidates.sort(key=lambda c: c["last_ts"])
    return candidates


# ── 2. Generation via Gemma ──────────────────────────────────────


# v3 prompt — looser than v2. v2 was correctly more careful about not
# proposing "continue your CV" but became too defensive, skipping 92%
# of candidates. v3 keeps the "no same-task chasing" + the hard
# deliverability list, but explicitly encourages creative adjacency
# and DEFAULTS to generating instead of defaulting to skipping.
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

SKIP ONLY when one of these clearly applies (output {"skip": true,
"reason": "..."}):
- Conversation involved suicide ideation, self-harm, severe distress
- User was upset, angry, or explicitly said goodbye / "done for today"
- Bot's last reply was an apology / "I can't help with that"
  (a nudge would just remind them of the failure)
- Conversation was clearly just a test or curiosity — no real task
  or domain to build on
- Truly no useful adjacent step exists — but try harder first,
  most real conversations have one

DEFAULT to generating. Skipping should be the exception.

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


def _build_user_context(msisdn: str, max_turns: int = 12) -> tuple[str, str]:
    """Return (short_term_conversation, long_term_facts) as plain text
    blocks the model can read directly."""
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
            for f in facts[:20]:
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
    cutoff_hour: int,
    min_lines: int,
    min_tokens: int,
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
            min_lines=min_lines,
            min_tokens=min_tokens,
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
    g.add_argument("--cutoff-hour", type=int, default=13,
                   help="Last-message hour (local). Default 13 (1pm).")
    g.add_argument("--min-lines", type=int, default=5,
                   help="Min chat lines today. Default 5.")
    g.add_argument("--min-tokens", type=int, default=500,
                   help="Min total tokens_out. Default 500.")
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
        out = asyncio.run(generate_batch(
            cutoff_hour=args.cutoff_hour,
            min_lines=args.min_lines,
            min_tokens=args.min_tokens,
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
