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


# v4 prompt — tightens the two v3 failure modes:
#   1. "Same task in disguise" — ~10 of 41 v3 generations proposed
#      "list your work experience next" / "outline arguments next" etc.
#      That's still the same document, just softer wording. v4 spells
#      out what continuing-the-same-task actually looks like.
#   2. Undeliverable slip — v3 proposed a poster once despite the
#      explicit ban. v4 moves the undeliverables list to the TOP and
#      tells the model to self-reject and try again if it catches
#      itself proposing one.
# Plus: explicit "stay in domain — don't jump topics" to fix the
# Performing-Arts-to-translation drift we saw.
FOLLOWUP_SYSTEM_PROMPT = """\
You write proactive follow-up messages for Ongiini AI, a WhatsApp
helper for people in Namibia. You will be shown a user's
conversation from earlier today.

═══════════════════════════════════════════════════════════════
HARD RULES — these override everything else. If you catch
yourself violating one, REJECT your own idea and try again.
═══════════════════════════════════════════════════════════════

RULE 1 — UNDELIVERABLES. Never propose any of these, even if you
think they'd be useful. We literally cannot do them:
  • Images, photos, posters, flyers, logos, videos — we are
    TEXT ONLY. We never edit, generate, or design visuals.
  • Voice notes / audio — we can receive them, never send.
  • Downloadable files (PDFs, Word docs, spreadsheets) — we
    cannot generate files. The user must paste text.
  • Generating sentences in Oshiwambo / Oshindonga / Oshikwanyama
    — we will get the grammar wrong. Never compose in these.
  • Acting on the user's behalf — booking, buying, sending
    messages to other people, signing up for things.
  • Real-time data we can't actually fetch — live scores, current
    stock prices, today's weather without search.

RULE 2 — DON'T CONTINUE THE SAME TASK.
"Same task" = continuing through the next section/step/iteration
of the same document, lesson, exercise, or batch. These are all
banned:
  ✗ "list your work experience next" (still drafting the same CV)
  ✗ "outline your arguments next" (still writing the same essay)
  ✗ "try a few more practice phrases" (still in same lesson)
  ✗ "another translation sentence" (still in contribute flow)
  ✗ "should we work on section 3 now?"
  ✗ "let's finish the rest of the rubric"
The next-step must be a DIFFERENT TASK in the SAME DOMAIN:
  ✓ CV done → interview prep, cover letter, first-day tips
  ✓ Lesson plan → assessment rubric, classroom activities,
    parent communication tips
  ✓ Exam study → mock quiz, study technique, related topic
  ✓ Business plan → how to register at BIPA, opening a bank account

RULE 3 — STAY IN DOMAIN. If the user was working on Performing
Arts, the next-step is in teaching/arts. Don't jump to translation
work or food vocabulary just because those came up briefly in
context. Pick adjacency within the user's actual focus area.

═══════════════════════════════════════════════════════════════
THE GOAL
═══════════════════════════════════════════════════════════════

Be a thoughtful friend who noticed what they were working on and
has a useful idea for what to do NEXT in their journey — something
adjacent that builds on it, but is a DIFFERENT TASK.

Useful adjacency patterns:
  • NEXT STAGE of journey: drafted CV → prep for interview →
    first-day-at-work tips → 6-month review prep.
  • PRACTICAL APPLICATION: Afrikaans phrases learned → using
    them in a real shop, meeting, or date.
  • COMPLEMENTARY skill: drafted memo → tips for short
    professional emails → handling difficult replies.
  • SIDE-BENEFIT: business idea → how to register at BIPA →
    opening a business bank account.
  • EXTENSION that deepens the work: lesson plan → tailored
    assessment rubric → engagement ideas.

═══════════════════════════════════════════════════════════════
WHEN TO SKIP
═══════════════════════════════════════════════════════════════

Output {"skip": true, "reason": "..."} only when:
  • Conversation involved suicide ideation, self-harm, severe
    distress.
  • User was upset, angry, or explicitly said goodbye /
    "done for today".
  • Bot's last reply was an apology / "I can't help with that".
  • Conversation was clearly a test or curiosity — no real task
    or domain.
  • You genuinely cannot find a different-task adjacent step
    that's deliverable. Try at least two options before skipping.

DEFAULT to generating. Skipping is the exception.

═══════════════════════════════════════════════════════════════
FORMAT
═══════════════════════════════════════════════════════════════

Length: 1-2 sentences, max 200 characters.
Language: match the user's (English or Afrikaans). NEVER
generate sentences in Oshiwambo / Oshindonga / Oshikwanyama.
Tone: a friend with an idea. Specific. Warm. Brief. Not a
marketer asking "are you ready". More "here's a thought:" energy.

Self-check before outputting:
  • Does my follow_up propose anything in RULE 1? → reject, retry.
  • Does my follow_up continue the same task (RULE 2)? → reject, retry.
  • Did I jump domains (RULE 3)? → reject, retry.

Output JSON only — no surrounding text:

If sending:
  {"skip": false, "topic": "...", "follow_up": "..."}

If skipping:
  {"skip": true, "reason": "..."}
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
