"""Per-action contribute_* tools — drive the community Oshiwambo
translation contribution loop.

Architectural note (2026-05-25 rewrite):
This is the second iteration of the contribute tool layer. The first
shipped a single multi-action ``contribute_translation`` tool and
relied on the model to choose the right action from a SKILL.md
trigger table. That failed in production — Gemma hallucinated tool
calls without invoking them, and we patched around it with regex
intercepts in api/main.py. The patches worked but were architecturally
wrong: Ongiini already has a classifier that runs on every inbound
message, and a policy table that can force-call any tool by name.
Per Sebastian's call, this rewrite uses what we already have.

Now each contribute action is its own tool:
- ``contribute_invite_check`` — query state (called when classifier
  decides the user is volunteering / asking about Oshiwambo support)
- ``contribute_set_dialect`` — parse + store dialect choice
- ``contribute_next`` — serve the next task, mark pending
- ``contribute_save`` — save the user's text as a translation
- ``contribute_skip`` — drop the current task, serve a different one
- ``contribute_decline`` — record decline + cooldown
- ``contribute_stats`` — counts

Each tool takes only ``ctx: ToolContext`` and reads everything else
from sqlite + ``ctx.msg.text``. The classifier outputs a
``CONTRIBUTE_*`` verdict; the policy table forces the matching tool
via ``force_tool(name)``; the executor turn 1 forces the call, the
model composes the user-facing reply on turn 2 using the tool result.

The model can no longer skip a save, invent a source sentence, or
fake a 'Saved!' reply — the call is non-optional, and the tool's
args come from state instead of the model.
"""
from __future__ import annotations

import json
import logging
import re

from owela import ToolContext, tool

from .. import contributions

log = logging.getLogger("ongiini.tools.contribute")


# ── Helpers ────────────────────────────────────────────────────────


def _hash_user(ctx: ToolContext) -> str | None:
    """Hash the contributor's msisdn or return None when the salt is
    missing (deployment misconfig). Tools soft-fail in that case."""
    try:
        return contributions.hash_msisdn(ctx.user_id)
    except RuntimeError as e:
        log.warning("contribute hash failed: %s", e)
        return None


def _user_text(ctx: ToolContext) -> str:
    return (ctx.msg.text or "").strip()


# Dialect-name normalisation. The user can write the dialect in many
# ways; we map them all to the canonical 'Oshindonga' / 'Oshikwanyama'
# values the database stores. "Either" / "both" defaults to Oshindonga
# (our primary target dialect — see project memories).
_OSHINDONGA_RE = re.compile(r"\b(oshindonga|ondonga|oshidonga|ndonga)\b", re.IGNORECASE)
_OSHIKWANYAMA_RE = re.compile(r"\b(oshikwanyama|kwanyama)\b", re.IGNORECASE)
_EITHER_RE = re.compile(r"\b(either|both|any|whatever)\b", re.IGNORECASE)


def _parse_dialect(text: str) -> str | None:
    """Return canonical dialect name from a user message, or None
    if no dialect mention is detected. Order matters: Oshindonga
    pattern includes 'ndonga' which could substring 'oshindonga',
    so we check the more specific labels first."""
    if _OSHIKWANYAMA_RE.search(text):
        return contributions.DIALECT_OSHIKWANYAMA
    if _OSHINDONGA_RE.search(text):
        return contributions.DIALECT_OSHINDONGA
    if _EITHER_RE.search(text):
        return contributions.DIALECT_OSHINDONGA
    return None


# ── Tools ──────────────────────────────────────────────────────────


@tool(
    name="contribute_invite_check",
    description=(
        "FORCED by classifier verdict CONTRIBUTE_INVITE — the user "
        "just volunteered to help translate or asked about Oshiwambo "
        "support. Reads the contributor's status from the contributions "
        "database and returns it as JSON: status ('new'/'unset'/"
        "'known:Oshindonga'/'known:Oshikwanyama'), recently_declined "
        "(bool), total_contributions (int). Compose a warm WhatsApp "
        "invitation based on the result. If recently_declined=true, "
        "DON'T re-invite — they declined within the cooldown window; "
        "answer their question normally instead."
    ),
)
async def contribute_invite_check(ctx: ToolContext) -> str:
    h = _hash_user(ctx)
    if h is None:
        return json.dumps({"error": "contributions temporarily unavailable"})
    return json.dumps({
        "status": contributions.whoami(h),
        "recently_declined": contributions.recently_declined(h),
        "total_contributions": contributions.contributor_total(h),
    })


@tool(
    name="contribute_set_dialect",
    description=(
        "FORCED by classifier verdict CONTRIBUTE_DIALECT — the user "
        "just told us which Oshiwambo dialect they speak. Parses their "
        "message ('Oshindonga' / 'Oshikwanyama' / 'Ndonga' / "
        "'Kwanyama' / 'either') and stores the canonical name. Returns "
        "JSON: ok (bool), dialect (the canonical name stored), or "
        "error (couldn't parse)."
    ),
)
async def contribute_set_dialect(ctx: ToolContext) -> str:
    h = _hash_user(ctx)
    if h is None:
        return json.dumps({"error": "contributions temporarily unavailable"})
    dialect = _parse_dialect(_user_text(ctx))
    if dialect is None:
        return json.dumps({
            "error": "could not detect dialect — ask the user to reply "
                     "'Oshindonga' or 'Oshikwanyama'",
        })
    contributions.set_dialect(h, dialect)
    # Chain into the first-task fetch so the user sees a real corpus
    # sentence in the same turn the dialect is stored. Without this
    # chain the user would need to say "yes I'm ready" on the NEXT
    # turn, which the classifier can't route to CONTRIBUTE_NEXT
    # (awaiting_followup is false until after a save). The chain
    # avoids an entire confused-state round-trip.
    task = contributions.next_task(h)
    if task is None:
        return json.dumps({"ok": True, "dialect": dialect, "task": None,
                           "message": "no more tasks in the pool"})
    contributions.set_pending_save(h, task["id"], dialect)
    return json.dumps({
        "ok": True,
        "dialect": dialect,
        "task": task,
    })


@tool(
    name="contribute_next",
    description=(
        "FORCED by classifier verdict CONTRIBUTE_NEXT — the user just "
        "said yes to 'want another sentence?', or is starting a fresh "
        "contribution after the dialect-asking step. Fetches the next "
        "English source sentence from the curated corpus AND marks "
        "pending state so the next user message is force-saved as the "
        "translation. Returns JSON: task ({id, source_en, category}) "
        "or {task: null, message: ...} when the pool is exhausted. "
        "After the tool returns, present the source_en sentence "
        "VERBATIM to the user (in quotes) and ask 'how would you say "
        "this in <dialect>?'"
    ),
)
async def contribute_next(ctx: ToolContext) -> str:
    h = _hash_user(ctx)
    if h is None:
        return json.dumps({"error": "contributions temporarily unavailable"})
    status = contributions.whoami(h)
    if not status.startswith("known:"):
        return json.dumps({
            "error": "contributor has no dialect set yet — ask which "
                     "dialect they speak before serving a task",
        })
    dialect = status.split(":", 1)[1]
    task = contributions.next_task(h)
    # Clear the awaiting_followup flag — we're now mid-task again,
    # not awaiting a yes/no. (No-op if it was already false.)
    contributions.clear_awaiting_followup(h)
    if task is None:
        return json.dumps({
            "task": None,
            "message": "no more tasks in the pool for this contributor",
        })
    contributions.set_pending_save(h, task["id"], dialect)
    return json.dumps({"task": task, "dialect": dialect})


@tool(
    name="contribute_save",
    description=(
        "FORCED by classifier verdict CONTRIBUTE_SAVE — the user just "
        "typed their Oshiwambo translation for the English sentence "
        "you most recently presented. Stores their literal message "
        "VERBATIM against the pending task in their stored dialect. "
        "Returns JSON: ok + contribution_id + total_for_contributor + "
        "task_id, OR error if there's no pending task (which means the "
        "save was misrouted by the classifier — apologise and ask what "
        "they meant). After the tool returns, thank the contributor "
        "warmly, mention their running total, and offer another sentence."
    ),
)
async def contribute_save(ctx: ToolContext) -> str:
    h = _hash_user(ctx)
    if h is None:
        return json.dumps({"error": "contributions temporarily unavailable"})
    pending = contributions.get_pending_save(h)
    if pending is None:
        return json.dumps({
            "error": "no pending task — the user's message was not in "
                     "reply to a served English sentence",
        })
    text = _user_text(ctx)
    if not text:
        return json.dumps({"error": "empty message"})
    try:
        result = contributions.save_contribution(
            contributor_hash=h,
            task_id=pending["task_id"],
            target_dialect=pending["dialect"],
            target_translation_raw=text,
        )
    except ValueError as e:
        log.warning("contribute_save rejected: %s", e)
        contributions.clear_pending_save(h)
        return json.dumps({"error": str(e)})
    # Mark awaiting_followup so the classifier routes the user's NEXT
    # reply ("yes, another" / "no, done") to CONTRIBUTE_NEXT or
    # CONTRIBUTE_DECLINE instead of falling back to free-form NONE.
    contributions.set_awaiting_followup(h)
    return json.dumps({
        "ok": True,
        "contribution_id": result["contribution_id"],
        "total_for_contributor": result["total_for_contributor"],
        "task_id": pending["task_id"],
        "dialect": pending["dialect"],
    })


@tool(
    name="contribute_skip",
    description=(
        "FORCED by classifier verdict CONTRIBUTE_SKIP — the user "
        "rejected the current sentence ('skip', 'too hard', 'I don't "
        "know this one', 'send me a different one'). Drops the pending "
        "task without saving, fetches a different one, and marks new "
        "pending state. Returns JSON: task ({id, source_en, category}) "
        "+ dialect, OR error. After it returns, acknowledge gently "
        "('No problem!') and present the new sentence VERBATIM."
    ),
)
async def contribute_skip(ctx: ToolContext) -> str:
    h = _hash_user(ctx)
    if h is None:
        return json.dumps({"error": "contributions temporarily unavailable"})
    # Capture which task is being skipped BEFORE clearing pending,
    # so we can pass it as an exclusion to next_task and not
    # immediately re-serve the same sentence the user just rejected.
    pending = contributions.get_pending_save(h)
    skipped_id = pending["task_id"] if pending else None
    contributions.clear_pending_save(h)
    contributions.clear_awaiting_followup(h)
    status = contributions.whoami(h)
    if not status.startswith("known:"):
        return json.dumps({"error": "contributor has no dialect set"})
    dialect = status.split(":", 1)[1]
    task = contributions.next_task(
        h, exclude_task_ids=[skipped_id] if skipped_id else None,
    )
    if task is None:
        return json.dumps({"task": None, "message": "no more tasks"})
    contributions.set_pending_save(h, task["id"], dialect)
    return json.dumps({"task": task, "dialect": dialect})


@tool(
    name="contribute_decline",
    description=(
        "FORCED by classifier verdict CONTRIBUTE_DECLINE — the user "
        "said no to contributing ('no thanks', 'maybe later', 'done', "
        "'enough for now'). Records the decline so we don't re-invite "
        "for 7 days, clears any pending task. Returns JSON: ok + "
        "cooldown_days. After it returns, send a warm closing reply "
        "thanking them for their existing contributions (if any) and "
        "letting them know the offer stays open."
    ),
)
async def contribute_decline(ctx: ToolContext) -> str:
    h = _hash_user(ctx)
    if h is None:
        return json.dumps({"error": "contributions temporarily unavailable"})
    contributions.record_decline(h)
    contributions.clear_pending_save(h)
    contributions.clear_awaiting_followup(h)
    return json.dumps({
        "ok": True,
        "cooldown_days": contributions.DECLINE_COOLDOWN_DAYS,
        "total_contributions": contributions.contributor_total(h),
    })


@tool(
    name="contribute_stats",
    description=(
        "FORCED by classifier verdict CONTRIBUTE_STATS — the user "
        "asked how many translations have been collected so far. "
        "Returns JSON: total_contributions, by_dialect (per-dialect "
        "breakdown), total_contributors, total_tasks. Quote the "
        "numbers naturally in the reply ('we're at N so far — every "
        "one helps')."
    ),
)
async def contribute_stats(ctx: ToolContext) -> str:
    return json.dumps(contributions.stats_summary())
