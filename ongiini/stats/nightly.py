"""Nightly qualitative analysis for /statistics — fixed categories.

Replaces the emergent-clustering loop (paused since 2026-05-27): that
loop's clusters came out as "Other 48 % / Lifestyle 41 %" and its
multi-minute synthesis calls stalled the chat path. This module
classifies into FIXED, readable categories with short batched calls,
once a night, inside the webhook process (mem0's local qdrant can only
be opened by one process, so a separate cron job can't read it).

Per user message (batched, cached by text hash in qualia.sqlite):
  - ``topic_category``   one of TOPIC_CATEGORIES
  - ``message_language`` en / af / ow / other
  - ``topics``           a short generic phrase for "Top topics",
                         through the same anti-PII prompt + label filter
Per user with mem0 facts (batched, cached by user + facts hash):
  - ``roles`` / ``regions`` / ``family`` from [PROFILE] facts
  - ``situations``                       from [SITUATION] facts
  ("not mentioned" is not stored, so a panel only counts users who said it)

Counts are computed over what is CURRENTLY stored: a deleted user's
history file is gone, an objecting user is skipped, so neither counts —
the right-to-object and deletion promises hold without purging the
cache. Results are written as ``synthesis-<name>.json`` (same shape the
aggregator already reads) plus ``synthesis-top_topics.json``.

Run: scheduled by ``run_nightly_forever`` (01:00 UTC = 03:00 Namibia);
``catch_up_if_stale`` runs once shortly after start-up when the last
run is more than a day old.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from owela import ModelRequest

from ..config import settings
from .safety import ANTI_PII_PROMPT, sanitise_label
from .taxonomy import ANALYSIS_VERSION

log = logging.getLogger("ongiini.stats.nightly")

METHOD = f"Fixed categories, local Gemma (analysis v{ANALYSIS_VERSION})"
RUN_HOUR_UTC = 1
_MSG_BATCH = 20
_USER_BATCH = 10
_MSG_CHARS = 300
_FACT_CHARS = 600
_PAUSE_S = 0.3          # between batches: leave room for live chat traffic
_IMAGE_MARKER = "[image attached]"
_VOICE_MARKER = "[voice note]"
_MIN_LEN = 8
LANGS = ("en", "af", "ow", "other")
# Phrases that say nothing about the request; never shown as a top topic.
_VAGUE_PHRASES = frozenset({
    "general inquiry", "general question", "question", "inquiry", "request",
    "help request", "assistance request", "general assistance", "information request",
    "greeting", "greetings", "small talk", "conversation", "chat",
    "requesting assistance", "assistance", "help",
})
_LANG_LABELS = {"en": "English", "af": "Afrikaans", "ow": "Oshiwambo", "other": "Other"}


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    summary: str


TOPIC_CATEGORIES = (
    Category("school", "School & homework", "Subjects, homework, exam prep, explaining a concept."),
    Category("writing", "Writing help", "CVs, cover letters, letters, essays, speeches, messages."),
    Category("translation", "Translation", "Translating a text or word between languages."),
    Category("language", "Language learning", "Learning a language, vocabulary, grammar practice."),
    Category("jobs", "Jobs & career", "Finding work, applications, interviews, career advice."),
    Category("health", "Health", "Symptoms, medicine, pregnancy, mental-health information."),
    Category("money", "Money & business", "Budgets, loans, starting or running a business."),
    Category("government", "Government services", "IDs, passports, permits, tax, registrations."),
    Category("local", "Local info & news", "Places, opening hours, prices, events, news, weather."),
    Category("personal", "Personal & relationships", "Family, relationships, feelings, life advice."),
    Category("tech", "Tech help", "Phones, apps, computers, internet."),
    Category("chat", "Small talk", "Greetings, thanks, chatting, questions about the assistant."),
    Category("other", "Other", "Anything that fits none of the above."),
)
ROLE_CATEGORIES = (
    Category("learner", "School learners", "Attending primary or secondary school."),
    Category("student", "University & college students", "Tertiary or vocational study."),
    Category("teacher", "Teachers & lecturers", "Teaching or training others."),
    Category("jobseeker", "Looking for work", "Unemployed or searching for a job."),
    Category("employed", "Employed", "Working for an employer."),
    Category("business", "Running a business", "Self-employed, trader or business owner."),
    Category("farmer", "Farmers", "Crop or livestock farming."),
    Category("other", "Other roles", "A role that fits none of the above."),
)
REGION_CATEGORIES = tuple(
    Category(k, k, "") for k in (
        "Erongo", "Hardap", "//Kharas", "Kavango East", "Kavango West", "Khomas",
        "Kunene", "Ohangwena", "Omaheke", "Omusati", "Oshana", "Oshikoto",
        "Otjozondjupa", "Zambezi", "Outside Namibia",
    )
)
FAMILY_CATEGORIES = (
    Category("parent", "Parents", "Has children or cares for children."),
    Category("expecting", "Expecting a child", "Pregnant or partner expecting."),
    Category("caregiver", "Caring for a relative", "Looks after a sick or elderly family member."),
    Category("partner", "In a relationship", "Married or in a relationship, no children mentioned."),
    Category("other", "Other", "A household situation that fits none of the above."),
)
SITUATION_CATEGORIES = (
    Category("studying", "Studying or exams", "Preparing for exams, assignments, applying to study."),
    Category("jobhunt", "Looking for work", "Searching for a job or a first job."),
    Category("business", "Starting or running a business", "Planning, starting or growing a business."),
    Category("health", "Health", "A health issue of their own or in the family."),
    Category("family", "Family & relationships", "A family, partner or friendship matter."),
    Category("money", "Money worries", "Debt, budgeting, paying for school or living costs."),
    Category("language", "Learning a language", "Working on a language."),
    Category("other", "Other", "A situation that fits none of the above."),
)

Complete = Callable[[ModelRequest], Awaitable[Any]]


# --------------------------------------------------------------------- storage

def _db_path() -> Path:
    return settings.data_dir / "qualia.sqlite"


def _init_db() -> None:
    with sqlite3.connect(_db_path()) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS extractions (
                 analysis TEXT NOT NULL, item_hash TEXT NOT NULL, version INTEGER NOT NULL,
                 label TEXT NOT NULL, extracted_at TEXT NOT NULL,
                 PRIMARY KEY (analysis, item_hash, version))"""
        )


def _labels(analysis: str) -> dict[str, str]:
    with sqlite3.connect(_db_path()) as conn:
        rows = conn.execute(
            "SELECT item_hash, label FROM extractions WHERE analysis = ? AND version = ?",
            (analysis, ANALYSIS_VERSION),
        ).fetchall()
    return dict(rows)


def _store(rows: Iterable[tuple[str, str, str]], *, replace_prefix: bool = False) -> None:
    """rows: (analysis, item_hash, label). With ``replace_prefix`` the
    item_hash is "<user>:<facts>" and older rows for the same user are
    removed first (facts changed)."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(_db_path()) as conn:
        for analysis, item_hash, label in rows:
            if replace_prefix:
                user = item_hash.split(":", 1)[0]
                conn.execute(
                    "DELETE FROM extractions WHERE analysis = ? AND version = ? AND item_hash LIKE ?",
                    (analysis, ANALYSIS_VERSION, user + ":%"),
                )
            conn.execute(
                "INSERT OR REPLACE INTO extractions VALUES (?, ?, ?, ?, ?)",
                (analysis, item_hash, ANALYSIS_VERSION, label, ts),
            )
        conn.commit()


def _write_synthesis(name: str, item_kind: str, counts: Counter, cats: tuple[Category, ...] | None) -> dict:
    summary = {c.label: c.summary for c in (cats or ())}
    clusters = [
        {"label": label, "summary": summary.get(label, ""), "count": n, "items": []}
        for label, n in counts.most_common() if n > 0
    ]
    out = {
        "analysis": name,
        "item_kind": item_kind,
        "method": METHOD,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_items_analysed": sum(counts.values()),
        "distinct_labels": len(clusters),
        "clusters": clusters,
    }
    target = settings.data_dir / f"synthesis-{name}.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return out


# --------------------------------------------------------------------- sources

def _clean_message(text: str) -> str:
    return text.replace(_IMAGE_MARKER, "").replace(_VOICE_MARKER, "").strip()


def iter_user_messages(excluded: frozenset[str]) -> Iterable[tuple[str, list[tuple[str, str]]]]:
    """(msisdn, [(text_hash, text), ...]) for every stored user history."""
    for path in sorted(settings.data_dir.glob("*.json")):
        if not path.stem.isdigit() or path.stem in excluded:
            continue
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        msgs = []
        for e in entries if isinstance(entries, list) else []:
            if not isinstance(e, dict) or e.get("role") != "user" or not isinstance(e.get("content"), str):
                continue
            text = _clean_message(e["content"])[:1500]
            if len(text) < _MIN_LEN:
                continue
            msgs.append((hashlib.sha256(text.encode("utf-8")).hexdigest(), text))
        if msgs:
            yield path.stem, msgs


def iter_user_facts(excluded: frozenset[str], list_facts: Callable[[str], list]) -> Iterable[tuple[str, str, str]]:
    """(msisdn, profile_text, situation_text) for users with mem0 facts."""
    for path in sorted(settings.data_dir.glob("*.json")):
        msisdn = path.stem
        if not msisdn.isdigit() or msisdn in excluded:
            continue
        try:
            facts = list_facts(msisdn) or []
        except Exception:                                 # noqa: BLE001 — skip unreadable users
            continue
        profile, situation = [], []
        for f in facts:
            text = str((f.get("memory") or f.get("text") or "") if isinstance(f, dict) else f)
            if "[PROFILE]" in text or "[PREFERENCE]" in text:
                profile.append(text.replace("[PROFILE]", "").replace("[PREFERENCE]", "").strip())
            elif "[SITUATION]" in text:
                situation.append(text.replace("[SITUATION]", "").strip())
        if profile or situation:
            yield msisdn, "; ".join(profile)[:_FACT_CHARS], "; ".join(situation)[:_FACT_CHARS]


def _user_key(msisdn: str, text: str) -> str:
    u = hashlib.sha256(msisdn.encode("utf-8")).hexdigest()[:24]
    t = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{u}:{t}"


# --------------------------------------------------------------------- prompts

def _cat_lines(cats: tuple[Category, ...]) -> str:
    return "\n".join(f"  {c.key:12} {c.summary or c.label}" for c in cats)


MESSAGE_PROMPT = (
    ANTI_PII_PROMPT
    + "\nTask: label each numbered WhatsApp message sent to a free AI assistant in Namibia.\n"
    "For each message give:\n"
    "  cat    exactly one category key:\n" + _cat_lines(TOPIC_CATEGORIES) + "\n"
    "  lang   en, af, ow (Oshiwambo: Oshindonga/Oshikwanyama) or other\n"
    "  topic  a SHORT GENERIC English phrase (2-6 words) for the type of request,\n"
    "         no specifics (e.g. 'grade 11 chemistry homework', 'cv improvement',\n"
    "         'fever symptoms'), never vague ones like 'general inquiry'; 'small talk'\n"
    "         for greetings or thanks; 'REDACTED'\n"
    "         if it cannot be said without identifying details.\n"
    'Return ONE JSON object: {"labels": [{"i": 1, "cat": "...", "lang": "...", "topic": "..."}, ...]}\n'
    "one entry per message, same numbering.\n\nMessages:\n"
)

USER_PROMPT = (
    ANTI_PII_PROMPT
    + "\nTask: for each numbered user, read what an AI assistant remembers about them and pick\n"
    "the best category for each field, or \"none\" if the facts don't say.\n"
    "role:\n" + _cat_lines(ROLE_CATEGORIES) + "\n"
    "region (Namibian region the user lives in, from any town or region named):\n  "
    + ", ".join(c.key for c in REGION_CATEGORIES) + "\n"
    "family:\n" + _cat_lines(FAMILY_CATEGORIES) + "\n"
    "situation (from the SITUATION line):\n" + _cat_lines(SITUATION_CATEGORIES) + "\n"
    'Return ONE JSON object: {"users": [{"i": 1, "role": "...", "region": "...", "family": "...", '
    '"situation": "..."}, ...]}\n\nUsers:\n'
)


async def _ask_json(complete: Complete, prompt: str, key: str, max_tokens: int) -> dict[int, dict]:
    req = ModelRequest(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=max_tokens, response_format="json_object", timeout_s=120,
    )
    try:
        resp = await complete(req)
        data = json.loads(resp.content or "{}")
        out = {}
        for x in data.get(key, []):
            try:
                out[int(x["i"])] = x
            except (TypeError, KeyError, ValueError):     # one bad entry, not the whole batch
                continue
        return out
    except Exception as exc:                              # noqa: BLE001 — a bad batch is skipped, retried next night
        log.warning("nightly batch failed: %s", exc)
        return {}


# --------------------------------------------------------------------- passes

async def classify_messages(complete: Complete, pending: list[tuple[str, str]]) -> int:
    """Classify unseen message texts; returns how many got stored."""
    keys = {c.key for c in TOPIC_CATEGORIES}
    stored = 0
    for start in range(0, len(pending), _MSG_BATCH):
        batch = pending[start:start + _MSG_BATCH]
        listing = "\n".join(
            f"{n + 1}. {text[:_MSG_CHARS].replace(chr(10), ' ')}" for n, (_, text) in enumerate(batch)
        )
        got = await _ask_json(complete, MESSAGE_PROMPT + listing, "labels", 2000)
        rows = []
        for n, (h, _) in enumerate(batch):
            x = got.get(n + 1)
            if not x or x.get("cat") not in keys:
                continue                                  # unlabelled: retried next run
            rows.append(("topic_category", h, x["cat"]))
            rows.append(("message_language", h, x.get("lang") if x.get("lang") in LANGS else "other"))
            phrase = sanitise_label(str(x.get("topic") or "").strip().lower())
            if phrase and phrase not in _VAGUE_PHRASES:
                rows.append(("topics", h, phrase))
            stored += 1
        if rows:
            _store(rows)
        await asyncio.sleep(_PAUSE_S)
    return stored


_WHO_FIELDS = (
    ("roles", "role", ROLE_CATEGORIES),
    ("regions", "region", REGION_CATEGORIES),
    ("family", "family", FAMILY_CATEGORIES),
    ("situations", "situation", SITUATION_CATEGORIES),
)


async def classify_users(complete: Complete, pending: list[tuple[str, str, str]]) -> int:
    """pending: (user_key, profile_text, situation_text)."""
    stored = 0
    for start in range(0, len(pending), _USER_BATCH):
        batch = pending[start:start + _USER_BATCH]
        listing = "\n".join(
            f"{n + 1}. PROFILE: {p or '-'} | SITUATION: {s or '-'}".replace("\n", " ")
            for n, (_, p, s) in enumerate(batch)
        )
        got = await _ask_json(complete, USER_PROMPT + listing, "users", 1500)
        rows = []
        for n, (key, _, _) in enumerate(batch):
            x = got.get(n + 1)
            if not x:
                continue
            for analysis, field, cats in _WHO_FIELDS:
                value = x.get(field)
                valid = {c.key for c in cats}
                # "none" is stored too, so the user isn't re-asked every night;
                # it is never counted.
                rows.append((analysis, key, value if value in valid else "none"))
            stored += 1
        if rows:
            _store(rows, replace_prefix=True)
        await asyncio.sleep(_PAUSE_S)
    return stored


# --------------------------------------------------------------------- counting

def write_outputs(users_msgs: list[tuple[str, list[tuple[str, str]]]],
                  user_keys: list[str]) -> dict[str, int]:
    """Count over what is stored NOW and write every synthesis file."""
    cat_by_key = {c.key: c.label for c in TOPIC_CATEGORIES}
    topic_cat, lang, phrase = _labels("topic_category"), _labels("message_language"), _labels("topics")

    cat_counts: Counter = Counter()
    phrase_counts: Counter = Counter()
    lang_users: Counter = Counter()
    for _, msgs in users_msgs:
        user_langs: Counter = Counter()
        for h, _ in msgs:
            if h in topic_cat:
                cat_counts[cat_by_key.get(topic_cat[h], "Other")] += 1
            if h in phrase and topic_cat.get(h) != "chat":   # thanks/greetings are not topics
                phrase_counts[phrase[h]] += 1
            if h in lang:
                user_langs[lang[h]] += 1
        if user_langs:
            lang_users[_LANG_LABELS[user_langs.most_common(1)[0][0]]] += 1

    out = {"topics": sum(cat_counts.values()), "languages": sum(lang_users.values())}
    _write_synthesis("topics", "message", cat_counts, TOPIC_CATEGORIES)
    _write_synthesis("languages", "user", lang_users, None)
    _write_top_topics(phrase_counts)

    current = set(user_keys)
    for analysis, _, cats in _WHO_FIELDS:
        label_by_key = {c.key: c.label for c in cats}
        counts: Counter = Counter()
        for key, value in _labels(analysis).items():
            if key in current and value in label_by_key:
                counts[label_by_key[value]] += 1
        _write_synthesis(analysis, "user", counts, cats)
        out[analysis] = sum(counts.values())
    return out


def _write_top_topics(counts: Counter) -> None:
    floor = settings.stats_minimum_bucket
    rows = [{"label": p, "count": n} for p, n in counts.most_common()
            if n >= floor and p != "small talk"]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_distinct": len(counts),
        "labels": rows[:20],
    }
    target = settings.data_dir / "synthesis-top_topics.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)


# --------------------------------------------------------------------- driver

async def run_once(complete: Complete, list_facts: Callable[[str], list],
                   excluded: frozenset[str]) -> dict[str, Any]:
    _init_db()
    started = datetime.now(timezone.utc)
    users_msgs = list(iter_user_messages(excluded))
    known = set(_labels("topic_category"))
    seen: set[str] = set()
    pending_msgs = []
    for _, msgs in users_msgs:
        for h, text in msgs:
            if h not in known and h not in seen:
                seen.add(h)
                pending_msgs.append((h, text))
    new_msgs = await classify_messages(complete, pending_msgs)

    facts = list(iter_user_facts(excluded, list_facts))
    keys = [_user_key(m, p + "|" + s) for m, p, s in facts]
    known_users = set(_labels("roles"))
    pending_users = [(k, p, s) for k, (_, p, s) in zip(keys, facts) if k not in known_users]
    new_users = await classify_users(complete, pending_users)

    counts = write_outputs(users_msgs, keys)
    report = {
        "started": started.isoformat(timespec="seconds"),
        "minutes": round((datetime.now(timezone.utc) - started).total_seconds() / 60, 1),
        "messages_new": new_msgs, "messages_pending": len(pending_msgs),
        "users_new": new_users, "users_pending": len(pending_users),
        "counted": counts,
    }
    log.info("nightly stats run: %s", report)
    return report


def _last_run() -> datetime | None:
    p = settings.data_dir / "synthesis-topics.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("method", "").startswith("Fixed categories"):
            return datetime.fromisoformat(data["generated_at"])
    except (OSError, ValueError, KeyError):
        pass
    return None


def seconds_until_next_run(now: datetime) -> float:
    target = now.replace(hour=RUN_HOUR_UTC, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _catch_up_forced() -> bool:
    return os.environ.get("ONGIINI_STATS_CATCH_UP_NOW", "").lower() in ("1", "true", "yes")


def in_night_window(now: datetime) -> bool:
    """23:00–04:00 UTC (01:00–06:00 in Namibia): low chat traffic."""
    return now.hour >= 23 or now.hour < 4


async def run_nightly_forever(complete: Complete, list_facts: Callable[[str], list],
                              load_excluded: Callable[[], frozenset[str]]) -> None:
    """Run nightly at RUN_HOUR_UTC. If the process starts inside the night
    window and the last run is over a day old (e.g. a restart missed it),
    catch up straight away. During the day only when the operator sets
    ONGIINI_STATS_CATCH_UP_NOW=1 — the first full run takes hours of
    batch calls on the same vLLM that serves chat."""
    last = _last_run()
    now = datetime.now(timezone.utc)
    stale = last is None or now - last > timedelta(hours=26)
    if stale and (in_night_window(now) or _catch_up_forced()):
        await asyncio.sleep(300)                        # let start-up traffic settle
        await _safe_run(complete, list_facts, load_excluded)
    while True:
        await asyncio.sleep(seconds_until_next_run(datetime.now(timezone.utc)))
        await _safe_run(complete, list_facts, load_excluded)


async def _safe_run(complete, list_facts, load_excluded) -> None:
    try:
        await run_once(complete, list_facts, load_excluded())
    except asyncio.CancelledError:
        raise
    except Exception:                                   # noqa: BLE001 — never take the webhook down
        log.exception("nightly stats run crashed; next attempt tomorrow")
