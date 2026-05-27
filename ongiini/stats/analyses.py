"""LLM-driven qualitative analysis framework.

The premise: we don't decide upfront what categories the data falls
into. The LLM reads the data and proposes the structure. Two passes:

  EXTRACT (cheap, per-item, incremental)
    For each new user message (or each new user profile, or any future
    source), the local LLM produces a SHORT free-form label describing
    what the item is about — e.g. "yellowing maize leaves",
    "grade 11 chemistry homework", "VAT registration form". The label
    is cached in /data/qualia.sqlite keyed by the analysis name +
    SHA-256 of the input + ANALYSIS_VERSION, so re-running is cheap.

  SYNTHESIZE (more expensive, periodic, batched)
    Periodically — and only when there are new extractions since the
    last run — the LLM is shown ALL the extracted labels with their
    counts and asked to cluster them into a small set of named themes.
    The clustering is emergent: no pre-defined taxonomy, no keyword
    lists. The output is a JSON file at /data/synthesis-{name}.json
    that the aggregator reads on each /stats.json request.

Adding a new analysis later is a one-config-block change: define the
source iterator, extraction prompt, and synthesis prompt. The same
machinery handles it.

All extractions and synthesis happen on the same vLLM endpoint that
powers the chat service — no data leaves the foundation's
infrastructure.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from openai import AsyncOpenAI

from ..config import settings
from ..memory import long_term as mem
from .safety import ANTI_PII_PROMPT, sanitise_label as _sanitise_label
from .taxonomy import ANALYSIS_VERSION

log = logging.getLogger("ongiini.stats")

# --- LLM client (shared across analyses) -----------------------------------

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(base_url=settings.vllm_base_url, api_key="not-needed")
    return _client


# --- Database (single sqlite file, one table per analysis via column) ------

def _db_path() -> Path:
    return settings.data_dir / "qualia.sqlite"


def _init_db_sync() -> None:
    with sqlite3.connect(_db_path()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS extractions (
              analysis    TEXT NOT NULL,
              item_hash   TEXT NOT NULL,
              version     INTEGER NOT NULL,
              label       TEXT NOT NULL,
              extracted_at TEXT NOT NULL,
              PRIMARY KEY (analysis, item_hash, version)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_extractions_analysis_version "
            "ON extractions(analysis, version)"
        )
        conn.commit()


def _known_hashes_sync(analysis_name: str) -> set[str]:
    with sqlite3.connect(_db_path()) as conn:
        rows = conn.execute(
            "SELECT item_hash FROM extractions WHERE analysis = ? AND version = ?",
            (analysis_name, ANALYSIS_VERSION),
        ).fetchall()
    return {r[0] for r in rows}


def _store_extraction_sync(analysis_name: str, item_hash: str, label: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(_db_path()) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO extractions
              (analysis, item_hash, version, label, extracted_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (analysis_name, item_hash, ANALYSIS_VERSION, label, ts),
        )
        conn.commit()


def load_label_counts_sync(analysis_name: str) -> dict[str, int]:
    """Return {label: count} for the latest version of one analysis.

    Used by aggregator (read-only, fast)."""
    try:
        with sqlite3.connect(_db_path()) as conn:
            rows = conn.execute(
                """
                SELECT label, COUNT(*) FROM extractions
                WHERE analysis = ? AND version = ?
                GROUP BY label
                """,
                (analysis_name, ANALYSIS_VERSION),
            ).fetchall()
        return {label: int(c) for label, c in rows}
    except sqlite3.Error:
        return {}


def load_extraction_total_sync(analysis_name: str) -> int:
    try:
        with sqlite3.connect(_db_path()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM extractions WHERE analysis = ? AND version = ?",
                (analysis_name, ANALYSIS_VERSION),
            ).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


# --- Source iterators (one per analysis) -----------------------------------

_IMAGE_MARKER = "[image attached]"
_MIN_LEN = 8


def _iter_user_messages(excluded: frozenset[str]) -> Iterator[tuple[str, str]]:
    """Yield (content_hash, text) over user messages in per-user memory files.

    Hash is SHA-256 of the cleaned text — same string from different
    users hashes the same, so we classify it once.
    """
    for path in settings.data_dir.glob("*.json"):
        if not path.stem.isdigit():
            continue
        if path.stem in excluded:
            continue
        try:
            arr = json.loads(path.read_text(encoding="utf-8"))
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
            text = content
            if _IMAGE_MARKER in text:
                text = text.replace(_IMAGE_MARKER, "", 1).strip()
            if len(text) < _MIN_LEN:
                continue
            text = text[:1500]
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            yield h, text


def _iter_user_facts_by_tags(
    tags: tuple[str, ...],
    excluded: frozenset[str],
) -> Iterator[tuple[str, str]]:
    """Yield (msisdn-hash, concatenated-fact-text) for every user with at
    least one fact carrying any of `tags`.

    `tags` is a tuple of mem0 tag markers like ("[PROFILE]",) or
    ("[PROFILE]", "[PREFERENCE]") — facts carrying any listed tag are
    included; their tag prefix is stripped. Users with no matching
    facts are skipped (so a WHO analysis only consumes LLM budget for
    users who actually have data on that dimension).
    """
    for path in settings.data_dir.glob("*.json"):
        if not path.stem.isdigit():
            continue
        msisdn = path.stem
        if msisdn in excluded:
            continue
        try:
            facts = mem.list_all(msisdn)
        except Exception:  # noqa: BLE001
            continue
        bits: list[str] = []
        for f in facts or []:
            txt = ""
            if isinstance(f, dict):
                txt = str(f.get("memory") or f.get("text") or "")
            elif isinstance(f, str):
                txt = f
            if not txt:
                continue
            if any(t in txt for t in tags):
                cleaned = txt
                for t in tags:
                    cleaned = cleaned.replace(t, "")
                cleaned = cleaned.strip()
                if cleaned:
                    bits.append(cleaned)
        if not bits:
            continue
        combined = "; ".join(bits)[:1500]
        h = hashlib.sha256(msisdn.encode("utf-8")).hexdigest()
        yield h, combined


def _iter_user_profiles(excluded: frozenset[str]) -> Iterator[tuple[str, str]]:
    return _iter_user_facts_by_tags(("[PROFILE]",), excluded)


def _iter_user_profile_pref(excluded: frozenset[str]) -> Iterator[tuple[str, str]]:
    return _iter_user_facts_by_tags(("[PROFILE]", "[PREFERENCE]"), excluded)


def _iter_user_situations(excluded: frozenset[str]) -> Iterator[tuple[str, str]]:
    return _iter_user_facts_by_tags(("[SITUATION]",), excluded)


# --- Analysis definitions --------------------------------------------------

@dataclass
class Analysis:
    name: str
    description: str
    source: Callable[[frozenset[str]], Iterator[tuple[str, str]]]
    extract_system: str
    synthesize_system: str
    item_kind: str  # "message" | "user" — used in synthesis prompt and aggregator labelling


# ---------------------------------------------------------------------------
# Topic analysis (HOW people use Ongiini.ai). Extracted per-message,
# synthesised into BROAD use-case buckets. The raw extractions
# themselves are surfaced separately on the page as "top topics" — one
# level down from the use-case buckets.
# ---------------------------------------------------------------------------

TOPICS_ANALYSIS = Analysis(
    name="topics",
    description="Per-message topic phrase; clusters become use-cases",
    source=_iter_user_messages,
    extract_system=(
        ANTI_PII_PROMPT
        + "Task: You read one short message from a user of a free Namibian AI "
        "helper. Reply with a SHORT GENERIC noun phrase (3-7 words) describing "
        "WHAT TYPE of question the user is asking — the topic category, not the "
        "specifics. Strip all identifying details per the rules above.\n\n"
        "Format rules:\n"
        "- Output ONLY the phrase. No quotes, no trailing punctuation, no explanation.\n"
        "- Use English regardless of the message's language.\n"
        "- Prefer concrete category nouns ('crop disease' over 'agriculture'; "
        "  'grade 11 chemistry homework' over 'school'; 'fever symptoms' over 'health').\n"
        "- For pure greetings, thanks, or yes/no with no topic, output 'small talk'.\n"
    ),
    synthesize_system=(
        "You are analysing how people use a free AI helper in Namibia. Below is a "
        "list of short topic phrases, each with a count of how often it appeared. "
        "Cluster them into a SMALL number of BROAD use-case buckets.\n\n"
        "Rules:\n"
        "- Produce 4-7 clusters total. Prefer FEWER, BIGGER buckets.\n"
        "- Each cluster label is 1-2 words at high-level category granularity — "
        "  e.g. 'Education', 'Health', 'Agriculture', 'Government', 'Daily life', "
        "  'Business & money', 'Parenting'. Avoid sub-categorisation here.\n"
        "- Each cluster summary is one sentence explaining what's in it.\n"
        "- Assign EVERY input phrase to exactly one cluster.\n"
        "- Output strict JSON in this shape, with no preamble or trailing text:\n"
        '  {"clusters": [{"label": "...", "summary": "...", "items": ["phrase1", "phrase2"]}]}\n'
        "- Items must be EXACT verbatim copies of the input phrases — JUST "
        "the phrase, NOT the '(count: N)' suffix. Example: an input line "
        "'small talk (count: 112)' becomes the item 'small talk' — no count, "
        "no parentheses.\n"
    ),
    item_kind="message",
)


# ---------------------------------------------------------------------------
# WHO analyses — five dimensions, all sourced from mem0 facts. Each is
# self-contained so the framework adds new dimensions by adding new
# Analysis objects to ALL_ANALYSES.
# ---------------------------------------------------------------------------

_WHO_SYNTH_RULES = (
    "Rules:\n"
    "- Produce 4-8 clusters total.\n"
    "- Each cluster label is 1-3 words, descriptive and dignified.\n"
    "- Each cluster summary is one sentence.\n"
    "- Assign EVERY input phrase to exactly one cluster.\n"
    "- Output strict JSON in this shape, with no preamble or trailing text:\n"
    '  {"clusters": [{"label": "...", "summary": "...", "items": ["phrase1", "phrase2"]}]}\n'
    "- Items must be EXACT verbatim copies of the input phrases.\n"
)


ROLES_ANALYSIS = Analysis(
    name="roles",
    description="What roles/professions describe the user base",
    source=_iter_user_profiles,
    extract_system=(
        ANTI_PII_PROMPT
        + "Task: You read profile facts about one user. Reply with a SHORT GENERIC "
        "role descriptor (1-4 words) capturing the person's MAIN role or occupation "
        "category — e.g. 'smallholder farmer', 'matric student', 'taxi driver', "
        "'shop owner', 'nurse', 'pastor', 'unemployed jobseeker'.\n\n"
        "Format rules:\n"
        "- Output ONLY the role phrase. No quotes, no punctuation, no explanation.\n"
        "- Strip ALL identifying details — never include employer name, school name, "
        "  specific shop/clinic name, family member names, or specific town.\n"
        "- If no role is stated or inferable, output 'unknown'.\n"
        "- Use English regardless of the facts' language.\n"
    ),
    synthesize_system=(
        "You are analysing roles in the user base of a Namibian AI helper. Below "
        "are role descriptors, each with a count of users.\n\n" + _WHO_SYNTH_RULES
    ),
    item_kind="user",
)


REGIONS_ANALYSIS = Analysis(
    name="regions",
    description="Which Namibian administrative region users are in",
    source=_iter_user_profiles,
    extract_system=(
        # The standard anti-PII prefix forbids town/village names. For
        # this analysis we override the geographic part: a REGION is a
        # large administrative unit (100k+ people), publishable in
        # aggregate. Names, dates, counts, employers etc. remain forbidden.
        "PRIVACY RULES — THIS OUTPUT WILL BE PUBLISHED IN AGGREGATE STATISTICS, "
        "SO IT MUST BE GENERIC:\n"
        "- NEVER include a person's name, employer, school, clinic, or any "
        "  identifying detail other than the region.\n"
        "- NEVER include specific numbers (ages, dates, quantities).\n"
        "- You MAY output one of Namibia's 14 administrative regions OR a "
        "  coarse cardinal descriptor: Erongo, Hardap, ǁKaras, Kavango East, "
        "  Kavango West, Khomas, Kunene, Ohangwena, Omaheke, Omusati, Oshana, "
        "  Oshikoto, Otjozondjupa, Zambezi, 'northern Namibia', 'central "
        "  Namibia', 'southern Namibia', 'coastal Namibia', 'rural Namibia'.\n"
        "- If the user mentioned a town (e.g. Oshakati), output the region it "
        "  is in (Oshana for Oshakati, Khomas for Windhoek, etc.) — NOT the "
        "  town itself.\n"
        "- If no location is stated or inferable, output 'unknown'.\n"
        "- If you cannot generalise without naming a specific town, output "
        "  'REDACTED'.\n\n"
        "Format rules:\n"
        "- Output ONLY the region name. No quotes, no punctuation, no explanation.\n"
    ),
    synthesize_system=(
        "You are analysing where users live, in the user base of a Namibian AI "
        "helper. Below are location descriptors with counts.\n\n" + _WHO_SYNTH_RULES
    ),
    item_kind="user",
)


FAMILY_ANALYSIS = Analysis(
    name="family",
    description="Family / household situation",
    source=_iter_user_profiles,
    extract_system=(
        ANTI_PII_PROMPT
        + "Task: You read profile facts about one user. Reply with a SHORT GENERIC "
        "family/household descriptor (1-4 words) — e.g. 'parent', 'single mother', "
        "'married couple', 'lives with parents', 'caring for elderly relative', "
        "'no dependents'.\n\n"
        "Format rules:\n"
        "- Output ONLY the descriptor. No quotes, no punctuation, no explanation.\n"
        "- NEVER include the count of children, their names, or specific ages — "
        "  'parent' is correct; 'parent of three' is forbidden.\n"
        "- If no family or household info is stated, output 'unknown'.\n"
    ),
    synthesize_system=(
        "You are analysing family / household situations in the user base of a "
        "Namibian AI helper. Below are family descriptors with counts.\n\n"
        + _WHO_SYNTH_RULES
    ),
    item_kind="user",
)


LANGUAGES_ANALYSIS = Analysis(
    name="languages",
    description="Preferred language",
    source=_iter_user_profile_pref,
    extract_system=(
        # Languages are inherently a small closed set of public names —
        # the anti-PII prefix doesn't add much here, but inclusion keeps
        # the model from echoing other parts of the profile by accident.
        ANTI_PII_PROMPT
        + "Task: Reply with a SHORT language descriptor (1-2 words) capturing the "
        "preferred reply language — e.g. 'English', 'Afrikaans', 'Oshiwambo', "
        "'Oshikwanyama', 'Khoekhoegowab', 'Otjiherero', 'mixed'.\n\n"
        "Format rules:\n"
        "- Output ONLY the language name. No quotes, no punctuation, no explanation.\n"
        "- If no preference is stated, output 'unknown'.\n"
    ),
    synthesize_system=(
        "You are analysing preferred languages in the user base of a Namibian AI "
        "helper. Below are language descriptors with counts.\n\n"
        "Rules:\n"
        "- Produce EXACTLY these four clusters, in this order:\n"
        "  1. \"English\"\n"
        "  2. \"Afrikaans\"\n"
        "  3. \"Oshiwambo\"\n"
        "  4. \"Other\"\n"
        "- Assign each input phrase to ONE cluster using this priority "
        "(check from top to bottom, stop at the first match):\n"
        "  a. If the phrase mentions Oshiwambo, Oshindonga, Oshikwanyama, "
        "     Ndonga, Kwanyama, or Ovambo → \"Oshiwambo\".\n"
        "  b. Else if the phrase mentions Afrikaans → \"Afrikaans\".\n"
        "  c. Else if the phrase mentions English → \"English\".\n"
        "  d. Else (other Namibian languages such as Khoekhoegowab, "
        "     Damara, Nama, Otjiherero, Rukwangali, Silozi; foreign "
        "     languages; or 'mixed' / 'unknown' / 'unspecified') → \"Other\".\n"
        "- Each cluster summary is one sentence about who is in it.\n"
        "- Include all four clusters even if some have zero items "
        "  (use an empty items list).\n"
        "- Assign EVERY input phrase to exactly one cluster.\n"
        "- Output strict JSON in this shape, with no preamble or trailing text:\n"
        '  {"clusters": [{"label": "...", "summary": "...", "items": ["phrase1", "phrase2"]}]}\n'
        "- Items must be EXACT verbatim copies of the input phrases.\n"
    ),
    item_kind="user",
)


SITUATIONS_ANALYSIS = Analysis(
    name="situations",
    description="Current life situations users are working through",
    source=_iter_user_situations,
    extract_system=(
        ANTI_PII_PROMPT
        + "Task: You read short notes about one user's current situation. Reply "
        "with a SHORT GENERIC phrase (3-6 words) describing the dominant current "
        "situation TYPE — e.g. 'planting season for crops', 'preparing for matric "
        "exams', 'navigating business registration', 'caring for a sick relative'.\n\n"
        "Format rules:\n"
        "- Output ONLY the phrase. No quotes, no punctuation, no explanation.\n"
        "- Strip identifying details: no names, no specific schools/businesses, no "
        "  specific places below country, no specific dates.\n"
        "- If situations are unclear or absent, output 'unspecified'.\n"
    ),
    synthesize_system=(
        "You are analysing current life situations in the user base of a "
        "Namibian AI helper. Below are situation descriptors with counts.\n\n"
        + _WHO_SYNTH_RULES
    ),
    item_kind="user",
)


# Order matters for display on the page: topics first, then WHO panels
# in the order that's most informative for understanding the user base.
ALL_ANALYSES: list[Analysis] = [
    TOPICS_ANALYSIS,
    ROLES_ANALYSIS,
    REGIONS_ANALYSIS,
    LANGUAGES_ANALYSIS,
    FAMILY_ANALYSIS,
    SITUATIONS_ANALYSIS,
]


# --- Extraction pass -------------------------------------------------------

async def _extract_one(prompt_system: str, text: str) -> str:
    try:
        resp = await _get_client().chat.completions.create(
            model=settings.vllm_model,
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=24,
        )
    except Exception as exc:  # noqa: BLE001 — broad to keep loop alive
        log.warning("extraction call failed: %s", exc)
        return ""
    raw = (resp.choices[0].message.content or "").strip()
    # Strip wrapping quotes / line breaks; keep the first line only.
    cleaned = raw.strip().strip("\"'.,;: \n\r\t")
    first_line = cleaned.split("\n", 1)[0].strip().strip("\"'.,;: \t")
    # Empty / refusal sentinel -> let caller skip storing.
    if not first_line:
        return ""
    return first_line[:80]  # cap label length defensively


async def run_extract_pass(analysis: Analysis, excluded: frozenset[str]) -> tuple[int, int]:
    """Extract labels for unseen items of one analysis. Returns (new, skipped).

    Every label is passed through `_sanitise_label` before storage; labels
    that fail safety checks are dropped and never written to the cache.
    Skipped counts include both LLM-empty-responses and labels rejected
    by the sanitiser — the loop logs both so a sudden spike is visible.
    """
    await asyncio.to_thread(_init_db_sync)
    known = await asyncio.to_thread(_known_hashes_sync, analysis.name)
    new = 0
    skipped = 0
    seen_this_pass: set[str] = set()
    for item_hash, text in analysis.source(excluded):
        if item_hash in known or item_hash in seen_this_pass:
            continue
        seen_this_pass.add(item_hash)
        raw = await _extract_one(analysis.extract_system, text)
        if not raw:
            skipped += 1
            await asyncio.sleep(0)
            continue
        safe = _sanitise_label(raw)
        if safe is None:
            # Model produced something we judged unsafe — drop entirely.
            # The sanitiser is the last gate before storage; once a
            # label is in qualia.sqlite it can flow to the API.
            log.info(
                "topic sanitiser rejected label for analysis=%s: %r",
                analysis.name, raw[:80],
            )
            skipped += 1
            await asyncio.sleep(0)
            continue
        await asyncio.to_thread(_store_extraction_sync, analysis.name, item_hash, safe)
        new += 1
        await asyncio.sleep(0)
    return new, skipped


# --- Synthesis pass --------------------------------------------------------

_MAX_SYNTH_ITEMS = 80     # cap descriptions fed to the LLM in one synthesis call.
                          # Hard ceiling — once the label set grew past ~150 in
                          # production, the synthesis call hit the 180s vLLM
                          # timeout and returned None, leaving the stale
                          # synthesis file in place (100% Other). 80 by-count-
                          # rank labels covers ~85% of items in our distribution
                          # while keeping the prompt under 4k chars and the
                          # response under 30s. Long-tail singletons drop into
                          # Other via the aggregator's existing fall-through,
                          # which is semantically correct anyway — singletons
                          # by definition aren't part of a broad bucket.
_SYNTH_TIMEOUT_SECONDS = 180
# Synthesis can return a lot of JSON when there are dozens of items to
# assign. 4000 was empirically too tight — the LLM truncated mid-cluster
# and we lost everything to the 'Other' fallback. 12000 gives plenty of
# headroom; the local model can handle it cheaply.
_SYNTH_MAX_TOKENS = 12000


from .synthesis_io import synthesis_path as _synthesis_path  # noqa: E402
from .synthesis_io import load_synthesis as load_synthesis_sync  # noqa: E402


from .synth_match import norm_label as _norm_label  # noqa: E402


def _serialise_label_counts(counts: dict[str, int]) -> str:
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    items = items[:_MAX_SYNTH_ITEMS]
    return "\n".join(f"{label} (count: {count})" for label, count in items)


def _parse_synthesis_json(raw: str) -> dict | None:
    raw = raw.strip()
    # Strip code fences if the model wrapped them around the JSON.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Try to recover by finding the first {...} block.
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    if not isinstance(obj, dict) or "clusters" not in obj:
        return None
    return obj


async def run_synthesis(analysis: Analysis) -> dict | None:
    """Synthesise clusters from current extractions and write JSON output.

    Returns the synthesis dict, or None on failure. On success the file
    /data/synthesis-{analysis.name}.json is also written atomically.
    """
    counts = await asyncio.to_thread(load_label_counts_sync, analysis.name)
    total_labels = sum(counts.values())
    distinct = len(counts)
    if distinct == 0:
        return None

    payload = _serialise_label_counts(counts)
    try:
        resp = await asyncio.wait_for(
            _get_client().chat.completions.create(
                model=settings.vllm_model,
                messages=[
                    {"role": "system", "content": analysis.synthesize_system},
                    {"role": "user", "content": payload},
                ],
                temperature=0.2,
                max_tokens=_SYNTH_MAX_TOKENS,
            ),
            timeout=_SYNTH_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        log.warning("synthesis '%s' timed out", analysis.name)
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("synthesis '%s' call failed: %s", analysis.name, exc)
        return None

    raw = resp.choices[0].message.content or ""
    parsed = _parse_synthesis_json(raw)
    if parsed is None:
        log.warning("synthesis '%s' returned unparsable JSON: %r", analysis.name, raw[:300])
        return None

    norm_to_label: dict[str, str] = {_norm_label(lbl): lbl for lbl in counts.keys()}

    cluster_out: list[dict] = []
    assigned_items: set[str] = set()
    for cluster in parsed.get("clusters", []):
        if not isinstance(cluster, dict):
            continue
        label = str(cluster.get("label", "")).strip()
        summary = str(cluster.get("summary", "")).strip()
        items = cluster.get("items") or []
        if not isinstance(items, list):
            items = []
        total = 0
        kept_items: list[str] = []
        for it in items:
            it_norm = _norm_label(str(it))
            if not it_norm:
                continue
            # Resolve back to the actual stored label via the
            # normalised-key map. If the model invented an item that
            # doesn't match any input phrase, skip it.
            matched = norm_to_label.get(it_norm)
            if matched is None or matched in assigned_items:
                continue
            assigned_items.add(matched)
            total += counts.get(matched, 0)
            kept_items.append(matched)
        if label and total > 0:
            cluster_out.append(
                {
                    "label": label,
                    "summary": summary,
                    "count": total,
                    "items": kept_items,
                }
            )

    # Long-tail bucket: anything not covered by the model goes into Other.
    other_total = sum(c for lbl, c in counts.items() if lbl not in assigned_items)
    if other_total > 0:
        cluster_out.append(
            {
                "label": "Other",
                "summary": "Items the synthesis did not assign to a named cluster.",
                "count": other_total,
                "items": [],
            }
        )

    cluster_out.sort(key=lambda c: c["count"], reverse=True)

    out = {
        "analysis": analysis.name,
        "item_kind": analysis.item_kind,
        "method": f"Local Gemma synthesis (analysis v{ANALYSIS_VERSION})",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_items_analysed": total_labels,
        "distinct_labels": distinct,
        "clusters": cluster_out,
    }

    # Atomic write.
    target = _synthesis_path(analysis.name)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return out


# --- Background loop -------------------------------------------------------

# How many new extractions must accumulate before we trigger a new
# synthesis pass. Keeps us from re-clustering after every single new
# message — synthesis is the expensive bit.
_SYNTH_TRIGGER_DELTA = 25
_synth_last_run_counts: dict[str, int] = {}


def _load_objections_at_call_time() -> frozenset[str]:
    # Lazy import to avoid circular import (aggregator imports nothing from
    # this module, but both share the objections helper).
    from .aggregator import _load_objections
    return _load_objections()


async def _one_full_pass() -> None:
    excluded = await asyncio.to_thread(_load_objections_at_call_time)
    for analysis in ALL_ANALYSES:
        try:
            new, skipped = await run_extract_pass(analysis, excluded)
            if new or skipped:
                log.info(
                    "extract '%s': %d new, %d skipped", analysis.name, new, skipped
                )
            # Decide whether to re-synthesize: enough new items since last,
            # OR no synthesis file yet.
            total = await asyncio.to_thread(load_extraction_total_sync, analysis.name)
            last_total = _synth_last_run_counts.get(analysis.name, -1)
            need_synth = (
                load_synthesis_sync(analysis.name) is None
                or (last_total < 0)
                or (total - last_total >= _SYNTH_TRIGGER_DELTA)
            )
            if need_synth and total > 0:
                log.info("running synthesis for '%s' (%d items)", analysis.name, total)
                out = await run_synthesis(analysis)
                if out is not None:
                    _synth_last_run_counts[analysis.name] = total
                    log.info(
                        "synthesis '%s' produced %d clusters", analysis.name, len(out["clusters"])
                    )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("analysis '%s' pass crashed", analysis.name)


async def run_forever() -> None:
    import os
    if os.environ.get("ONGIINI_STATS_LOOP_DISABLED", "").lower() in ("1", "true", "yes"):
        log.warning(
            "qualitative analysis loop DISABLED via ONGIINI_STATS_LOOP_DISABLED — "
            "stats will be served from existing /data/synthesis-*.json snapshots"
        )
        return
    log.info(
        "qualitative analysis loop starting (interval %ds, analyses: %s)",
        settings.topic_classify_interval_seconds,
        ", ".join(a.name for a in ALL_ANALYSES),
    )
    while True:
        try:
            await _one_full_pass()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("qualitative analysis loop crashed; sleeping and retrying")
        await asyncio.sleep(settings.topic_classify_interval_seconds)
