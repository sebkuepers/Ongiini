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
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from openai import AsyncOpenAI

from .. import mem
from ..config import settings
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


def _iter_user_profiles(excluded: frozenset[str]) -> Iterator[tuple[str, str]]:
    """Yield (msisdn-hash, profile_text) — one item per user.

    profile_text is the concatenation of that user's [PROFILE] facts
    from mem0. Users with no PROFILE facts are skipped.
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
        profile_bits: list[str] = []
        for f in facts or []:
            txt = ""
            if isinstance(f, dict):
                txt = str(f.get("memory") or f.get("text") or "")
            elif isinstance(f, str):
                txt = f
            if not txt:
                continue
            if "[PROFILE]" in txt:
                # Strip the tag, keep just the descriptor text.
                cleaned = txt.replace("[PROFILE]", "").strip()
                if cleaned:
                    profile_bits.append(cleaned)
        if not profile_bits:
            continue
        combined = "; ".join(profile_bits)[:1500]
        # Hash the msisdn (not the profile text) so we get one row per
        # user; if the profile evolves the label can update (we INSERT
        # OR REPLACE on extraction store).
        h = hashlib.sha256(msisdn.encode("utf-8")).hexdigest()
        yield h, combined


# --- Analysis definitions --------------------------------------------------

@dataclass
class Analysis:
    name: str
    description: str
    source: Callable[[frozenset[str]], Iterator[tuple[str, str]]]
    extract_system: str
    synthesize_system: str
    item_kind: str  # "message" | "user" — used in synthesis prompt and aggregator labelling


TOPICS_ANALYSIS = Analysis(
    name="topics",
    description="What people use Ongiini for, emergent from message content",
    source=_iter_user_messages,
    extract_system=(
        "You read one short message from a user of a free Namibian AI helper. "
        "Reply with a SHORT noun phrase (3-7 words) describing WHAT the user is "
        "asking about — the topic, not the answer. Be specific and concrete.\n\n"
        "Rules:\n"
        "- Output ONLY the phrase. No quotes, no punctuation at the end, no explanation.\n"
        "- Use English regardless of the message's language.\n"
        "- Prefer concrete nouns over generic categories ('yellowing maize leaves' "
        "  beats 'agriculture'; 'grade 11 chemistry homework' beats 'school').\n"
        "- For pure greetings, thanks, or yes/no with no topic, output 'small talk'.\n"
    ),
    synthesize_system=(
        "You are analysing how people use a free AI helper in Namibia. Below is a "
        "list of short topic phrases, each with a count of how often it appeared. "
        "Cluster them into meaningful named themes.\n\n"
        "Rules:\n"
        "- Produce 6-12 clusters. Don't over-fragment — closely related phrases "
        "  belong together.\n"
        "- Each cluster label is 1-4 words, clear and concrete (e.g. 'Crop health', "
        "  'School homework', 'Government forms', 'Childhood illness').\n"
        "- Each cluster summary is one sentence describing what's in it.\n"
        "- Assign EVERY input phrase to exactly one cluster.\n"
        "- Output strict JSON in this shape, with no preamble or trailing text:\n"
        '  {"clusters": [{"label": "...", "summary": "...", "items": ["phrase1", "phrase2"]}]}\n'
        "- Items must be EXACT verbatim copies of the input phrases.\n"
    ),
    item_kind="message",
)


ROLES_ANALYSIS = Analysis(
    name="roles",
    description="What roles/professions describe the user base, emergent from profile facts",
    source=_iter_user_profiles,
    extract_system=(
        "You read a short profile description of one user of a free Namibian AI "
        "helper. Reply with a SHORT role description (1-4 words) capturing the "
        "person's dominant role or situation — e.g. 'smallholder farmer', "
        "'matric student', 'first-time mother', 'taxi driver', 'small shop owner'.\n\n"
        "Rules:\n"
        "- Output ONLY the role phrase. No quotes, no punctuation at the end, "
        "  no explanation.\n"
        "- Use English regardless of the profile's language.\n"
        "- If the profile only says a location with no role, output 'unspecified'.\n"
    ),
    synthesize_system=(
        "You are analysing the user base of a free AI helper in Namibia. Below "
        "is a list of short role descriptions, each with a count of how many "
        "users had that description. Cluster them into meaningful named groups.\n\n"
        "Rules:\n"
        "- Produce 5-10 clusters.\n"
        "- Each cluster label is 1-4 words (e.g. 'Farmers', 'Students', "
        "  'Health workers').\n"
        "- Each cluster summary is one sentence describing the group.\n"
        "- Assign EVERY input phrase to exactly one cluster.\n"
        "- Output strict JSON in this shape, with no preamble or trailing text:\n"
        '  {"clusters": [{"label": "...", "summary": "...", "items": ["phrase1", "phrase2"]}]}\n'
        "- Items must be EXACT verbatim copies of the input phrases.\n"
    ),
    item_kind="user",
)


ALL_ANALYSES: list[Analysis] = [TOPICS_ANALYSIS, ROLES_ANALYSIS]


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
    """Extract labels for unseen items of one analysis. Returns (new, skipped)."""
    await asyncio.to_thread(_init_db_sync)
    known = await asyncio.to_thread(_known_hashes_sync, analysis.name)
    new = 0
    skipped = 0
    seen_this_pass: set[str] = set()
    for item_hash, text in analysis.source(excluded):
        if item_hash in known or item_hash in seen_this_pass:
            continue
        seen_this_pass.add(item_hash)
        label = await _extract_one(analysis.extract_system, text)
        if not label:
            skipped += 1
            await asyncio.sleep(0)
            continue
        await asyncio.to_thread(_store_extraction_sync, analysis.name, item_hash, label)
        new += 1
        await asyncio.sleep(0)
    return new, skipped


# --- Synthesis pass --------------------------------------------------------

_MAX_SYNTH_ITEMS = 1000   # cap descriptions fed to the LLM in one synthesis call
_SYNTH_TIMEOUT_SECONDS = 120


from .synthesis_io import synthesis_path as _synthesis_path  # noqa: E402
from .synthesis_io import load_synthesis as load_synthesis_sync  # noqa: E402


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
                max_tokens=4000,
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

    # Compute per-cluster counts by summing counts of assigned items.
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
            it = str(it).strip()
            if not it or it in assigned_items:
                continue
            assigned_items.add(it)
            total += counts.get(it, 0)
            kept_items.append(it)
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
