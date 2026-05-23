"""Per-user source-URL index for cross-turn citation recall.

WHY this tier exists (in addition to short_term + mem0):
  The short-term tier caps at ~50 turns and gets folded into a rolling
  summary beyond that. When a user has been chatting for 60+ turns and
  asks "give me the sources you used earlier", the cited URLs from
  turn 1 are no longer in the model's visible context — they were
  collapsed into prose by the summariser.

  This index persists JUST the URLs (no surrounding prose) so the
  MemoryProvider can re-inject them as a compact system block when
  history doesn't carry them anymore.

Storage shape:
  data/source_index/<msisdn>.json -- list[dict], newest-first, capped
  at _MAX_ENTRIES. Each entry: {"url": str, "ts": ISO-8601}.

Atomic writes (tempfile + os.replace), same crash-safety pattern as
short_term. Dedup by URL keeping the newest timestamp.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import settings
from ..filters import normalize

log = logging.getLogger("ongiini.memory.source_index")


# How many URLs to keep on disk per user. A long research conversation
# might cite 5-15 URLs total; 30 leaves headroom for chatty users.
_MAX_ENTRIES = 30

# How many to inject into the model context when assembling messages.
# 10 fits the "give me sources" replay case without bloating the
# system block.
_INJECT_LIMIT = 10


def _path_for(msisdn: str) -> Path:
    return settings.data_dir / "source_index" / f"{normalize(msisdn)}.json"


def load(msisdn: str) -> list[dict[str, Any]]:
    """Return the user's source index newest-first. Empty list on any
    failure — never raises."""
    p = _path_for(msisdn)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("source_index.load failed for %s: %s", msisdn, exc)
        return []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict) and isinstance(e.get("url"), str)]


def append(msisdn: str, urls: list[str]) -> None:
    """Merge ``urls`` into the user's index. Dedup by URL keeping the
    newest timestamp; cap at ``_MAX_ENTRIES``; persist atomically.

    Soft-fails on any IO error — the caller (the Hook) is also
    soft-fail, so a broken append never crashes a successful reply.
    """
    fresh = [u for u in urls if isinstance(u, str) and u]
    if not fresh:
        return
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    existing = load(msisdn)
    by_url: dict[str, dict[str, Any]] = {}
    for e in existing:
        by_url[e["url"]] = e
    for u in fresh:
        by_url[u] = {"url": u, "ts": now}    # newest ts wins on collision
    combined = sorted(
        by_url.values(),
        key=lambda x: x.get("ts", ""),
        reverse=True,
    )[:_MAX_ENTRIES]
    p = _path_for(msisdn)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(combined, ensure_ascii=False, indent=2))
        os.replace(tmp, p)
    except OSError as exc:
        log.warning("source_index.append write failed for %s: %s", msisdn, exc)


def delete(msisdn: str) -> bool:
    """Wipe the user's source index. Called from ``delete_my_data``.
    Returns True if a file was actually removed."""
    p = _path_for(msisdn)
    if p.exists():
        try:
            p.unlink()
            return True
        except OSError as exc:
            log.warning("source_index.delete failed for %s: %s", msisdn, exc)
    return False


def format_for_injection(entries: list[dict[str, Any]]) -> str:
    """Compact system-message body listing recent URLs. Empty string if
    there's nothing to inject — caller should skip the message append."""
    if not entries:
        return ""
    recent = entries[:_INJECT_LIMIT]
    lines = [
        "Sources cited in earlier turns of this conversation (re-list "
        "these if asked for sources, references, or links):",
    ]
    for e in recent:
        lines.append(f"- {e['url']}")
    return "\n".join(lines)
