import asyncio
import json
import os
from collections import defaultdict
from typing import Any

from ..config import settings
from ..filters import normalize

# Per-MSISDN asyncio locks so two rapid messages from the SAME user can't
# race on load → modify → save. Different users still run concurrently.
# The dict grows lazily; at pilot scale (small user base) the memory cost
# is trivial. If we ever need to bound it, switch to a weakref dict or
# add a periodic GC of unused locks.
_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def lock_for(msisdn: str) -> asyncio.Lock:
    """Return the per-user lock used to serialize memory access."""
    return _locks[normalize(msisdn)]


def _path_for(msisdn: str):
    return settings.data_dir / f"{normalize(msisdn)}.json"


def load(msisdn: str) -> list[dict[str, Any]]:
    p = _path_for(msisdn)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []


def save(msisdn: str, messages: list[dict[str, Any]]) -> None:
    cap = settings.memory_window * 2
    # A leading system message is the rolling-summary placeholder. Preserve
    # it across trims — losing it defeats the entire point of summarization.
    if messages and messages[0].get("role") == "system":
        trimmed = [messages[0]] + messages[1:][-cap:]
    else:
        trimmed = messages[-cap:]
    p = _path_for(msisdn)
    # Atomic write: serialize to a sibling tempfile then rename. On POSIX
    # os.replace is atomic — readers see either the old file or the new
    # one, never a half-written truncated state. Crash-safe under SIGKILL
    # and power loss. The tempfile is in the same directory so the rename
    # never crosses filesystem boundaries.
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(trimmed, ensure_ascii=False, indent=2))
    os.replace(tmp, p)


def append_synthetic_assistant_turn(msisdn: str, text: str) -> None:
    """Append an assistant-role entry to per-user history WITHOUT a
    preceding user message. Used by the proactive-broadcast path: the
    bot sent the user a template-driven announcement, and we want
    the AI's memory to reflect "I said this to them" so the next
    inbound reply lands with context.

    Distinct from the normal flow (BaseMemoryProvider.record_turn)
    because:
      - no user turn — we initiated
      - no mem0 / long-term write — broadcasts aren't profile facts
      - no hooks fired — billing / tracing skipped (no inference)

    The text IS PII-sanitised before disk write, mirroring the unconditional
    "MUST go through pii.sanitize for any new persistence path" contract
    in ongiini/CLAUDE.md. In practice operator broadcasts won't contain
    email / phone — but if one ever did ("reach me at s@example.com")
    we'd otherwise persist that to every recipient's history.

    Caller MUST hold ``lock_for(msisdn)`` while invoking this so a
    concurrent inbound from the same user can't race the broadcast
    write.
    """
    # Lazy import — short_term is imported widely; avoid the always-on
    # cost of loading pii (regex compilation) on every webhook process.
    from ..pii import sanitize as pii_sanitize

    cleaned = pii_sanitize(text or "")
    msgs = load(msisdn)
    msgs.append({"role": "assistant", "content": cleaned})
    save(msisdn, msgs)


def delete(msisdn: str) -> bool:
    """Wipe a user's stored conversation history.

    Returns True if a file was actually removed, False if nothing was stored.
    """
    p = _path_for(msisdn)
    if p.exists():
        p.unlink()
        return True
    return False
