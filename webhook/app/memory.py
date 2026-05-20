import asyncio
import json
from collections import defaultdict
from typing import Any

from .config import settings
from .filters import normalize

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
    p.write_text(json.dumps(trimmed, ensure_ascii=False, indent=2))


def delete(msisdn: str) -> bool:
    """Wipe a user's stored conversation history.

    Returns True if a file was actually removed, False if nothing was stored.
    """
    p = _path_for(msisdn)
    if p.exists():
        p.unlink()
        return True
    return False
