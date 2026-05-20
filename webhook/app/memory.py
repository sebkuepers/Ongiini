import json
from typing import Any

from .config import settings
from .filters import normalize


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
    trimmed = messages[-settings.memory_window * 2 :]
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
