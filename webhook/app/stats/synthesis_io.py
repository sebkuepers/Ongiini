"""Lightweight read-only access to synthesis JSON files.

Pure stdlib — no mem0, no openai, no aiosqlite. Imported by the
aggregator (which runs on every /stats.json request and must be cheap
to import) without dragging in the LLM-side dependencies that the
analyses module needs.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import settings


def synthesis_path(analysis_name: str) -> Path:
    return settings.data_dir / f"synthesis-{analysis_name}.json"


def load_synthesis(analysis_name: str) -> dict | None:
    path = synthesis_path(analysis_name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
