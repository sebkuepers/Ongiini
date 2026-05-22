"""Stdlib-only readers for the qualia.sqlite extraction cache.

Mirrors the synthesis_io.py pattern: the aggregator must be importable
without dragging in mem0 / openai / qdrant via analyses.py. This file
exposes just the read-side SQLite helpers, keyed by the current
analysis version from taxonomy.py.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..config import settings
from .taxonomy import ANALYSIS_VERSION


def _db_path() -> Path:
    return settings.data_dir / "qualia.sqlite"


def load_label_counts_via_io(analysis_name: str) -> dict[str, int]:
    """Return {label: count} for the current ANALYSIS_VERSION of one
    analysis. Empty dict if the table or file doesn't exist yet."""
    path = _db_path()
    if not path.exists():
        return {}
    try:
        with sqlite3.connect(path) as conn:
            rows = conn.execute(
                """
                SELECT label, COUNT(*)
                FROM extractions
                WHERE analysis = ? AND version = ?
                GROUP BY label
                """,
                (analysis_name, ANALYSIS_VERSION),
            ).fetchall()
        return {lbl: int(c) for lbl, c in rows}
    except sqlite3.Error:
        return {}


def load_extraction_total_via_io(analysis_name: str) -> int:
    path = _db_path()
    if not path.exists():
        return 0
    try:
        with sqlite3.connect(path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM extractions WHERE analysis = ? AND version = ?",
                (analysis_name, ANALYSIS_VERSION),
            ).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
