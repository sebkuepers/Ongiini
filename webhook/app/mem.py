"""Long-term semantic memory via mem0.

This is the "tier 2" memory layer that sits ALONGSIDE the existing
short-term JSON memory in app.memory. The split:

- app.memory  → recent raw turns for THIS conversation (10 turns,
                rolling summary kicks in beyond that)
- app.mem     → durable semantic facts about THIS user, extracted by
                an LLM call when new info appears, retrieved by vector
                similarity to the current query

Stack:
- LLM (for fact extraction): the same vLLM-served Gemma 4 we use for
  replies, via mem0's "vllm" provider against settings.vllm_base_url.
- Vector store: qdrant in embedded-file mode under /data/qdrant/.
  No separate qdrant container needed at pilot scale.
- Embedder: sentence-transformers/all-MiniLM-L6-v2, CPU-only (image
  pre-downloads it during build).

The mem0 client is initialised lazily on first use because constructing
it loads ~80MB of model weights into RAM — we don't want to pay that
cost just to import the module.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

from .config import settings
from .filters import normalize

log = logging.getLogger("ongiini.mem")

_memory_singleton: Any = None
_init_lock = Lock()


def _qdrant_storage_path() -> str:
    p = settings.data_dir / "qdrant"
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _build_config() -> dict:
    return {
        "llm": {
            "provider": "vllm",
            "config": {
                "model": settings.vllm_model,
                "vllm_base_url": settings.vllm_base_url,
                "api_key": "not-needed",
                # Keep fact-extraction calls cheap and deterministic.
                "temperature": 0.1,
                "max_tokens": 400,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "ongiini_memories",
                "path": _qdrant_storage_path(),
                "embedding_model_dims": 384,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dims": 384,
            },
        },
    }


def _client():
    """Lazy singleton — the first call pays the model-load cost, subsequent calls are free."""
    global _memory_singleton
    if _memory_singleton is not None:
        return _memory_singleton
    with _init_lock:
        if _memory_singleton is not None:
            return _memory_singleton
        # Defer the heavy imports until the first call so that simply
        # importing app.mem doesn't load torch.
        from mem0 import Memory  # noqa: WPS433
        log.info("initialising mem0 (this loads the embedding model — ~80MB RAM)")
        _memory_singleton = Memory.from_config(_build_config())
    return _memory_singleton


def add_turn(msisdn: str, user_text: str, assistant_text: str) -> None:
    """Feed one completed turn to mem0 so it can extract / update facts.

    mem0 itself decides what (if anything) is worth remembering. Most
    turns will be no-ops (e.g. "what's 2+2?" yields no durable facts).
    A turn like "I'm a farmer in Oshakati" produces one or two facts.

    Never raises — long-term memory is a soft enhancement; if mem0 hiccups
    we don't want to break the live reply path.
    """
    try:
        m = _client()
        msgs = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        m.add(msgs, user_id=normalize(msisdn))
    except Exception as exc:
        log.warning("mem0 add_turn failed for %s: %s", msisdn, exc)


def search(msisdn: str, query: str, limit: int = 5) -> list[dict]:
    """Return up to `limit` memories relevant to the query, ranked by similarity.

    Each item is a dict from mem0; the only field we rely on is
    "memory" (the stored fact string). Returns [] on any error.
    """
    try:
        m = _client()
        result = m.search(query=query, user_id=normalize(msisdn), limit=limit)
        # mem0 0.1.x returns {"results": [...]}; older versions returned a bare list.
        if isinstance(result, dict):
            return list(result.get("results", []))
        return list(result or [])
    except Exception as exc:
        log.warning("mem0 search failed for %s: %s", msisdn, exc)
        return []


def list_all(msisdn: str) -> list[dict]:
    """Return every stored memory for a user. Used by the whats_in_my_memory tool."""
    try:
        m = _client()
        result = m.get_all(user_id=normalize(msisdn))
        if isinstance(result, dict):
            return list(result.get("results", []))
        return list(result or [])
    except Exception as exc:
        log.warning("mem0 list_all failed for %s: %s", msisdn, exc)
        return []


def delete_all(msisdn: str) -> bool:
    """Wipe every stored memory for a user. Returns True on success."""
    try:
        m = _client()
        m.delete_all(user_id=normalize(msisdn))
        return True
    except Exception as exc:
        log.warning("mem0 delete_all failed for %s: %s", msisdn, exc)
        return False
