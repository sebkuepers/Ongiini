"""Regression guard for the 2026-05-23 mem0 data-loss bug.

mem0 0.1.x's QdrantDB.__init__ (vendored at
``site-packages/mem0/vector_stores/qdrant.py:65-67``) does this:

    if not on_disk:
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)

When ``on_disk`` defaults to False, mem0 wipes the entire qdrant storage
directory on every init. We hit this on every container restart, which
destroyed 87% of today's mem0 facts before we identified the cause.

The fix is to pass ``on_disk=True`` in our vector_store config. This
test locks that in so a future config edit can't silently restore the
bug.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock


def _import_long_term():
    """Import ongiini.memory.long_term with the heavy mem0/qdrant/
    transformers dependency chain stubbed out. The module's
    ``_build_config()`` is pure — it only constructs a dict — so the
    stubs are never actually called by the test."""
    for mod in (
        "mem0", "mem0.llms", "mem0.llms.vllm", "mem0.embeddings",
        "mem0.embeddings.huggingface", "mem0.vector_stores",
        "mem0.vector_stores.qdrant", "mem0.memory", "mem0.memory.main",
        "mem0.utils", "mem0.utils.factory",
    ):
        sys.modules.setdefault(mod, MagicMock())
    if "ongiini.memory.long_term" in sys.modules:
        return importlib.reload(sys.modules["ongiini.memory.long_term"])
    return importlib.import_module("ongiini.memory.long_term")


def test_qdrant_config_has_on_disk_true():
    """The PRIMARY regression guard. If on_disk is missing/False,
    mem0 will rmtree /data/qdrant on every container restart and
    silently destroy all accumulated facts."""
    lt = _import_long_term()
    cfg = lt._build_config()
    qdrant_cfg = cfg["vector_store"]["config"]
    assert "on_disk" in qdrant_cfg, (
        "vector_store.config must include 'on_disk' — without it, mem0 "
        "defaults to on_disk=False and wipes the qdrant directory on "
        "every restart (2026-05-23 production data-loss bug)."
    )
    assert qdrant_cfg["on_disk"] is True, (
        f"vector_store.config['on_disk'] must be True, got "
        f"{qdrant_cfg['on_disk']!r}. See mem0 vector_stores/qdrant.py "
        f"lines 65-67 — when on_disk is falsy, mem0 calls "
        f"shutil.rmtree(path) on init."
    )


def test_qdrant_config_keeps_required_fields():
    """Sanity: don't accidentally drop the other required qdrant config
    when editing on_disk."""
    lt = _import_long_term()
    qdrant_cfg = lt._build_config()["vector_store"]["config"]
    assert qdrant_cfg["collection_name"] == "ongiini_memories"
    assert qdrant_cfg["embedding_model_dims"] == 384
    assert "path" in qdrant_cfg
    assert qdrant_cfg["path"].endswith("qdrant")
