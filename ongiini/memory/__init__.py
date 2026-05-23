"""Memory subpackage — short-term JSON + mem0 long-term + source index + provider.

Submodules:
  - ``short_term``   — per-user JSON rolling history (~50 turns, capped)
  - ``long_term``    — mem0 vector store with typed [TAG] facts
  - ``source_index`` — per-user URL index for cross-turn source recall (v1.6-B)
  - ``provider``     — ``OngiiniMemoryProvider`` that combines all three
                       and implements ``owela.MemoryProvider``.

The provider is the public surface for the Owela runtime; the backend
modules are importable directly when application code (e.g. the
``whats_in_my_memory`` tool, or the ``SourceIndexHook``) wants to
reach behind the provider abstraction.
"""

from .provider import OngiiniMemoryProvider

__all__ = ["OngiiniMemoryProvider"]
