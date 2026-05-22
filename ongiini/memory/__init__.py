"""Memory subpackage — short-term JSON + mem0 long-term + provider.

Three submodules:
  - ``short_term`` — per-user JSON rolling history (~50 turns, capped)
  - ``long_term`` — mem0 vector store with typed [TAG] facts
  - ``provider`` — ``OngiiniMemoryProvider`` that combines the two
                   and implements ``owela.MemoryProvider``.

The provider is the public surface for the Owela runtime; the two
backend modules are importable directly when application code (e.g.
the ``whats_in_my_memory`` tool) wants to reach behind the provider
abstraction.
"""

from .provider import OngiiniMemoryProvider

__all__ = ["OngiiniMemoryProvider"]
