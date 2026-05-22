"""Classifier protocol — the upstream orchestration decision.

The Classifier looks at one inbound message and returns a
``ClassifierResult`` with a verdict (NONE / ADMIN / DOCS / SEARCH) and
a depth (SHALLOW / DEEP). The verdict + depth pair selects a Policy
from the PolicyTable, and the Policy drives the rest of the turn.

The result is a small value object (NOT a Step) so adapter impls
never have to know about the Step contract or stamp timing fields —
the executor wraps the result into a RouterStep with its own timing.

Why a protocol rather than a function: implementations may need state
(an LLM client for Gemma-as-classifier, a cached embedding model, etc.)
and may have setup/teardown concerns the executor shouldn't see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .policy import DEPTH_SHALLOW, VERDICT_NONE
from .transport import InboundMessage


@dataclass
class ClassifierResult:
    """Value returned by ``Classifier.classify``. Owela's executor wraps
    this into a ``RouterStep`` and is responsible for timing.

    ``tokens_in`` / ``tokens_out`` / ``cached_tokens`` are propagated
    into the RouterStep so billing hooks see classifier cost. ``attrs``
    is for adapter-specific extras (e.g. the raw classifier response
    object for debug logging).
    """
    verdict: str = VERDICT_NONE
    depth: str = DEPTH_SHALLOW
    tokens_in: int = 0
    tokens_out: int = 0
    cached_tokens: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Classifier(Protocol):
    """Single method: classify an inbound message.

    Implementations must be fail-safe: any error (timeout, parse failure,
    network blip) should yield ``ClassifierResult(verdict="NONE")`` rather
    than raising. That way the executor falls back to the global default
    policy and the user still gets a reply.

    The Ongiini impl uses Gemma 4 as the classifier with a prefix-cached
    prompt (~270 tokens) and a 2s timeout. See
    ``ongiini/routers/gemma_classifier.py``.
    """

    async def classify(self, msg: InboundMessage) -> ClassifierResult:
        ...
