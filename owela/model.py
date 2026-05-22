"""Model protocol — one LLM round-trip.

Wraps whatever client the application uses to talk to its model.
Today the only impl is the vLLM/OpenAI-compatible adapter living in
``webhook/app/models/vllm_gemma.py``; the protocol is intentionally
designed around the OpenAI chat-completions shape because that's what
Gemma + vLLM serves and that's what every other major engine will
serve too.

Why an explicit protocol rather than just using ``AsyncOpenAI`` directly:
the Gemma-on-vLLM call carries Owela-specific extras the bare client
doesn't model — selective ``enable_thinking``, ``reasoning_budget``,
prefix-cache–aware token reporting via ``cached_tokens``. The adapter
hides those behind a uniform contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .policy import Policy, ToolChoice


@dataclass
class ModelRequest:
    """All inputs to one chat.completions.create call."""
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    tool_choice: ToolChoice
    policy: Policy
    enable_thinking: bool = False


@dataclass
class ModelResponse:
    """Normalised output. Token counts are already cache-corrected — i.e.
    ``tokens_in`` is the BILLABLE input (cached subtracted), and
    ``cached_tokens`` is reported separately for observability.

    ``raw`` is the underlying provider response object (e.g. an
    ``openai.ChatCompletion``). Tests pass; production code should not
    depend on its shape because it varies by adapter."""
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cached_tokens: int = 0
    raw: Any = None


@runtime_checkable
class Model(Protocol):
    """Single method: send a request, get a response.

    Implementations should NOT retry transparently — if a single attempt
    fails, raise. Retry policy lives one layer up (in the application's
    webhook handler, where the per-user lock is held and the duplicate
    message detection runs).
    """

    async def complete(self, req: ModelRequest) -> ModelResponse:
        ...
