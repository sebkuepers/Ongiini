"""Model protocol — one LLM round-trip.

Wraps whatever client the application uses to talk to its model.
Today the only impl is the vLLM/OpenAI-compatible adapter living in
``ongiini/models/vllm_gemma.py``; the protocol is intentionally
designed around the OpenAI chat-completions shape because that's what
Gemma + vLLM serves and that's what every other major engine will
serve too.

Why an explicit protocol rather than just using ``AsyncOpenAI`` directly:
the call carries engine knobs the bare client doesn't model — a thinking
mode with its own budget, prefix-cache–aware token reporting via
``cached_tokens``, output sanitising. The adapter hides those behind a
uniform contract.

``ModelRequest`` is self-describing: everything the adapter needs is on
the request, so the same ``complete`` path serves the act loop AND
auxiliary calls (classifier, planner, reviewer, summariser). One adapter,
one output-sanitising path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .policy import AUTO, THINKING_OFF, Policy, ToolChoice


@dataclass
class ModelRequest:
    """All inputs to one chat.completions.create call.

    ``max_tokens`` bounds the visible reply; when ``thinking`` is not
    "off" the adapter adds ``thinking_budget`` on top so reasoning cannot
    consume the answer. ``response_format`` is "json_object" or None.
    ``timeout_s`` is a per-request ceiling (None = adapter default).
    ``policy`` is metadata for adapters that want it — nothing an adapter
    needs to build the call may live only there.
    """
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: ToolChoice = AUTO
    max_tokens: int | None = None
    temperature: float | None = None
    thinking: str = THINKING_OFF
    thinking_budget: int | None = None
    response_format: str | None = None
    timeout_s: float | None = None
    policy: Policy | None = None


@dataclass
class ModelResponse:
    """Normalised output. Token counts are already cache-corrected — i.e.
    ``tokens_in`` is the BILLABLE input (cached subtracted), and
    ``cached_tokens`` is reported separately for observability.

    ``raw`` is the underlying provider response object (e.g. an
    ``openai.ChatCompletion``). Tests pass; production code should not
    depend on its shape because it varies by adapter.

    ``attrs`` is a free-form metadata bag the adapter can use to
    surface engine-specific audit signals to the executor (which
    merges them into ``ModelCallStep.attrs``). Examples: a Gemma 4
    adapter setting ``reasoning_leak_stripped=N`` when it scrubbed
    leaked channel tokens; a future Claude adapter recording
    ``thinking_tokens_used``. Owela treats the contents as opaque."""
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cached_tokens: int = 0
    raw: Any = None
    attrs: dict[str, Any] = field(default_factory=dict)


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
