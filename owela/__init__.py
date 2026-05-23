"""Owela — an opinionated chat-agent framework.

Owela (Oshiwambo: "thread") wires a chat turn as a thread of typed
steps: router → (plan) → act → tool(s) → (critique) → reply. Loop
shape is decided UPFRONT by an explicit Policy table driven by a
classifier, not by the model's emergent behaviour.

See the plan at
``~/.claude/plans/lets-improve-the-website-lexical-island.md``
for the architectural rationale and anti-trap principles.

Quick start for application code::

    from owela import (
        Agent, Runtime, Policy, PolicyTable, HookRegistry, ToolRegistry,
        InboundMessage, tool, AUTO, force_tool,
        VERDICT_NONE, VERDICT_ADMIN, VERDICT_DOCS, VERDICT_SEARCH,
        DEPTH_SHALLOW, DEPTH_DEEP,
    )

    @tool(name="web_search", params={"query": "Search query."})
    async def web_search(query: str) -> str:
        ...

    runtime = Runtime(
        model=MyModel(),
        transport=MyTransport(),
        memory=MyMemory(),
        classifier=MyClassifier(),
        tools=ToolRegistry([web_search]),
        policies=PolicyTable().set(
            VERDICT_SEARCH, DEPTH_SHALLOW,
            Policy(name="search_shallow", first_tool=force_tool("web_search")),
        ),
        hooks=HookRegistry(),
    )
    agent = Agent(runtime)

    result = await agent.handle(InboundMessage(...))
"""

from __future__ import annotations

from .agent import Agent, HandleResult
from .errors import ModelError, OwelaError, PolicyNotFound, ToolError
from .executor import execute_turn
from .hooks import Hook, HookRegistry, TurnContext
from .hooks_builtin import MemoryRecordingHook
from .memory import MemoryProvider
from .model import Model, ModelRequest, ModelResponse
from .policy import (
    ALL_DEPTHS, ALL_VERDICTS, AUTO, DEPTH_DEEP, DEPTH_SHALLOW, Policy,
    PolicyTable, ToolChoice, VERDICT_ADMIN, VERDICT_DOCS, VERDICT_NONE,
    VERDICT_SEARCH, force_tool,
)
from .router import Classifier, ClassifierResult
from .runtime import Planner, Reviewer, Runtime
from .step import (
    CritiqueStep, ModelCallStep, PlanStep, QueryVariant, ReplyStep,
    ReviseStep, RouterStep, Step, ToolStep,
)
from .tools import (
    ToolContext, ToolRegistry, ToolSpec, reset_global_registry, tool,
)
from .transport import InboundMessage, Transport

__all__ = [
    # Agent / Runtime / Executor
    "Agent",
    "HandleResult",
    "Runtime",
    "execute_turn",
    # Protocols
    "Model",
    "Transport",
    "MemoryProvider",
    "Classifier",
    "ClassifierResult",
    "Planner",
    "Reviewer",
    "Hook",
    # Hooks
    "HookRegistry",
    "TurnContext",
    "MemoryRecordingHook",
    # Model
    "ModelRequest",
    "ModelResponse",
    # Policy
    "Policy",
    "PolicyTable",
    "ToolChoice",
    "AUTO",
    "force_tool",
    "VERDICT_NONE",
    "VERDICT_ADMIN",
    "VERDICT_DOCS",
    "VERDICT_SEARCH",
    "DEPTH_SHALLOW",
    "DEPTH_DEEP",
    "ALL_VERDICTS",
    "ALL_DEPTHS",
    # Steps
    "Step",
    "RouterStep",
    "PlanStep",
    "QueryVariant",
    "ModelCallStep",
    "ToolStep",
    "CritiqueStep",
    "ReviseStep",
    "ReplyStep",
    # Tools
    "tool",
    "ToolRegistry",
    "ToolSpec",
    "ToolContext",
    "reset_global_registry",
    # Transport
    "InboundMessage",
    # Errors
    "OwelaError",
    "ToolError",
    "ModelError",
    "PolicyNotFound",
]
