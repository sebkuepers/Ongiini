"""@tool decorator, ToolRegistry, parallel dispatch.

Tools are plain async Python functions; the decorator inspects their
signature + docstring to autogenerate the OpenAI function-call schema.
Schema and implementation can never drift because they live next to
each other and are derived from the same source.

Tools that need runtime access (e.g. ``delete_my_data`` needs the
user_id, ``my_token_usage`` needs the usage hook) declare their first
parameter as ``ctx: ToolContext`` — the decorator detects this by type
annotation, excludes ctx from the schema, and the registry injects it
at execute time.

Anti-trap principle #2: tools are decorated functions, not dict schemas.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING, Any, Awaitable, Callable, get_args, get_origin, get_type_hints,
)

from .errors import ToolError
from .step import ToolStep
from .transport import InboundMessage

if TYPE_CHECKING:
    from .runtime import Runtime

log = logging.getLogger("owela.tools")


@dataclass
class ToolContext:
    """Passed to tools whose first parameter is typed ``ToolContext``."""
    user_id: str
    runtime: "Runtime"
    msg: InboundMessage


# Python → JSON Schema type map. Keep small and explicit — adding a new
# type is a deliberate decision, not a "just supports everything" surprise.
_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


@dataclass
class ToolSpec:
    """Captured metadata for one tool: name, description, JSON schema,
    callable, and whether it expects a ToolContext as first arg."""
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Awaitable[str]]
    needs_context: bool = False
    param_descriptions: dict[str, str] = field(default_factory=dict)


# Process-global registry for tools that register themselves at import
# time. The Runtime can pull from this OR be constructed with an explicit
# list — tests prefer explicit. Avoid relying on the global in production
# code; treat it as a convenience for the common case.
#
# Tests should give each @tool a uniquely-named registration to avoid
# overwriting each other across the same Python session, OR call
# ``reset_global_registry()`` in a fixture to start each test clean.
_GLOBAL_REGISTRY: dict[str, ToolSpec] = {}


def reset_global_registry() -> None:
    """Clear the process-global @tool registry. For tests only."""
    _GLOBAL_REGISTRY.clear()


def _python_type_to_json(py_type: Any, param_name: str, tool_name: str) -> dict[str, Any]:
    """Map a Python type annotation to a JSON Schema fragment.

    Supports ``str``, ``int``, ``float``, ``bool``, and ``list[T]`` for
    those T. Anything else raises ``TypeError`` at decoration time — we
    want a loud failure at import, not a silently malformed schema at
    request time.
    """
    if py_type in _JSON_TYPES:
        return {"type": _JSON_TYPES[py_type]}
    origin = get_origin(py_type)
    if origin in (list, tuple):
        args = get_args(py_type)
        item_type = args[0] if args else str
        if item_type not in _JSON_TYPES:
            raise TypeError(
                f"@tool {tool_name!r}: parameter {param_name!r} has "
                f"unsupported list item type {item_type!r}"
            )
        return {"type": "array", "items": {"type": _JSON_TYPES[item_type]}}
    raise TypeError(
        f"@tool {tool_name!r}: parameter {param_name!r} has unsupported "
        f"type {py_type!r}. Supported: str, int, float, bool, list[str|int|float|bool]."
    )


def _build_schema(
    fn: Callable[..., Awaitable[Any]],
    tool_name: str,
    param_descriptions: dict[str, str],
) -> tuple[dict[str, Any], bool]:
    """Return (parameters_schema, needs_context). Inspects signature."""
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    params_iter = iter(sig.parameters.values())
    needs_context = False

    # Detect a leading ctx: ToolContext parameter and skip it from the schema.
    first = next(params_iter, None)
    if first is not None and hints.get(first.name) is ToolContext:
        needs_context = True
    else:
        # Put the first param back at the front of our processing list.
        params_iter = iter([first, *params_iter]) if first is not None else iter([])

    properties: dict[str, Any] = {}
    required: list[str] = []
    for p in params_iter:
        if p.name == "self":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(
                f"@tool {tool_name!r}: variadic params (*args / **kwargs) are not supported"
            )
        if p.name not in hints:
            raise TypeError(
                f"@tool {tool_name!r}: parameter {p.name!r} needs a type annotation"
            )
        properties[p.name] = _python_type_to_json(hints[p.name], p.name, tool_name)
        if p.name in param_descriptions:
            properties[p.name]["description"] = param_descriptions[p.name]
        if p.default is inspect.Parameter.empty:
            required.append(p.name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema, needs_context


def tool(
    name: str | None = None,
    description: str | None = None,
    params: dict[str, str] | None = None,
) -> Callable[[Callable[..., Awaitable[str]]], Callable[..., Awaitable[str]]]:
    """Decorator: register an async function as an Owela tool.

    Usage::

        @tool(
            name="web_search",
            description="Search the web for current info...",
            params={"query": "Search query in natural language."},
        )
        async def web_search(query: str) -> str:
            ...

    The decorated function is unchanged for direct callers (e.g. unit
    tests can ``await web_search("...")`` normally). The schema lives on
    ``fn.__owela_tool__`` and in the global registry under ``name``.
    """
    def deco(fn: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
        if not asyncio.iscoroutinefunction(fn):
            raise TypeError(f"@tool: {fn.__name__} must be `async def`")
        resolved_name = name or fn.__name__
        doc = (description or (fn.__doc__ or "").strip()).strip()
        if not doc:
            raise ValueError(
                f"@tool {resolved_name!r}: must have either an explicit description "
                f"or a non-empty docstring"
            )
        param_descs = dict(params or {})
        schema, needs_ctx = _build_schema(fn, resolved_name, param_descs)
        spec = ToolSpec(
            name=resolved_name,
            description=doc,
            parameters=schema,
            fn=fn,
            needs_context=needs_ctx,
            param_descriptions=param_descs,
        )
        fn.__owela_tool__ = spec   # type: ignore[attr-defined]
        _GLOBAL_REGISTRY[resolved_name] = spec
        return fn
    return deco


class ToolRegistry:
    """A registry the Runtime hands to the executor. Holds ToolSpecs and
    runs tool executions, including PARALLEL dispatch via
    ``asyncio.gather`` when the model emits multiple tool_calls in one turn.

    Construct with an explicit list of tools (preferred for tests) or
    call ``.from_global()`` to pick up everything ``@tool``-registered
    at import time.
    """
    def __init__(self, tools: list[ToolSpec | Callable] | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for t in (tools or []):
            self.add(t)

    @classmethod
    def from_global(cls) -> "ToolRegistry":
        return cls(list(_GLOBAL_REGISTRY.values()))

    def add(self, tool_or_fn: ToolSpec | Callable) -> "ToolRegistry":
        if isinstance(tool_or_fn, ToolSpec):
            spec = tool_or_fn
        elif hasattr(tool_or_fn, "__owela_tool__"):
            spec = tool_or_fn.__owela_tool__   # type: ignore[attr-defined]
        else:
            raise TypeError(
                "ToolRegistry.add expects a ToolSpec or a @tool-decorated function"
            )
        self._tools[spec.name] = spec
        return self

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self, expose: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """Return the OpenAI tool-list shape. ``expose=None`` returns all;
        otherwise only those names."""
        names = self._tools.keys() if expose is None else expose
        out: list[dict[str, Any]] = []
        for n in names:
            spec = self._tools.get(n)
            if spec is None:
                continue
            out.append({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            })
        return out

    async def execute(
        self,
        tool_call: dict[str, Any],
        ctx: ToolContext,
    ) -> ToolStep:
        """Run one tool_call. The OpenAI shape is::

            {"id": "...", "type": "function",
             "function": {"name": "...", "arguments": "<json string>"}}

        Returns a ToolStep with the result text in ``attrs["result"]``.
        Tool failures are caught and turned into ``error`` text — the
        executor still gets a ToolStep and can show a sensible message
        to the model on the next turn.
        """
        started = time.monotonic()
        fn_block = tool_call.get("function") or {}
        name = fn_block.get("name", "")
        raw_args = fn_block.get("arguments") or "{}"

        step = ToolStep(
            started_at=started,
            tool_name=name,
            tool_call_id=tool_call.get("id", ""),
            args_len=len(raw_args),
        )

        spec = self._tools.get(name)
        if spec is None:
            step.error = f"unknown tool: {name}"
            step.attrs["result"] = f"Error: tool {name!r} is not available."
            step.result_len = len(step.attrs["result"])
            step.ended_at = time.monotonic()
            return step

        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}

        try:
            if spec.needs_context:
                result = await spec.fn(ctx, **args)
            else:
                result = await spec.fn(**args)
        except ToolError as exc:
            step.error = str(exc)
            step.attrs["result"] = f"Tool error: {exc}"
        except Exception as exc:                     # noqa: BLE001 — tool runs are sandboxed
            log.exception("tool %s raised: %s", name, exc)
            step.error = repr(exc)
            step.attrs["result"] = f"Tool {name} failed: {exc}"
        else:
            step.attrs["result"] = result if isinstance(result, str) else str(result)

        step.result_len = len(step.attrs["result"])
        step.ended_at = time.monotonic()
        return step

    async def execute_parallel(
        self,
        tool_calls: list[dict[str, Any]],
        ctx: ToolContext,
    ) -> list[ToolStep]:
        """Run a batch of tool_calls concurrently. ``asyncio.gather`` —
        every call lands at the same time so total wall time is roughly
        ``max(t_i)`` rather than ``sum(t_i)``."""
        if not tool_calls:
            return []
        return list(await asyncio.gather(*[self.execute(tc, ctx) for tc in tool_calls]))
