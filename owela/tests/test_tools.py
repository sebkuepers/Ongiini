"""@tool decorator, ToolRegistry, parallel dispatch tests."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from owela.tools import ToolContext, ToolRegistry, tool
from owela.transport import InboundMessage


# ---------- Decorator: schema generation ----------

def test_tool_basic_string_param():
    @tool(name="search_test_1", params={"query": "Search query."})
    async def fn(query: str) -> str:
        """Search the web."""
        return query

    spec = fn.__owela_tool__   # type: ignore[attr-defined]
    assert spec.name == "search_test_1"
    assert spec.description == "Search the web."
    assert spec.parameters == {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query."}},
        "required": ["query"],
    }
    assert spec.needs_context is False


def test_tool_list_param():
    @tool(name="fetch_urls_test", params={"urls": "List of URLs to fetch."})
    async def fn(urls: list[str]) -> str:
        """Fetch many URLs."""
        return ",".join(urls)

    spec = fn.__owela_tool__   # type: ignore[attr-defined]
    assert spec.parameters["properties"]["urls"] == {
        "type": "array",
        "items": {"type": "string"},
        "description": "List of URLs to fetch.",
    }


def test_tool_no_args():
    @tool(name="ping_test", description="Returns pong.")
    async def fn() -> str:
        return "pong"

    spec = fn.__owela_tool__   # type: ignore[attr-defined]
    assert spec.parameters == {"type": "object", "properties": {}}
    assert spec.needs_context is False


def test_tool_with_context():
    @tool(name="needs_ctx_test")
    async def fn(ctx: ToolContext) -> str:
        """Returns the user_id."""
        return ctx.user_id

    spec = fn.__owela_tool__   # type: ignore[attr-defined]
    assert spec.needs_context is True
    # ctx is NOT in the schema — the model doesn't see it.
    assert spec.parameters == {"type": "object", "properties": {}}


def test_tool_with_context_and_args():
    @tool(name="mixed_test", params={"q": "A query."})
    async def fn(ctx: ToolContext, q: str) -> str:
        """Does a thing."""
        return f"{ctx.user_id}:{q}"

    spec = fn.__owela_tool__   # type: ignore[attr-defined]
    assert spec.needs_context is True
    assert "ctx" not in spec.parameters["properties"]
    assert spec.parameters["properties"]["q"] == {"type": "string", "description": "A query."}


def test_tool_requires_description():
    with pytest.raises(ValueError):
        @tool(name="no_desc")
        async def fn(q: str) -> str:    # no docstring, no description
            return q


def test_tool_must_be_async():
    with pytest.raises(TypeError):
        @tool(name="sync_test", description="sync.")
        def fn(q: str) -> str:           # not async
            return q


def test_tool_rejects_unannotated_param():
    with pytest.raises(TypeError):
        @tool(name="bad_test", description="bad.")
        async def fn(q) -> str:          # type: ignore[no-untyped-def]
            return str(q)


def test_tool_rejects_unsupported_type():
    with pytest.raises(TypeError):
        @tool(name="bad_type_test", description="bad type.")
        async def fn(q: dict) -> str:    # dict not in supported types
            return str(q)


# ---------- Registry: schemas and execute ----------

@pytest.mark.asyncio
async def test_registry_schemas_and_execute():
    @tool(name="echo_reg_test", params={"x": "Value to echo."})
    async def echo(x: str) -> str:
        """Echoes x."""
        return x

    reg = ToolRegistry([echo])
    schemas = reg.schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "echo_reg_test"

    ctx = ToolContext(user_id="u", runtime=None, msg=_dummy_msg())   # type: ignore[arg-type]
    step = await reg.execute(
        {"id": "c1", "type": "function",
         "function": {"name": "echo_reg_test", "arguments": json.dumps({"x": "hi"})}},
        ctx,
    )
    assert step.tool_name == "echo_reg_test"
    assert step.tool_call_id == "c1"
    assert step.error is None
    assert step.attrs["result"] == "hi"
    assert step.result_len == 2
    assert step.ended_at is not None


@pytest.mark.asyncio
async def test_registry_unknown_tool():
    reg = ToolRegistry([])
    ctx = ToolContext(user_id="u", runtime=None, msg=_dummy_msg())   # type: ignore[arg-type]
    step = await reg.execute(
        {"id": "c2", "type": "function",
         "function": {"name": "missing", "arguments": "{}"}},
        ctx,
    )
    assert step.error == "unknown tool: missing"
    assert "not available" in step.attrs["result"]


@pytest.mark.asyncio
async def test_registry_executes_context_tool():
    @tool(name="who_am_i_test")
    async def who(ctx: ToolContext) -> str:
        """Returns the user."""
        return ctx.user_id

    reg = ToolRegistry([who])
    ctx = ToolContext(user_id="+264123", runtime=None, msg=_dummy_msg())  # type: ignore[arg-type]
    step = await reg.execute(
        {"id": "c3", "type": "function",
         "function": {"name": "who_am_i_test", "arguments": "{}"}},
        ctx,
    )
    assert step.attrs["result"] == "+264123"


@pytest.mark.asyncio
async def test_registry_unpacks_tuple_return_into_step_attrs():
    """v1.3 contract: tools may optionally return ``(text, attrs_dict)``
    to attach structured metadata (e.g. URL lists from web_search) to
    the resulting ToolStep without exposing it to the model in the
    visible result text. The registry detects the tuple shape and
    merges the dict into ``step.attrs`` while keeping ``attrs["result"]``
    as the plain text."""
    @tool(name="returns_tuple_test", params={"q": "Query."})
    async def returns_tuple(q: str) -> tuple[str, dict]:
        """Returns text plus attrs."""
        return f"results for {q}", {"urls": ["https://a", "https://b"]}

    reg = ToolRegistry([returns_tuple])
    ctx = ToolContext(user_id="u", runtime=None, msg=_dummy_msg())  # type: ignore[arg-type]
    step = await reg.execute(
        {"id": "c-tuple", "type": "function",
         "function": {"name": "returns_tuple_test",
                      "arguments": '{"q": "ham"}'}},
        ctx,
    )
    assert step.attrs["result"] == "results for ham"
    assert step.attrs["urls"] == ["https://a", "https://b"]
    assert step.result_len == len("results for ham")
    assert step.error is None


@pytest.mark.asyncio
async def test_registry_string_return_still_works():
    """Backwards compat: tools returning a plain string land in
    ``attrs["result"]`` and no extra attrs are merged."""
    @tool(name="plain_string_test", params={"q": "Query."})
    async def plain(q: str) -> str:
        """Returns text."""
        return f"plain for {q}"

    reg = ToolRegistry([plain])
    ctx = ToolContext(user_id="u", runtime=None, msg=_dummy_msg())  # type: ignore[arg-type]
    step = await reg.execute(
        {"id": "c-str", "type": "function",
         "function": {"name": "plain_string_test",
                      "arguments": '{"q": "spam"}'}},
        ctx,
    )
    assert step.attrs["result"] == "plain for spam"
    assert "urls" not in step.attrs


@pytest.mark.asyncio
async def test_registry_handles_tool_exception():
    @tool(name="boom_test")
    async def boom() -> str:
        """Boom."""
        raise RuntimeError("kaboom")

    reg = ToolRegistry([boom])
    ctx = ToolContext(user_id="u", runtime=None, msg=_dummy_msg())  # type: ignore[arg-type]
    step = await reg.execute(
        {"id": "c4", "type": "function",
         "function": {"name": "boom_test", "arguments": "{}"}},
        ctx,
    )
    # Errors are captured, NOT raised — the executor keeps going.
    assert step.error is not None
    assert "kaboom" in step.attrs["result"]


# ---------- Parallel dispatch ----------

@pytest.mark.asyncio
async def test_parallel_dispatch_is_concurrent():
    @tool(name="slow_test", params={"delay_ms": "Delay in ms."})
    async def slow(delay_ms: int) -> str:
        """Sleeps then returns."""
        await asyncio.sleep(delay_ms / 1000.0)
        return f"slept {delay_ms}ms"

    reg = ToolRegistry([slow])
    ctx = ToolContext(user_id="u", runtime=None, msg=_dummy_msg())  # type: ignore[arg-type]

    calls = [
        {"id": f"c{i}", "type": "function",
         "function": {"name": "slow_test", "arguments": json.dumps({"delay_ms": 80})}}
        for i in range(3)
    ]

    start = time.monotonic()
    steps = await reg.execute_parallel(calls, ctx)
    elapsed = time.monotonic() - start

    # Three 80ms sleeps in parallel should finish in ~80ms, definitely less than 200ms.
    # Serial would be ~240ms. We pick 0.2s as a forgiving threshold for CI noise.
    assert elapsed < 0.2, f"expected concurrent execution, took {elapsed*1000:.0f}ms"
    assert len(steps) == 3
    assert all(s.attrs["result"] == "slept 80ms" for s in steps)


@pytest.mark.asyncio
async def test_parallel_dispatch_empty():
    reg = ToolRegistry([])
    ctx = ToolContext(user_id="u", runtime=None, msg=_dummy_msg())   # type: ignore[arg-type]
    assert await reg.execute_parallel([], ctx) == []


# ---------- helpers ----------

def _dummy_msg() -> InboundMessage:
    return InboundMessage(
        user_id="u", msg_id="m", text="t", content_parts=[{"type": "text", "text": "t"}],
    )
