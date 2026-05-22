"""Executor tests — drive execute_turn against mocked protocols.

Covers the v0 behaviours: router → act loop → reply, plus the gating
flags (planner/critique/interstitial) which should be no-ops in v0
because the runtime supplies None for the optional components.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from owela.agent import Agent
from owela.hooks import HookRegistry
from owela.hooks_builtin import MemoryRecordingHook
from owela.memory import MemoryProvider
from owela.model import Model, ModelRequest, ModelResponse
from owela.policy import (
    AUTO, DEPTH_DEEP, DEPTH_SHALLOW, Policy, PolicyTable,
    VERDICT_NONE, VERDICT_SEARCH, force_tool,
)
from owela.router import Classifier, ClassifierResult
from owela.runtime import Runtime
from owela.step import (
    ModelCallStep, ReplyStep, RouterStep, ToolStep,
)
from owela.tools import ToolContext, ToolRegistry, tool
from owela.transport import InboundMessage, Transport


# ---------- Fakes ----------

class FakeTransport:
    name = "fake"
    typing_window_s = 30.0
    max_message_chars = 4096
    format = "plain_text"

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.interstitials: list[str] = []
        self.acknowledged: list[str] = []

    async def acknowledge(self, msg):
        self.acknowledged.append(msg.user_id)

    async def send_interstitial(self, user_id, policy):
        self.interstitials.append(user_id)

    async def send(self, user_id, body, policy, *, used_search: bool = False) -> bool:
        self.sent.append((user_id, body))
        self.last_used_search = used_search
        return True


class FakeClassifier:
    def __init__(
        self,
        verdict: str = VERDICT_NONE,
        depth: str = DEPTH_SHALLOW,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        self.verdict = verdict
        self.depth = depth
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out

    async def classify(self, msg) -> ClassifierResult:
        return ClassifierResult(
            verdict=self.verdict,
            depth=self.depth,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
        )


class FakeMemory:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, str]] = []

    async def assemble_messages(self, msg, policy, prior_steps):
        # Minimal: system + user only. No mem0, no history.
        return [
            {"role": "system", "content": "you are a test bot"},
            {"role": "user", "content": msg.text},
        ]

    async def record_turn(self, user_id, user_text, reply):
        self.records.append((user_id, user_text, reply))

    async def delete_all(self, user_id) -> bool:
        return True

    async def list_all(self, user_id):
        return []


@dataclass
class _ScriptedResponse:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tokens_in: int = 10
    tokens_out: int = 20
    cached_tokens: int = 0
    finish_reason: str = "stop"


class ScriptedModel:
    """Returns a pre-recorded sequence of responses, one per call."""
    def __init__(self, script: list[_ScriptedResponse]) -> None:
        self.script = list(script)
        self.calls: list[ModelRequest] = []

    async def complete(self, req: ModelRequest) -> ModelResponse:
        self.calls.append(req)
        if not self.script:
            raise RuntimeError("ScriptedModel exhausted")
        s = self.script.pop(0)
        return ModelResponse(
            content=s.content,
            tool_calls=list(s.tool_calls),
            finish_reason=s.finish_reason,
            tokens_in=s.tokens_in,
            tokens_out=s.tokens_out,
            cached_tokens=s.cached_tokens,
        )


# ---------- Fixtures ----------

def _make_runtime(
    model: Model,
    transport: Transport | None = None,
    classifier: Classifier | None = None,
    memory: MemoryProvider | None = None,
    tools: list = None,
    policies: PolicyTable | None = None,
    hooks: HookRegistry | None = None,
) -> Runtime:
    if policies is None:
        policies = PolicyTable().set(
            VERDICT_NONE, DEPTH_SHALLOW, Policy(name="none", first_tool=AUTO),
        )
    return Runtime(
        model=model,
        transport=transport or FakeTransport(),
        memory=memory or FakeMemory(),
        classifier=classifier or FakeClassifier(),
        tools=ToolRegistry(tools or []),
        policies=policies,
        hooks=hooks or HookRegistry(),
    )


def _msg(text: str = "hi") -> InboundMessage:
    return InboundMessage(
        user_id="+264user",
        msg_id="wamid.1",
        text=text,
        content_parts=[{"type": "text", "text": text}],
    )


# ---------- Tests ----------

@pytest.mark.asyncio
async def test_simple_turn_no_tools():
    model = ScriptedModel([_ScriptedResponse(content="hello back")])
    transport = FakeTransport()
    rt = _make_runtime(model=model, transport=transport)
    agent = Agent(rt)

    result = await agent.handle(_msg("hi"))

    assert result.sent is True
    assert result.reply_text == "hello back"
    assert transport.acknowledged == ["+264user"]
    assert transport.sent == [("+264user", "hello back")]
    # Steps: router, model_call, reply.
    assert [s.kind for s in result.steps] == ["router", "model_call", "reply"]


@pytest.mark.asyncio
async def test_turn_with_one_tool_call():
    @tool(name="echo_test_exec", params={"x": "Echo this."})
    async def echo(x: str) -> str:
        """Echoes x."""
        return f"echoed:{x}"

    # First model call asks for the tool. Second responds with a final reply.
    model = ScriptedModel([
        _ScriptedResponse(
            content="",
            tool_calls=[{
                "id": "tc1", "type": "function",
                "function": {"name": "echo_test_exec",
                             "arguments": json.dumps({"x": "abc"})},
            }],
            finish_reason="tool_calls",
        ),
        _ScriptedResponse(content="here is your answer using echoed:abc"),
    ])
    rt = _make_runtime(model=model, tools=[echo])
    result = await Agent(rt).handle(_msg("please echo abc"))

    assert result.sent is True
    assert "echoed:abc" in result.reply_text
    # Steps: router, model_call, tool, model_call, reply.
    kinds = [s.kind for s in result.steps]
    assert kinds == ["router", "model_call", "tool", "model_call", "reply"]
    # The tool step has a result.
    tool_step = [s for s in result.steps if isinstance(s, ToolStep)][0]
    assert tool_step.attrs["result"] == "echoed:abc"


@pytest.mark.asyncio
async def test_parallel_tool_calls_in_one_turn():
    @tool(name="echo_par_test", params={"x": "Value."})
    async def echo(x: str) -> str:
        """Echoes x."""
        return f"echo:{x}"

    # Model emits TWO tool_calls in one turn. The executor should dispatch
    # them in parallel and both ToolSteps land before the next model call.
    model = ScriptedModel([
        _ScriptedResponse(
            content="",
            tool_calls=[
                {"id": "a", "type": "function",
                 "function": {"name": "echo_par_test",
                              "arguments": json.dumps({"x": "1"})}},
                {"id": "b", "type": "function",
                 "function": {"name": "echo_par_test",
                              "arguments": json.dumps({"x": "2"})}},
            ],
            finish_reason="tool_calls",
        ),
        _ScriptedResponse(content="got 1 and 2"),
    ])
    rt = _make_runtime(model=model, tools=[echo])
    result = await Agent(rt).handle(_msg("two echos please"))

    tool_steps = [s for s in result.steps if isinstance(s, ToolStep)]
    assert len(tool_steps) == 2
    assert {s.attrs["result"] for s in tool_steps} == {"echo:1", "echo:2"}


@pytest.mark.asyncio
async def test_router_forces_tool_on_first_turn():
    @tool(name="search_force_test", params={"q": "Query."})
    async def search(q: str) -> str:
        """Search."""
        return f"results for {q}"

    table = PolicyTable().set(
        VERDICT_SEARCH, DEPTH_SHALLOW,
        Policy(name="search_shallow", first_tool=force_tool("search_force_test")),
    )
    model = ScriptedModel([
        _ScriptedResponse(
            content="",
            tool_calls=[{
                "id": "c1", "type": "function",
                "function": {"name": "search_force_test",
                             "arguments": json.dumps({"q": "windhoek"})},
            }],
            finish_reason="tool_calls",
        ),
        _ScriptedResponse(content="The capital is Windhoek."),
    ])
    rt = _make_runtime(
        model=model,
        tools=[search],
        classifier=FakeClassifier(verdict=VERDICT_SEARCH, depth=DEPTH_SHALLOW),
        policies=table,
    )
    result = await Agent(rt).handle(_msg("what's the capital of namibia?"))

    # Inspect the first model call: tool_choice should be the forced dict.
    first_call_req = model.calls[0]
    assert first_call_req.tool_choice == {"type": "function", "function": {"name": "search_force_test"}}
    # Subsequent calls fall back to AUTO.
    second_call_req = model.calls[1]
    assert second_call_req.tool_choice == AUTO
    assert "Windhoek" in result.reply_text


@pytest.mark.asyncio
async def test_max_steps_truncates_with_fallback_reply():
    # Model keeps asking for tool calls forever; executor must bail at max_steps.
    @tool(name="loop_tool_test")
    async def loop_tool() -> str:
        """Loops."""
        return "more"

    forever_call = _ScriptedResponse(
        content="",
        tool_calls=[{"id": "x", "type": "function",
                     "function": {"name": "loop_tool_test", "arguments": "{}"}}],
        finish_reason="tool_calls",
    )
    model = ScriptedModel([forever_call] * 10)   # plenty
    rt = _make_runtime(
        model=model, tools=[loop_tool],
        policies=PolicyTable().set(
            VERDICT_NONE, DEPTH_SHALLOW,
            Policy(name="capped", first_tool=AUTO, max_steps=3),
        ),
    )
    result = await Agent(rt).handle(_msg())
    assert "trouble answering" in result.reply_text
    # Exactly max_steps model calls.
    assert sum(1 for s in result.steps if isinstance(s, ModelCallStep)) == 3


@pytest.mark.asyncio
async def test_reasoning_enabled_after_long_tool_result():
    # Tool returns a "long" result -> next model call should have enable_thinking=True.
    @tool(name="big_tool_test")
    async def big_tool() -> str:
        """Returns a big result."""
        return "x" * 2000

    model = ScriptedModel([
        _ScriptedResponse(
            content="",
            tool_calls=[{"id": "1", "type": "function",
                         "function": {"name": "big_tool_test", "arguments": "{}"}}],
            finish_reason="tool_calls",
        ),
        _ScriptedResponse(content="digested it"),
    ])
    rt = _make_runtime(
        model=model, tools=[big_tool],
        policies=PolicyTable().set(
            VERDICT_NONE, DEPTH_SHALLOW,
            Policy(name="reason", first_tool=AUTO,
                   long_result_threshold_chars=1000,
                   enable_thinking_after_long_results=True),
        ),
    )
    await Agent(rt).handle(_msg())
    # First call: no reasoning. Second call: reasoning ON because prior tool
    # result exceeded the threshold.
    assert model.calls[0].enable_thinking is False
    assert model.calls[1].enable_thinking is True


@pytest.mark.asyncio
async def test_executor_does_not_persist_without_hook():
    """Persistence is opt-in via MemoryRecordingHook. The executor itself
    no longer calls memory.record_turn — that responsibility moved to a
    Hook for cleaner separation."""
    model = ScriptedModel([_ScriptedResponse(content="hi user")])
    memory = FakeMemory()
    rt = _make_runtime(model=model, memory=memory, hooks=HookRegistry())
    await Agent(rt).handle(_msg("hello"))
    assert memory.records == []   # no hook → no persistence


@pytest.mark.asyncio
async def test_memory_recording_hook_persists_turn():
    model = ScriptedModel([_ScriptedResponse(content="hi user")])
    memory = FakeMemory()
    rt = _make_runtime(
        model=model,
        memory=memory,
        hooks=HookRegistry([MemoryRecordingHook()]),
    )
    await Agent(rt).handle(_msg("hello"))
    assert memory.records == [("+264user", "hello", "hi user")]


@pytest.mark.asyncio
async def test_memory_recording_hook_skips_unsent_replies():
    """If the transport returns False (delivery failed), the hook must
    NOT persist the turn — otherwise the user's history shows replies
    they never received."""
    class FailingTransport(FakeTransport):
        async def send(self, user_id, body, policy, *, used_search: bool = False) -> bool:
            return False

    model = ScriptedModel([_ScriptedResponse(content="undelivered")])
    memory = FakeMemory()
    rt = _make_runtime(
        model=model,
        memory=memory,
        transport=FailingTransport(),
        hooks=HookRegistry([MemoryRecordingHook()]),
    )
    await Agent(rt).handle(_msg("hello"))
    assert memory.records == []


@pytest.mark.asyncio
async def test_hooks_observe_steps():
    seen_steps: list[str] = []
    seen_complete: list[int] = []

    class TestHook:
        async def on_step(self, step, ctx):
            seen_steps.append(step.kind)

        async def on_turn_complete(self, steps, ctx):
            seen_complete.append(len(steps))

    model = ScriptedModel([_ScriptedResponse(content="reply")])
    rt = _make_runtime(model=model, hooks=HookRegistry([TestHook()]))
    await Agent(rt).handle(_msg())
    assert seen_steps == ["router", "model_call", "reply"]
    assert seen_complete == [3]


@pytest.mark.asyncio
async def test_hook_failure_does_not_break_turn():
    class BadHook:
        async def on_step(self, step, ctx):
            raise RuntimeError("hook broken")

    model = ScriptedModel([_ScriptedResponse(content="still works")])
    rt = _make_runtime(model=model, hooks=HookRegistry([BadHook()]))
    result = await Agent(rt).handle(_msg())
    assert result.sent is True
    assert result.reply_text == "still works"


@pytest.mark.asyncio
async def test_v1_flags_are_noop_without_components():
    # Policy has enable_planner=True but runtime.planner is None.
    # Executor should skip the phase, not crash.
    model = ScriptedModel([_ScriptedResponse(content="ok")])
    rt = _make_runtime(
        model=model,
        policies=PolicyTable().set(
            VERDICT_NONE, DEPTH_SHALLOW,
            Policy(name="planner_on_but_no_planner",
                   first_tool=AUTO, enable_planner=True),
        ),
    )
    result = await Agent(rt).handle(_msg())
    assert result.sent is True
    # No PlanStep produced.
    assert not any(s.kind == "plan" for s in result.steps)


@pytest.mark.asyncio
async def test_used_search_hint_set_when_search_tool_fires():
    """Executor passes used_search=True to transport.send when any
    search-shaped tool fired during the turn. Transports use this to
    gate dead-URL hygiene."""
    @tool(name="search_used_hint_test")
    async def fake_search() -> str:
        """fake."""
        return "result"

    @tool(name="web_search")
    async def web_search() -> str:
        """real-name search."""
        return "result"

    model = ScriptedModel([
        _ScriptedResponse(
            content="",
            tool_calls=[{"id": "1", "type": "function",
                         "function": {"name": "web_search", "arguments": "{}"}}],
            finish_reason="tool_calls",
        ),
        _ScriptedResponse(content="ok"),
    ])
    transport = FakeTransport()
    rt = _make_runtime(model=model, transport=transport, tools=[web_search])
    await Agent(rt).handle(_msg())
    assert transport.last_used_search is True


@pytest.mark.asyncio
async def test_used_search_hint_false_when_no_search_tool():
    model = ScriptedModel([_ScriptedResponse(content="just chatting")])
    transport = FakeTransport()
    rt = _make_runtime(model=model, transport=transport)
    await Agent(rt).handle(_msg())
    assert transport.last_used_search is False


@pytest.mark.asyncio
async def test_classifier_tokens_propagate_to_router_step():
    """The executor wraps ClassifierResult into a RouterStep with timing
    + token usage stamped. Verifies the wrapping contract so adapters
    only need to return the value object."""
    model = ScriptedModel([_ScriptedResponse(content="ok")])
    rt = _make_runtime(
        model=model,
        classifier=FakeClassifier(verdict=VERDICT_NONE, tokens_in=42, tokens_out=5),
    )
    result = await Agent(rt).handle(_msg())
    rstep = next(s for s in result.steps if isinstance(s, RouterStep))
    assert rstep.tokens_in == 42
    assert rstep.tokens_out == 5
    assert rstep.started_at is not None
    assert rstep.ended_at is not None
    assert rstep.latency_ms() >= 0


@pytest.mark.asyncio
async def test_classifier_attrs_propagate_to_router_step():
    """ClassifierResult.attrs is merged into the RouterStep so adapters
    can attach debug info (e.g. raw classifier response)."""
    class AttrsClassifier:
        async def classify(self, msg):
            return ClassifierResult(
                verdict=VERDICT_NONE, attrs={"raw_verdict": "NoNe"},
            )

    model = ScriptedModel([_ScriptedResponse(content="ok")])
    rt = _make_runtime(model=model, classifier=AttrsClassifier())
    result = await Agent(rt).handle(_msg())
    rstep = next(s for s in result.steps if isinstance(s, RouterStep))
    assert rstep.attrs.get("raw_verdict") == "NoNe"


@pytest.mark.asyncio
async def test_depth_routes_to_different_policy():
    """DEEP and SHALLOW for the same verdict map to different policies."""
    shallow_policy = Policy(name="shallow_only", first_tool=force_tool("nope_shallow_test"))
    deep_policy = Policy(name="deep_only", first_tool=force_tool("nope_deep_test"))
    table = (PolicyTable()
             .set(VERDICT_SEARCH, DEPTH_SHALLOW, shallow_policy)
             .set(VERDICT_SEARCH, DEPTH_DEEP, deep_policy)
             .set(VERDICT_NONE, DEPTH_SHALLOW, Policy(name="fallback")))

    # SHALLOW classification -> shallow policy
    model_s = ScriptedModel([_ScriptedResponse(content="shallow reply")])
    rt_s = _make_runtime(
        model=model_s,
        classifier=FakeClassifier(verdict=VERDICT_SEARCH, depth=DEPTH_SHALLOW),
        policies=table,
    )
    await Agent(rt_s).handle(_msg())
    assert model_s.calls[0].tool_choice == {"type": "function", "function": {"name": "nope_shallow_test"}}

    # DEEP classification -> deep policy
    model_d = ScriptedModel([_ScriptedResponse(content="deep reply")])
    rt_d = _make_runtime(
        model=model_d,
        classifier=FakeClassifier(verdict=VERDICT_SEARCH, depth=DEPTH_DEEP),
        policies=table,
    )
    await Agent(rt_d).handle(_msg())
    assert model_d.calls[0].tool_choice == {"type": "function", "function": {"name": "nope_deep_test"}}
