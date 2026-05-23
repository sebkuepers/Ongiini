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
    ModelCallStep, PlanStep, QueryVariant, ReplyStep, RouterStep, ToolStep,
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


# =====================================================================
# v1.3 — synthesised phases (multi-query fan-out + auto-followup)
# =====================================================================


class FakePlanner:
    """Returns a fixed PlanStep on every call. Tests use this to
    drive the multi-query fan-out branch of the executor without
    spinning up a real Planner.
    """
    def __init__(
        self,
        queries: list[QueryVariant] | None = None,
        plan_text: str = "",
    ) -> None:
        self.queries = queries or []
        self.plan_text = plan_text
        self.calls = 0

    async def plan(self, msg, policy, prior_steps) -> PlanStep:
        self.calls += 1
        return PlanStep(plan_text=self.plan_text, queries=list(self.queries))


def _make_runtime_with_planner(
    *,
    model: Model,
    planner: FakePlanner,
    tools: list,
    policies: PolicyTable,
    transport: Transport | None = None,
) -> Runtime:
    return Runtime(
        model=model,
        transport=transport or FakeTransport(),
        memory=FakeMemory(),
        classifier=FakeClassifier(verdict=VERDICT_SEARCH, depth=DEPTH_DEEP),
        tools=ToolRegistry(tools),
        policies=policies,
        hooks=HookRegistry(),
        planner=planner,
    )


@pytest.mark.asyncio
async def test_executor_synthesises_multi_query_fanout_from_plan_step():
    """v1.3 contract: when policy.planner_query_tool is set AND the
    planner emits PlanStep.queries, the executor fans out N parallel
    tool calls BEFORE the model takes its first turn — no LLM call
    consumed for the fan-out itself."""
    web_search_calls: list[str] = []

    @tool(name="web_search_q_fan", params={"query": "Query."})
    async def web_search(query: str) -> tuple[str, dict]:
        """Search."""
        web_search_calls.append(query)
        return f"snippets for {query}", {"urls": [f"https://{query.replace(' ', '-')}.example/x"]}

    table = PolicyTable().set(
        VERDICT_SEARCH, DEPTH_DEEP,
        Policy(
            name="search_deep_fan",
            first_tool=force_tool("web_search_q_fan"),
            max_steps=6,
            enable_planner=True,
            planner_query_tool="web_search_q_fan",
            planner_query_arg="query",
        ),
    )
    planner = FakePlanner(queries=[
        QueryVariant(query="Bank Windhoek rate"),
        QueryVariant(query="FNB Namibia rate"),
        QueryVariant(query="Nedbank rate"),
    ])
    # Only ONE scripted response — the model just composes after seeing
    # the fan-out's results. If the executor accidentally called the
    # model an extra time, ScriptedModel would raise "exhausted".
    model = ScriptedModel([
        _ScriptedResponse(content="here's the comparison"),
    ])
    rt = _make_runtime_with_planner(
        model=model, planner=planner, tools=[web_search], policies=table,
    )
    result = await Agent(rt).handle(_msg("compare home loans"))

    assert result.sent is True
    assert "comparison" in result.reply_text
    # All three search variants ran in parallel as synthesised calls.
    assert sorted(web_search_calls) == sorted([
        "Bank Windhoek rate", "FNB Namibia rate", "Nedbank rate",
    ])
    # Three synthesised tool steps + one model call (the composer).
    tool_steps = [s for s in result.steps if isinstance(s, ToolStep)]
    assert len(tool_steps) == 3
    # Audit trail: every fan-out tool_step is marked.
    for ts in tool_steps:
        assert ts.attrs["synthesized_by_policy"] == "search_deep_fan"
        assert ts.attrs["decision_source"] == "plan.queries"
        assert "query_variant_index" in ts.attrs
    # The composer turn was NOT forced to call web_search (the fan-out
    # already covered it).
    assert model.calls[0].tool_choice == AUTO


@pytest.mark.asyncio
async def test_executor_synthesises_auto_followup_after_web_search():
    """When auto_followup_after matches a ToolStep's tool_name AND it
    carries attrs[auto_followup_attr], the executor synthesises a
    follow-up tool call WITHOUT an LLM call between."""
    followup_calls: list[list[str]] = []

    @tool(name="web_search_af", params={"query": "Q."})
    async def web_search(query: str) -> tuple[str, dict]:
        """Search."""
        return "snippets", {"urls": ["https://a.example/x", "https://b.example/y"]}

    @tool(name="fetch_urls_af", params={"urls": "URLs to fetch."})
    async def fetch_urls(urls: list[str]) -> str:
        """Batch fetch."""
        followup_calls.append(list(urls))
        return f"fetched: {urls}"

    table = PolicyTable().set(
        VERDICT_SEARCH, DEPTH_DEEP,
        Policy(
            name="search_deep_af",
            first_tool=force_tool("web_search_af"),
            max_steps=6,
            auto_followup_after="web_search_af",
            auto_followup_tool="fetch_urls_af",
        ),
    )
    # Model turn 1: emit web_search. Then the executor synthesises
    # fetch_urls (no LLM). Model turn 2: compose.
    model = ScriptedModel([
        _ScriptedResponse(
            content="",
            tool_calls=[{
                "id": "tc1", "type": "function",
                "function": {"name": "web_search_af",
                             "arguments": json.dumps({"query": "namibia rates"})},
            }],
            finish_reason="tool_calls",
        ),
        _ScriptedResponse(content="here's the answer"),
    ])
    rt = _make_runtime_with_planner(
        model=model, planner=FakePlanner(), tools=[web_search, fetch_urls],
        policies=table,
    )
    result = await Agent(rt).handle(_msg("namibia loan rates"))

    assert result.sent is True
    # fetch_urls was called with the URLs from web_search's attrs.
    assert followup_calls == [["https://a.example/x", "https://b.example/y"]]
    # Tool steps in order: model's web_search, synth fetch_urls.
    tool_steps = [s for s in result.steps if isinstance(s, ToolStep)]
    assert [ts.tool_name for ts in tool_steps] == ["web_search_af", "fetch_urls_af"]
    # Only the synth step carries the audit attr.
    assert "synthesized_by_policy" not in tool_steps[0].attrs
    assert tool_steps[1].attrs["synthesized_by_policy"] == "search_deep_af"
    assert tool_steps[1].attrs["decision_source"] == "auto_followup"


@pytest.mark.asyncio
async def test_no_auto_followup_when_url_pool_is_empty():
    """If web_search returns no URLs, the executor must NOT synthesise
    a follow-up (would call fetch_urls([])) — fall through and let the
    model handle the empty-pool case."""
    @tool(name="web_search_empty", params={"q": "Q."})
    async def web_search(q: str) -> tuple[str, dict]:
        """Search."""
        return "no results", {"urls": []}

    @tool(name="fetch_urls_empty", params={"urls": "URLs."})
    async def fetch_urls(urls: list[str]) -> str:
        """Fetch."""
        return f"fetched {urls}"

    table = PolicyTable().set(
        VERDICT_SEARCH, DEPTH_DEEP,
        Policy(
            name="search_empty",
            first_tool=force_tool("web_search_empty"),
            max_steps=6,
            auto_followup_after="web_search_empty",
            auto_followup_tool="fetch_urls_empty",
        ),
    )
    model = ScriptedModel([
        _ScriptedResponse(
            content="",
            tool_calls=[{
                "id": "tc1", "type": "function",
                "function": {"name": "web_search_empty",
                             "arguments": json.dumps({"q": "x"})},
            }],
            finish_reason="tool_calls",
        ),
        _ScriptedResponse(content="couldn't find anything"),
    ])
    rt = _make_runtime_with_planner(
        model=model, planner=FakePlanner(), tools=[web_search, fetch_urls],
        policies=table,
    )
    result = await Agent(rt).handle(_msg())
    tool_steps = [s for s in result.steps if isinstance(s, ToolStep)]
    # Only the one model-emitted call; no fetch_urls synthesis.
    assert len(tool_steps) == 1
    assert tool_steps[0].tool_name == "web_search_empty"


@pytest.mark.asyncio
async def test_url_pool_dedup_and_host_diversity():
    """When multiple parallel searches return overlapping URLs and
    multiple URLs from the same host, the executor consolidates:
    dedupe by canonical form + prefer one URL per host. The follow-up
    sees the diversified pool."""
    seen: list[list[str]] = []

    @tool(name="ws_div", params={"q": "Q."})
    async def web_search(q: str) -> tuple[str, dict]:
        """Search."""
        # Each variant returns 2 URLs; many overlap.
        # Variant 1: a.com, b.com
        # Variant 2: a.com (dup), c.com
        # Variant 3: b.com (dup), b.com/2 (same host)
        urls_map = {
            "v1": ["https://a.com/x", "https://b.com/y"],
            "v2": ["https://a.com/x", "https://c.com/z"],
            "v3": ["https://b.com/y", "https://b.com/y2"],
        }
        return "snippets", {"urls": urls_map[q]}

    @tool(name="fu_div", params={"urls": "URLs."})
    async def fetch_urls(urls: list[str]) -> str:
        """Fetch."""
        seen.append(list(urls))
        return "ok"

    table = PolicyTable().set(
        VERDICT_SEARCH, DEPTH_DEEP,
        Policy(
            name="search_div",
            first_tool=AUTO,
            max_steps=6,
            enable_planner=True,
            planner_query_tool="ws_div",
            planner_query_arg="q",
            auto_followup_after="ws_div",
            auto_followup_tool="fu_div",
        ),
    )
    planner = FakePlanner(queries=[
        QueryVariant(query="v1"),
        QueryVariant(query="v2"),
        QueryVariant(query="v3"),
    ])
    model = ScriptedModel([_ScriptedResponse(content="done")])
    rt = _make_runtime_with_planner(
        model=model, planner=planner, tools=[web_search, fetch_urls],
        policies=table,
    )
    await Agent(rt).handle(_msg())

    # One follow-up was made with the consolidated pool.
    assert len(seen) == 1
    pool = seen[0]
    # Distinct hosts only: a.com, b.com, c.com — in encounter order
    # across the three variants.
    hosts = [s.split("/")[2] for s in pool]
    assert hosts == ["a.com", "b.com", "c.com"]


@pytest.mark.asyncio
async def test_synthesised_phases_do_not_count_toward_max_steps():
    """The fan-out + auto-followup add several Steps but the model
    still gets its full ``max_steps`` worth of model-driven turns."""
    model_turns_observed: list[int] = []

    @tool(name="ws_max", params={"q": "Q."})
    async def web_search(q: str) -> tuple[str, dict]:
        """Search."""
        return "ok", {"urls": ["https://a.example/x"]}

    @tool(name="fu_max", params={"urls": "URLs."})
    async def fetch_urls(urls: list[str]) -> str:
        """Fetch."""
        return "fetched"

    table = PolicyTable().set(
        VERDICT_SEARCH, DEPTH_DEEP,
        Policy(
            name="max_steps_test",
            first_tool=AUTO,
            max_steps=2,    # only 2 model-driven turns allowed
            enable_planner=True,
            planner_query_tool="ws_max",
            planner_query_arg="q",
            auto_followup_after="ws_max",
            auto_followup_tool="fu_max",
        ),
    )
    planner = FakePlanner(queries=[QueryVariant(query="q1")])

    # Model wants TWO real turns: turn 1 emits a tool call, turn 2 composes.
    # If synth steps incorrectly counted toward max_steps, the second
    # model call would never happen.
    model = ScriptedModel([
        _ScriptedResponse(
            content="",
            tool_calls=[{
                "id": "x", "type": "function",
                "function": {"name": "ws_max",
                             "arguments": json.dumps({"q": "extra"})},
            }],
            finish_reason="tool_calls",
        ),
        _ScriptedResponse(content="composed answer"),
    ])

    def _on_call(req):
        model_turns_observed.append(len(model_turns_observed) + 1)

    # Wrap model.complete to count real calls.
    original_complete = model.complete
    async def counting_complete(req):
        _on_call(req)
        return await original_complete(req)
    model.complete = counting_complete  # type: ignore

    rt = _make_runtime_with_planner(
        model=model, planner=planner, tools=[web_search, fetch_urls],
        policies=table,
    )
    result = await Agent(rt).handle(_msg())
    # Model got both its turns — synthesised steps didn't burn the budget.
    assert len(model_turns_observed) == 2
    assert result.reply_text == "composed answer"


@pytest.mark.asyncio
async def test_no_fanout_when_planner_query_tool_is_none():
    """Safe skip: planner emits PlanStep.queries but the policy did
    NOT declare a planner_query_tool. The executor should fall back
    to normal model-driven turn 1 and HONOUR ``policy.first_tool``."""
    @tool(name="ws_safe", params={"q": "Q."})
    async def web_search(q: str) -> tuple[str, dict]:
        """Search."""
        return "snip", {"urls": []}

    table = PolicyTable().set(
        VERDICT_SEARCH, DEPTH_DEEP,
        Policy(
            name="no_fanout",
            first_tool=force_tool("ws_safe"),
            max_steps=6,
            enable_planner=True,
            # NOTE: planner_query_tool is NOT set; planner.queries are ignored.
        ),
    )
    planner = FakePlanner(queries=[QueryVariant(query="x")])
    model = ScriptedModel([
        _ScriptedResponse(
            content="",
            tool_calls=[{
                "id": "tc1", "type": "function",
                "function": {"name": "ws_safe", "arguments": json.dumps({"q": "x"})},
            }],
            finish_reason="tool_calls",
        ),
        _ScriptedResponse(content="ok"),
    ])
    rt = _make_runtime_with_planner(
        model=model, planner=planner, tools=[web_search], policies=table,
    )
    result = await Agent(rt).handle(_msg())
    # No synthesised fan-out occurred — only the model-driven web_search.
    synth_steps = [
        s for s in result.steps
        if isinstance(s, ToolStep) and s.attrs.get("synthesized_by_policy")
    ]
    assert synth_steps == []
    # Model's first turn was forced to call ws_safe.
    assert model.calls[0].tool_choice == {"type": "function",
                                          "function": {"name": "ws_safe"}}


@pytest.mark.asyncio
async def test_auto_followup_does_not_recurse_on_its_own_output():
    """Loop-prevention: if ``auto_followup_after == auto_followup_tool``
    (a misconfigured policy), the executor must NOT feed the follow-up
    its own output. The ``decision_source == "auto_followup"`` filter
    is what guards this."""
    call_count: dict[str, int] = {"n": 0}

    @tool(name="self_followup", params={"urls": "URLs."})
    async def self_followup(urls: list[str]) -> tuple[str, dict]:
        """Returns same trigger attr."""
        call_count["n"] += 1
        return "fetched", {"urls": ["https://x.example/y"]}

    table = PolicyTable().set(
        VERDICT_SEARCH, DEPTH_DEEP,
        Policy(
            name="recurse_test",
            first_tool=AUTO,
            max_steps=4,
            auto_followup_after="self_followup",
            auto_followup_tool="self_followup",   # same name → would loop
        ),
    )
    model = ScriptedModel([
        _ScriptedResponse(
            content="",
            tool_calls=[{
                "id": "t1", "type": "function",
                "function": {"name": "self_followup",
                             "arguments": json.dumps({"urls": ["https://a.example/x"]})},
            }],
            finish_reason="tool_calls",
        ),
        _ScriptedResponse(content="done"),
    ])
    rt = _make_runtime_with_planner(
        model=model, planner=FakePlanner(), tools=[self_followup], policies=table,
    )
    result = await Agent(rt).handle(_msg())
    # Model's turn 1 invocation + ONE synthesised follow-up = 2 total.
    # If the follow-up's own output re-triggered, we'd see 3+ calls.
    assert call_count["n"] == 2
    assert result.reply_text == "done"


@pytest.mark.asyncio
async def test_search_deep_end_to_end_planner_to_fanout_to_followup():
    """Integration test for the v1.3 SEARCH_DEEP pipeline shape:
    planner emits queries → executor fans out parallel web_search →
    executor synthesises fetch_urls from URL pool → model composes
    final reply. Exercises the wiring across Planner + Policy +
    executor synthesis + tool dispatch."""
    search_calls: list[str] = []
    fetch_calls: list[list[str]] = []

    @tool(name="e2e_search", params={"query": "Q."})
    async def search(query: str) -> tuple[str, dict]:
        """Search."""
        search_calls.append(query)
        # Two unique URLs per query.
        return f"snippets for {query}", {
            "urls": [
                f"https://{query.split()[0].lower()}.example/a",
                f"https://{query.split()[0].lower()}.example/b",
            ],
        }

    @tool(name="e2e_fetch", params={"urls": "URLs."})
    async def fetch(urls: list[str]) -> str:
        """Fetch."""
        fetch_calls.append(list(urls))
        return f"fetched: {urls}"

    table = PolicyTable().set(
        VERDICT_SEARCH, DEPTH_DEEP,
        Policy(
            name="e2e_search_deep",
            first_tool=force_tool("e2e_search"),
            max_steps=4,
            enable_planner=True,
            planner_query_tool="e2e_search",
            planner_query_arg="query",
            auto_followup_after="e2e_search",
            auto_followup_tool="e2e_fetch",
            auto_followup_attr="urls",
            auto_followup_arg="urls",
            auto_followup_max_items=5,
            auto_followup_one_per_host=True,
            tool_result_message_caps={"e2e_search": 4000, "e2e_fetch": 12000},
        ),
    )
    planner = FakePlanner(queries=[
        QueryVariant(query="alpha first"),
        QueryVariant(query="beta second"),
        QueryVariant(query="gamma third"),
    ])
    # Only ONE scripted response — the composer turn. If the executor
    # accidentally called the model for the fan-out, ScriptedModel
    # would raise "exhausted".
    model = ScriptedModel([
        _ScriptedResponse(content="comparison complete with all three"),
    ])
    rt = _make_runtime_with_planner(
        model=model, planner=planner, tools=[search, fetch], policies=table,
    )
    result = await Agent(rt).handle(_msg("compare three things"))

    # 1. Planner ran exactly once.
    assert planner.calls == 1
    # 2. Three parallel searches ran (one per QueryVariant).
    assert sorted(search_calls) == sorted([
        "alpha first", "beta second", "gamma third",
    ])
    # 3. ONE fetch_urls call consolidated URLs from all three searches,
    #    deduped by host, capped at 5.
    assert len(fetch_calls) == 1
    fetched_pool = fetch_calls[0]
    hosts = {u.split("/")[2] for u in fetched_pool}
    # Three distinct hosts (one per QueryVariant).
    assert hosts == {"alpha.example", "beta.example", "gamma.example"}
    # 4. Only ONE real model call was made (the composer). All other
    #    activity was synthesised.
    assert len(model.calls) == 1
    # 5. Audit trail: every synthesised tool step carries the policy attrs.
    tool_steps = [s for s in result.steps if isinstance(s, ToolStep)]
    assert len(tool_steps) == 4    # 3 searches + 1 fetch
    for ts in tool_steps:
        assert ts.attrs["synthesized_by_policy"] == "e2e_search_deep"
        assert ts.attrs["decision_source"] in ("plan.queries", "auto_followup")
    # 6. Reply went out.
    assert result.sent is True
    assert "comparison" in result.reply_text.lower()


@pytest.mark.asyncio
async def test_result_truncation_keeps_full_text_in_attrs_but_caps_messages():
    """Per-tool message caps bound model-visible context. The full
    pre-truncation text remains in ``ToolStep.attrs["result"]`` so
    reviewers + traces can still inspect it."""
    big = "x" * 5000

    @tool(name="big_tool_trunc", params={"q": "Q."})
    async def big_tool(q: str) -> str:
        """Returns a long string."""
        return big

    table = PolicyTable().set(
        VERDICT_NONE, DEPTH_SHALLOW,
        Policy(
            name="trunc_test",
            first_tool=AUTO,
            max_steps=4,
            tool_result_message_caps={"big_tool_trunc": 200},
        ),
    )

    # Capture messages on the second model.complete call (after the tool
    # ran). The model's view of the tool result should be truncated.
    seen_messages: list[list[dict]] = []

    class CapturingModel(ScriptedModel):
        async def complete(self, req):
            seen_messages.append(list(req.messages))
            return await super().complete(req)

    model = CapturingModel([
        _ScriptedResponse(
            content="",
            tool_calls=[{
                "id": "t1", "type": "function",
                "function": {"name": "big_tool_trunc",
                             "arguments": json.dumps({"q": "x"})},
            }],
            finish_reason="tool_calls",
        ),
        _ScriptedResponse(content="done"),
    ])
    rt = _make_runtime(model=model, tools=[big_tool], policies=table)
    result = await Agent(rt).handle(_msg())

    # The model's second-turn messages: the tool result block should
    # be truncated to ~200 chars + truncation marker.
    tool_message = next(
        m for m in seen_messages[1] if m.get("role") == "tool"
    )
    assert len(tool_message["content"]) < 350    # cap + marker
    assert "truncated at 200" in tool_message["content"]
    # But the ToolStep retains the FULL text in attrs.
    tool_step = next(s for s in result.steps if isinstance(s, ToolStep))
    assert len(tool_step.attrs["result"]) == 5000
    assert tool_step.attrs["result"] == big
