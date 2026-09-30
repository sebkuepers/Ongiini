"""Executor guarantees added in Owela v2: capability degrade, forced-tool
verification + synthesis fallback, message-derived synthesis, soft
deadline, interstitial timer, guaranteed reply on errors, critique/revise.

Reuses the protocol fakes from ``test_executor``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from owela.agent import Agent
from owela.errors import ToolError
from owela.health import BreakerConfig, CircuitBreaker
from owela.hooks import HookRegistry
from owela.policy import (
    AUTO, DEPTH_DEEP, DEPTH_SHALLOW, Policy, PolicyTable, VERDICT_NONE,
    VERDICT_SEARCH, force_tool,
)
from owela.runtime import Runtime
from owela.step import (
    CritiqueStep, DegradeStep, ErrorStep, ModelCallStep, QueryVariant,
    ReplyStep, ReviseStep, ToolStep,
)
from owela.tools import ToolRegistry, tool

from .test_executor import (
    FakeClassifier, FakeMemory, FakePlanner, FakeTransport, ScriptedModel,
    _ScriptedResponse, _msg,
)


def _call(name: str, args: dict | None = None, call_id: str = "c1") -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}


def _runtime(
    *,
    model,
    policies: PolicyTable,
    tools: list | None = None,
    breaker: CircuitBreaker | None = None,
    transport: FakeTransport | None = None,
    memory=None,
    classifier=None,
    hooks: HookRegistry | None = None,
    planner=None,
    reviewer=None,
) -> Runtime:
    return Runtime(
        model=model,
        transport=transport or FakeTransport(),
        memory=memory or FakeMemory(),
        classifier=classifier or FakeClassifier(),
        tools=ToolRegistry(tools or [], breaker=breaker),
        policies=policies,
        hooks=hooks or HookRegistry(),
        planner=planner,
        reviewer=reviewer,
    )


class RecordingMemory(FakeMemory):
    def __init__(self) -> None:
        super().__init__()
        self.prior_step_kinds: list[str] = []
        self.policy_names: list[str] = []

    async def assemble_messages(self, msg, policy, prior_steps):
        self.prior_step_kinds = [s.kind for s in prior_steps]
        self.policy_names.append(policy.name)
        return await super().assemble_messages(msg, policy, prior_steps)


class FakeReviewer:
    def __init__(self, verdict: str = "PASS", revised: str = "revised text") -> None:
        self.verdict = verdict
        self.revised = revised
        self.critiques = 0
        self.revisions = 0

    async def critique(self, msg, draft, prior_steps, policy):
        self.critiques += 1
        return CritiqueStep(verdict=self.verdict, reasons=["r"] if self.verdict == "REVISE" else [])

    async def revise(self, msg, draft, critique, prior_steps, policy):
        self.revisions += 1
        step = ReviseStep()
        step.attrs["revised_reply"] = self.revised
        return step


# ---------------------------------------------------------------------
# Capability health / degrade
# ---------------------------------------------------------------------


def _search_table(*, on_unavailable: str | None = "search_degraded", critique: bool = False) -> PolicyTable:
    table = PolicyTable().set(
        VERDICT_SEARCH, DEPTH_SHALLOW,
        Policy(
            name="search",
            first_tool=force_tool("ws_health"),
            requires_tools=("ws_health",),
            on_unavailable=on_unavailable,
            enable_critique=critique,
        ),
    )
    table.set(VERDICT_NONE, DEPTH_SHALLOW, Policy(name="none"))
    table.add(Policy(name="search_degraded", expose_tools=(), reply_notice="search is down"))
    return table


@pytest.mark.asyncio
async def test_open_breaker_degrades_policy_before_assemble_messages():
    @tool(name="ws_health")
    async def ws(query: str = "") -> str:
        """search."""
        return "unused"

    breaker = CircuitBreaker(BreakerConfig(failure_threshold=1))
    breaker.record("ws_health", ok=False)                     # already open
    memory = RecordingMemory()
    transport = FakeTransport()
    model = ScriptedModel([_ScriptedResponse(content="general-knowledge answer")])
    rt = _runtime(
        model=model, policies=_search_table(), tools=[ws], breaker=breaker,
        memory=memory, transport=transport, classifier=FakeClassifier(verdict=VERDICT_SEARCH),
    )
    result = await Agent(rt).handle(_msg("what's the repo rate"))

    degrade = next(s for s in result.steps if isinstance(s, DegradeStep))
    assert (degrade.from_policy, degrade.to_policy) == ("search", "search_degraded")
    assert degrade.missing_tools == ("ws_health",)
    assert degrade.trigger == "breaker_open"
    # The provider saw the DegradeStep and was handed the degraded policy.
    assert "degrade" in memory.prior_step_kinds
    assert memory.policy_names == ["search_degraded"]
    # The degraded policy exposes no tools and forces nothing.
    assert model.calls[0].tools == []
    assert model.calls[0].tool_choice == AUTO
    assert transport.last_ctx.degraded is True
    assert result.steps[-1].degraded is True


@pytest.mark.asyncio
async def test_required_tool_error_degrades_mid_turn_and_skips_critique():
    @tool(name="ws_health")
    async def ws(query: str = "") -> str:
        """search."""
        raise ToolError("provider returned 402")

    reviewer = FakeReviewer(verdict="REVISE")
    model = ScriptedModel([
        _ScriptedResponse(content="", tool_calls=[_call("ws_health", {"query": "x"})],
                          finish_reason="tool_calls"),
        _ScriptedResponse(content="honest answer without search"),
    ])
    rt = _runtime(
        model=model, policies=_search_table(critique=True), tools=[ws],
        classifier=FakeClassifier(verdict=VERDICT_SEARCH), reviewer=reviewer,
    )
    result = await Agent(rt).handle(_msg())

    degrade = next(s for s in result.steps if isinstance(s, DegradeStep))
    assert degrade.trigger == "tool_error"
    assert degrade.to_policy == "search_degraded"
    # After the swap the model composes with the degraded policy's tools.
    assert model.calls[1].tools == []
    assert reviewer.critiques == 0
    assert result.reply_text == "honest answer without search"


@pytest.mark.asyncio
async def test_degrade_without_fallback_policy_still_emits_step():
    @tool(name="ws_health")
    async def ws(query: str = "") -> str:
        """search."""
        raise ToolError("down")

    reviewer = FakeReviewer(verdict="REVISE")
    model = ScriptedModel([
        _ScriptedResponse(content="", tool_calls=[_call("ws_health")], finish_reason="tool_calls"),
        _ScriptedResponse(content="answer"),
    ])
    rt = _runtime(
        model=model, policies=_search_table(on_unavailable=None, critique=True), tools=[ws],
        classifier=FakeClassifier(verdict=VERDICT_SEARCH), reviewer=reviewer,
    )
    result = await Agent(rt).handle(_msg())
    degrade = next(s for s in result.steps if isinstance(s, DegradeStep))
    assert degrade.from_policy == degrade.to_policy == "search"
    assert reviewer.critiques == 0                              # degraded turns aren't critiqued
    assert result.steps[-1].degraded is True


@pytest.mark.asyncio
async def test_only_one_degrade_per_turn():
    @tool(name="ws_health")
    async def ws(query: str = "") -> str:
        """search."""
        raise ToolError("down")

    model = ScriptedModel([
        _ScriptedResponse(content="", tool_calls=[_call("ws_health")], finish_reason="tool_calls"),
        _ScriptedResponse(content="", tool_calls=[_call("ws_health", call_id="c2")], finish_reason="tool_calls"),
        _ScriptedResponse(content="done"),
    ])
    rt = _runtime(
        model=model, policies=_search_table(on_unavailable=None), tools=[ws],
        classifier=FakeClassifier(verdict=VERDICT_SEARCH),
    )
    result = await Agent(rt).handle(_msg())
    assert sum(isinstance(s, DegradeStep) for s in result.steps) == 1


# ---------------------------------------------------------------------
# Synthesis from the message + forced-tool verification
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synth_first_call_from_message_dispatches_without_model_call():
    seen: list[str] = []

    @tool(name="docs_synth_test")
    async def docs(query: str = "<none>") -> str:
        """docs."""
        seen.append(query)
        return "the docs"

    model = ScriptedModel([_ScriptedResponse(content="answer from docs")])
    table = PolicyTable().set(
        VERDICT_NONE, DEPTH_SHALLOW,
        Policy(name="docs", synth_tool="docs_synth_test", synth_arg="",
               synth_first_call_from_message=True),
    )
    rt = _runtime(model=model, policies=table, tools=[docs])
    result = await Agent(rt).handle(_msg("who runs this?"))

    assert seen == ["<none>"]                                    # synth_arg="" → no text arg
    assert len(model.calls) == 1                                 # only the compose call
    assert model.calls[0].tool_choice == AUTO
    synth = next(s for s in result.steps if isinstance(s, ModelCallStep) and s.synthesized)
    assert synth.attrs["decision_source"] == "message"
    tool_step = next(s for s in result.steps if isinstance(s, ToolStep))
    assert tool_step.attrs["decision_source"] == "message"
    assert result.reply_text == "answer from docs"


@pytest.mark.asyncio
async def test_forced_tool_not_honoured_is_traced_and_recovered_by_synthesis():
    seen: list[str] = []

    @tool(name="ws_force_test", params={"query": "q"})
    async def ws(query: str) -> str:
        """search."""
        seen.append(query)
        return "fresh results"

    model = ScriptedModel([
        _ScriptedResponse(content="I think the answer is ... (from memory)"),   # ignored the force
        _ScriptedResponse(content="grounded answer"),
    ])
    table = PolicyTable().set(
        VERDICT_NONE, DEPTH_SHALLOW,
        Policy(name="search", first_tool=force_tool("ws_force_test"), synth_tool="ws_force_test"),
    )
    rt = _runtime(model=model, policies=table, tools=[ws])
    result = await Agent(rt).handle(_msg("latest fuel price windhoek"))

    real_calls = [s for s in result.steps if isinstance(s, ModelCallStep) and not s.synthesized]
    assert real_calls[0].forced_tool == "ws_force_test"
    assert real_calls[0].forced_tool_honoured is False
    synth = next(s for s in result.steps if isinstance(s, ModelCallStep) and s.synthesized)
    assert synth.attrs["decision_source"] == "forced_tool_fallback"
    assert seen == ["latest fuel price windhoek"]
    # The discarded prose never reaches the user.
    assert result.reply_text == "grounded answer"
    assert model.calls[1].tool_choice == AUTO


@pytest.mark.asyncio
async def test_forced_tool_not_honoured_without_synth_accepts_prose():
    @tool(name="ws_force_test2")
    async def ws(query: str = "") -> str:
        """search."""
        return "x"

    model = ScriptedModel([_ScriptedResponse(content="prose answer")])
    table = PolicyTable().set(
        VERDICT_NONE, DEPTH_SHALLOW,
        Policy(name="forced", first_tool=force_tool("ws_force_test2")),
    )
    rt = _runtime(model=model, policies=table, tools=[ws])
    result = await Agent(rt).handle(_msg())
    call = next(s for s in result.steps if isinstance(s, ModelCallStep))
    assert call.forced_tool_honoured is False
    assert result.reply_text == "prose answer"


@pytest.mark.asyncio
async def test_forced_tool_honoured_is_recorded():
    @tool(name="ws_force_test3")
    async def ws(query: str = "") -> str:
        """search."""
        return "x"

    model = ScriptedModel([
        _ScriptedResponse(content="", tool_calls=[_call("ws_force_test3", {"query": "q"})],
                          finish_reason="tool_calls"),
        _ScriptedResponse(content="done"),
    ])
    table = PolicyTable().set(
        VERDICT_NONE, DEPTH_SHALLOW,
        Policy(name="forced", first_tool=force_tool("ws_force_test3"), synth_tool="ws_force_test3"),
    )
    rt = _runtime(model=model, policies=table, tools=[ws])
    result = await Agent(rt).handle(_msg())
    calls = [s for s in result.steps if isinstance(s, ModelCallStep)]
    assert calls[0].forced_tool_honoured is True
    assert calls[1].forced_tool is None                          # AUTO on later turns
    assert not any(s.synthesized for s in calls)


@pytest.mark.asyncio
async def test_plan_queries_take_precedence_over_message_synthesis():
    seen: list[str] = []

    @tool(name="ws_plan_prec", params={"query": "q"})
    async def ws(query: str) -> str:
        """search."""
        seen.append(query)
        return "r"

    model = ScriptedModel([_ScriptedResponse(content="done")])
    table = PolicyTable().set(
        VERDICT_SEARCH, DEPTH_DEEP,
        Policy(name="deep", enable_planner=True, synth_tool="ws_plan_prec",
               synth_first_call_from_message=True),
    )
    planner = FakePlanner(queries=[QueryVariant(query="a"), QueryVariant(query="b")])
    rt = _runtime(
        model=model, policies=table, tools=[ws], planner=planner,
        classifier=FakeClassifier(verdict=VERDICT_SEARCH, depth=DEPTH_DEEP),
    )
    result = await Agent(rt).handle(_msg("raw user text"))
    assert sorted(seen) == ["a", "b"]
    sources = {s.attrs.get("decision_source") for s in result.steps if isinstance(s, ToolStep)}
    assert sources == {"plan.queries"}


# ---------------------------------------------------------------------
# Deadline + interstitial
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deadline_forces_compose_without_tools():
    @tool(name="ws_deadline")
    async def ws(query: str = "") -> str:
        """search."""
        return "x"

    model = ScriptedModel([_ScriptedResponse(content="best effort")])
    table = PolicyTable().set(
        VERDICT_NONE, DEPTH_SHALLOW,
        Policy(name="tight", first_tool=force_tool("ws_deadline"), deadline_s=0.0),
    )
    transport = FakeTransport()
    rt = _runtime(model=model, policies=table, tools=[ws], transport=transport)
    result = await Agent(rt).handle(_msg())
    assert model.calls[0].tools == []
    assert model.calls[0].tool_choice == AUTO
    reply = result.steps[-1]
    assert reply.reason == "deadline"
    assert reply.deadline_exceeded is True
    assert transport.last_ctx.deadline_exceeded is True


@pytest.mark.asyncio
async def test_deadline_skips_critique_and_marks_reply():
    @tool(name="slow_deadline")
    async def slow() -> str:
        """slow."""
        await asyncio.sleep(0.05)
        return "late result"

    reviewer = FakeReviewer(verdict="REVISE")
    model = ScriptedModel([
        _ScriptedResponse(content="", tool_calls=[_call("slow_deadline")], finish_reason="tool_calls"),
        _ScriptedResponse(content="composed after deadline"),
    ])
    table = PolicyTable().set(
        VERDICT_NONE, DEPTH_SHALLOW,
        Policy(name="tight", deadline_s=0.02, enable_critique=True),
    )
    rt = _runtime(model=model, policies=table, tools=[slow], reviewer=reviewer)
    result = await Agent(rt).handle(_msg())
    assert model.calls[1].tools == []                            # second call composes
    assert reviewer.critiques == 0
    assert result.reply_text == "composed after deadline"
    assert result.steps[-1].deadline_exceeded is True


@pytest.mark.asyncio
async def test_interstitial_fires_after_delay():
    @tool(name="slow_inter")
    async def slow() -> str:
        """slow."""
        await asyncio.sleep(0.08)
        return "r"

    transport = FakeTransport()
    model = ScriptedModel([
        _ScriptedResponse(content="", tool_calls=[_call("slow_inter")], finish_reason="tool_calls"),
        _ScriptedResponse(content="done"),
    ])
    table = PolicyTable().set(
        VERDICT_NONE, DEPTH_SHALLOW, Policy(name="p", interstitial_after_s=0.01),
    )
    rt = _runtime(model=model, policies=table, tools=[slow], transport=transport)
    await Agent(rt).handle(_msg())
    assert transport.interstitials == ["+264user"]


@pytest.mark.asyncio
async def test_interstitial_cancelled_on_fast_reply():
    transport = FakeTransport()
    model = ScriptedModel([_ScriptedResponse(content="quick")])
    table = PolicyTable().set(
        VERDICT_NONE, DEPTH_SHALLOW, Policy(name="p", interstitial_after_s=0.05),
    )
    rt = _runtime(model=model, policies=table, transport=transport)
    await Agent(rt).handle(_msg())
    await asyncio.sleep(0.1)
    assert transport.interstitials == []


@pytest.mark.asyncio
async def test_interstitial_capped_at_typing_window():
    @tool(name="slow_cap")
    async def slow() -> str:
        """slow."""
        await asyncio.sleep(0.08)
        return "r"

    transport = FakeTransport()
    transport.typing_window_s = 0.01
    model = ScriptedModel([
        _ScriptedResponse(content="", tool_calls=[_call("slow_cap")], finish_reason="tool_calls"),
        _ScriptedResponse(content="done"),
    ])
    table = PolicyTable().set(
        VERDICT_NONE, DEPTH_SHALLOW, Policy(name="p", interstitial_after_s=100.0),
    )
    rt = _runtime(model=model, policies=table, tools=[slow], transport=transport)
    await Agent(rt).handle(_msg())
    assert transport.interstitials == ["+264user"]


@pytest.mark.asyncio
async def test_no_interstitial_without_policy_field():
    transport = FakeTransport()
    model = ScriptedModel([_ScriptedResponse(content="x")])
    rt = _runtime(model=model, policies=PolicyTable().set(VERDICT_NONE, DEPTH_SHALLOW, Policy(name="p")),
                  transport=transport)
    await Agent(rt).handle(_msg())
    assert transport.interstitials == []


# ---------------------------------------------------------------------
# Guaranteed reply
# ---------------------------------------------------------------------


class _RaisingModel:
    async def complete(self, req):
        raise RuntimeError("engine unreachable")


@pytest.mark.asyncio
async def test_model_exception_yields_error_step_reply_step_and_turn_complete():
    completed: list[list[str]] = []

    class Spy:
        async def on_turn_complete(self, steps, ctx):
            completed.append([s.kind for s in steps])

    transport = FakeTransport()
    table = PolicyTable().set(
        VERDICT_NONE, DEPTH_SHALLOW, Policy(name="p", fallback_reply="please try again"),
    )
    rt = _runtime(model=_RaisingModel(), policies=table, transport=transport,
                  hooks=HookRegistry([Spy()]))
    result = await Agent(rt).handle(_msg())

    assert completed == [["router", "error", "reply"]]
    err = next(s for s in result.steps if isinstance(s, ErrorStep))
    assert (err.phase, err.exc_type) == ("turn", "RuntimeError")
    reply = result.steps[-1]
    assert isinstance(reply, ReplyStep) and reply.reason == "error"
    assert transport.sent == [("+264user", "please try again")]
    assert result.error == "turn:RuntimeError"


@pytest.mark.asyncio
async def test_classifier_exception_yields_router_fallback():
    class BrokenClassifier:
        async def classify(self, msg):
            raise TimeoutError("classifier hung")

    model = ScriptedModel([_ScriptedResponse(content="still answered")])
    rt = _runtime(
        model=model, classifier=BrokenClassifier(),
        policies=PolicyTable().set(VERDICT_NONE, DEPTH_SHALLOW, Policy(name="none")),
    )
    result = await Agent(rt).handle(_msg())
    router = result.steps[0]
    assert router.kind == "router"
    assert router.verdict == VERDICT_NONE
    assert router.fallback_reason == "exception:TimeoutError"
    assert any(isinstance(s, ErrorStep) and s.phase == "router" for s in result.steps)
    assert result.reply_text == "still answered"


@pytest.mark.asyncio
async def test_classifier_fallback_reason_propagates():
    from owela.router import ClassifierResult

    class TimingOutClassifier:
        async def classify(self, msg):
            return ClassifierResult(fallback_reason="timeout")

    model = ScriptedModel([_ScriptedResponse(content="ok")])
    rt = _runtime(
        model=model, classifier=TimingOutClassifier(),
        policies=PolicyTable().set(VERDICT_NONE, DEPTH_SHALLOW, Policy(name="none")),
    )
    result = await Agent(rt).handle(_msg())
    assert result.steps[0].fallback_reason == "timeout"
    assert not any(isinstance(s, ErrorStep) for s in result.steps)


@pytest.mark.asyncio
async def test_transport_exception_still_completes_turn():
    class BrokenTransport(FakeTransport):
        async def send(self, user_id, body, policy, ctx):
            raise ConnectionError("graph api down")

    completed: list[bool] = []

    class Spy:
        async def on_turn_complete(self, steps, ctx):
            completed.append(True)

    model = ScriptedModel([_ScriptedResponse(content="hi")])
    rt = _runtime(model=model, transport=BrokenTransport(),
                  policies=PolicyTable().set(VERDICT_NONE, DEPTH_SHALLOW, Policy(name="p")),
                  hooks=HookRegistry([Spy()]))
    result = await Agent(rt).handle(_msg())
    assert result.sent is False
    assert result.steps[-1].attrs["send_error"] == "ConnectionError"
    assert completed == [True]


# ---------------------------------------------------------------------
# Critique / revise
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_critique_revise_ships_revised_draft():
    reviewer = FakeReviewer(verdict="REVISE", revised="the corrected reply")
    transport = FakeTransport()
    model = ScriptedModel([_ScriptedResponse(content="draft with a mistake")])
    rt = _runtime(
        model=model, reviewer=reviewer, transport=transport,
        policies=PolicyTable().set(VERDICT_NONE, DEPTH_SHALLOW, Policy(name="p", enable_critique=True)),
    )
    result = await Agent(rt).handle(_msg())
    assert [s.kind for s in result.steps] == ["router", "model_call", "critique", "revise", "reply"]
    assert transport.sent == [("+264user", "the corrected reply")]
    assert result.reply_text == "the corrected reply"


@pytest.mark.asyncio
async def test_critique_pass_ships_draft():
    reviewer = FakeReviewer(verdict="PASS")
    model = ScriptedModel([_ScriptedResponse(content="good draft")])
    rt = _runtime(
        model=model, reviewer=reviewer,
        policies=PolicyTable().set(VERDICT_NONE, DEPTH_SHALLOW, Policy(name="p", enable_critique=True)),
    )
    result = await Agent(rt).handle(_msg())
    assert reviewer.revisions == 0
    assert result.reply_text == "good draft"


@pytest.mark.asyncio
async def test_critique_skipped_on_max_steps_fallback():
    @tool(name="loop_crit")
    async def loop() -> str:
        """loop."""
        return "again"

    reviewer = FakeReviewer(verdict="REVISE")
    model = ScriptedModel([
        _ScriptedResponse(content="", tool_calls=[_call("loop_crit")], finish_reason="tool_calls"),
    ] * 2)
    rt = _runtime(
        model=model, reviewer=reviewer, tools=[loop],
        policies=PolicyTable().set(
            VERDICT_NONE, DEPTH_SHALLOW,
            Policy(name="p", max_steps=2, enable_critique=True, fallback_reply="fallback"),
        ),
    )
    result = await Agent(rt).handle(_msg())
    assert reviewer.critiques == 0
    assert result.reply_text == "fallback"
