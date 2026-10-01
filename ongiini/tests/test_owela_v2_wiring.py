"""Ongiini wiring for the Owela v2 primitives: policy table invariants,
per-policy context assembly, health alerts, reviewer URL check."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest


from owela import (
    DEPTH_DEEP, DEPTH_SHALLOW, DegradeStep, InboundMessage, ModelRequest,
    ModelResponse, Policy, Skill, SkillRegistry, ToolStep, VERDICT_ADMIN,
    VERDICT_DOCS, VERDICT_NONE, VERDICT_SEARCH, forced_tool_name,
)
from ongiini.hooks.health_alert_hook import HealthAlertHook
from ongiini.memory.provider import OngiiniMemoryProvider
from ongiini.reviewer import OngiiniReviewer
from ongiini.policies import build_policy_table
from ongiini.system_prompt import SYSTEM_PROMPT, build_system_prompt
from ongiini.tools import ALL_TOOLS


# ---------------------------------------------------------------------
# Policy table
# ---------------------------------------------------------------------

def _all_policies():
    table = build_policy_table()
    named = dict(table._named)            # includes name-only rows
    return table, named


def test_every_policy_has_budget_deadline_and_fallback():
    _, named = _all_policies()
    for name, p in named.items():
        assert p.max_reply_tokens, f"{name} has no reply budget"
        assert p.deadline_s, f"{name} has no deadline"
        assert p.fallback_reply, f"{name} has no fallback reply"


def test_exposed_and_synth_tools_exist():
    registered = {t.__owela_tool__.name for t in ALL_TOOLS}
    _, named = _all_policies()
    for name, p in named.items():
        for tool_name in (p.expose_tools or ()):
            assert tool_name in registered, f"{name} exposes unknown tool {tool_name}"
        if p.synth_tool:
            assert p.synth_tool in registered, f"{name} synthesises unknown tool"
        forced = forced_tool_name(p.first_tool)
        if forced:
            assert forced in registered


def test_casual_chat_splits_by_depth():
    table, _ = _all_policies()
    shallow = table.lookup(VERDICT_NONE, DEPTH_SHALLOW)
    deep = table.lookup(VERDICT_NONE, DEPTH_DEEP)
    assert shallow.name == "none" and deep.name == "none_deep"
    assert deep.max_reply_tokens > 2 * shallow.max_reply_tokens
    # State-changing contribute tools are not exposed on casual turns.
    assert not any(t.startswith("contribute_") for t in shallow.expose_tools)


def test_search_degrades_to_a_toolless_honest_policy():
    table, _ = _all_policies()
    for depth in (DEPTH_SHALLOW, DEPTH_DEEP):
        p = table.lookup(VERDICT_SEARCH, depth)
        assert p.requires_tools == ("web_search",)
        assert p.on_unavailable == "search_degraded"
        # The model writes the query; synthesis is only the recovery path.
        assert forced_tool_name(p.first_tool) == "web_search"
        assert p.synth_tool == "web_search"
    degraded = table.by_name("search_degraded")
    assert degraded.expose_tools == ()
    assert degraded.enable_critique is False
    assert degraded.reply_notice


def test_argumentless_tools_are_dispatched_without_a_model_call():
    table, _ = _all_policies()
    docs = table.lookup(VERDICT_DOCS)
    assert docs.synth_first_call_from_message and docs.synth_arg == ""
    assert forced_tool_name(docs.first_tool) is None
    save = table.lookup("CONTRIBUTE_SAVE")
    assert save.synth_tool == "contribute_save"
    assert save.synth_first_call_from_message
    assert "skill:contribute" in save.prompt_sections
    rate = table.lookup("RATE_INVITE")
    assert rate.synth_tool == "rate_link" and rate.synth_arg == ""
    assert "skill:rate" in rate.prompt_sections and "skill:contribute" not in rate.prompt_sections


def test_thinking_is_off_everywhere():
    """Thinking leaked reasoning into content on search compose calls
    (eval 2026-09-30); off until an eval shows it helps."""
    _, named = _all_policies()
    assert all(p.thinking == "off" for p in named.values())


def test_admin_policy_exposes_data_tools_only():
    table, _ = _all_policies()
    admin = table.lookup(VERDICT_ADMIN)
    assert "delete_my_data" in admin.expose_tools
    assert "web_search" not in admin.expose_tools


# ---------------------------------------------------------------------
# System prompt sections
# ---------------------------------------------------------------------

def test_full_prompt_is_every_section():
    assert build_system_prompt({"first_turn", "grounding", "citations", "admin"}) == SYSTEM_PROMPT


def test_sections_are_selected_in_canonical_order():
    core = build_system_prompt(())
    chat = build_system_prompt(("admin",))
    search = build_system_prompt(("citations", "grounding"))
    assert "FIRST-MESSAGE DISCLOSURE" not in core
    assert "GROUNDING" not in chat and "TOOL DISPATCH" in chat
    assert "GROUNDING" in search and "CITATIONS" in search
    assert search.index("GROUNDING") < search.index("CITATIONS")
    # Re-listing earlier sources works on casual turns too.
    assert "When the user asks for sources" in core
    assert len(chat) < 0.6 * len(SYSTEM_PROMPT)


# ---------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------

def _long(hits=None):
    long = MagicMock()
    long.search = MagicMock(return_value=hits or [])
    long.format_relevant = lambda facts: (
        "What you know:\n" + "\n".join(f"- {f['memory']}" for f in facts) if facts else ""
    )
    return long


def _provider(long=None, skills=None):
    return OngiiniMemoryProvider(
        prompt_builder=build_system_prompt,
        short_term=MagicMock(),
        long_term=long or _long(),
        skills=skills,
    )


def _msg(history=None, text="hello"):
    return InboundMessage(user_id="u", msg_id="m", text=text,
                          content_parts=[{"type": "text", "text": text}],
                          history=history or [])


@pytest.mark.asyncio
async def test_first_turn_section_only_without_assistant_history():
    p = _provider()
    first = await p.assemble_messages(_msg(), Policy(name="none", prompt_sections=("admin",)), [])
    later = await p.assemble_messages(
        _msg(history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]),
        Policy(name="none", prompt_sections=("admin",)), [],
    )
    assert "FIRST-MESSAGE DISCLOSURE" in first[0]["content"]
    assert "FIRST-MESSAGE DISCLOSURE" not in later[0]["content"]


@pytest.mark.asyncio
async def test_prompt_sections_follow_the_policy():
    p = _provider()
    history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    search = await p.assemble_messages(
        _msg(history), Policy(name="s", prompt_sections=("grounding", "citations")), [])
    chat = await p.assemble_messages(_msg(history), Policy(name="c", prompt_sections=("admin",)), [])
    assert "GROUNDING" in search[0]["content"]
    assert "GROUNDING" not in chat[0]["content"]


@pytest.mark.asyncio
async def test_volatile_context_sits_after_history_so_the_prefix_caches():
    p = _provider()
    history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    one = await p.assemble_messages(_msg(history), Policy(name="c"), [])
    two = await p.assemble_messages(_msg(history, text="different"), Policy(name="c"), [])
    # Everything up to and including the history is identical turn to turn.
    assert one[:3] == two[:3]
    assert "Right now in Namibia" in one[-2]["content"]
    assert one[-1] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_mem0_facts_filtered_by_score_and_capped():
    hits = [
        {"memory": "relevant A", "score": 0.8},
        {"memory": "irrelevant", "score": 0.1},
        {"memory": "relevant B", "score": 0.5},
        {"memory": "relevant C", "score": 0.4},
        {"memory": "relevant D", "score": 0.35},
    ]
    p = _provider(long=_long(hits))
    out = await p.assemble_messages(_msg(), Policy(name="c"), [])
    context = out[-2]["content"]
    assert "relevant A" in context and "relevant C" in context
    assert "irrelevant" not in context
    assert "relevant D" not in context              # capped at 3


@pytest.mark.asyncio
async def test_degrade_note_reaches_the_model():
    p = _provider()
    step = DegradeStep(from_policy="search_shallow", to_policy="search_degraded",
                       missing_tools=("web_search",), trigger="breaker_open")
    out = await p.assemble_messages(_msg(), Policy(name="search_degraded"), [step])
    assert "Live tools unavailable this turn (web_search)" in out[-2]["content"]


@pytest.mark.asyncio
async def test_contribute_skill_only_on_policies_that_ask_for_it():
    skills = SkillRegistry([
        Skill(name="oshiwambo", description="greetings", content="OSHI", load="always"),
        Skill(name="contribute", description="phrasing", content="CONTRIB", load="on_demand"),
    ])
    p = _provider(skills=skills)
    chat = await p.assemble_messages(_msg(), Policy(name="none"), [])
    contrib = await p.assemble_messages(
        _msg(), Policy(name="contribute_save", prompt_sections=("admin", "skill:contribute")), [])
    chat_text = "\n".join(m["content"] for m in chat if m["role"] == "system")
    contrib_text = "\n".join(m["content"] for m in contrib if m["role"] == "system")
    assert "OSHI" in chat_text and "CONTRIB" not in chat_text
    assert "load_skill" not in chat_text             # on-demand skills are not advertised
    assert "CONTRIB" in contrib_text


# ---------------------------------------------------------------------
# Health alerts
# ---------------------------------------------------------------------

class _Clock:
    t = 0.0

    def __call__(self):
        return self.t


@pytest.mark.asyncio
async def test_health_alert_hook_rate_limits_per_tool():
    sent: list[tuple[str, str]] = []

    async def sender(to, text):
        sent.append((to, text))

    clock = _Clock()
    hook = HealthAlertHook(operator_msisdn="264811234567", sender=sender, clock=clock)
    tripped = ToolStep(tool_name="web_search", error="402")
    tripped.attrs["breaker_tripped"] = True
    await hook.on_step(tripped, MagicMock())
    await hook.on_step(tripped, MagicMock())             # same hour → suppressed
    clock.t += 3601
    await hook.on_step(tripped, MagicMock())
    await asyncio.sleep(0)
    await asyncio.gather(*list(hook._tasks))
    assert len(sent) == 2
    assert sent[0][0] == "264811234567"
    assert "web_search" in sent[0][1]


@pytest.mark.asyncio
async def test_health_alert_hook_without_operator_only_logs(caplog):
    hook = HealthAlertHook(operator_msisdn="")
    tripped = ToolStep(tool_name="web_search", error="402")
    tripped.attrs["breaker_tripped"] = True
    with caplog.at_level("ERROR"):
        await hook.on_step(tripped, MagicMock())
    assert any("capability down" in r.message for r in caplog.records)
    assert not hook._tasks


@pytest.mark.asyncio
async def test_health_alert_delivery_failure_never_raises():
    async def broken(to, text):
        raise ConnectionError("graph api down")

    hook = HealthAlertHook(operator_msisdn="264811234567", sender=broken)
    tripped = ToolStep(tool_name="web_search", error="402")
    tripped.attrs["breaker_tripped"] = True
    await hook.on_step(tripped, MagicMock())
    await asyncio.gather(*list(hook._tasks))           # swallowed inside _deliver


# ---------------------------------------------------------------------
# Reviewer
# ---------------------------------------------------------------------

class _RecordingModel:
    def __init__(self, content="VERDICT: PASS"):
        self.content = content
        self.requests: list[ModelRequest] = []

    async def complete(self, req):
        self.requests.append(req)
        return ModelResponse(content=self.content)


def _search_step(urls):
    ts = ToolStep(tool_name="web_search")
    ts.attrs["urls"] = urls
    ts.attrs["result"] = "results"
    return ts


@pytest.mark.asyncio
async def test_invented_url_is_a_deterministic_revise_without_a_model_call():
    model = _RecordingModel()
    rev = OngiiniReviewer(model=model)
    draft = "Rates rose.\n— source: https://made-up.example/rates"
    crit = await rev.critique(_msg(text="repo rate?"), draft,
                              [_search_step(["https://bon.org.na/rates"])], Policy(name="s"))
    assert crit.verdict == "REVISE"
    assert crit.attrs["mode"] == "deterministic"
    assert "made-up.example" in crit.reasons[0]
    assert model.requests == []


@pytest.mark.asyncio
async def test_allowed_url_goes_to_the_llm_critique():
    model = _RecordingModel("1. OK\nVERDICT: PASS")
    rev = OngiiniReviewer(model=model)
    draft = "Rates rose.\n— source: https://www.bon.org.na/rates/"
    crit = await rev.critique(_msg(text="repo rate?"), draft,
                              [_search_step(["https://bon.org.na/rates"])], Policy(name="s"))
    assert crit.verdict == "PASS"
    assert crit.attrs["mode"] == "llm"
    assert len(model.requests) == 1


@pytest.mark.asyncio
async def test_critique_sees_the_composers_tool_caps():
    model = _RecordingModel("VERDICT: PASS")
    rev = OngiiniReviewer(model=model)
    step = ToolStep(tool_name="web_search", result_len=9000)
    step.attrs["result"] = "x" * 9000
    policy = Policy(name="s", tool_result_message_caps={"web_search": 3500})
    await rev.critique(_msg(text="q?"), "draft", [step], policy)
    prompt = model.requests[0].messages[0]["content"]
    assert "x" * 3500 in prompt and "x" * 3501 not in prompt


@pytest.mark.asyncio
async def test_revise_keeps_persona_and_reply_budget():
    from owela import CritiqueStep
    model = _RecordingModel("revised reply")
    rev = OngiiniReviewer(model=model, system_prompt="PERSONA")
    crit = CritiqueStep(verdict="REVISE", reasons=["claim X unsupported"])
    out = await rev.revise(_msg(text="q?"), "draft", crit, [], Policy(name="s", max_reply_tokens=450))
    req = model.requests[0]
    assert req.messages[0] == {"role": "system", "content": "PERSONA"}
    assert req.max_tokens == 450
    assert req.thinking == "off"
    assert out.attrs["revised_reply"] == "revised reply"
