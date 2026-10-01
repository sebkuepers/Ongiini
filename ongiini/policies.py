"""The Ongiini policy table — classifier verdict + depth → loop shape.

Pure configuration: no adapters, no I/O. Lives apart from
``runtime.py`` (the composition root, which imports the heavy memory
and model stacks) so the table can be imported and tested on its own.
"""

from __future__ import annotations

from dataclasses import replace

from owela import (
    DEPTH_DEEP, DEPTH_SHALLOW, Policy, PolicyTable,
    VERDICT_ADMIN, VERDICT_DOCS, VERDICT_NONE, VERDICT_SEARCH, force_tool,
)

from .config import settings
from .routers.gemma_classifier import (
    VERDICT_CONTRIB_DECLINE, VERDICT_CONTRIB_DIALECT, VERDICT_CONTRIB_INVITE,
    VERDICT_CONTRIB_NEXT, VERDICT_CONTRIB_SAVE, VERDICT_CONTRIB_SKIP,
    VERDICT_CONTRIB_STATS, VERDICT_OPT_OUT_BROADCAST, VERDICT_RATE_INVITE,
)


# Prompt sections (see system_prompt.py). "first_turn" is added by the
# memory provider when the history has no assistant message yet.
SECTIONS_CHAT = ("admin",)
SECTIONS_SEARCH = ("grounding", "citations")

# Fallback replies (Policy.fallback_reply): sent when the loop runs out
# of steps or a phase raises. Short, honest, and invite a retry.
_FALLBACK_CHAT = "Sorry, I got stuck on that one. Could you ask me again, maybe in a slightly different way?"
_FALLBACK_SEARCH = (
    "Sorry, I couldn't finish looking that up. Could you ask again in a moment?"
)
_DEGRADED_SEARCH_NOTICE = (
    "Live web search is temporarily unavailable, so this comes from general "
    "knowledge — please double-check anything time-sensitive."
)

# Search tools whose failure should degrade a search turn rather than
# let the model compose over an error string.
_SEARCH_REQUIRES = ("web_search",)


def build_policy_table() -> PolicyTable:
    """The orchestration policy table — verdict + depth → Policy.

    Budgets are real, not prose: ``max_reply_tokens`` caps the visible
    reply (thinking gets its own ``max_thinking_tokens`` on top), and
    ``deadline_s`` keeps turns inside WhatsApp's 25 s typing window where
    possible. The classifier's depth splits casual chat into SHALLOW
    (short answers) and DEEP (translations, CVs, homework explanations —
    the top tasks — which need room).

    Thinking is OFF everywhere. The 2026-09-30 eval showed that on this
    stack (vLLM 0.20, gemma4 reasoning parser) a thinking compose call
    after search results writes its reasoning into ``content``; with a
    real reply budget it runs out of tokens and the leak guard has to
    drop the whole reply. Re-enable per policy only with an eval that
    shows reasoning separated cleanly AND better answers.

    Phase gating:
      - **Planner** only on SEARCH_DEEP.
      - **Critique** on DOCS + both SEARCH depths, skipped automatically
        on degraded or past-deadline turns.
      - **Interstitial** only on SEARCH_DEEP, after 18 s.

    ``force_tool`` stays only where the model must formulate arguments
    (search queries). Argument-less tools (docs, contribute, opt-out) are
    dispatched deterministically from the message — no model call, no
    dependence on the engine honouring a named tool_choice. Search keeps
    ``synth_tool`` as a recovery path when the forced call is ignored.
    """
    table = PolicyTable()

    # Casual chat. Tools stay available for misroute recovery (a search
    # or data request the classifier missed); state-changing contribute
    # tools are not exposed here.
    casual_tools = ("web_search", "fetch_url", "lookup_ongiini_docs",
                    "delete_my_data", "whats_in_my_memory", "my_token_usage")
    table.set(
        VERDICT_NONE, DEPTH_SHALLOW,
        Policy(
            name="none", max_steps=4,
            max_reply_tokens=320, deadline_s=20.0,
            expose_tools=casual_tools, prompt_sections=SECTIONS_CHAT,
            fallback_reply=_FALLBACK_CHAT,
        ),
    )
    table.set(
        VERDICT_NONE, DEPTH_DEEP,
        Policy(
            name="none_deep", max_steps=4,
            max_reply_tokens=750, deadline_s=35.0,
            expose_tools=casual_tools, prompt_sections=SECTIONS_CHAT,
            fallback_reply=_FALLBACK_CHAT,
        ),
    )

    # ADMIN — actions on the user's own data. The model picks the tool
    # from the phrasing; templated tool replies need no critique.
    table.set(
        VERDICT_ADMIN, DEPTH_SHALLOW,
        Policy(
            name="admin", max_steps=4,
            max_reply_tokens=300, deadline_s=20.0,
            expose_tools=("delete_my_data", "whats_in_my_memory", "my_token_usage",
                          "opt_out_broadcast", "contribute_stats"),
            prompt_sections=SECTIONS_CHAT, fallback_reply=_FALLBACK_CHAT,
        ),
    )

    # DOCS — questions about Ongiini itself. The docs lookup takes no
    # arguments, so it is dispatched without a model call; the one model
    # call composes from product.md. Thinking stays off (production leak
    # 2026-05-23: reasoning about a one-line capability question ran out
    # the budget and shipped raw chain-of-thought).
    table.set(
        VERDICT_DOCS, DEPTH_SHALLOW,
        Policy(
            name="docs", max_steps=3,
            enable_critique=_critique_on(),
            max_reply_tokens=420, deadline_s=22.0,
            synth_tool="lookup_ongiini_docs", synth_arg="",
            synth_first_call_from_message=True,
            expose_tools=("lookup_ongiini_docs",),
            prompt_sections=SECTIONS_CHAT, fallback_reply=_FALLBACK_CHAT,
        ),
    )

    # SEARCH_SHALLOW — one search, maybe one fetch. The model writes the
    # query (it resolves "and in Walvis Bay?" from history better than the
    # raw text would); if the engine ignores the forced call, the executor
    # synthesises it from the message.
    table.set(
        VERDICT_SEARCH, DEPTH_SHALLOW,
        Policy(
            name="search_shallow",
            first_tool=force_tool("web_search"),
            max_steps=4,
            enable_critique=_critique_on(),
            max_reply_tokens=450,
            deadline_s=24.0,
            requires_tools=_SEARCH_REQUIRES, on_unavailable="search_degraded",
            synth_tool="web_search", synth_arg="query",
            expose_tools=("web_search", "fetch_url"),
            prompt_sections=SECTIONS_SEARCH, fallback_reply=_FALLBACK_SEARCH,
            tool_result_message_caps={
                "web_search": 6000,
                "fetch_url": 12000,
            },
        ),
    )

    # SEARCH_DEEP — multi-source research, fully deterministic tool plan:
    # planner query variants → parallel web_search fan-out → auto
    # fetch_urls on the consolidated URLs. ``first_tool`` forces
    # web_search if the planner soft-fails. Thinking ON for the compose
    # call over several sources (see the thinking note above).
    table.set(
        VERDICT_SEARCH, DEPTH_DEEP,
        Policy(
            name="search_deep",
            first_tool=force_tool("web_search"),
            max_steps=6,
            enable_planner=_planner_on(),
            enable_critique=_critique_on(),
            max_reply_tokens=700,
            deadline_s=45.0,
            interstitial_after_s=18.0 if _interstitial_on() else None,
            requires_tools=_SEARCH_REQUIRES, on_unavailable="search_degraded",
            synth_tool="web_search", synth_arg="query",
            # The auto-followup fetch_urls supplies depth; raw content in
            # the search response would double-fetch the same pages.
            synth_default_args={"include_raw_content": False},
            auto_followup_after="web_search",
            auto_followup_tool="fetch_urls",
            auto_followup_attr="urls",
            auto_followup_arg="urls",
            auto_followup_max_items=5,
            auto_followup_one_per_host=True,
            expose_tools=("web_search", "fetch_urls", "fetch_url"),
            prompt_sections=SECTIONS_SEARCH, fallback_reply=_FALLBACK_SEARCH,
            tool_result_message_caps={
                "web_search": 3500,
                "fetch_urls": 12000,
                "fetch_url": 12000,
            },
        ),
    )

    # Degraded search — reached by name when web_search is down (breaker
    # open) or errors mid-turn. No tools, no critique, a short honest
    # answer plus the notice the transport appends.
    table.add(
        Policy(
            # 2 steps: a mid-turn swap has already used one on the failed search.
            name="search_degraded", max_steps=2,
            max_reply_tokens=320, deadline_s=20.0,
            expose_tools=(), prompt_sections=("grounding",),
            reply_notice=_DEGRADED_SEARCH_NOTICE, fallback_reply=_FALLBACK_SEARCH,
        ),
    )

    # ---------- Contribution loop + broadcast opt-out ----------
    #
    # Each verdict maps to one argument-less tool that reads state +
    # ctx.msg.text and writes to the contributions DB. The executor
    # dispatches it deterministically (no model call, model cannot skip
    # or fake it), then one model call composes the reply from the tool
    # result. Critique off: the effect is binary and the reply templated.
    # The contribute phrasing skill is only injected on these turns
    # (``skill:contribute`` section), not on every chat turn.
    contribute_sections = (*SECTIONS_CHAT, "skill:contribute")

    def _state_tool(name: str, tool_name: str, sections: tuple[str, ...]) -> Policy:
        return Policy(
            name=name, max_steps=2,
            max_reply_tokens=220, deadline_s=20.0,
            synth_tool=tool_name, synth_arg="",
            synth_first_call_from_message=True,
            expose_tools=(tool_name,),
            prompt_sections=sections, fallback_reply=_FALLBACK_CHAT,
        )

    for verdict, name, tool_name in (
        (VERDICT_CONTRIB_INVITE, "contribute_invite", "contribute_invite_check"),
        (VERDICT_CONTRIB_DIALECT, "contribute_dialect", "contribute_set_dialect"),
        (VERDICT_CONTRIB_NEXT, "contribute_next", "contribute_next"),
        (VERDICT_CONTRIB_SAVE, "contribute_save", "contribute_save"),
        (VERDICT_CONTRIB_SKIP, "contribute_skip", "contribute_skip"),
        (VERDICT_CONTRIB_DECLINE, "contribute_decline", "contribute_decline"),
        (VERDICT_CONTRIB_STATS, "contribute_stats", "contribute_stats"),
    ):
        table.set(verdict, DEPTH_SHALLOW, _state_tool(name, tool_name, contribute_sections))
    # Checking translations (ongiini.ai/rate) is a separate flow from
    # translating. The tool returns the finished one-line reply with the
    # link; it is sent verbatim (no model call — the link can't be
    # mangled, and the message stays small: data is expensive).
    table.set(
        VERDICT_RATE_INVITE, DEPTH_SHALLOW,
        replace(_state_tool("rate_invite", "rate_link", SECTIONS_CHAT), reply_from_tool="reply"),
    )
    table.set(
        VERDICT_OPT_OUT_BROADCAST, DEPTH_SHALLOW,
        _state_tool("opt_out_broadcast", "opt_out_broadcast", SECTIONS_CHAT),
    )

    return table


# v1 quality-phase kill switches. Each is gated by an env var in
# ``ongiini.config`` so we can disable a phase WITHOUT redeploying if
# we discover (e.g.) the REVISE rate is too high or the planner is
# misfiring on questions the classifier mis-tagged as DEEP. Default
# is ON for every flag.
def _planner_on() -> bool:
    return not settings.disable_planner


def _critique_on() -> bool:
    return not settings.disable_critique


def _interstitial_on() -> bool:
    return not settings.disable_interstitial
