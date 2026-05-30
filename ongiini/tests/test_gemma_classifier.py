"""Unit tests for GemmaClassifier — uses an injected fake AsyncOpenAI.

The classifier returns a JSON object whose top-level ``verdict`` field
drives PolicyTable dispatch. The rest of the JSON lands in
``ClassifierResult.attrs``. Fixtures here wrap a verdict string into a
minimal valid JSON object; the helpers ``_json`` and ``_client_with_json``
keep that ergonomic.
"""

from __future__ import annotations

import json as _json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import asyncio

import pytest

from owela import (
    ClassifierResult, DEPTH_DEEP, DEPTH_SHALLOW, InboundMessage,
    VERDICT_ADMIN, VERDICT_DOCS, VERDICT_NONE, VERDICT_SEARCH,
)
from ongiini.routers.gemma_classifier import (
    GemmaClassifier, VERDICT_CONTRIB_SAVE, _has_pronoun_or_reference,
    _format_age,
)


def _json_for(
    verdict: str,
    *,
    confidence: str = "high",
    reasoning: str = "test reasoning",
    extracted: dict | None = None,
    state_relevance: str | None = None,
    secondary_verdict: str | None = None,
) -> str:
    """Render a minimal valid classifier JSON reply."""
    payload = {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "extracted": extracted or {
            "named_dialect": None,
            "looks_like_translation": False,
            "looks_like_button_confirmation": False,
            "looks_like_decline": False,
            "active_topic_domain": None,
        },
        "state_relevance": state_relevance,
        "secondary_verdict": secondary_verdict,
    }
    return _json.dumps(payload)


def _client_returning(content: str, prompt_tokens: int = 270, cached: int = 250) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock(message=msg)
    resp = MagicMock(choices=[choice])
    resp.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=5)
    resp.usage.prompt_tokens_details.cached_tokens = cached

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


def _client_json(verdict: str, **kwargs) -> MagicMock:
    """Shortcut: build a fake client that returns the JSON form of ``verdict``."""
    return _client_returning(_json_for(verdict, **kwargs))


def _msg(text: str, history=None, has_image: bool = False, user_id: str = "u") -> InboundMessage:
    return InboundMessage(
        user_id=user_id, msg_id="m", text=text,
        content_parts=[{"type": "text", "text": text}],
        has_image=has_image,
        history=history or [],
    )


# Default-state patch — classifier reads contributions sqlite for the
# state block; in unit tests we don't want a real DB hop. _default_state
# returns the "no active state" snapshot the real ``_read_contribute_state``
# would produce for a brand-new contributor.

def _no_state(user_id: str):  # noqa: ARG001
    return {
        "pending_save":      None,
        "awaiting_followup": False,
        "dialect":           "unknown",
        "recently_declined": False,
    }


def _make_classifier(client, **kwargs):
    """Construct a GemmaClassifier with the contribute-state reader
    patched to return the no-active-state default. Individual tests
    that care about state override this with patch.object()."""
    c = GemmaClassifier(base_url="x", model_id="g", client=client, **kwargs)
    c._read_contribute_state = _no_state    # type: ignore[assignment]
    return c


# ---------- Parse / verdict mapping ----------

@pytest.mark.asyncio
async def test_search_shallow_parses_correctly():
    c = _make_classifier(_client_json("SEARCH_SHALLOW"))
    result = await c.classify(_msg("what's the BoN exchange rate?"))
    assert result.verdict == VERDICT_SEARCH
    assert result.depth == DEPTH_SHALLOW


@pytest.mark.asyncio
async def test_search_deep_parses_correctly():
    c = _make_classifier(_client_json("SEARCH_DEEP"))
    result = await c.classify(_msg("compare home loan rates at 3 banks"))
    assert result.verdict == VERDICT_SEARCH
    assert result.depth == DEPTH_DEEP


@pytest.mark.asyncio
async def test_bare_search_degrades_to_shallow():
    """Backwards compat: a JSON reply with bare verdict 'SEARCH' still parses."""
    c = _make_classifier(_client_json("SEARCH"))
    result = await c.classify(_msg("BoN exchange rate?"))
    assert result.verdict == VERDICT_SEARCH
    assert result.depth == DEPTH_SHALLOW


@pytest.mark.asyncio
async def test_docs_parses_correctly():
    c = _make_classifier(_client_json("DOCS"))
    result = await c.classify(_msg("what's your privacy policy?"))
    assert result.verdict == VERDICT_DOCS


@pytest.mark.asyncio
async def test_admin_parses_correctly():
    c = _make_classifier(_client_json("ADMIN"))
    result = await c.classify(_msg("delete my data"))
    assert result.verdict == VERDICT_ADMIN


@pytest.mark.asyncio
async def test_none_parses_correctly():
    c = _make_classifier(_client_json("NONE"))
    result = await c.classify(_msg("explain photosynthesis"))
    assert result.verdict == VERDICT_NONE


@pytest.mark.asyncio
async def test_unrecognised_verdict_falls_back_to_none():
    c = _make_classifier(_client_json("UNKNOWN_LABEL"))
    result = await c.classify(_msg("ambiguous"))
    assert result.verdict == VERDICT_NONE
    assert result.depth == DEPTH_SHALLOW


@pytest.mark.asyncio
async def test_json_parse_failure_falls_back_to_none():
    """JSON parse failure → NONE+SHALLOW with empty attrs. Mirrors the
    old token-parser fallback exactly so the safety net stays loud and
    visible (the warning log) but invisible to the user."""
    c = _make_classifier(_client_returning("this is not json {{{"))
    result = await c.classify(_msg("ambiguous"))
    assert result.verdict == VERDICT_NONE
    assert result.depth == DEPTH_SHALLOW
    assert result.attrs == {}


@pytest.mark.asyncio
async def test_json_not_an_object_falls_back_to_none():
    """A JSON list / string / number is valid JSON but not the shape we
    asked for. Treat the same as a parse failure."""
    c = _make_classifier(_client_returning('["SEARCH_SHALLOW"]'))
    result = await c.classify(_msg("ambiguous"))
    assert result.verdict == VERDICT_NONE


@pytest.mark.asyncio
async def test_json_missing_verdict_key_falls_back_to_none():
    """Defensive: a JSON object without a `verdict` key (e.g. model
    returned only reasoning) must NOT raise — it falls back to NONE
    the same way an unrecognised verdict does. Pins behaviour so future
    parse-tightening can't silently start raising KeyError."""
    c = _make_classifier(_client_returning(
        '{"reasoning": "I considered it but did not commit to a label"}'
    ))
    result = await c.classify(_msg("ambiguous"))
    assert result.verdict == VERDICT_NONE
    assert result.depth == DEPTH_SHALLOW


@pytest.mark.asyncio
async def test_json_non_string_verdict_falls_back_to_none():
    """Defensive: Gemma is occasionally creative and may emit
    `{"verdict": 7}` or `{"verdict": null}`. The parser must not raise
    AttributeError on .strip() / .upper() — it must fall back cleanly."""
    for bogus in ('{"verdict": 7}', '{"verdict": null}', '{"verdict": true}', '{"verdict": [1,2]}'):
        c = _make_classifier(_client_returning(bogus))
        result = await c.classify(_msg("ambiguous"))
        assert result.verdict == VERDICT_NONE, f"failed for {bogus!r}"
        assert result.depth == DEPTH_SHALLOW


# ---------- Attrs population ----------

@pytest.mark.asyncio
async def test_attrs_populated_from_json():
    """JSON fields beyond verdict flow into ClassifierResult.attrs so
    hooks and future policy gates can read confidence, reasoning, the
    extracted dict, etc."""
    extracted = {
        "named_dialect": "Oshindonga",
        "looks_like_translation": True,
        "looks_like_button_confirmation": False,
        "looks_like_decline": False,
        "active_topic_domain": "translation",
    }
    body = _json_for(
        VERDICT_CONTRIB_SAVE,
        confidence="high",
        reasoning="User answered in Oshindonga phonology",
        extracted=extracted,
        state_relevance="fresh",
        secondary_verdict="NONE",
    )
    c = _make_classifier(_client_returning(body))
    result = await c.classify(_msg("Onkalo yombepo ombwaanawa nena"))
    assert result.verdict == VERDICT_CONTRIB_SAVE
    assert result.attrs["confidence"] == "high"
    assert result.attrs["reasoning"] == "User answered in Oshindonga phonology"
    assert result.attrs["extracted"] == extracted
    assert result.attrs["state_relevance"] == "fresh"
    assert result.attrs["secondary_verdict"] == "NONE"
    assert result.attrs["verdict_raw"] == VERDICT_CONTRIB_SAVE


# ---------- Token reporting ----------

@pytest.mark.asyncio
async def test_classify_reports_cache_corrected_tokens():
    c = _make_classifier(
        _client_returning(_json_for("NONE"), prompt_tokens=500, cached=480),
    )
    result = await c.classify(_msg("hello"))
    assert result.tokens_in == 20    # 500 - 480
    assert result.cached_tokens == 480
    assert result.tokens_out == 5


# ---------- Fail-safe paths ----------

@pytest.mark.asyncio
async def test_empty_text_returns_none_without_calling_model():
    client = _client_json("SEARCH_DEEP")
    c = _make_classifier(client)
    result = await c.classify(_msg(""))
    assert result.verdict == VERDICT_NONE
    client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_short_text_returns_none_without_calling_model():
    """Very short messages aren't worth classifying — they fall through
    to NONE / default policy."""
    client = _client_json("SEARCH_DEEP")
    c = _make_classifier(client)
    result = await c.classify(_msg("hi"))
    assert result.verdict == VERDICT_NONE
    client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_image_message_skips_classification():
    """Image-bearing turns route to NONE so tool_choice=auto and the
    model itself decides what to do with the image."""
    client = _client_json("SEARCH_DEEP")
    c = _make_classifier(client)
    result = await c.classify(_msg("what is this?", has_image=True))
    assert result.verdict == VERDICT_NONE
    client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_classifier_timeout_falls_back_to_none():
    async def slow(*args, **kwargs):
        await asyncio.sleep(10.0)
        return None

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = slow
    c = _make_classifier(client, timeout_s=0.05)
    result = await c.classify(_msg("a typical question"))
    assert result.verdict == VERDICT_NONE


@pytest.mark.asyncio
async def test_classifier_exception_falls_back_to_none():
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("model down"))
    c = _make_classifier(client)
    result = await c.classify(_msg("a typical question"))
    assert result.verdict == VERDICT_NONE


# ---------- Response_format wiring ----------

@pytest.mark.asyncio
async def test_classifier_requests_json_response_format():
    """Confirms we ask vLLM for ``response_format=json_object`` — the
    same pattern in_window_followup uses."""
    client = _client_json("NONE")
    c = _make_classifier(client)
    await c.classify(_msg("a typical question"))
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs.get("response_format") == {"type": "json_object"}
    assert kwargs.get("max_tokens") == 500


# ---------- Pronoun + short-message context ----------

@pytest.mark.asyncio
async def test_pronoun_triggers_prev_context():
    history = [
        {"role": "user", "content": "Who is the President of Namibia?"},
        {"role": "assistant", "content": "Netumbo Nandi-Ndaitwah is the President."},
    ]
    client = _client_json("SEARCH_DEEP")
    c = _make_classifier(client)
    await c.classify(_msg("what is her policy on healthcare?", history=history))
    sent_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Previous user message: Who is the President of Namibia?" in sent_msg
    # v1.6: previous assistant reply is now also surfaced — needed for
    # source-listing questions where the cited URLs live in the reply.
    assert "Previous assistant reply: Netumbo Nandi-Ndaitwah is the President." in sent_msg


@pytest.mark.asyncio
async def test_short_message_includes_both_prev_user_and_assistant():
    """v1.6-A: 'give me sources' is short → context is included. The
    classifier needs to see the previous ASSISTANT reply (which has the
    sources) to route this to NONE rather than SEARCH/DOCS."""
    history = [
        {"role": "user", "content": "Compare the 3 biggest Namibian banks."},
        {"role": "assistant", "content": (
            "Bank Windhoek leads on retail — source: https://bankwindhoek.com.na/about\n"
            "FNB Namibia is largest by assets — source: https://fnbnamibia.com.na/ir"
        )},
    ]
    client = _client_json("NONE")
    c = _make_classifier(client)
    await c.classify(_msg("give me your sources", history=history))
    sent_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Previous user message: Compare the 3 biggest Namibian banks." in sent_msg
    assert "Previous assistant reply:" in sent_msg
    assert "bankwindhoek.com.na" in sent_msg


@pytest.mark.asyncio
async def test_assistant_only_history_still_surfaces_context():
    """First-turn edge case: assistant said something, user replies with
    a short follow-up — there's no prior user message but the assistant
    reply should still be visible to the classifier."""
    history = [
        {"role": "assistant", "content": "Welcome! I can help with anything in Namibia."},
    ]
    client = _client_json("NONE")
    c = _make_classifier(client)
    await c.classify(_msg("ok thanks", history=history))
    sent_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Previous assistant reply: Welcome!" in sent_msg
    assert "Previous user message:" not in sent_msg


@pytest.mark.asyncio
async def test_short_message_triggers_prev_context_without_pronoun():
    """Short follow-ups almost always rely on prior context. Example:
    prev 'what's happening in Windhoek this weekend?', curr 'whats in
    the movies?' — no pronoun but clearly continuing the Windhoek topic."""
    history = [{"role": "user", "content": "what's happening in Windhoek this weekend?"}]
    client = _client_json("SEARCH_SHALLOW")
    c = _make_classifier(client)
    await c.classify(_msg("whats in the movies?", history=history))
    sent_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Previous user message: what's happening in Windhoek" in sent_msg


@pytest.mark.asyncio
async def test_prev_user_text_handles_multipart_content():
    """Image-bearing prior turns have list content; extractor must
    flatten the text parts rather than crashing or stringifying the list."""
    history = [
        {"role": "user", "content": [
            {"type": "text", "text": "earlier multipart text"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]},
    ]
    client = _client_json("SEARCH_SHALLOW")
    c = _make_classifier(client)
    await c.classify(_msg("hm?", history=history))
    sent_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "earlier multipart text" in sent_msg


@pytest.mark.asyncio
async def test_long_message_without_pronoun_skips_prev_context():
    """A long, self-contained message doesn't need context. Saves prompt tokens."""
    history = [{"role": "user", "content": "earlier"}]
    long_text = "Please tell me what the typical fees are for company registration in Namibia, including BIPA and stamp duty"
    client = _client_json("SEARCH_DEEP")
    c = _make_classifier(client)
    await c.classify(_msg(long_text, history=history))
    sent_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Previous user message:" not in sent_msg


# ---------- Contribute-state-driven context inclusion ----------

@pytest.mark.asyncio
async def test_active_state_forces_prev_assistant_even_on_long_self_contained_message():
    """For any contributor with active state (pending_save / awaiting_
    followup / known dialect), prev_assistant + prev_user MUST be
    included regardless of pronoun / length. The bot's previous question
    is exactly the context the classifier needs to decide
    fresh-vs-stale."""
    history = [
        {"role": "user", "content": "earlier message"},
        {"role": "assistant", "content": (
            "How would you say this in Oshindonga? \"The weather is "
            "beautiful today.\""
        )},
    ]
    long_text = (
        "Please consider this as my final translation attempt for the "
        "weather sentence I just got asked about thanks"
    )
    client = _client_json(VERDICT_CONTRIB_SAVE)
    c = _make_classifier(client)
    # Override the state hook to indicate pending_save + known dialect
    c._read_contribute_state = lambda uid: {  # type: ignore[assignment]
        "pending_save": {"task_id": 99, "dialect": "Oshindonga", "set_at": None},
        "awaiting_followup": False,
        "dialect": "Oshindonga",
        "recently_declined": False,
    }
    await c.classify(_msg(long_text, history=history))
    sent_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Previous assistant reply: How would you say this in Oshindonga?" in sent_msg
    assert "Previous user message: earlier message" in sent_msg


@pytest.mark.asyncio
async def test_state_block_renders_age_for_stale_pending_save():
    """When pending_save has a stored set_at >1h ago, the state block
    renders 'set Xh ago' so the model can judge staleness."""
    one_day_ago = (datetime.now(timezone.utc) - timedelta(hours=23)).isoformat(timespec="seconds")
    history = [
        {"role": "assistant", "content": "Earlier the bot answered about BIPA."},
    ]
    client = _client_json("NONE", state_relevance="stale")
    c = _make_classifier(client)
    c._read_contribute_state = lambda uid: {  # type: ignore[assignment]
        "pending_save": {"task_id": 3500, "dialect": "Oshindonga", "set_at": one_day_ago},
        "awaiting_followup": False,
        "dialect": "Oshindonga",
        "recently_declined": False,
    }
    await c.classify(_msg("Yes, let's do that", history=history))
    sent_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    # The exact rendered form is "set 23h ago" given the freeze above.
    assert "pending_save:" in sent_msg
    assert "task_id=3500" in sent_msg
    assert "set 23h ago" in sent_msg


@pytest.mark.asyncio
async def test_stale_pending_save_plus_button_click_does_not_get_save_verdict():
    """The regression test for today's production bug: a button click
    'Yes, let's do that' arrives an hour after pending_save was set, and
    Gemma (in this test, a script) reasonably picks NONE. The classifier
    must respect that — no internal rule should overrule a non-SAVE
    verdict back into CONTRIBUTE_SAVE."""
    one_day_ago = (datetime.now(timezone.utc) - timedelta(hours=23)).isoformat(timespec="seconds")
    history = [
        {"role": "assistant", "content": (
            "Want to keep contributing translations whenever you have a "
            "moment? I'll save your dialect choice."
        )},
    ]
    # Gemma returns NONE with state_relevance=stale and the button-
    # confirmation hint — exactly what we want the prompt to elicit.
    body = _json_for(
        "NONE",
        confidence="high",
        reasoning="State is stale; user just clicked a button on the latest offer.",
        extracted={
            "named_dialect": None,
            "looks_like_translation": False,
            "looks_like_button_confirmation": True,
            "looks_like_decline": False,
            "active_topic_domain": None,
        },
        state_relevance="stale",
    )
    client = _client_returning(body)
    c = _make_classifier(client)
    c._read_contribute_state = lambda uid: {  # type: ignore[assignment]
        "pending_save": {"task_id": 3500, "dialect": "Oshindonga", "set_at": one_day_ago},
        "awaiting_followup": False,
        "dialect": "Oshindonga",
        "recently_declined": False,
    }
    result = await c.classify(_msg("Yes, let's do that", history=history))
    assert result.verdict != VERDICT_CONTRIB_SAVE
    assert result.verdict == VERDICT_NONE
    assert result.attrs.get("state_relevance") == "stale"
    assert result.attrs["extracted"]["looks_like_button_confirmation"] is True


# ---------- State block always present ----------

@pytest.mark.asyncio
async def test_state_block_always_emitted_even_for_brand_new_contributor():
    """The state block ships with default values for a brand-new user
    so the prompt prefix stays consistent and cacheable."""
    client = _client_json("NONE")
    c = _make_classifier(client)
    await c.classify(_msg("a generic question about photosynthesis"))
    sent_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Contribute state" in sent_msg
    assert "pending_save:       none" in sent_msg
    assert "awaiting_followup:  false" in sent_msg
    assert "dialect:            unknown" in sent_msg
    assert "recently_declined:  false" in sent_msg


# ---------- Mem0 facts injection ----------

@pytest.mark.asyncio
async def test_mem0_facts_injected_when_state_is_active():
    """When contribute state is non-empty, the classifier pulls up to 3
    short mem0 facts and renders them into the prompt for
    disambiguation. Stateless turns skip this entirely (tested below)."""
    history = [{"role": "assistant", "content": "earlier reply"}]
    client = _client_json("NONE")
    c = _make_classifier(client)
    c._read_contribute_state = lambda uid: {  # type: ignore[assignment]
        "pending_save": None,
        "awaiting_followup": False,
        "dialect": "Oshindonga",
        "recently_declined": False,
    }
    fake_facts = [
        {"memory": "[PROFILE] Lives in Oshakati"},
        {"memory": "[PREFERENCE] Prefers English replies"},
    ]
    # Inject a fake ongiini.memory.long_term BEFORE the classifier's
    # lazy import fires. mem0 isn't installed in the local dev env, so
    # we can't import the real module just to patch its list_all.
    import sys
    import types
    fake_lt = types.ModuleType("ongiini.memory.long_term")
    fake_lt.list_all = lambda _uid: fake_facts  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"ongiini.memory.long_term": fake_lt}):
        await c.classify(_msg("hello there", history=history))
    sent_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Recent facts about this user" in sent_msg
    assert "Lives in Oshakati" in sent_msg
    assert "Prefers English replies" in sent_msg


@pytest.mark.asyncio
async def test_mem0_facts_skipped_when_no_active_state():
    """Stateless turns skip the mem0 hop — keeps the bulk of traffic
    fast and the prompt prefix small."""
    client = _client_json("NONE")
    c = _make_classifier(client)
    called = {"n": 0}

    def _spy(_uid):
        called["n"] += 1
        return []

    import sys
    import types
    fake_lt = types.ModuleType("ongiini.memory.long_term")
    fake_lt.list_all = _spy  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"ongiini.memory.long_term": fake_lt}):
        await c.classify(_msg("a generic stateless question"))
    assert called["n"] == 0


# ---------- Helper sanity ----------

def test_pronoun_regex_matches_english():
    assert _has_pronoun_or_reference("what is HER stance on AI?")
    assert _has_pronoun_or_reference("tell me about it")
    assert _has_pronoun_or_reference("this is great")


def test_pronoun_regex_matches_afrikaans():
    assert _has_pronoun_or_reference("wat is haar standpunt?")
    assert _has_pronoun_or_reference("hierdie is goed")


def test_pronoun_regex_does_not_match_neutral_text():
    assert not _has_pronoun_or_reference("what time does BIPA close?")
    assert not _has_pronoun_or_reference("explain photosynthesis briefly")


def test_format_age_minutes():
    eight_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=8)).isoformat(timespec="seconds")
    assert _format_age(eight_min_ago) == "set 8m ago"


def test_format_age_hours():
    five_hr_ago = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(timespec="seconds")
    assert _format_age(five_hr_ago) == "set 5h ago"


def test_format_age_days():
    three_d_ago = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(timespec="seconds")
    assert _format_age(three_d_ago) == "set 3d ago"


def test_format_age_empty_input():
    assert _format_age(None) == ""
    assert _format_age("") == ""
    assert _format_age("not-an-iso-timestamp") == ""
