"""Unit tests for GemmaClassifier — uses an injected fake AsyncOpenAI."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import asyncio

import pytest

from owela import (
    ClassifierResult, DEPTH_DEEP, DEPTH_SHALLOW, InboundMessage,
    VERDICT_ADMIN, VERDICT_DOCS, VERDICT_NONE, VERDICT_SEARCH,
)
from ongiini.routers.gemma_classifier import (
    GemmaClassifier, _has_pronoun_or_reference,
)


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


def _msg(text: str, history=None, has_image: bool = False) -> InboundMessage:
    return InboundMessage(
        user_id="u", msg_id="m", text=text,
        content_parts=[{"type": "text", "text": text}],
        has_image=has_image,
        history=history or [],
    )


# ---------- Parse / verdict mapping ----------

@pytest.mark.asyncio
async def test_search_shallow_parses_correctly():
    c = GemmaClassifier(base_url="x", model_id="g", client=_client_returning("SEARCH_SHALLOW"))
    result = await c.classify(_msg("what's the BoN exchange rate?"))
    assert result.verdict == VERDICT_SEARCH
    assert result.depth == DEPTH_SHALLOW


@pytest.mark.asyncio
async def test_search_deep_parses_correctly():
    c = GemmaClassifier(base_url="x", model_id="g", client=_client_returning("SEARCH_DEEP"))
    result = await c.classify(_msg("compare home loan rates at 3 banks"))
    assert result.verdict == VERDICT_SEARCH
    assert result.depth == DEPTH_DEEP


@pytest.mark.asyncio
async def test_bare_search_degrades_to_shallow():
    """Backwards compat with the old 4-way prompt."""
    c = GemmaClassifier(base_url="x", model_id="g", client=_client_returning("SEARCH"))
    result = await c.classify(_msg("BoN exchange rate?"))
    assert result.verdict == VERDICT_SEARCH
    assert result.depth == DEPTH_SHALLOW


@pytest.mark.asyncio
async def test_docs_parses_correctly():
    c = GemmaClassifier(base_url="x", model_id="g", client=_client_returning("DOCS"))
    result = await c.classify(_msg("what's your privacy policy?"))
    assert result.verdict == VERDICT_DOCS


@pytest.mark.asyncio
async def test_admin_parses_correctly():
    c = GemmaClassifier(base_url="x", model_id="g", client=_client_returning("ADMIN"))
    result = await c.classify(_msg("delete my data"))
    assert result.verdict == VERDICT_ADMIN


@pytest.mark.asyncio
async def test_none_parses_correctly():
    c = GemmaClassifier(base_url="x", model_id="g", client=_client_returning("NONE"))
    result = await c.classify(_msg("explain photosynthesis"))
    assert result.verdict == VERDICT_NONE


@pytest.mark.asyncio
async def test_unparseable_verdict_falls_back_to_none():
    c = GemmaClassifier(base_url="x", model_id="g", client=_client_returning("uhhh I'm not sure"))
    result = await c.classify(_msg("ambiguous"))
    assert result.verdict == VERDICT_NONE
    assert result.depth == DEPTH_SHALLOW


# ---------- Label disambiguation ----------

@pytest.mark.asyncio
async def test_search_shallow_not_confused_with_search():
    """The token order in the parser must put SEARCH_SHALLOW first so
    we don't match SEARCH inside SEARCH_SHALLOW and lose the depth."""
    c = GemmaClassifier(base_url="x", model_id="g", client=_client_returning("SEARCH_SHALLOW"))
    result = await c.classify(_msg("x"))
    assert result.depth == DEPTH_SHALLOW


# ---------- Token reporting ----------

@pytest.mark.asyncio
async def test_classify_reports_cache_corrected_tokens():
    c = GemmaClassifier(
        base_url="x", model_id="g",
        client=_client_returning("NONE", prompt_tokens=500, cached=480),
    )
    result = await c.classify(_msg("hello"))
    assert result.tokens_in == 20    # 500 - 480
    assert result.cached_tokens == 480
    assert result.tokens_out == 5


# ---------- Fail-safe paths ----------

@pytest.mark.asyncio
async def test_empty_text_returns_none_without_calling_model():
    client = _client_returning("SEARCH_DEEP")
    c = GemmaClassifier(base_url="x", model_id="g", client=client)
    result = await c.classify(_msg(""))
    assert result.verdict == VERDICT_NONE
    client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_short_text_returns_none_without_calling_model():
    """Very short messages aren't worth classifying — they fall through
    to NONE / default policy."""
    client = _client_returning("SEARCH_DEEP")
    c = GemmaClassifier(base_url="x", model_id="g", client=client)
    result = await c.classify(_msg("hi"))
    assert result.verdict == VERDICT_NONE
    client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_image_message_skips_classification():
    """Image-bearing turns route to NONE so tool_choice=auto and the
    model itself decides what to do with the image."""
    client = _client_returning("SEARCH_DEEP")
    c = GemmaClassifier(base_url="x", model_id="g", client=client)
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
    c = GemmaClassifier(base_url="x", model_id="g", client=client, timeout_s=0.05)
    result = await c.classify(_msg("a typical question"))
    assert result.verdict == VERDICT_NONE


@pytest.mark.asyncio
async def test_classifier_exception_falls_back_to_none():
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("model down"))
    c = GemmaClassifier(base_url="x", model_id="g", client=client)
    result = await c.classify(_msg("a typical question"))
    assert result.verdict == VERDICT_NONE


# ---------- Pronoun + short-message context ----------

@pytest.mark.asyncio
async def test_pronoun_triggers_prev_context():
    history = [
        {"role": "user", "content": "Who is the President of Namibia?"},
        {"role": "assistant", "content": "Netumbo Nandi-Ndaitwah is the President."},
    ]
    client = _client_returning("SEARCH_DEEP")
    c = GemmaClassifier(base_url="x", model_id="g", client=client)
    await c.classify(_msg("what is her policy on healthcare?", history=history))
    sent_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Previous user message: Who is the President of Namibia?" in sent_msg


@pytest.mark.asyncio
async def test_short_message_triggers_prev_context_without_pronoun():
    """Short follow-ups almost always rely on prior context. Example:
    prev 'what's happening in Windhoek this weekend?', curr 'whats in
    the movies?' — no pronoun but clearly continuing the Windhoek topic."""
    history = [{"role": "user", "content": "what's happening in Windhoek this weekend?"}]
    client = _client_returning("SEARCH_SHALLOW")
    c = GemmaClassifier(base_url="x", model_id="g", client=client)
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
    client = _client_returning("SEARCH_SHALLOW")
    c = GemmaClassifier(base_url="x", model_id="g", client=client)
    await c.classify(_msg("hm?", history=history))
    sent_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "earlier multipart text" in sent_msg


@pytest.mark.asyncio
async def test_long_message_without_pronoun_skips_prev_context():
    """A long, self-contained message doesn't need context. Saves prompt tokens."""
    history = [{"role": "user", "content": "earlier"}]
    long_text = "Please tell me what the typical fees are for company registration in Namibia, including BIPA and stamp duty"
    client = _client_returning("SEARCH_DEEP")
    c = GemmaClassifier(base_url="x", model_id="g", client=client)
    await c.classify(_msg(long_text, history=history))
    sent_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Previous user message:" not in sent_msg


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
