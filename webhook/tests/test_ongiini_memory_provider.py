"""Unit tests for OngiiniMemoryProvider — uses injected fake backends
so we don't import mem0 (which pulls torch + sentence-transformers and
isn't installed on dev boxes outside the Docker image)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from owela import InboundMessage, Policy
from webhook.app.memory_provider.ongiini_memory import OngiiniMemoryProvider


def _make_short(load_value=None, delete_value=False) -> MagicMock:
    backend = MagicMock()
    backend.load = MagicMock(return_value=load_value or [])
    backend.save = MagicMock()
    backend.delete = MagicMock(return_value=delete_value)
    return backend


def _make_long(
    search_value=None,
    list_value=None,
    delete_value=False,
    format_relevant_value="",
) -> MagicMock:
    backend = MagicMock()
    backend.search = MagicMock(return_value=search_value or [])
    backend.add_turn = MagicMock()
    backend.add_image_turn = MagicMock()
    backend.list_all = MagicMock(return_value=list_value or [])
    backend.delete_all = MagicMock(return_value=delete_value)
    backend.format_relevant = MagicMock(return_value=format_relevant_value)
    return backend


def _provider(short=None, long=None, system_prompt: str = "SYS") -> OngiiniMemoryProvider:
    return OngiiniMemoryProvider(
        system_prompt=system_prompt,
        short_term=short or _make_short(),
        long_term=long or _make_long(),
    )


# ---------- assemble_messages ----------

@pytest.mark.asyncio
async def test_assemble_messages_text_only_no_memory():
    provider = _provider()
    msg = InboundMessage(
        user_id="u", msg_id="m", text="hi",
        content_parts=[{"type": "text", "text": "hi"}],
    )
    result = await provider.assemble_messages(msg, Policy(name="p"), [])
    assert result == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
    ]


@pytest.mark.asyncio
async def test_assemble_messages_with_history_and_mem0():
    long = _make_long(
        search_value=[{"memory": "[PROFILE] Lives in Oshakati"}],
        format_relevant_value="What you know:\n- Lives in Oshakati",
    )
    provider = _provider(long=long)
    msg = InboundMessage(
        user_id="u", msg_id="m", text="weather?",
        content_parts=[{"type": "text", "text": "weather?"}],
        history=[
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "reply"},
        ],
    )
    result = await provider.assemble_messages(msg, Policy(name="p"), [])

    assert result[0] == {"role": "system", "content": "SYS"}
    assert result[1]["role"] == "system"
    assert "Lives in Oshakati" in result[1]["content"]
    assert result[2] == {"role": "user", "content": "earlier"}
    assert result[3] == {"role": "assistant", "content": "reply"}
    assert result[4] == {"role": "user", "content": "weather?"}

    long.search.assert_called_once_with("u", "weather?", 5)


@pytest.mark.asyncio
async def test_assemble_messages_image_uses_multipart_content():
    multipart = [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    ]
    msg = InboundMessage(
        user_id="u", msg_id="m", text="what is this?",
        content_parts=multipart, has_image=True,
    )
    provider = _provider()
    result = await provider.assemble_messages(msg, Policy(name="p"), [])
    user_msg = result[-1]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == multipart


@pytest.mark.asyncio
async def test_assemble_messages_skips_empty_mem0_block():
    """No mem0 hits → no extra system message (avoid wasting prompt tokens
    on an empty 'What you know about this user' header)."""
    long = _make_long(format_relevant_value="")   # empty → no block
    provider = _provider(long=long)
    msg = InboundMessage(
        user_id="u", msg_id="m", text="hi", content_parts=[],
        history=[{"role": "user", "content": "earlier"}],
    )
    result = await provider.assemble_messages(msg, Policy(name="p"), [])
    system_msgs = [m for m in result if m["role"] == "system"]
    assert len(system_msgs) == 1


# ---------- record_turn ----------

@pytest.mark.asyncio
async def test_record_turn_writes_both_tiers():
    short = _make_short(load_value=[])
    long = _make_long()
    provider = _provider(short=short, long=long)
    await provider.record_turn("+264u", "user said hi", "bot replied")

    short.load.assert_called_once_with("+264u")
    short.save.assert_called_once()
    saved_history = short.save.call_args.args[1]
    assert saved_history[-2] == {"role": "user", "content": "user said hi"}
    assert saved_history[-1] == {"role": "assistant", "content": "bot replied"}
    long.add_turn.assert_called_once_with("+264u", "user said hi", "bot replied")


@pytest.mark.asyncio
async def test_record_turn_short_term_failure_does_not_skip_long_term():
    """A broken short-term write must not prevent the long-term mem0 add."""
    short = _make_short()
    short.load.side_effect = OSError("disk full")
    long = _make_long()
    provider = _provider(short=short, long=long)
    await provider.record_turn("+264u", "u", "b")
    long.add_turn.assert_called_once()


@pytest.mark.asyncio
async def test_record_turn_long_term_failure_does_not_skip_short_term():
    short = _make_short(load_value=[])
    long = _make_long()
    long.add_turn.side_effect = RuntimeError("mem0 down")
    provider = _provider(short=short, long=long)
    await provider.record_turn("+264u", "u", "b")
    short.save.assert_called_once()


# ---------- record_image_turn ----------

@pytest.mark.asyncio
async def test_record_image_turn_uses_text_caption_short_term():
    """Image bytes are NEVER persisted; only the '[image attached] caption' + reply."""
    short = _make_short(load_value=[])
    long = _make_long()
    provider = _provider(short=short, long=long)
    await provider.record_image_turn("+264u", "look at my maize", "I see four leaves...")
    saved = short.save.call_args.args[1]
    # Format is "[image attached] <caption>" to match the original
    # llm.py behaviour — covered in detail by the other test below.
    assert saved[-2]["content"] == "[image attached] look at my maize"
    assert saved[-1]["content"] == "I see four leaves..."
    long.add_image_turn.assert_called_once_with("+264u", "look at my maize", "I see four leaves...")


@pytest.mark.asyncio
async def test_record_image_turn_with_empty_caption_uses_placeholder():
    short = _make_short(load_value=[])
    long = _make_long()
    provider = _provider(short=short, long=long)
    await provider.record_image_turn("+264u", "", "reply text")
    saved = short.save.call_args.args[1]
    assert saved[-2]["content"] == "[image attached]"


@pytest.mark.asyncio
async def test_record_image_turn_with_caption_combines_placeholder():
    """Parity with original behavior: '[image attached] <caption>' goes
    to short-term storage so the next text turn sees both the marker
    and the user's caption text."""
    short = _make_short(load_value=[])
    long = _make_long()
    provider = _provider(short=short, long=long)
    await provider.record_image_turn("+264u", "look at this maize", "I see leaves")
    saved = short.save.call_args.args[1]
    assert saved[-2]["content"] == "[image attached] look at this maize"


# ---------- delete_all ----------

@pytest.mark.asyncio
async def test_delete_all_returns_true_if_either_tier_had_data():
    short = _make_short(delete_value=False)
    long = _make_long(delete_value=True)
    provider = _provider(short=short, long=long)
    assert await provider.delete_all("+264u") is True

    short = _make_short(delete_value=False)
    long = _make_long(delete_value=False)
    provider = _provider(short=short, long=long)
    assert await provider.delete_all("+264u") is False


@pytest.mark.asyncio
async def test_delete_all_tries_both_tiers_even_if_one_raises():
    """Privacy is critical — a failure in one tier cannot leak data
    from the other."""
    short = _make_short()
    short.delete.side_effect = RuntimeError("boom")
    long = _make_long(delete_value=True)
    provider = _provider(short=short, long=long)
    result = await provider.delete_all("+264u")
    long.delete_all.assert_called_once_with("+264u")
    assert result is True


# ---------- list_all ----------

@pytest.mark.asyncio
async def test_delete_all_returns_false_when_both_tiers_raise():
    """Privacy: if BOTH tiers raise, return False (nothing deleted) but
    do not propagate — the user gets a 'nothing was stored' confirmation
    rather than a crash."""
    short = _make_short()
    short.delete.side_effect = OSError("disk gone")
    long = _make_long()
    long.delete_all.side_effect = RuntimeError("mem0 unreachable")
    provider = _provider(short=short, long=long)
    result = await provider.delete_all("+264u")
    assert result is False


def test_format_facts_uses_long_term_grouped_formatter():
    long = _make_long()
    long.format_grouped_by_tag = MagicMock(return_value="About you:\n  - Lives in X")
    provider = _provider(long=long)
    out = provider.format_facts([{"memory": "[PROFILE] Lives in X"}])
    assert "About you:" in out
    long.format_grouped_by_tag.assert_called_once()


def test_format_facts_falls_back_to_flat_list():
    long = _make_long()
    delattr(long, "format_grouped_by_tag")    # remove the override
    provider = _provider(long=long)
    out = provider.format_facts([{"memory": "fact 1"}, {"memory": "fact 2"}])
    assert "- fact 1" in out
    assert "- fact 2" in out


@pytest.mark.asyncio
async def test_list_all_returns_mem0_facts():
    facts = [{"memory": "[PROFILE] Lives in Oshakati"}]
    long = _make_long(list_value=facts)
    provider = _provider(long=long)
    result = await provider.list_all("+264u")
    assert result == facts
