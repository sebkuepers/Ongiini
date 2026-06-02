"""Unit tests for WebChatTransport — capture pattern + reply hygiene."""
from __future__ import annotations

import asyncio

import pytest

from owela import InboundMessage, Policy
from ongiini.transports.web_chat_transport import WebChatTransport


def _policy() -> Policy:
    return Policy(name="test", first_tool="AUTO")


@pytest.mark.asyncio
async def test_send_captures_body():
    t = WebChatTransport()
    ok = await t.send("session-id", "hello world", _policy())
    assert ok is True
    assert t.reply_received is True
    reply = await t.await_reply()
    assert reply == "hello world"


@pytest.mark.asyncio
async def test_send_strips_whitespace():
    t = WebChatTransport()
    await t.send("session-id", "  spaced  ", _policy())
    reply = await t.await_reply()
    assert reply == "spaced"


@pytest.mark.asyncio
async def test_send_default_when_empty():
    t = WebChatTransport()
    await t.send("session-id", "", _policy())
    reply = await t.await_reply()
    assert "Sorry" in reply


@pytest.mark.asyncio
async def test_send_caps_at_max_message_chars():
    t = WebChatTransport()
    huge = "x" * 20_000
    await t.send("session-id", huge, _policy())
    reply = await t.await_reply()
    assert len(reply) == t.max_message_chars


@pytest.mark.asyncio
async def test_double_send_ignored_with_warning(caplog):
    """Owela's contract is one send per turn. A double-call must NOT
    silently overwrite the first reply."""
    t = WebChatTransport()
    await t.send("session-id", "first", _policy())
    with caplog.at_level("WARNING"):
        ok = await t.send("session-id", "second", _policy())
    assert ok is True
    # First reply wins
    reply = await t.await_reply()
    assert reply == "first"
    assert any("called twice" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_acknowledge_is_noop():
    t = WebChatTransport()
    msg = InboundMessage(user_id="x", msg_id="m", text="hi", content_parts=[])
    # Must not raise; must return None
    result = await t.acknowledge(msg)
    assert result is None


@pytest.mark.asyncio
async def test_send_interstitial_is_noop():
    t = WebChatTransport()
    result = await t.send_interstitial("session-id", _policy())
    assert result is None


@pytest.mark.asyncio
async def test_fail_unblocks_await_reply_with_exception():
    t = WebChatTransport(reply_timeout_s=5.0)

    async def fail_after_delay():
        await asyncio.sleep(0.05)
        t.fail(RuntimeError("boom"))

    asyncio.create_task(fail_after_delay())
    with pytest.raises(RuntimeError, match="boom"):
        await t.await_reply()


@pytest.mark.asyncio
async def test_fail_ignored_after_successful_send(caplog):
    t = WebChatTransport()
    await t.send("session-id", "all good", _policy())
    with caplog.at_level("WARNING"):
        t.fail(RuntimeError("late failure"))
    # await_reply still returns the original body, not the exception
    reply = await t.await_reply()
    assert reply == "all good"
    assert any("already set" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_await_reply_times_out_when_nothing_sent():
    t = WebChatTransport(reply_timeout_s=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await t.await_reply()


@pytest.mark.asyncio
async def test_concurrent_send_and_await():
    """The HTTP handler awaits the reply while the executor (running
    concurrently inside the same event loop) fires send. Verify the
    handoff works under realistic concurrency."""
    t = WebChatTransport()

    async def producer():
        await asyncio.sleep(0.01)
        await t.send("session-id", "from executor", _policy())

    async def consumer():
        return await t.await_reply()

    prod, cons = await asyncio.gather(producer(), consumer())
    assert cons == "from executor"


@pytest.mark.asyncio
async def test_dead_url_strip_is_gated_on_used_search():
    """Dead-URL HEAD-check costs latency; only runs when used_search
    is True. With it off, every URL stays in the body verbatim
    regardless of liveness."""
    t = WebChatTransport()
    body = "Check this: https://this-domain-does-not-exist-12345.invalid/page"
    await t.send("session-id", body, _policy(), used_search=False)
    reply = await t.await_reply()
    # URL preserved because used_search=False
    assert "this-domain-does-not-exist-12345" in reply


@pytest.mark.asyncio
async def test_send_preserves_markdown_unchanged():
    """The web frontend renders markdown — unlike WhatsAppTransport,
    we don't flatten ** to *."""
    t = WebChatTransport()
    await t.send("session-id", "**bold** and [link](https://ex.com)", _policy())
    reply = await t.await_reply()
    assert "**bold**" in reply
    assert "[link](https://ex.com)" in reply


def test_transport_protocol_attrs():
    """WebChatTransport must surface the Owela transport contract:
    name, typing_window_s, max_message_chars, format."""
    t = WebChatTransport()
    assert t.name == "web_chat"
    assert isinstance(t.typing_window_s, float)
    assert isinstance(t.max_message_chars, int)
    assert t.format == "markdown"
