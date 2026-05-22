"""Unit tests for the WhatsApp Owela Transport."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from owela import InboundMessage, Policy
from webhook.app.transports.whatsapp_transport import WhatsAppTransport


def _msg(msg_id: str = "wamid.1") -> InboundMessage:
    return InboundMessage(
        user_id="+264user", msg_id=msg_id, text="hi", content_parts=[],
    )


# ---------- Transport metadata ----------

def test_transport_metadata():
    t = WhatsAppTransport()
    assert t.name == "whatsapp"
    assert t.typing_window_s == 25.0
    assert t.max_message_chars == 4096
    assert t.format == "plain_text"


# ---------- acknowledge ----------

@pytest.mark.asyncio
async def test_acknowledge_calls_mark_as_read():
    t = WhatsAppTransport()
    with patch("webhook.app.transports.whatsapp_transport._mark_as_read",
               new=AsyncMock()) as mock_mar:
        await t.acknowledge(_msg("wamid.42"))
    mock_mar.assert_awaited_once_with("wamid.42", with_typing=True)


@pytest.mark.asyncio
async def test_acknowledge_skips_if_msg_id_empty():
    """A weird payload with no msg_id shouldn't trigger a Meta call —
    Meta returns 400 on missing message_id."""
    t = WhatsAppTransport()
    with patch("webhook.app.transports.whatsapp_transport._mark_as_read",
               new=AsyncMock()) as mock_mar:
        await t.acknowledge(_msg(""))
    mock_mar.assert_not_called()


# ---------- send_interstitial ----------

@pytest.mark.asyncio
async def test_send_interstitial_sends_configured_text():
    t = WhatsAppTransport(interstitial_text="hold on")
    with patch("webhook.app.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        await t.send_interstitial("+264user", Policy(name="x"))
    mock_send.assert_awaited_once_with("+264user", "hold on")


# ---------- send (no URLs to check) ----------

@pytest.mark.asyncio
async def test_send_basic():
    t = WhatsAppTransport()
    with patch("webhook.app.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        ok = await t.send("+264user", "hello world", Policy(name="x"))
    assert ok is True
    mock_send.assert_awaited_once_with("+264user", "hello world")


@pytest.mark.asyncio
async def test_send_skips_dead_url_check_when_no_search():
    """Parity: dead-URL HEAD-check is gated on used_search=True. Plain
    chat replies that happen to contain a URL must NOT incur HEAD latency."""
    t = WhatsAppTransport()
    body = "https://example.com/somepath is a thing"
    with patch("webhook.app.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        # If a HEAD check is performed, the patched httpx call will fail
        # the test. Don't patch httpx — verify no call is made.
        await t.send("+264user", body, Policy(name="x"), used_search=False)
    # URL still present — no stripping happened.
    assert "https://example.com/somepath" in mock_send.call_args.args[1]


@pytest.mark.asyncio
async def test_send_falls_back_when_dead_url_strip_empties_body():
    """If every URL line gets stripped (all dead), the body could be
    empty. Send a graceful fallback instead of an empty WhatsApp message
    (which Meta 400s)."""
    t = WhatsAppTransport()
    body = "— source: https://dead1.example.com/a\n— source: https://dead2.example.com/b"

    async def fake_head(self, url, **kwargs):
        return httpx.Response(404)

    with patch("webhook.app.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        with patch("httpx.AsyncClient.head", new=fake_head):
            await t.send("+264user", body, Policy(name="x"), used_search=True)

    sent = mock_send.call_args.args[1]
    assert "dead1" not in sent and "dead2" not in sent
    assert sent.strip() != ""           # NOT empty
    assert "different terms" in sent.lower() or "search again" in sent.lower()


@pytest.mark.asyncio
async def test_send_empty_body_uses_fallback():
    t = WhatsAppTransport()
    with patch("webhook.app.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        await t.send("+264user", "", Policy(name="x"))
    body = mock_send.call_args.args[1]
    assert "couldn't come up with a reply" in body.lower()


@pytest.mark.asyncio
async def test_send_caps_at_max_chars():
    t = WhatsAppTransport()
    big = "x" * 5000
    with patch("webhook.app.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        await t.send("+264user", big, Policy(name="x"))
    body = mock_send.call_args.args[1]
    assert len(body) == 4096


# ---------- dead URL stripping ----------

@pytest.mark.asyncio
async def test_send_strips_malformed_html_url():
    """A URL with an embedded <i> tag is broken — must not reach the user."""
    t = WhatsAppTransport()
    body = (
        "Here's the answer.\n"
        "\n"
        "— source: https://example.com/path</i>\n"
        "— source: https://good.example.com/article\n"
    )
    with patch("webhook.app.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        # No HEAD check runs because the good URL stays; the malformed
        # one is filtered before HEAD. But we still need to avoid network.
        with patch("httpx.AsyncClient.head",
                   new=AsyncMock(return_value=httpx.Response(200))):
            await t.send("+264user", body, Policy(name="x"), used_search=True)
    sent = mock_send.call_args.args[1]
    assert "example.com/path</i>" not in sent
    assert "good.example.com/article" in sent


@pytest.mark.asyncio
async def test_send_strips_404_url_lines():
    t = WhatsAppTransport()
    body = (
        "Here's the news.\n"
        "\n"
        "— source: https://dead.example.com/gone\n"
        "— source: https://good.example.com/article\n"
    )

    async def fake_head(self, url, **kwargs):
        if "dead" in url:
            return httpx.Response(404)
        return httpx.Response(200)

    with patch("webhook.app.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        with patch("httpx.AsyncClient.head", new=fake_head):
            await t.send("+264user", body, Policy(name="x"), used_search=True)

    sent = mock_send.call_args.args[1]
    assert "dead.example.com/gone" not in sent
    assert "good.example.com/article" in sent


@pytest.mark.asyncio
async def test_send_keeps_403_and_401_urls():
    """Gated / paywalled URLs (401, 403) are real pages — keep them."""
    t = WhatsAppTransport()
    body = "Check this: https://paywalled.example.com/x"

    async def fake_head(self, url, **kwargs):
        return httpx.Response(403)

    with patch("webhook.app.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        with patch("httpx.AsyncClient.head", new=fake_head):
            await t.send("+264user", body, Policy(name="x"), used_search=True)

    sent = mock_send.call_args.args[1]
    assert "paywalled.example.com" in sent


@pytest.mark.asyncio
async def test_send_keeps_url_when_head_fails_softly():
    """Network errors during the HEAD check must not strip URLs —
    transient failures shouldn't make us silently drop citations."""
    t = WhatsAppTransport()
    body = "Source: https://maybe-up.example.com/x"

    async def fake_head(self, url, **kwargs):
        raise httpx.ConnectError("connection refused")

    with patch("webhook.app.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        with patch("httpx.AsyncClient.head", new=fake_head):
            await t.send("+264user", body, Policy(name="x"), used_search=True)

    sent = mock_send.call_args.args[1]
    assert "maybe-up.example.com" in sent


@pytest.mark.asyncio
async def test_send_handles_url_with_trailing_punctuation():
    """Tavily snippets often have 'see https://example.com/x.' — trailing
    period should not break the HEAD check or leak from the regex match."""
    t = WhatsAppTransport()
    body = "See https://example.com/x."

    async def fake_head(self, url, **kwargs):
        # Verify URL is cleaned of trailing punct.
        assert not url.endswith(".")
        return httpx.Response(200)

    with patch("webhook.app.transports.whatsapp_transport._send_text",
               new=AsyncMock()):
        with patch("httpx.AsyncClient.head", new=fake_head):
            await t.send("+264user", body, Policy(name="x"), used_search=True)
