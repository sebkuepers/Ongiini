"""Unit tests for the WhatsApp Owela Transport."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from owela import InboundMessage, Policy, ReplyContext
from ongiini.transports.whatsapp_transport import WhatsAppTransport


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
    with patch("ongiini.transports.whatsapp_transport._mark_as_read",
               new=AsyncMock()) as mock_mar:
        await t.acknowledge(_msg("wamid.42"))
    mock_mar.assert_awaited_once_with("wamid.42", with_typing=True)


@pytest.mark.asyncio
async def test_acknowledge_skips_if_msg_id_empty():
    """A weird payload with no msg_id shouldn't trigger a Meta call —
    Meta returns 400 on missing message_id."""
    t = WhatsAppTransport()
    with patch("ongiini.transports.whatsapp_transport._mark_as_read",
               new=AsyncMock()) as mock_mar:
        await t.acknowledge(_msg(""))
    mock_mar.assert_not_called()


# ---------- send_interstitial ----------

@pytest.mark.asyncio
async def test_send_interstitial_sends_configured_text():
    t = WhatsAppTransport(interstitial_text="hold on")
    with patch("ongiini.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        await t.send_interstitial("+264user", Policy(name="x"))
    mock_send.assert_awaited_once_with("+264user", "hold on")


# ---------- send ----------

SEARCH_CTX = ReplyContext(
    used_tools=("web_search",),
    allowed_urls=("https://www.namibian.com.na/national/story-1",),
)


@pytest.mark.asyncio
async def test_send_basic():
    t = WhatsAppTransport()
    with patch("ongiini.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        result = await t.send("+264user", "hello world", Policy(name="x"), ReplyContext())
    assert result.sent is True
    mock_send.assert_awaited_once_with("+264user", "hello world")


@pytest.mark.asyncio
async def test_send_leaves_urls_alone_without_citation_tools():
    """No citation tool ran → URLs came from the conversation or the
    model's own knowledge; the transport doesn't judge them."""
    t = WhatsAppTransport()
    body = "See https://example.com/page for the details."
    with patch("ongiini.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        result = await t.send("+264user", body, Policy(name="x"), ReplyContext())
    mock_send.assert_awaited_once_with("+264user", body)
    assert "urls_dropped" not in result.attrs


@pytest.mark.asyncio
async def test_send_drops_urls_the_tools_did_not_return():
    t = WhatsAppTransport()
    body = (
        "The story is here.\n\n"
        "— source: https://www.namibian.com.na/national/story-1\n"
        "— source: https://invented.example.org/made-up-path"
    )
    with patch("ongiini.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        result = await t.send("+264user", body, Policy(name="x"), SEARCH_CTX)
    sent = mock_send.await_args.args[1]
    assert "https://www.namibian.com.na/national/story-1" in sent
    assert "invented.example.org" not in sent
    assert "— source:" in sent                     # the real citation line survives
    assert sent.count("— source:") == 1            # the empty stub line is gone
    assert result.attrs["urls_dropped"] == 1


@pytest.mark.asyncio
async def test_send_url_match_ignores_www_trailing_slash_and_fragment():
    t = WhatsAppTransport()
    body = "Details: https://namibian.com.na/national/story-1/#top."
    with patch("ongiini.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        result = await t.send("+264user", body, Policy(name="x"), SEARCH_CTX)
    assert "namibian.com.na/national/story-1" in mock_send.await_args.args[1]
    assert result.attrs["urls_dropped"] == 0


@pytest.mark.asyncio
async def test_send_strips_malformed_html_url():
    t = WhatsAppTransport()
    body = "Info\n— source: https://www.namibian.com.na/national/story-1<i>x</i>"
    with patch("ongiini.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        result = await t.send("+264user", body, Policy(name="x"), SEARCH_CTX)
    sent = mock_send.await_args.args[1]
    assert "<i>" not in sent
    assert result.attrs["urls_dropped"] == 1


@pytest.mark.asyncio
async def test_send_with_citation_tool_but_no_urls_only_drops_malformed():
    """A search that failed returns no URLs; nothing is verifiable, so only
    malformed URLs go (user-supplied links must survive)."""
    t = WhatsAppTransport()
    body = "You mentioned https://user-shared.example/page earlier."
    with patch("ongiini.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        await t.send("+264user", body, Policy(name="x"),
                     ReplyContext(used_tools=("web_search",)))
    assert "user-shared.example" in mock_send.await_args.args[1]


@pytest.mark.asyncio
async def test_send_falls_back_when_url_strip_empties_body():
    t = WhatsAppTransport()
    body = "https://invented.example.org/a\nhttps://invented.example.org/b"
    with patch("ongiini.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        result = await t.send("+264user", body, Policy(name="x"), SEARCH_CTX)
    assert mock_send.await_args.args[1] == t.empty_reply_text
    assert result.attrs["empty_fallback"] is True


@pytest.mark.asyncio
async def test_send_empty_body_uses_fallback():
    t = WhatsAppTransport()
    with patch("ongiini.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        result = await t.send("+264user", "", Policy(name="x"), ReplyContext())
    mock_send.assert_awaited_once_with("+264user", "Sorry, I couldn't come up with a reply.")
    assert result.attrs["empty_fallback"] is True


@pytest.mark.asyncio
async def test_send_caps_at_max_chars_on_a_boundary():
    t = WhatsAppTransport()
    big = ("This is one sentence of a very long reply. " * 200).strip()
    with patch("ongiini.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        result = await t.send("+264user", big, Policy(name="x"), ReplyContext())
    sent = mock_send.await_args.args[1]
    assert len(sent) <= 4096
    assert sent.endswith(".")
    assert result.attrs["chars_capped"] is True


@pytest.mark.asyncio
async def test_truncated_draft_is_trimmed_to_sentence_and_offers_more():
    t = WhatsAppTransport()
    body = "First full sentence here. Second full sentence here. Third one stops mid"
    with patch("ongiini.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        result = await t.send("+264user", body, Policy(name="x"),
                              ReplyContext(truncated=True))
    sent = mock_send.await_args.args[1]
    assert "stops mid" not in sent
    assert sent.startswith("First full sentence here. Second full sentence here.")
    assert sent.endswith(t.continue_offer_text)
    assert result.attrs["trimmed_at_boundary"] is True


@pytest.mark.asyncio
async def test_reply_notice_is_appended_in_italics():
    t = WhatsAppTransport()
    policy = Policy(name="search_degraded", reply_notice="Live search is down.")
    with patch("ongiini.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        await t.send("+264user", "Answer.", policy, ReplyContext(degraded=True))
    assert mock_send.await_args.args[1] == "Answer.\n\n_Live search is down._"


@pytest.mark.asyncio
async def test_reply_notice_survives_the_char_cap():
    t = WhatsAppTransport()
    policy = Policy(name="p", reply_notice="Notice.")
    big = ("Sentence number one. " * 400).strip()
    with patch("ongiini.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        await t.send("+264user", big, policy, ReplyContext())
    sent = mock_send.await_args.args[1]
    assert len(sent) <= 4096
    assert sent.endswith("_Notice._")



# ---------- Markdown → WhatsApp normalisation ----------

def test_normalise_double_asterisks_to_single():
    """Gemma 4's most common formatting failure: **double-asterisk**
    Markdown bold that doesn't render in WhatsApp. Transform to single
    asterisks deterministically."""
    n = WhatsAppTransport._normalise_markdown_for_whatsapp
    assert n("This is **bold** text.") == "This is *bold* text."
    assert n("Two **here** and **there**.") == "Two *here* and *there*."


def test_normalise_does_not_break_intra_word_underscores():
    """Don't munge file_names_with_underscores or variable_names."""
    n = WhatsAppTransport._normalise_markdown_for_whatsapp
    assert n("File a__b__c.txt") == "File a__b__c.txt"
    assert "my_var" in n("Variable my_var here.")


def test_normalise_markdown_headers_to_bold():
    n = WhatsAppTransport._normalise_markdown_for_whatsapp
    assert n("# Heading") == "*Heading*"
    assert n("## Sub-heading") == "*Sub-heading*"
    assert n("### Section three") == "*Section three*"
    assert n("see #1234") == "see #1234"


def test_normalise_markdown_links_to_text_plus_url():
    """Markdown link syntax doesn't render. Convert to 'text (url)' so
    BOTH the link text AND the URL are visible — and the URL is bare,
    so the downstream dead-URL check can probe it."""
    n = WhatsAppTransport._normalise_markdown_for_whatsapp
    assert n("Read [The Namibian](https://namibian.com.na/x) today.") == \
        "Read The Namibian (https://namibian.com.na/x) today."


def test_normalise_leaves_whatsapp_native_syntax_alone():
    """Don't munge the formatting WhatsApp ACTUALLY renders."""
    n = WhatsAppTransport._normalise_markdown_for_whatsapp
    assert n("*single asterisk bold*") == "*single asterisk bold*"
    assert n("_italic here_") == "_italic here_"
    assert n("- bullet point") == "- bullet point"
    assert n("1. first\n2. second") == "1. first\n2. second"
    assert n("> quote") == "> quote"
    assert n("`inline code`") == "`inline code`"


@pytest.mark.asyncio
async def test_send_applies_markdown_normalisation():
    """Markdown links become 'text (url)' before the URL check, so a
    cited Markdown link to an allowed URL survives."""
    t = WhatsAppTransport()
    body = "**Key:** see [The Namibian](https://www.namibian.com.na/national/story-1)"
    with patch("ongiini.transports.whatsapp_transport._send_text",
               new=AsyncMock()) as mock_send:
        await t.send("+264user", body, Policy(name="x"), SEARCH_CTX)
    sent = mock_send.await_args.args[1]
    assert sent == "*Key:* see The Namibian (https://www.namibian.com.na/national/story-1)"



# ---------- v1.3 second interstitial ----------

@pytest.mark.asyncio
async def test_send_interstitial_schedules_followup_task():
    """v1.3: send_interstitial sends the first message AND schedules a
    follow-up task that will fire at T+followup_delay_s if the real
    reply hasn't arrived."""
    import asyncio
    t = WhatsAppTransport(
        interstitial_text="first",
        followup_interstitial_text="follow",
        followup_delay_s=0.05,
    )
    sent: list[str] = []

    async def fake_send(uid, body):
        sent.append(body)

    with patch("ongiini.transports.whatsapp_transport._send_text",
               new=AsyncMock(side_effect=fake_send)):
        await t.send_interstitial("+264user", Policy(name="x"))
        # First interstitial sent synchronously.
        assert sent == ["first"]
        # Task is alive and pending.
        task = t._followup_tasks.get("+264user")
        assert task is not None and not task.done()
        # Wait for the followup to fire.
        await asyncio.sleep(0.08)
    assert sent == ["first", "follow"]


@pytest.mark.asyncio
async def test_send_cancels_pending_followup_task():
    """When send() runs (the real reply lands), the pending followup
    interstitial is cancelled cleanly — no second interstitial is sent."""
    import asyncio
    t = WhatsAppTransport(
        interstitial_text="first",
        followup_interstitial_text="follow",
        followup_delay_s=1.0,    # plenty of time to cancel
    )
    sent: list[str] = []

    async def fake_send(uid, body):
        sent.append(body)

    async def fake_head(self, url, **kwargs):
        return httpx.Response(200)

    with patch("ongiini.transports.whatsapp_transport._send_text",
               new=AsyncMock(side_effect=fake_send)):
        with patch("httpx.AsyncClient.head", new=fake_head):
            await t.send_interstitial("+264user", Policy(name="x"))
            assert sent == ["first"]
            # Reply arrives BEFORE the 1s followup_delay elapses.
            await t.send("+264user", "real reply", Policy(name="x"), ReplyContext())
            # Give the cancelled task a chance to settle.
            await asyncio.sleep(0.02)
    # Only "first" interstitial + "real reply" — no follow-up.
    assert sent == ["first", "real reply"]
    # Task entry removed from the tracking dict.
    assert "+264user" not in t._followup_tasks


@pytest.mark.asyncio
async def test_followup_task_per_user_does_not_clobber_other_users():
    """The followup task dict is keyed by user_id; one user's
    interstitial doesn't affect another's."""
    import asyncio
    t = WhatsAppTransport(
        followup_interstitial_text="follow",
        followup_delay_s=10.0,    # long enough we never reach it
    )
    with patch("ongiini.transports.whatsapp_transport._send_text",
               new=AsyncMock()):
        await t.send_interstitial("+264alice", Policy(name="x"))
        await t.send_interstitial("+264bob", Policy(name="x"))
        assert "+264alice" in t._followup_tasks
        assert "+264bob" in t._followup_tasks
        # Cancel cleanly to avoid leaving tasks running in test teardown.
        for task in list(t._followup_tasks.values()):
            task.cancel()
        await asyncio.sleep(0.01)


# ---------- v1.3.1 Markdown table → labelled bullets ----------

def test_normalise_simple_table_to_bullets():
    n = WhatsAppTransport._normalise_markdown_for_whatsapp
    table = (
        "| Provider | Example | Best For |\n"
        "| :--- | :--- | :--- |\n"
        "| Local (NA) | Paratus | Hosting |\n"
        "| Global | AWS | Massive AI |\n"
    )
    out = n(table)
    # First column becomes the bolded row header; other columns become
    # labelled bullets paired with the header row's labels.
    assert "*Local (NA)*" in out
    assert "- Example: Paratus" in out
    assert "- Best For: Hosting" in out
    assert "*Global*" in out
    assert "- Example: AWS" in out
    # Raw pipes are gone.
    assert "|" not in out


def test_normalise_table_embedded_in_prose():
    n = WhatsAppTransport._normalise_markdown_for_whatsapp
    text = (
        "Here is the comparison:\n"
        "\n"
        "| Bank | Rate |\n"
        "| :--- | :--- |\n"
        "| Bank Windhoek | 13.0% |\n"
        "\n"
        "Hope that helps."
    )
    out = n(text)
    assert "Here is the comparison:" in out
    assert "*Bank Windhoek*" in out
    assert "- Rate: 13.0%" in out
    assert "Hope that helps." in out


def test_normalise_malformed_table_passes_through():
    """Header row with no alignment separator → not a valid Markdown
    table; pass through unchanged so we don't accidentally break
    something that wasn't a table."""
    n = WhatsAppTransport._normalise_markdown_for_whatsapp
    text = (
        "Some text with | pipes | in it but no\n"
        "alignment row below it.\n"
    )
    out = n(text)
    # Pipes preserved — we didn't try to convert.
    assert "|" in out


def test_normalise_table_skips_empty_cells():
    n = WhatsAppTransport._normalise_markdown_for_whatsapp
    table = (
        "| Name | A | B |\n"
        "| :--- | :--- | :--- |\n"
        "| Row1 | x |  |\n"
    )
    out = n(table)
    assert "*Row1*" in out
    assert "- A: x" in out
    # Empty cell B should not emit a bullet.
    assert "- B:" not in out


def test_normalise_table_alignment_with_no_colons_also_recognised():
    """Alignment cells can be ``---`` (no colons) too — common when
    Gemma emits plain Markdown."""
    n = WhatsAppTransport._normalise_markdown_for_whatsapp
    table = (
        "| H1 | H2 |\n"
        "| --- | --- |\n"
        "| a | b |\n"
    )
    out = n(table)
    assert "*a*" in out
    assert "- H2: b" in out


def test_normalise_table_inside_fenced_code_block_is_left_alone():
    """Markdown tables inside triple-backtick fences are part of a
    code block — WhatsApp renders those in monospace. Do NOT convert
    them to bullets."""
    n = WhatsAppTransport._normalise_markdown_for_whatsapp
    text = (
        "Here's an example of Markdown syntax:\n"
        "\n"
        "```\n"
        "| A | B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
        "```\n"
        "\n"
        "Use it like that."
    )
    out = n(text)
    # The pipes inside the fence are preserved.
    assert "| A | B |" in out
    assert "| 1 | 2 |" in out
    # No bullet conversion happened.
    assert "*1*" not in out
