"""Transport protocol + the InboundMessage value object.

The Transport adapter is everything the framework knows about the
outside world: where messages come from, where they go, what the
medium's constraints are (max length, allowed formatting, typing
indicator window). All WhatsApp-specific code lives in the impl; the
executor only sees the protocol.

Anti-trap principle #5: transport is an adapter. Adding Signal,
Telegram, or a CLI is "implement Transport + register the impl into
the Runtime" — no executor changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .policy import Policy


@dataclass
class InboundMessage:
    """A normalised inbound message, transport-agnostic.

    ``content_parts`` is the OpenAI-multipart shape so the model adapter
    can pass it straight through (this is how text + image_url are
    represented). ``text`` is the same content flattened to a single
    string — used for routing and search-query extraction. For an audio
    message, the application transcribes first and presents the result
    as text + content_parts=[{"type": "text", "text": transcript}].

    ``storage_text`` is what the application wants persisted as the
    user-side turn entry (post-PII-sanitisation, with any media
    placeholder like "[image attached]" or "[voice note]" applied).
    Empty string means "use ``text`` verbatim". The model sees ``text``;
    persistence sees ``storage_text``. This decoupling is what lets us
    show raw text to the LLM while storing a redacted/labelled version
    on disk.

    ``history`` is the conversation history this transport already has
    persisted for this user. The application loads it before constructing
    the InboundMessage; Owela does not perform any I/O to read history
    on its own.
    """
    user_id: str
    msg_id: str
    text: str
    content_parts: list[dict[str, Any]]
    has_image: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)
    storage_text: str = ""
    raw_payload: dict[str, Any] | None = None


@runtime_checkable
class Transport(Protocol):
    """The outbound side of the transport — everything that sends bytes
    back to the user, plus transport metadata the executor inspects.

    Inbound parsing (webhook payload → InboundMessage) lives outside
    Owela. The application's webhook handler is responsible for
    constructing InboundMessages and handing them to Agent.handle().
    """

    name: str
    typing_window_s: float
    max_message_chars: int
    format: str                        # "plain_text" | "markdown" | "html"

    async def acknowledge(self, msg: InboundMessage) -> None:
        """Surface the medium's "we got your message" UX — read receipt,
        typing indicator, etc. Soft-fail: should never raise. Called by
        the executor immediately on entering a turn."""
        ...

    async def send_interstitial(self, user_id: str, policy: Policy) -> None:
        """Send a "still working" interstitial during long turns. Only
        called when ``policy.enable_interstitial`` is True (v1 feature)."""
        ...

    async def send(
        self,
        user_id: str,
        body: str,
        policy: Policy,
        *,
        used_search: bool = False,
    ) -> bool:
        """Deliver the final reply. The transport is free (and expected)
        to post-process the body — dead-URL strip, format normalisation,
        char cap. Returns True if the recipient's transport accepted the
        payload.

        ``used_search`` is a hint set by the executor when any
        web_search / fetch_url / fetch_urls tool fired during the turn.
        Transports use it to gate expensive reply-hygiene checks (e.g.
        dead-URL HEAD probing) that are only meaningful when the model
        is citing tool output, not when it's chatting plainly.
        """
        ...
