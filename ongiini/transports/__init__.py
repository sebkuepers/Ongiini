"""Owela Transport adapters for Ongiini."""

from .web_chat_transport import WebChatTransport
from .whatsapp_transport import WhatsAppTransport

__all__ = ["WebChatTransport", "WhatsAppTransport"]
