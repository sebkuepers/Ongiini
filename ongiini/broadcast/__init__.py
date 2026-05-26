"""Proactive WhatsApp broadcast subsystem.

Outbound-only path for sending pre-approved MARKETING template
messages to opted-in users, with three integrations the regular
agent loop doesn't need:

- per-user memory write (so the AI has context when the user replies)
- opt-out store (STOP keyword honoured before any send)
- Meta Cloud API template-message payload (distinct from session sends)

CLI entrypoint is ``scripts/broadcast.py``.
"""
