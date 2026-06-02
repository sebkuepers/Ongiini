"""Adaptive language-learning module for the learn.ongiini.ai surface.

This package owns the persistent learning experience: a learner identity
that survives across sessions, an intake state machine, a Leitner-style
spaced-repetition queue, and a SQLite store for progress.

It is a deliberate departure from the WhatsApp and chat.ongiini.ai
channels, both of which were either ephemeral or scoped to single
turns. Learning needs continuity — the same card has to come back
tomorrow, harder cards come back sooner, mastery shows up over weeks.
So this is the first part of Ongiini that writes user-linked progress
to disk on every turn.

Layout:
  db.py       — sqlite warmup + connection helper (mirrors contributions.py)
  srs.py      — pure-function Leitner system
  tokens.py   — HMAC-signed magic-link tokens
  intake.py   — 4-step intake state machine (name → age → level → objective)
  store.py    — LearnerStore: high-level data access over the schema
  cards.py    — card generation (model-driven, cached in DB)
  grading.py  — answer-grading (model call, returns rating + feedback)
"""
from .db import warmup
from .srs import promote, next_due_at, BOX_INTERVALS_HOURS

__all__ = ["warmup", "promote", "next_due_at", "BOX_INTERVALS_HOURS"]
