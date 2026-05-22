"""Owela error hierarchy.

All Owela-raised exceptions inherit from ``OwelaError`` so callers can
catch the framework's errors distinctly from arbitrary Python exceptions
that bubble up from adapter implementations.
"""

from __future__ import annotations


class OwelaError(Exception):
    """Base class for all Owela-raised exceptions."""


class ToolError(OwelaError):
    """Raised by a tool implementation to signal a recoverable failure.

    The executor catches this and turns it into a tool result the model
    can read, instead of aborting the turn. Use ``RuntimeError`` (or any
    other non-Owela exception) for unrecoverable bugs.
    """


class ModelError(OwelaError):
    """Generic wrapper for model adapter failures."""


class PolicyNotFound(OwelaError):
    """No Policy matches the router verdict + depth combination."""
