"""Owela @tool implementations for Ongiini.

Each tool is an async function with the @tool decorator that
auto-generates the OpenAI function-call schema from the signature
+ docstring. Tools that need runtime access (delete_my_data,
my_token_usage, etc.) take a ToolContext as their first parameter;
the schema generator excludes ToolContext from what the model sees.

The application's runtime construction (``ongiini.runtime``) builds
a ``ToolRegistry`` from the list exported here.
"""

from .contribute import (
    contribute_decline,
    contribute_invite_check,
    contribute_next,
    contribute_save,
    contribute_set_dialect,
    contribute_skip,
    contribute_stats,
)
from .ongiini_tools import (
    ALL_TOOLS as _PRODUCT_TOOLS,
    delete_my_data,
    fetch_url,
    fetch_urls,
    lookup_ongiini_docs,
    my_token_usage,
    web_search,
    whats_in_my_memory,
)
from .opt_out import opt_out_broadcast
from .skill_tools import load_skill

# Canonical tool list passed to ToolRegistry at runtime build time.
# load_skill is appended last because the model interacts with it less
# than the product tools (it's only relevant when an on-demand skill
# matches the user's message), so it gets the smallest prior.
# The contribute_* tools come AFTER the product tools — they're
# domain-specific, force-called by the classifier-driven policy table
# (no model selection), so their position in the list is essentially
# cosmetic (force_tool targets by name).
_CONTRIBUTE_TOOLS = (
    contribute_invite_check,
    contribute_set_dialect,
    contribute_next,
    contribute_save,
    contribute_skip,
    contribute_decline,
    contribute_stats,
)
ALL_TOOLS = (*_PRODUCT_TOOLS, *_CONTRIBUTE_TOOLS, opt_out_broadcast, load_skill)

__all__ = [
    "ALL_TOOLS",
    "contribute_decline",
    "contribute_invite_check",
    "contribute_next",
    "contribute_save",
    "contribute_set_dialect",
    "contribute_skip",
    "contribute_stats",
    "delete_my_data",
    "fetch_url",
    "fetch_urls",
    "load_skill",
    "lookup_ongiini_docs",
    "my_token_usage",
    "opt_out_broadcast",
    "web_search",
    "whats_in_my_memory",
]
