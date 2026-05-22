"""Owela @tool implementations for Ongiini.

Each tool is an async function with the @tool decorator that
auto-generates the OpenAI function-call schema from the signature
+ docstring. Tools that need runtime access (delete_my_data,
my_token_usage, etc.) take a ToolContext as their first parameter;
the schema generator excludes ToolContext from what the model sees.

The application's runtime construction (``ongiini.runtime``) builds
a ``ToolRegistry`` from the list exported here.
"""

from .ongiini_tools import (
    ALL_TOOLS,
    delete_my_data,
    fetch_url,
    fetch_urls,
    lookup_ongiini_docs,
    my_token_usage,
    web_search,
    whats_in_my_memory,
)

__all__ = [
    "ALL_TOOLS",
    "delete_my_data",
    "fetch_url",
    "fetch_urls",
    "lookup_ongiini_docs",
    "my_token_usage",
    "web_search",
    "whats_in_my_memory",
]
