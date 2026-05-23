"""``load_skill`` tool — fetches the full content of an on-demand skill.

Companion to the Owela skill framework: the system prompt's manifest
lists registered skills with their descriptions; on-demand skills tell
the model to call ``load_skill("<name>")`` to retrieve their full
content. Always-loaded skills are already inlined in the manifest and
don't need this tool — but calling it for an always-loaded skill is
harmless (returns the same content, no side effects).

Lives in ``ongiini/`` (application layer) rather than ``owela/`` because
``owela/`` is a pure library that knows nothing about how applications
expose tools to the model.
"""

from __future__ import annotations

from owela import ToolContext, tool


@tool(
    name="load_skill",
    description=(
        "Load the full content of a registered skill by name. The system "
        "prompt lists available skills under 'AVAILABLE SKILLS'; call this "
        "tool with a skill's name to retrieve its full reference content. "
        "Use ONLY when the skill's description matches the user's current "
        "message. Always-loaded skills are already shown in the system "
        "prompt and don't need to be loaded again."
    ),
    params={"name": "Exact skill name as shown in AVAILABLE SKILLS (e.g. 'oshiwambo')."},
)
async def load_skill(ctx: ToolContext, name: str) -> str:
    """Look up the skill on the Runtime and return its content."""
    registry = ctx.runtime.skills
    if registry is None:
        return "Error: this runtime has no skills registry."
    skill = registry.get(name)
    if skill is None:
        available = registry.names()
        return (
            f"Error: skill {name!r} not found. "
            f"Available: {available if available else '(none registered)'}"
        )
    return skill.content
