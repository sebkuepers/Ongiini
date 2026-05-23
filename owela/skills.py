"""Skill — a named, optionally on-demand context block for the model.

A Skill is a named bundle of static reference content (typically markdown)
that the application registers into the Runtime at startup.

Frontmatter format matches Claude's skill standard for cross-compatibility:
required fields are ``name`` and ``description``; the description is the
primary signal the model uses to decide when the skill is relevant (so
"when to use this skill" guidance goes IN the description, Claude-style).

Owela adds one optional extension field, ``load``, ignored by Claude but
honoured by Owela's renderer:

  - ``load="always"`` — the full content is folded into the system prompt
    on every turn (via the application's ``MemoryProvider.assemble_messages``,
    which renders ``Runtime.skills.manifest()``).

  - ``load="on_demand"`` (default) — only the manifest entry (name +
    description) appears in the system prompt. The model retrieves the
    full content by calling a ``load_skill`` tool — application-provided,
    since the tool needs to look up the skill via ``ToolContext.runtime``.

Use ``always`` for small, frequently-needed content where the cost of an
extra tool round-trip exceeds the cost of carrying the content. Use
``on_demand`` for larger or rarely-needed content where saving prompt
tokens on most turns is worth the round-trip on the few turns that need it.

A skill written for Claude (``~/skills/<name>/SKILL.md`` with just ``name``
+ ``description``) loads cleanly in Owela: ``load`` defaults to ``on_demand``.
An Owela skill with the ``load: always`` extension loads cleanly in Claude:
Claude ignores the unknown key.

Anti-trap fit: Skills don't touch the executor, don't introduce new step
types, and don't bundle tools/hooks/policies. They're a registry on the
Runtime; rendering and tool wiring stays in the application layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Skill:
    """One named reference block.

    ``name`` and ``description`` mirror Claude's skill frontmatter spec.
    The description should explain BOTH what the skill is AND when the
    model should use it — that's the Claude convention and it works
    well: the model sees the description in the manifest and decides
    relevance from it.

    ``content`` is what the model eventually sees when the skill is
    loaded (either inlined for ``load="always"`` or returned by a tool
    for ``load="on_demand"``). Typically markdown with worked examples.
    Frontmatter parsing happens at the application layer; by the time a
    Skill instance exists, ``content`` is plain text ready to embed in a
    system message or return from a tool call.

    ``load`` is an Owela-specific extension. Claude treats every skill
    as on-demand; Owela lets the skill declare itself as always-loaded
    if its content is small enough that prompt-time inclusion beats a
    tool round-trip.
    """
    name: str
    description: str
    content: str
    load: Literal["always", "on_demand"] = "on_demand"


class SkillRegistry:
    """Holds the Skills registered for one Runtime.

    Tests typically construct one explicitly with a small skill list.
    Production wires it from a filesystem loader (e.g. parsing markdown
    files with YAML frontmatter).
    """

    def __init__(self, skills: list[Skill] | None = None) -> None:
        self._by_name: dict[str, Skill] = {}
        for s in (skills or []):
            self.add(s)

    def add(self, skill: Skill) -> "SkillRegistry":
        """Register one Skill. Replaces any existing entry with the same name."""
        self._by_name[skill.name] = skill
        return self

    def get(self, name: str) -> Skill | None:
        """Lookup by name. Returns None if absent."""
        return self._by_name.get(name)

    def all(self) -> tuple[Skill, ...]:
        """All registered skills, in insertion order."""
        return tuple(self._by_name.values())

    def names(self) -> list[str]:
        """Names of all registered skills."""
        return list(self._by_name.keys())

    def manifest(self) -> str:
        """Render the system-message block: a manifest plus the inline
        content of every ``always``-loaded skill.

        Format::

            AVAILABLE SKILLS:
            - **<name>**: <description> [loaded below]
            - **<other>**: ... [call load_skill("other") to load]

            ## Skill: <name>

            <content>

            ## Skill: <other-always-loaded>

            ...

        If no skills are registered, returns an empty string so the
        caller can safely skip injecting the block.
        """
        if not self._by_name:
            return ""

        lines: list[str] = ["AVAILABLE SKILLS:"]
        always_loaded: list[Skill] = []
        for s in self._by_name.values():
            entry = f"- **{s.name}**: {s.description}"
            if s.load == "always":
                always_loaded.append(s)
                entry += " [loaded below]"
            else:
                entry += f' [call load_skill("{s.name}") to load]'
            lines.append(entry)

        for s in always_loaded:
            lines.append("")
            lines.append(f"## Skill: {s.name}")
            lines.append("")
            lines.append(s.content)

        return "\n".join(lines)
