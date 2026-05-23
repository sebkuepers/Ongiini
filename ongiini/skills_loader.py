"""Load Ongiini skills from disk into an Owela ``SkillRegistry``.

Layout matches Claude's skill standard for cross-compatibility::

    ongiini/skills/
        oshiwambo/
            SKILL.md
        namibian_regions/
            SKILL.md
        ...

Each ``SKILL.md`` has YAML frontmatter with the Claude-spec fields
(``name``, ``description``) and may include the Owela-specific ``load``
extension (``"always"`` or ``"on_demand"`` — defaults to ``"on_demand"``
for Claude-compat). Unknown frontmatter keys (e.g. ``sources``) are
parsed but otherwise ignored — they're metadata for humans, not the
runtime.

A skill file that drops into ``~/.claude/skills/<name>/SKILL.md`` works
without changes; an Owela-extended skill (with ``load: always``) loads
cleanly in Claude too because Claude ignores unknown keys.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from owela import Skill, SkillRegistry

log = logging.getLogger("ongiini.skills_loader")

SKILLS_DIR = Path(__file__).parent / "skills"

# Match a leading ``---\n<yaml>\n---\n<body>`` block. Re-DOTALL so the
# YAML body may contain newlines; non-greedy so we stop at the FIRST
# closing ``---`` line, not the last one in the file (a markdown body
# might contain `---` as a horizontal rule).
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def _parse_skill_file(path: Path) -> Skill:
    """Parse one SKILL.md file. Raises ValueError on malformed input."""
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(
            f"Skill file {path} is missing YAML frontmatter (expected "
            f"'---\\n<yaml>\\n---\\n<body>' at the top)."
        )
    raw_meta = m.group(1)
    body = m.group(2).strip()

    try:
        meta = yaml.safe_load(raw_meta) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Skill file {path}: invalid YAML frontmatter: {exc}") from exc

    if not isinstance(meta, dict):
        raise ValueError(f"Skill file {path}: frontmatter must be a YAML mapping.")

    # Claude-spec required fields
    name = meta.get("name")
    description = meta.get("description")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Skill file {path}: 'name' is required and must be a non-empty string.")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            f"Skill file {path}: 'description' is required and must be a non-empty string."
        )

    # Owela extension — defaults to on_demand (Claude's behaviour).
    load = meta.get("load", "on_demand")
    if load not in ("always", "on_demand"):
        raise ValueError(
            f"Skill file {path}: 'load' must be 'always' or 'on_demand' (got {load!r})."
        )

    return Skill(
        name=name.strip(),
        description=description.strip(),
        content=body,
        load=load,
    )


def load_skills(skills_dir: Path | None = None) -> SkillRegistry:
    """Scan ``skills_dir`` for ``<name>/SKILL.md`` files and build a
    SkillRegistry.

    Defaults to ``ongiini/skills/``. Tests can pass a tmp_path with a
    constructed layout.

    A malformed skill file logs a warning and is skipped — one bad
    skill must not prevent the application from starting. The application
    can still observe what loaded via ``runtime.skills.names()``.
    """
    target = skills_dir or SKILLS_DIR
    skills: list[Skill] = []

    if not target.exists():
        log.info("skills dir %s does not exist; no skills loaded", target)
        return SkillRegistry()

    for skill_dir in sorted(target.iterdir()):
        if not skill_dir.is_dir():
            continue
        md_path = skill_dir / "SKILL.md"
        if not md_path.exists():
            log.warning("skipping %s: no SKILL.md inside", skill_dir.name)
            continue
        try:
            skills.append(_parse_skill_file(md_path))
        except ValueError as exc:
            log.warning("failed to load skill %s: %s", skill_dir.name, exc)
            continue

    registry = SkillRegistry(skills)
    log.info(
        "loaded %d skill(s): %s",
        len(registry.all()),
        ", ".join(registry.names()) or "(none)",
    )
    return registry
