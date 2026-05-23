"""Tests for ongiini/skills_loader.py — parses Claude-format SKILL.md files."""

from __future__ import annotations

import logging
from pathlib import Path
from textwrap import dedent

import pytest

from ongiini.skills_loader import _parse_skill_file, load_skills


def _write_skill(root: Path, name: str, body: str) -> Path:
    """Helper: write root/<name>/SKILL.md with the given body, return its path."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    md_path = skill_dir / "SKILL.md"
    md_path.write_text(body, encoding="utf-8")
    return md_path


def test_parse_minimal_claude_compatible_skill(tmp_path: Path) -> None:
    """A skill with just name + description (Claude-spec minimum) loads cleanly."""
    md = _write_skill(tmp_path, "minimal", dedent("""\
        ---
        name: minimal
        description: A minimal Claude-compatible skill for testing.
        ---

        Hello world.
        """))
    skill = _parse_skill_file(md)
    assert skill.name == "minimal"
    assert skill.description == "A minimal Claude-compatible skill for testing."
    assert skill.content == "Hello world."
    assert skill.load == "on_demand"  # default for Claude-compat


def test_parse_owela_extended_skill(tmp_path: Path) -> None:
    """The Owela ``load`` extension is honoured when present."""
    md = _write_skill(tmp_path, "always_loaded", dedent("""\
        ---
        name: always_loaded
        description: An always-loaded skill.
        load: always
        ---

        Body content.
        """))
    skill = _parse_skill_file(md)
    assert skill.load == "always"


def test_parse_ignores_unknown_frontmatter_keys(tmp_path: Path) -> None:
    """Custom frontmatter (sources, author, version) is parsed but ignored.

    This is what keeps Owela skills cross-compatible with Claude — extra
    keys exist for human readers but the runtime contract is just
    name + description (+ optional Owela `load`)."""
    md = _write_skill(tmp_path, "rich", dedent("""\
        ---
        name: rich
        description: A skill with extra metadata.
        load: always
        sources:
          - Some source
          - Another source
        author: Daisy
        version: 0.1
        ---

        Body.
        """))
    skill = _parse_skill_file(md)
    assert skill.name == "rich"
    assert skill.load == "always"


def test_parse_missing_frontmatter_raises(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, "broken", "No frontmatter here.\n")
    with pytest.raises(ValueError, match="missing YAML frontmatter"):
        _parse_skill_file(md)


def test_parse_missing_name_raises(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, "noname", dedent("""\
        ---
        description: Has description but no name.
        ---

        Body.
        """))
    with pytest.raises(ValueError, match="'name' is required"):
        _parse_skill_file(md)


def test_parse_missing_description_raises(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, "nodesc", dedent("""\
        ---
        name: nodesc
        ---

        Body.
        """))
    with pytest.raises(ValueError, match="'description' is required"):
        _parse_skill_file(md)


def test_parse_invalid_load_raises(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, "badload", dedent("""\
        ---
        name: badload
        description: Has an invalid load value.
        load: maybe
        ---

        Body.
        """))
    with pytest.raises(ValueError, match="'load' must be"):
        _parse_skill_file(md)


def test_parse_markdown_with_horizontal_rule_in_body(tmp_path: Path) -> None:
    """The frontmatter regex must be non-greedy — `---` inside the body
    is a markdown horizontal rule, not a second frontmatter delimiter."""
    md = _write_skill(tmp_path, "withrule", dedent("""\
        ---
        name: withrule
        description: Skill whose body contains a horizontal rule.
        ---

        First paragraph.

        ---

        Second paragraph after the rule.
        """))
    skill = _parse_skill_file(md)
    assert "First paragraph." in skill.content
    assert "Second paragraph after the rule." in skill.content
    assert skill.content.count("---") == 1  # The horizontal rule is preserved.


def test_load_skills_from_directory(tmp_path: Path) -> None:
    """Scan a directory with multiple skills."""
    _write_skill(tmp_path, "a", dedent("""\
        ---
        name: a
        description: Skill A.
        load: always
        ---

        A content.
        """))
    _write_skill(tmp_path, "b", dedent("""\
        ---
        name: b
        description: Skill B.
        ---

        B content.
        """))
    registry = load_skills(skills_dir=tmp_path)
    assert set(registry.names()) == {"a", "b"}
    assert registry.get("a").load == "always"
    assert registry.get("b").load == "on_demand"


def test_load_skills_skips_directories_without_skill_md(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """A directory without SKILL.md is skipped with a warning, not an error."""
    (tmp_path / "incomplete").mkdir()
    _write_skill(tmp_path, "good", dedent("""\
        ---
        name: good
        description: Has a SKILL.md.
        ---

        Body.
        """))
    with caplog.at_level(logging.WARNING):
        registry = load_skills(skills_dir=tmp_path)
    assert registry.names() == ["good"]
    assert any("no SKILL.md" in rec.message for rec in caplog.records)


def test_load_skills_continues_after_malformed_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """One bad skill must not prevent the others from loading."""
    _write_skill(tmp_path, "bad", "this is not a valid skill file\n")
    _write_skill(tmp_path, "good", dedent("""\
        ---
        name: good
        description: A valid skill.
        ---

        Body.
        """))
    with caplog.at_level(logging.WARNING):
        registry = load_skills(skills_dir=tmp_path)
    assert "good" in registry.names()
    assert "bad" not in registry.names()
    assert any("failed to load skill bad" in rec.message for rec in caplog.records)


def test_load_skills_empty_directory_returns_empty_registry(tmp_path: Path) -> None:
    registry = load_skills(skills_dir=tmp_path)
    assert registry.all() == ()
    assert registry.manifest() == ""


def test_load_skills_nonexistent_directory_returns_empty_registry(tmp_path: Path) -> None:
    registry = load_skills(skills_dir=tmp_path / "does-not-exist")
    assert registry.all() == ()


def test_real_oshiwambo_skill_loads(tmp_path: Path) -> None:  # pragma: no cover
    """Smoke test: the actual oshiwambo skill in the repo parses correctly."""
    from ongiini import skills_loader
    real_dir = skills_loader.SKILLS_DIR
    if not (real_dir / "oshiwambo" / "SKILL.md").exists():
        pytest.skip("oshiwambo skill not present (not a regression)")
    registry = load_skills()
    assert "oshiwambo" in registry.names()
    osh = registry.get("oshiwambo")
    assert osh.load == "always"
    # Spot-check the content has the key Oshiwambo phrases
    assert "Ongiini" in osh.content
    assert "Kandi udite ko" in osh.content
    assert "Wa lele po" in osh.content
