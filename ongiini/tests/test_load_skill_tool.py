"""Tests for ``ongiini.tools.skill_tools.load_skill``."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from owela import Skill, SkillRegistry, ToolContext
from ongiini.tools.skill_tools import load_skill


@dataclass
class _FakeRuntime:
    """Minimal Runtime stand-in — load_skill only touches ``skills``."""
    skills: SkillRegistry | None = None


def _ctx(skills: SkillRegistry | None) -> ToolContext:
    """Build a ToolContext with a fake runtime carrying the given skills."""
    return ToolContext(
        user_id="test-user",
        runtime=_FakeRuntime(skills=skills),  # type: ignore[arg-type]
        msg=None,  # type: ignore[arg-type]  # the tool doesn't read msg
    )


@pytest.mark.asyncio
async def test_load_skill_returns_content_when_found() -> None:
    skill = Skill(
        name="oshiwambo", description="basics", content="hello world content",
    )
    registry = SkillRegistry([skill])
    out = await load_skill(_ctx(registry), name="oshiwambo")
    assert out == "hello world content"


@pytest.mark.asyncio
async def test_load_skill_returns_error_when_not_found() -> None:
    registry = SkillRegistry([
        Skill(name="oshiwambo", description="d", content="c"),
    ])
    out = await load_skill(_ctx(registry), name="missing")
    assert "not found" in out
    assert "oshiwambo" in out  # tells the model what IS available


@pytest.mark.asyncio
async def test_load_skill_returns_error_when_no_registry() -> None:
    out = await load_skill(_ctx(None), name="anything")
    assert "no skills registry" in out


@pytest.mark.asyncio
async def test_load_skill_returns_error_with_empty_registry() -> None:
    out = await load_skill(_ctx(SkillRegistry()), name="anything")
    assert "not found" in out


@pytest.mark.asyncio
async def test_load_skill_works_for_always_loaded_skills_too() -> None:
    """Calling load_skill for an always-loaded skill is harmless and
    returns the same content as the manifest already embedded."""
    skill = Skill(
        name="big", description="d", content="full content here", load="always",
    )
    registry = SkillRegistry([skill])
    out = await load_skill(_ctx(registry), name="big")
    assert out == "full content here"


@pytest.mark.asyncio
async def test_load_skill_is_registered_as_owela_tool() -> None:
    """The function carries a ToolSpec with the right name + ctx flag."""
    spec = load_skill.__owela_tool__  # type: ignore[attr-defined]
    assert spec.name == "load_skill"
    assert spec.needs_context is True
    assert "name" in spec.parameters["properties"]
