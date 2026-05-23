"""Tests for owela/skills.py — Skill dataclass and SkillRegistry."""

from __future__ import annotations

import pytest

from owela import Skill, SkillRegistry


def test_skill_defaults_to_on_demand():
    s = Skill(name="x", description="d", content="c")
    assert s.load == "on_demand"


def test_skill_is_frozen():
    s = Skill(name="x", description="d", content="c")
    with pytest.raises(Exception):
        s.name = "y"  # type: ignore[misc]


def test_registry_empty_manifest_is_empty_string():
    reg = SkillRegistry()
    assert reg.manifest() == ""


def test_registry_add_and_lookup():
    s = Skill(name="oshi", description="basics", content="hello")
    reg = SkillRegistry([s])
    assert reg.get("oshi") is s
    assert reg.get("missing") is None
    assert reg.names() == ["oshi"]
    assert reg.all() == (s,)


def test_registry_replaces_same_name():
    s1 = Skill(name="x", description="v1", content="a")
    s2 = Skill(name="x", description="v2", content="b")
    reg = SkillRegistry([s1])
    reg.add(s2)
    assert reg.get("x") is s2
    assert len(reg.all()) == 1


def test_manifest_always_loaded_embeds_content():
    s = Skill(name="oshi", description="basics", content="hello world", load="always")
    reg = SkillRegistry([s])
    out = reg.manifest()
    assert "AVAILABLE SKILLS:" in out
    assert "**oshi**: basics" in out
    assert "[loaded below]" in out
    assert "## Skill: oshi" in out
    assert "hello world" in out


def test_manifest_on_demand_only_lists_manifest_entry():
    s = Skill(
        name="big", description="big skill", content="huge content", load="on_demand",
    )
    reg = SkillRegistry([s])
    out = reg.manifest()
    assert "**big**: big skill" in out
    assert 'call load_skill("big") to load' in out
    # Content is NOT embedded for on_demand skills
    assert "huge content" not in out
    assert "## Skill: big" not in out


def test_manifest_mixed_load_modes():
    a = Skill(name="a", description="A", content="a-content", load="always")
    b = Skill(name="b", description="B", content="b-content", load="on_demand")
    reg = SkillRegistry([a, b])
    out = reg.manifest()
    # Both appear in the listing
    assert "**a**: A" in out
    assert "**b**: B" in out
    # Only the always-loaded one is inlined
    assert "a-content" in out
    assert "b-content" not in out


def test_manifest_preserves_insertion_order():
    a = Skill(name="a", description="A", content="x")
    b = Skill(name="b", description="B", content="y")
    c = Skill(name="c", description="C", content="z")
    reg = SkillRegistry([b, a, c])
    assert reg.names() == ["b", "a", "c"]
    out = reg.manifest()
    # b comes before a comes before c in the listing
    assert out.index("**b**") < out.index("**a**") < out.index("**c**")
