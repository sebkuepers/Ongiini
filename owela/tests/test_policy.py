"""Policy and PolicyTable behaviour tests."""

from __future__ import annotations

import pytest

from owela.errors import PolicyNotFound
from owela.policy import (
    AUTO, DEPTH_DEEP, DEPTH_SHALLOW, Policy, PolicyTable,
    VERDICT_DOCS, VERDICT_NONE, VERDICT_SEARCH, force_tool,
)


def test_policy_is_frozen():
    p = Policy(name="test")
    with pytest.raises(Exception):
        p.name = "other"   # type: ignore[misc]


def test_policy_defaults():
    p = Policy(name="x")
    assert p.first_tool == AUTO
    assert p.max_steps == 6
    assert p.enable_planner is False
    assert p.enable_critique is False
    assert p.enable_interstitial is False
    assert p.reasoning_budget == 500
    assert p.long_result_threshold_chars == 1000
    assert p.expose_tools is None


def test_force_tool_shape():
    tc = force_tool("web_search")
    assert tc == {"type": "function", "function": {"name": "web_search"}}


def test_policy_table_exact_lookup():
    table = PolicyTable()
    p = Policy(name="search_deep")
    table.set(VERDICT_SEARCH, DEPTH_DEEP, p)
    assert table.lookup(VERDICT_SEARCH, DEPTH_DEEP).name == "search_deep"


def test_policy_table_falls_back_to_shallow():
    table = PolicyTable()
    shallow = Policy(name="search_shallow")
    table.set(VERDICT_SEARCH, DEPTH_SHALLOW, shallow)
    # Querying for DEEP should fall back to SHALLOW for the same verdict.
    assert table.lookup(VERDICT_SEARCH, DEPTH_DEEP).name == "search_shallow"


def test_policy_table_falls_back_to_global_none():
    table = PolicyTable()
    fallback = Policy(name="fallback")
    table.set(VERDICT_NONE, DEPTH_SHALLOW, fallback)
    # An unknown verdict with no specific row should fall back to NONE/SHALLOW.
    assert table.lookup(VERDICT_DOCS, DEPTH_SHALLOW).name == "fallback"


def test_policy_table_raises_when_no_fallback():
    table = PolicyTable()
    with pytest.raises(PolicyNotFound):
        table.lookup(VERDICT_SEARCH, DEPTH_DEEP)


def test_policy_table_all_returns_copy():
    table = PolicyTable()
    p = Policy(name="x")
    table.set(VERDICT_NONE, DEPTH_SHALLOW, p)
    snapshot = table.all()
    # Mutating the snapshot must not mutate the table.
    snapshot.clear()
    assert (VERDICT_NONE, DEPTH_SHALLOW) in table.all()
