"""Operator alerts: template first, free text as fallback."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ongiini import ops_alert
from ongiini.broadcast.register_template import TEMPLATES


def test_template_param_is_one_safe_line():
    out = ops_alert.template_param("line one\n- line two\n\tindented    gap")
    assert "\n" not in out and "\t" not in out and "    " not in out
    assert out == "line one; - line two; indented gap"


def test_template_param_is_capped():
    assert len(ops_alert.template_param("x" * 5000)) <= 900


def test_ops_alert_template_is_registered_as_utility():
    t = TEMPLATES[ops_alert.TEMPLATE_NAME]
    assert t["category"] == "UTILITY"
    body = next(c for c in t["components"] if c["type"] == "BODY")
    assert "{{1}}" in body["text"] and not body["text"].rstrip().endswith("}}")


@pytest.mark.asyncio
async def test_sends_template_when_possible():
    with patch.object(ops_alert, "send_template", new=AsyncMock()) as tmpl, \
         patch.object(ops_alert, "send_text", new=AsyncMock()) as txt:
        path = await ops_alert.send_ops_alert("491700000000", "search down\nsince 07:00")
    assert path == "template"
    tmpl.assert_awaited_once_with("491700000000", "ongiini_ops_alert", "en", ["search down; since 07:00"])
    txt.assert_not_called()


@pytest.mark.asyncio
async def test_falls_back_to_text_when_template_fails():
    with patch.object(ops_alert, "send_template", new=AsyncMock(side_effect=RuntimeError("not approved"))), \
         patch.object(ops_alert, "send_text", new=AsyncMock()) as txt:
        path = await ops_alert.send_ops_alert("491700000000", "search down")
    assert path == "text"
    txt.assert_awaited_once_with("491700000000", "search down")
