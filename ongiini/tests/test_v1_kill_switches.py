"""Verify the v1 quality-phase kill switches in ongiini.config.

The kill switches gate the policy-table flags so that a bad post-deploy
behaviour (e.g. REVISE rate too high) can be turned off via env var
without a code redeploy. Each switch defaults to OFF (i.e. the phase
stays ON); setting the env var to "1" / "true" / "yes" disables it.

This file imports ``ongiini.config`` only, not ``ongiini.runtime``,
because runtime transitively imports mem0 which isn't installed
locally outside the Docker image.
"""

from __future__ import annotations

import importlib

import pytest


def _reload_config_with_env(monkeypatch, **env: str):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # Clear any prior env vars not in this set to avoid cross-test bleed.
    for var in ("ONGIINI_DISABLE_PLANNER", "ONGIINI_DISABLE_CRITIQUE", "ONGIINI_DISABLE_INTERSTITIAL"):
        if var not in env:
            monkeypatch.delenv(var, raising=False)

    import ongiini.config
    return importlib.reload(ongiini.config).settings


def test_kill_switches_default_off(monkeypatch):
    """Default (no env vars set): all three phases are ON, meaning
    the disable flags are False."""
    settings = _reload_config_with_env(monkeypatch)
    assert settings.disable_planner is False
    assert settings.disable_critique is False
    assert settings.disable_interstitial is False


def test_planner_kill_switch_set(monkeypatch):
    settings = _reload_config_with_env(monkeypatch, ONGIINI_DISABLE_PLANNER="1")
    assert settings.disable_planner is True
    assert settings.disable_critique is False
    assert settings.disable_interstitial is False


def test_critique_kill_switch_set(monkeypatch):
    settings = _reload_config_with_env(monkeypatch, ONGIINI_DISABLE_CRITIQUE="true")
    assert settings.disable_critique is True


def test_interstitial_kill_switch_set(monkeypatch):
    settings = _reload_config_with_env(monkeypatch, ONGIINI_DISABLE_INTERSTITIAL="yes")
    assert settings.disable_interstitial is True


def test_kill_switch_recognises_uppercase_and_mixed_case(monkeypatch):
    """env vars are commonly set in uppercase or with mixed case — be
    permissive."""
    settings = _reload_config_with_env(monkeypatch, ONGIINI_DISABLE_CRITIQUE="TRUE")
    assert settings.disable_critique is True


def test_kill_switch_ignores_falsy_values(monkeypatch):
    """Empty / "0" / "false" must NOT disable. We want explicit opt-in
    to disabling, not accidental disabling from a typo'd env var."""
    settings = _reload_config_with_env(monkeypatch, ONGIINI_DISABLE_CRITIQUE="0")
    assert settings.disable_critique is False

    settings = _reload_config_with_env(monkeypatch, ONGIINI_DISABLE_CRITIQUE="")
    assert settings.disable_critique is False

    settings = _reload_config_with_env(monkeypatch, ONGIINI_DISABLE_CRITIQUE="no")
    assert settings.disable_critique is False
