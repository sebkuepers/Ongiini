"""Tests for ReviseEvalCaptureHook — the v1.7 critique-revise validation
capture mechanism. Verifies the hook only fires when env is set, captures
the right data, and skips turns without ReviseStep.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from owela import CritiqueStep, InboundMessage, Policy, ReplyStep, ReviseStep, ToolStep, TurnContext

from ongiini.config import settings
from ongiini.hooks import ReviseEvalCaptureHook


@pytest.fixture
def tmp_capture_dir(tmp_path, monkeypatch):
    """Tmp capture dir + force capture on for this test."""
    monkeypatch.setattr(settings, "capture_revise_eval", True)
    return tmp_path


def _ctx(msg_id: str = "wamid_test", text: str = "compare 3 banks") -> TurnContext:
    msg = InboundMessage(
        user_id="+264u", msg_id=msg_id, text=text,
        content_parts=[{"type": "text", "text": text}],
        has_image=False, history=[],
    )
    return TurnContext(msg=msg, policy=Policy(name="search_deep"), runtime=MagicMock())


def _make_steps(
    *,
    compose: str = "compose draft",
    revised: str = "revised draft",
    critique_verdict: str = "REVISE",
    critique_reasons=None,
    with_tools: bool = False,
):
    steps = []
    if with_tools:
        ts = ToolStep(tool_name="web_search", result_len=4500)
        steps.append(ts)
    crit = CritiqueStep(verdict=critique_verdict, reasons=critique_reasons or ["claim X ungrounded"])
    crit.attrs["raw_critique"] = "1. ... FAIL: claim X"
    steps.append(crit)
    revise = ReviseStep()
    revise.attrs["compose_draft"] = compose
    revise.attrs["revised_reply"] = revised
    steps.append(revise)
    steps.append(ReplyStep(reply_len=len(revised), sent=True))
    return steps


# ---------- happy path ----------

@pytest.mark.asyncio
async def test_writes_capture_file_with_both_drafts(tmp_capture_dir):
    hook = ReviseEvalCaptureHook(base_dir=tmp_capture_dir)
    await hook.on_turn_complete(_make_steps(compose="A", revised="B"), _ctx("wamid_42"))
    path = tmp_capture_dir / "wamid_42.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["compose_draft"] == "A"
    assert data["revised_reply"] == "B"
    assert data["compose_len"] == 1
    assert data["revised_len"] == 1


@pytest.mark.asyncio
async def test_capture_includes_critique_metadata(tmp_capture_dir):
    hook = ReviseEvalCaptureHook(base_dir=tmp_capture_dir)
    steps = _make_steps(critique_reasons=["reason A", "reason B"])
    await hook.on_turn_complete(steps, _ctx("wamid_meta"))
    data = json.loads((tmp_capture_dir / "wamid_meta.json").read_text())
    assert data["critique_verdict"] == "REVISE"
    assert data["critique_reasons"] == ["reason A", "reason B"]
    assert "FAIL: claim X" in data["raw_critique"]


@pytest.mark.asyncio
async def test_capture_includes_user_question(tmp_capture_dir):
    """A reviewer can't judge revise-better-than-compose without the user
    question. So we capture it, accepting the PII trade-off documented
    in the hook's module docstring."""
    hook = ReviseEvalCaptureHook(base_dir=tmp_capture_dir)
    await hook.on_turn_complete(_make_steps(), _ctx(text="how many banks in Namibia?"))
    data = json.loads(list(tmp_capture_dir.glob("*.json"))[0].read_text())
    assert data["user_question"] == "how many banks in Namibia?"


@pytest.mark.asyncio
async def test_capture_includes_tool_metadata_not_bodies(tmp_capture_dir):
    """Tool metadata (name, result_len) is captured; the actual result
    text isn't, to keep capture files small."""
    hook = ReviseEvalCaptureHook(base_dir=tmp_capture_dir)
    steps = _make_steps(with_tools=True)
    await hook.on_turn_complete(steps, _ctx())
    data = json.loads(list(tmp_capture_dir.glob("*.json"))[0].read_text())
    assert data["tool_results"] == [
        {"name": "web_search", "result_len": 4500, "error": None}
    ]


# ---------- gating ----------

@pytest.mark.asyncio
async def test_no_capture_when_env_disabled(tmp_path, monkeypatch):
    """Default state: hook is a no-op."""
    monkeypatch.setattr(settings, "capture_revise_eval", False)
    hook = ReviseEvalCaptureHook(base_dir=tmp_path)
    await hook.on_turn_complete(_make_steps(), _ctx())
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.asyncio
async def test_no_capture_when_no_revise_step(tmp_capture_dir):
    """Compose-only turns (critique PASS) shouldn't write anything —
    nothing to compare."""
    hook = ReviseEvalCaptureHook(base_dir=tmp_capture_dir)
    crit = CritiqueStep(verdict="PASS", reasons=[])
    reply = ReplyStep(reply_len=42, sent=True)
    await hook.on_turn_complete([crit, reply], _ctx())
    assert list(tmp_capture_dir.glob("*.json")) == []


@pytest.mark.asyncio
async def test_no_capture_when_drafts_missing(tmp_capture_dir):
    """Defensive: if the ReviseStep somehow lacks compose_draft (e.g.
    pre-v1.7 code path), skip rather than write a half-populated record."""
    hook = ReviseEvalCaptureHook(base_dir=tmp_capture_dir)
    revise = ReviseStep()
    revise.attrs["revised_reply"] = "x"
    # No compose_draft key.
    await hook.on_turn_complete([revise], _ctx())
    assert list(tmp_capture_dir.glob("*.json")) == []


# ---------- robustness ----------

@pytest.mark.asyncio
async def test_unsafe_msg_id_sanitised_in_filename(tmp_capture_dir):
    """msg_id is used as filename — sanitiser strips path-traversal
    chars even though WhatsApp msg_ids are normally alphanumeric."""
    hook = ReviseEvalCaptureHook(base_dir=tmp_capture_dir)
    await hook.on_turn_complete(_make_steps(), _ctx(msg_id="../escape/badness"))
    files = list(tmp_capture_dir.glob("*.json"))
    assert len(files) == 1
    # No path separators in the resulting filename.
    assert "/" not in files[0].name


@pytest.mark.asyncio
async def test_hook_soft_fails_on_write_error(tmp_capture_dir, monkeypatch):
    """A broken filesystem must NOT crash the turn."""
    hook = ReviseEvalCaptureHook(base_dir=tmp_capture_dir)

    # Patch Path.write_text to raise.
    from pathlib import Path
    orig = Path.write_text

    def _raise(self, *a, **kw):
        raise OSError("disk full simulation")

    monkeypatch.setattr(Path, "write_text", _raise)
    try:
        # Should not raise.
        await hook.on_turn_complete(_make_steps(), _ctx())
    finally:
        monkeypatch.setattr(Path, "write_text", orig)
