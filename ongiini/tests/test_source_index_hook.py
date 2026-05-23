"""Unit tests for SourceIndexHook.

Walks fake step lists with web_search / fetch_url / fetch_urls
ToolSteps, asserts the URLs land in the user's source_index.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from owela import InboundMessage, Policy, ToolStep, TurnContext

from ongiini.config import settings
from ongiini.hooks import SourceIndexHook
from ongiini.memory import source_index


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


def _ctx(user_id: str = "+264811234567") -> TurnContext:
    msg = InboundMessage(
        user_id=user_id, msg_id="m", text="q",
        content_parts=[{"type": "text", "text": "q"}],
        has_image=False, history=[],
    )
    policy = Policy(name="search_deep")
    return TurnContext(msg=msg, policy=policy, runtime=MagicMock())


def _web_search_step(urls: list[str]) -> ToolStep:
    """ToolStep mirroring how the executor stashes web_search results."""
    s = ToolStep(tool_name="web_search")
    s.attrs["urls"] = list(urls)
    return s


def _fetch_urls_step(urls: list[str]) -> ToolStep:
    """ToolStep mirroring fetch_urls — the URLs are embedded in the
    result body via '## <url>' headers."""
    s = ToolStep(tool_name="fetch_urls")
    s.attrs["result"] = "\n\n".join(f"## {u}\n<body>" for u in urls)
    return s


# ---------- happy path ----------

@pytest.mark.asyncio
async def test_web_search_urls_are_persisted(tmp_data_dir):
    hook = SourceIndexHook()
    step = _web_search_step([
        "https://www.namibian.com.na/article1",
        "https://www.allgemeine-zeitung.com.na/post",
    ])
    await hook.on_turn_complete([step], _ctx())
    out = source_index.load("+264811234567")
    urls = {e["url"] for e in out}
    assert urls == {
        "https://www.namibian.com.na/article1",
        "https://www.allgemeine-zeitung.com.na/post",
    }


@pytest.mark.asyncio
async def test_fetch_urls_inline_urls_are_extracted(tmp_data_dir):
    hook = SourceIndexHook()
    step = _fetch_urls_step([
        "https://www.namibian.com.na/article1",
        "https://www.allgemeine-zeitung.com.na/post",
    ])
    await hook.on_turn_complete([step], _ctx())
    urls = {e["url"] for e in source_index.load("+264811234567")}
    assert "https://www.namibian.com.na/article1" in urls
    assert "https://www.allgemeine-zeitung.com.na/post" in urls


@pytest.mark.asyncio
async def test_mixed_tools_in_one_turn_all_collected(tmp_data_dir):
    hook = SourceIndexHook()
    search = _web_search_step(["https://search-result.example/page"])
    fetch = _fetch_urls_step(["https://fetched.example/article"])
    await hook.on_turn_complete([search, fetch], _ctx())
    urls = {e["url"] for e in source_index.load("+264811234567")}
    assert urls == {
        "https://search-result.example/page",
        "https://fetched.example/article",
    }


# ---------- filtering / robustness ----------

@pytest.mark.asyncio
async def test_unrelated_tools_are_ignored(tmp_data_dir):
    """delete_my_data and lookup_ongiini_docs aren't search tools and
    their results shouldn't be source-indexed even if they happen to
    contain URLs."""
    hook = SourceIndexHook()
    irrelevant = ToolStep(tool_name="lookup_ongiini_docs")
    irrelevant.attrs["result"] = "## https://this-should-not-be-indexed.example"
    await hook.on_turn_complete([irrelevant], _ctx())
    assert source_index.load("+264811234567") == []


@pytest.mark.asyncio
async def test_no_tool_steps_is_noop(tmp_data_dir):
    """A pure-chat NONE turn has no ToolSteps — hook must not crash."""
    hook = SourceIndexHook()
    await hook.on_turn_complete([], _ctx())
    assert source_index.load("+264811234567") == []


@pytest.mark.asyncio
async def test_dedups_within_one_turn(tmp_data_dir):
    """If the same URL appears in both web_search.urls AND a follow-up
    fetch_urls result, store it once."""
    hook = SourceIndexHook()
    url = "https://www.namibian.com.na/article1"
    search = _web_search_step([url])
    fetch = _fetch_urls_step([url])
    await hook.on_turn_complete([search, fetch], _ctx())
    out = source_index.load("+264811234567")
    assert len(out) == 1
    assert out[0]["url"] == url


@pytest.mark.asyncio
async def test_hook_soft_fails_on_storage_error(tmp_data_dir, monkeypatch):
    """If source_index.append raises, the hook must NOT propagate the
    exception — broken persistence must never crash a successful reply."""
    def _raise(*_args, **_kwargs):
        raise OSError("disk full simulation")

    monkeypatch.setattr(source_index, "append", _raise)
    hook = SourceIndexHook()
    await hook.on_turn_complete([_web_search_step(["https://a.example"])], _ctx())
    # No exception escaped — that's the assertion.
