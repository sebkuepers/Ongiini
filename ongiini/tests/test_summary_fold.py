"""The folded history must be persisted, so long chats don't re-summarise
(and break the prefix cache) on every turn."""

from __future__ import annotations

import pytest
from owela import ModelResponse

from ongiini import summary
from ongiini.config import settings


class _Model:
    def __init__(self):
        self.calls = 0

    async def complete(self, req):
        self.calls += 1
        return ModelResponse(content="The user is a student in Oshakati.")


class _Store:
    def __init__(self, history):
        self.data = {"u": list(history)}
        self.saves = 0

    def load(self, msisdn):
        return list(self.data[msisdn])

    def save(self, msisdn, messages):
        self.saves += 1
        self.data[msisdn] = list(messages)


def _turns(n):
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"q{i}"})
        out.append({"role": "assistant", "content": f"a{i}"})
    return out


@pytest.mark.asyncio
async def test_fold_is_saved_and_not_repeated(monkeypatch):
    model = _Model()
    monkeypatch.setattr(summary, "_model", model)
    store = _Store(_turns(40))                      # 80 entries > threshold 70

    first = await summary.load_history_folded("u", load=store.load, save=store.save)
    assert model.calls == 1 and store.saves == 1
    assert first[0]["role"] == "system" and first[0]["content"].startswith(summary.SUMMARY_PREFIX)
    assert len(first) == 1 + settings.memory_keep_recent

    # Next turn: the stored history is the folded one plus a new exchange.
    store.data["u"] += _turns(1)
    second = await summary.load_history_folded("u", load=store.load, save=store.save)
    assert model.calls == 1                         # no re-summarise
    assert store.saves == 1
    assert second[: len(first)] == first            # stable prefix → cacheable


@pytest.mark.asyncio
async def test_short_history_is_left_alone(monkeypatch):
    model = _Model()
    monkeypatch.setattr(summary, "_model", model)
    store = _Store(_turns(5))
    out = await summary.load_history_folded("u", load=store.load, save=store.save)
    assert out == _turns(5)
    assert model.calls == 0 and store.saves == 0
