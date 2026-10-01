"""stats.nightly: fixed-category classification for /statistics."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pytest
from owela import ModelResponse

from ongiini.config import settings
from ongiini.stats import nightly


class FakeModel:
    """Answers message batches with cat=school / lang=en / a topic phrase,
    and user batches with fixed WHO values."""

    def __init__(self, topic="grade 11 chemistry homework"):
        self.calls = 0
        self.topic = topic

    async def complete(self, req):
        self.calls += 1
        prompt = req.messages[0]["content"]
        n = len(re.findall(r"^\d+\. ", prompt.split("Messages:\n")[-1].split("Users:\n")[-1], re.M))
        if '"labels"' in prompt and "Messages:" in prompt:
            body = {"labels": [{"i": i + 1, "cat": "school", "lang": "en", "topic": self.topic}
                               for i in range(n)]}
        else:
            body = {"users": [{"i": i + 1, "role": "student", "region": "Oshana",
                               "family": "none", "situation": "studying"} for i in range(n)]}
        return ModelResponse(content=json.dumps(body))


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(nightly, "_PAUSE_S", 0)
    return tmp_path


def _history(dir_, msisdn, texts):
    entries = []
    for t in texts:
        entries += [{"role": "user", "content": t}, {"role": "assistant", "content": "ok"}]
    (dir_ / f"{msisdn}.json").write_text(json.dumps(entries))


def _facts(msisdn):
    return [{"memory": "[PROFILE] studies nursing at university"},
            {"memory": "[SITUATION] preparing for exams"}]


def _read(dir_, name):
    return json.loads((dir_ / f"synthesis-{name}.json").read_text())


@pytest.mark.asyncio
async def test_run_writes_fixed_category_outputs(data_dir):
    _history(data_dir, "264810000001", ["help me with my chemistry homework please"] * 1
             + ["what is photosynthesis exactly", "explain osmosis to me"])
    _history(data_dir, "264810000002", ["can you check my essay on water"])
    model = FakeModel()
    report = await nightly.run_once(model.complete, _facts, frozenset())

    topics = _read(data_dir, "topics")
    assert topics["method"].startswith("Fixed categories")
    assert topics["clusters"][0]["label"] == "School & homework"
    assert topics["total_items_analysed"] == 4
    assert _read(data_dir, "languages")["clusters"] == [
        {"label": "English", "summary": "", "count": 2, "items": []}]
    assert _read(data_dir, "roles")["clusters"][0]["label"] == "University & college students"
    assert _read(data_dir, "regions")["clusters"][0]["label"] == "Oshana"
    assert _read(data_dir, "family")["clusters"] == []          # "none" is never counted
    assert report["messages_new"] == 4 and report["users_new"] == 2


@pytest.mark.asyncio
async def test_second_run_only_classifies_new_items(data_dir):
    _history(data_dir, "264810000001", ["what is photosynthesis exactly"])
    model = FakeModel()
    await nightly.run_once(model.complete, _facts, frozenset())
    first_calls = model.calls
    report = await nightly.run_once(model.complete, _facts, frozenset())
    assert model.calls == first_calls                            # everything cached
    assert report["messages_pending"] == 0 and report["users_pending"] == 0


@pytest.mark.asyncio
async def test_deleted_and_objecting_users_drop_out_of_the_counts(data_dir):
    _history(data_dir, "264810000001", ["what is photosynthesis exactly"])
    _history(data_dir, "264810000002", ["explain osmosis to me please"])
    model = FakeModel()
    await nightly.run_once(model.complete, _facts, frozenset())
    assert _read(data_dir, "topics")["total_items_analysed"] == 2

    (data_dir / "264810000001.json").unlink()                    # delete_my_data
    await nightly.run_once(model.complete, _facts, frozenset({"264810000002"}))  # objection
    assert _read(data_dir, "topics")["total_items_analysed"] == 0
    assert _read(data_dir, "roles")["total_items_analysed"] == 0


@pytest.mark.asyncio
async def test_identifying_topic_phrases_are_never_published(data_dir, monkeypatch):
    monkeypatch.setattr(settings, "stats_minimum_bucket", 1)
    _history(data_dir, "264810000001", ["my exam results for 2026 came back"])
    model = FakeModel(topic="exam results 2026")                 # 4-digit number → dropped
    await nightly.run_once(model.complete, _facts, frozenset())
    top = json.loads((data_dir / "synthesis-top_topics.json").read_text())
    assert top["labels"] == []
    assert _read(data_dir, "topics")["total_items_analysed"] == 1   # category still counted


@pytest.mark.asyncio
async def test_top_topics_respect_the_privacy_floor(data_dir, monkeypatch):
    monkeypatch.setattr(settings, "stats_minimum_bucket", 5)
    _history(data_dir, "264810000001", [f"question number {w} about cells" for w in "abcdef"])
    await nightly.run_once(FakeModel().complete, _facts, frozenset())
    top = json.loads((data_dir / "synthesis-top_topics.json").read_text())
    assert top["labels"] == [{"label": "grade 11 chemistry homework", "count": 6}]


@pytest.mark.asyncio
async def test_failed_batch_is_retried_next_run(data_dir):
    _history(data_dir, "264810000001", ["what is photosynthesis exactly"])

    async def broken(req):
        raise TimeoutError("vllm busy")

    report = await nightly.run_once(broken, _facts, frozenset())
    assert report["messages_new"] == 0
    report = await nightly.run_once(FakeModel().complete, _facts, frozenset())
    assert report["messages_new"] == 1


def test_seconds_until_next_run():
    at_2359 = datetime(2026, 10, 1, 23, 59, tzinfo=timezone.utc)
    assert nightly.seconds_until_next_run(at_2359) == 61 * 60
    at_0100 = datetime(2026, 10, 1, 1, 0, tzinfo=timezone.utc)
    assert nightly.seconds_until_next_run(at_0100) == 24 * 3600


def test_aggregator_prefers_the_nightly_top_topics_file(data_dir):
    from ongiini.stats import aggregator
    (data_dir / "synthesis-top_topics.json").write_text(json.dumps({
        "generated_at": "2026-10-02T01:05:00+00:00", "n_distinct": 40,
        "labels": [{"label": "cv improvement", "count": 9}],
    }))
    block = aggregator._top_topics_block()
    assert block["labels"] == [{"label": "cv improvement", "count": 9}]
    assert block["n_distinct"] == 40


def test_catch_up_only_at_night():
    assert nightly.in_night_window(datetime(2026, 10, 1, 23, 30, tzinfo=timezone.utc))
    assert nightly.in_night_window(datetime(2026, 10, 2, 2, 0, tzinfo=timezone.utc))
    assert not nightly.in_night_window(datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc))


@pytest.mark.asyncio
async def test_vague_phrases_are_not_top_topics(data_dir, monkeypatch):
    monkeypatch.setattr(settings, "stats_minimum_bucket", 1)
    _history(data_dir, "264810000001", ["hello can you help me with something"])
    await nightly.run_once(FakeModel(topic="general inquiry").complete, _facts, frozenset())
    top = json.loads((data_dir / "synthesis-top_topics.json").read_text())
    assert top["labels"] == []
