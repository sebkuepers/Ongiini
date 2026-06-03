"""Tests for the conversation module (Track C — chat mode).

Mirrors the card_critic / curriculum_critic test patterns. The
conversation module is soft-fail by design — a bad model response or
crash returns a ConversationTurn(reply='') and the API layer surfaces
a friendly "couldn't reply" coach_text. Nothing here should raise."""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from owela import ModelRequest, ModelResponse

from ongiini.learning import conversation as cv
from ongiini.learning import context as ctx_mod
from ongiini.learning import db


@dataclass
class FakeModel:
    response: str = ""
    raise_exc: Exception | None = None
    last_request: ModelRequest | None = None

    async def complete(self, req: ModelRequest) -> ModelResponse:
        self.last_request = req
        if self.raise_exc:
            raise self.raise_exc
        return ModelResponse(
            content=self.response, tool_calls=[],
            finish_reason="stop",
            tokens_in=5, tokens_out=5, cached_tokens=0, raw=None,
        )


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    from ongiini.learning import db as dbmod
    from ongiini import config
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    dbmod.warmup()
    return tmp_path / "learning.sqlite"


def _ctx(temp_db):
    from ongiini.learning import store
    learner_id = store.create_anonymous_learner()
    store.save_profile_field(learner_id, "name", "Sebastian")
    store.save_profile_field(learner_id, "current_level", "beginner")
    store.save_profile_field(learner_id, "objective", "travel to Germany")
    store.mark_intake_complete(learner_id)
    return ctx_mod.build_learner_context(learner_id)


# ──────────────────────────────────────────────────────────────────
# Happy path + parsing
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_happy_reply_with_corrections_and_new_words(temp_db):
    fm = FakeModel(response=json.dumps({
        "reply": "Hallo Sebastian! Wie geht es dir heute?",
        "corrections": [
            {"learner": "ich habe ein Hund",
             "correct": "ich habe einen Hund",
             "note": "accusative after 'haben'"},
        ],
        "new_words": [
            {"word": "der Hund", "meaning": "dog"},
        ],
    }))
    out = await cv.chat_turn(
        _ctx(temp_db),
        user_text="Hallo! ich habe ein Hund.",
        history=[],
        model=fm, skill_content="SKILL",
    )
    assert out.reply.startswith("Hallo Sebastian!")
    assert len(out.corrections) == 1
    assert out.corrections[0]["correct"] == "ich habe einen Hund"
    assert out.new_words == [{"word": "der Hund", "meaning": "dog"}]


@pytest.mark.asyncio
async def test_empty_notes_blocks_default_to_empty_lists(temp_db):
    """A clean exchange — coach replies but has nothing worth
    correcting and no new word to introduce. Empty arrays, not
    missing keys, are normal."""
    fm = FakeModel(response=json.dumps({
        "reply": "Sehr gut!",
        "corrections": [],
        "new_words": [],
    }))
    out = await cv.chat_turn(
        _ctx(temp_db),
        user_text="Mir geht es gut.",
        history=[],
        model=fm, skill_content="SKILL",
    )
    assert out.reply == "Sehr gut!"
    assert out.corrections == []
    assert out.new_words == []


@pytest.mark.asyncio
async def test_corrections_capped_at_three(temp_db):
    bigger = [
        {"learner": f"l{i}", "correct": f"c{i}", "note": f"n{i}"}
        for i in range(8)
    ]
    fm = FakeModel(response=json.dumps({
        "reply": "Gut.",
        "corrections": bigger,
        "new_words": [],
    }))
    out = await cv.chat_turn(
        _ctx(temp_db),
        user_text="x",
        history=[],
        model=fm, skill_content="SKILL",
    )
    assert len(out.corrections) == 3


@pytest.mark.asyncio
async def test_new_words_capped_at_three(temp_db):
    bigger = [
        {"word": f"w{i}", "meaning": f"m{i}"}
        for i in range(10)
    ]
    fm = FakeModel(response=json.dumps({
        "reply": "Gut.",
        "corrections": [],
        "new_words": bigger,
    }))
    out = await cv.chat_turn(
        _ctx(temp_db),
        user_text="x",
        history=[],
        model=fm, skill_content="SKILL",
    )
    assert len(out.new_words) == 3


@pytest.mark.asyncio
async def test_malformed_note_entries_dropped(temp_db):
    """Each notes entry must have the right keys with non-empty
    string values; bad entries are dropped, good ones kept."""
    fm = FakeModel(response=json.dumps({
        "reply": "OK",
        "corrections": [
            {"learner": "a", "correct": "b", "note": "c"},
            {"learner": "a"},                                   # missing keys
            {"learner": "", "correct": "x", "note": "y"},       # empty
            "not a dict",                                        # wrong type
        ],
        "new_words": [],
    }))
    out = await cv.chat_turn(
        _ctx(temp_db),
        user_text="x",
        history=[],
        model=fm, skill_content="SKILL",
    )
    assert len(out.corrections) == 1
    assert out.corrections[0] == {"learner": "a", "correct": "b", "note": "c"}


# ──────────────────────────────────────────────────────────────────
# Soft-fail paths — chat MUST NOT raise
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_user_text_returns_empty_reply(temp_db):
    fm = FakeModel(response="(should not be called)")
    out = await cv.chat_turn(
        _ctx(temp_db),
        user_text="   ",
        history=[],
        model=fm, skill_content="SKILL",
    )
    assert out.reply == ""
    # The model should NOT have been called for an empty turn —
    # we'd be burning tokens on nothing.
    assert fm.last_request is None


@pytest.mark.asyncio
async def test_model_crash_returns_empty_reply(temp_db):
    fm = FakeModel(raise_exc=RuntimeError("vLLM offline"))
    out = await cv.chat_turn(
        _ctx(temp_db),
        user_text="Hallo",
        history=[],
        model=fm, skill_content="SKILL",
    )
    assert out.reply == ""
    assert out.corrections == []
    assert out.new_words == []


@pytest.mark.asyncio
async def test_model_returns_garbage_json_returns_empty_reply(temp_db):
    fm = FakeModel(response="here's what i think...")
    out = await cv.chat_turn(
        _ctx(temp_db),
        user_text="Hallo",
        history=[],
        model=fm, skill_content="SKILL",
    )
    assert out.reply == ""


@pytest.mark.asyncio
async def test_model_returns_error_field_returns_empty_reply(temp_db):
    fm = FakeModel(response=json.dumps({"error": "couldn't decide"}))
    out = await cv.chat_turn(
        _ctx(temp_db),
        user_text="Hallo",
        history=[],
        model=fm, skill_content="SKILL",
    )
    assert out.reply == ""


@pytest.mark.asyncio
async def test_missing_reply_field_returns_empty_reply(temp_db):
    fm = FakeModel(response=json.dumps({
        "corrections": [], "new_words": [],
    }))
    out = await cv.chat_turn(
        _ctx(temp_db),
        user_text="Hallo",
        history=[],
        model=fm, skill_content="SKILL",
    )
    assert out.reply == ""


# ──────────────────────────────────────────────────────────────────
# History + system prompt routing
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_history_is_surfaced_to_the_user_prompt(temp_db):
    fm = FakeModel(response=json.dumps({
        "reply": "Ja, gerne.",
        "corrections": [],
        "new_words": [],
    }))
    history = [
        {"role": "learner", "text": "Hallo!"},
        {"role": "coach",   "text": "Hallo Sebastian! Wie geht es dir?"},
        {"role": "learner", "text": "Mir geht's gut. Möchtest du Kaffee?"},
    ]
    await cv.chat_turn(
        _ctx(temp_db),
        user_text="Sehr gut.",
        history=history,
        model=fm, skill_content="SKILL",
    )
    user_msg = fm.last_request.messages[1]["content"]
    assert "CONVERSATION SO FAR" in user_msg
    assert "Hallo!" in user_msg
    assert "Möchtest du Kaffee?" in user_msg


@pytest.mark.asyncio
async def test_history_window_caps_at_window_size(temp_db):
    """Older entries beyond the recent window are dropped — keeps
    the prompt budget predictable on long sessions. Asserts against
    the module-level constant so a future tuning of the window only
    requires bumping one number."""
    fm = FakeModel(response=json.dumps({
        "reply": "OK", "corrections": [], "new_words": [],
    }))
    n = cv._HISTORY_WINDOW + 10
    history = [
        {"role": "learner", "text": f"OLD_{i}"} for i in range(n)
    ]
    await cv.chat_turn(
        _ctx(temp_db),
        user_text="x",
        history=history,
        model=fm, skill_content="SKILL",
    )
    user_msg = fm.last_request.messages[1]["content"]
    # Most recent entry kept.
    assert f"OLD_{n - 1}" in user_msg
    # Entries within the recent window kept.
    assert f"OLD_{n - cv._HISTORY_WINDOW}" in user_msg
    # Entries past the window dropped.
    assert "OLD_0" not in user_msg


@pytest.mark.asyncio
async def test_history_char_budget_drops_oldest_first(temp_db):
    """A long-text turn in the recent window would otherwise blow
    the prompt budget; the char-budget guard drops from the OLDEST
    end so the most recent context is preserved."""
    fm = FakeModel(response=json.dumps({
        "reply": "OK", "corrections": [], "new_words": [],
    }))
    big = "X" * (cv._HISTORY_CHAR_BUDGET + 100)
    history = [
        {"role": "learner", "text": "FIRST"},
        {"role": "coach",   "text": big},
        {"role": "learner", "text": "LAST"},
    ]
    await cv.chat_turn(
        _ctx(temp_db),
        user_text="x",
        history=history,
        model=fm, skill_content="SKILL",
    )
    user_msg = fm.last_request.messages[1]["content"]
    # Most recent ("LAST") kept; the oversized middle entry gets
    # trimmed off; the FIRST entry (older still) also drops.
    assert "LAST" in user_msg
    assert "FIRST" not in user_msg


@pytest.mark.asyncio
async def test_system_prompt_carries_skill_content_and_focus(temp_db):
    fm = FakeModel(response=json.dumps({
        "reply": "Hallo", "corrections": [], "new_words": [],
    }))
    await cv.chat_turn(
        _ctx(temp_db),
        user_text="Hi",
        history=[],
        model=fm, skill_content="SKILL-MARKER",
    )
    sys_msg = fm.last_request.messages[0]["content"]
    assert "SKILL-MARKER" in sys_msg
    # The learner's stated focus must be in the system prompt so
    # the coach can anchor replies to "travel to Germany".
    assert "travel to Germany" in sys_msg


# ──────────────────────────────────────────────────────────────────
# build_history_from_messages
# ──────────────────────────────────────────────────────────────────

def test_build_history_filters_to_chat_kinds_only():
    """Cards-mode messages (lesson / exercise / feedback / coach_text)
    must NOT leak into the chat-mode history — that would confuse the
    coach into thinking the learner just got a vocab card."""
    rows = [
        {"kind": db.MSG_LESSON,         "payload": {"title": "Greetings"}},
        {"kind": db.MSG_EXERCISE,       "payload": {"prompt_text": "x?"}},
        {"kind": db.MSG_CHAT_LEARNER,   "payload": {"text": "Hallo"}},
        {"kind": db.MSG_CHAT_COACH,     "payload": {"reply": "Hallo!"}},
        {"kind": db.MSG_CHAT_NOTES,     "payload": {"corrections": [], "new_words": []}},
        {"kind": db.MSG_COACH_TEXT,     "payload": {"text": "Welcome back."}},
    ]
    out = cv.build_history_from_messages(rows)
    assert out == [
        {"role": "learner", "text": "Hallo"},
        {"role": "coach",   "text": "Hallo!"},
    ]
