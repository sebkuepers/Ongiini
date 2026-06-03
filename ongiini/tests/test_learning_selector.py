"""Selector tests — the deterministic decision table for "what card next?".

These tests don't touch the LLM at all. The selector is a pure
function over (outline, module_digest); these cases construct
synthetic shapes and assert the phase + chosen card_type + topic_id.

If a future bug shows up where the wrong topic gets emitted, the
fix lands here as a new test before any selector code changes.
"""
from __future__ import annotations

import pytest

from ongiini.learning import selector
from ongiini.learning.db import (
    CARD_CLOZE,
    CARD_LESSON,
    CARD_MULTIPLE_CHOICE,
    CARD_TRANSLATION,
    CARD_VOCAB,
)


def _module(
    *,
    id_: str = "mod-1",
    title: str = "Greetings",
    status: str = "in_progress",
    lesson_topics: int = 2,
    practice_topics: int = 2,
    extra_kwargs: dict | None = None,
) -> dict:
    topics = []
    for i in range(lesson_topics):
        topics.append({
            "id": f"l{i+1}", "title": f"Lesson topic {i+1}", "kind": "lesson",
        })
    for i in range(practice_topics):
        topics.append({
            "id": f"p{i+1}", "title": f"Practice topic {i+1}", "kind": "practice",
        })
    out = {
        "id": id_, "title": title, "status": status,
        "estimated_cards": 8,
        "topics": topics,
    }
    if extra_kwargs:
        out.update(extra_kwargs)
    return out


def _outline(modules):
    return {"summary": "test", "modules": modules}


# ──────────────────────────────────────────────────────────────────
# Teach phase
# ──────────────────────────────────────────────────────────────────

def test_teach_first_turn_picks_first_lesson_topic():
    """Fresh learner, empty digest → teach the first lesson topic."""
    outline = _outline([_module()])
    sel = selector.select_next_card(outline=outline, module_digest={})
    assert sel.phase == "teach"
    assert sel.card_type == CARD_LESSON
    assert sel.module_id == "mod-1"
    assert sel.topic_id == "l1"
    assert sel.topic_title == "Lesson topic 1"


def test_teach_after_first_lesson_picks_second_lesson_topic():
    """After topic l1 is taught, the next lesson goes to l2 (NOT a
    repeat of l1)."""
    outline = _outline([_module()])
    digest = {"mod-1": {"topics_taught": {"l1": 1}, "topics_drilled": {}}}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    assert sel.phase == "teach"
    assert sel.card_type == CARD_LESSON
    assert sel.topic_id == "l2"


def test_teach_skips_already_taught_topics_in_order():
    """If l1 is somehow skipped and l2 was taught (out of order),
    the selector still picks the first untaught topic in outline
    order — l1."""
    outline = _outline([_module(lesson_topics=3, practice_topics=0)])
    digest = {"mod-1": {"topics_taught": {"l2": 1}, "topics_drilled": {}}}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    assert sel.phase == "teach"
    assert sel.topic_id == "l1"


# ──────────────────────────────────────────────────────────────────
# Drill phase
# ──────────────────────────────────────────────────────────────────

def test_drill_phase_starts_when_all_lessons_taught():
    """Once every lesson topic in the module has at least
    TARGET_LESSONS_PER_TOPIC lessons, drill begins on the first
    practice topic."""
    outline = _outline([_module()])
    digest = {"mod-1": {
        "topics_taught": {"l1": 1, "l2": 1},
        "topics_drilled": {},
    }}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    assert sel.phase == "drill"
    assert sel.card_type == CARD_VOCAB   # first in rotation
    assert sel.topic_id == "p1"


def test_drill_first_exercise_per_topic_is_vocab():
    """The 0th drill of a topic uses the first card_type in the
    rotation (vocab)."""
    outline = _outline([_module(lesson_topics=0, practice_topics=2)])
    digest = {"mod-1": {"topics_taught": {}, "topics_drilled": {}}}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    assert sel.card_type == CARD_VOCAB


def test_drill_second_exercise_in_module_is_cloze():
    """The 1st drill in the module uses the second card_type in the
    rotation (cloze) — proving the round-robin. The rotation index is
    the module-level total of drills, not per-topic."""
    outline = _outline([_module(lesson_topics=0, practice_topics=2)])
    digest = {"mod-1": {
        "topics_taught": {},
        "topics_drilled": {"p1": 1},
    }}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    # p1 still has headroom (TARGET=2), and the module already has 1
    # drill, so the picker returns cloze (rotation slot 1).
    assert sel.topic_id == "p1"
    assert sel.card_type == CARD_CLOZE


def test_drill_rotates_per_module_not_per_topic():
    """Per-module rotation: drilling 4 practice topics in a row
    should cycle through 4 different card_types, not return vocab
    every time. Each new topic gets the NEXT type in the rotation."""
    outline = _outline([_module(lesson_topics=0, practice_topics=4)])
    # First drill in a fresh module — module total = 0 → vocab.
    sel = selector.select_next_card(
        outline=outline,
        module_digest={"mod-1": {"topics_taught": {}, "topics_drilled": {}}},
    )
    assert sel.topic_id == "p1" and sel.card_type == CARD_VOCAB

    # After p1 hits its quota (2 drills), the 3rd drill in the
    # module goes to p2 — and the rotation index is 2, so it's
    # translation (NOT vocab again).
    sel = selector.select_next_card(
        outline=outline,
        module_digest={"mod-1": {
            "topics_taught": {},
            "topics_drilled": {"p1": 2},
        }},
    )
    assert sel.topic_id == "p2"
    assert sel.card_type == CARD_TRANSLATION   # rotation slot 2

    # After p2 also done, module total = 4 → rotation slot 4 = grammar.
    sel = selector.select_next_card(
        outline=outline,
        module_digest={"mod-1": {
            "topics_taught": {},
            "topics_drilled": {"p1": 2, "p2": 2},
        }},
    )
    assert sel.topic_id == "p3"
    assert sel.card_type == selector.EXERCISE_TYPE_ROTATION[4]   # grammar


def test_drill_advances_to_next_practice_topic_after_quota():
    """When p1's drill quota is met, drill moves to p2. Under the
    per-module rotation, p2's first drill is NOT vocab again — the
    rotation index is the module-level total (here: 2), so the third
    drill in the module is translation."""
    outline = _outline([_module(lesson_topics=0, practice_topics=2)])
    digest = {"mod-1": {
        "topics_taught": {},
        "topics_drilled": {"p1": selector.TARGET_DRILLS_PER_PRACTICE_TOPIC},
    }}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    assert sel.topic_id == "p2"
    assert sel.card_type == CARD_TRANSLATION   # rotation index = 2


# ──────────────────────────────────────────────────────────────────
# Recycle phase
# ──────────────────────────────────────────────────────────────────

def test_recycle_picks_taught_lesson_topic_with_fewest_drills():
    """When all practice topics have hit their drill quota, the
    selector recycles: drill a taught lesson topic. Tie-break is
    outline order; pick the topic with the fewest recycled drills
    so the recycling spaces evenly. The recycle cap is now the
    same as TARGET_DRILLS_PER_PRACTICE_TOPIC (was 2× before)."""
    outline = _outline([_module()])
    digest = {"mod-1": {
        "topics_taught": {"l1": 1, "l2": 1},
        "topics_drilled": {
            "p1": selector.TARGET_DRILLS_PER_PRACTICE_TOPIC,
            "p2": selector.TARGET_DRILLS_PER_PRACTICE_TOPIC,
            # Neither lesson topic has been recycled yet; pick l1 by
            # outline order tie-break.
        },
    }}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    assert sel.phase == "recycle"
    assert sel.topic_id == "l1"


# ──────────────────────────────────────────────────────────────────
# Advance / graduation
# ──────────────────────────────────────────────────────────────────

def test_advance_first_when_module_fully_drilled_and_recycled():
    """When even recycling has hit its cap, the selector tells the
    coach to advance the module."""
    outline = _outline([_module()])
    cap = selector.TARGET_DRILLS_PER_PRACTICE_TOPIC
    digest = {"mod-1": {
        "topics_taught": {"l1": 1, "l2": 1},
        "topics_drilled": {
            "p1": selector.TARGET_DRILLS_PER_PRACTICE_TOPIC,
            "p2": selector.TARGET_DRILLS_PER_PRACTICE_TOPIC,
            "l1": cap,
            "l2": cap,
        },
    }}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    assert sel.advance_first is True
    assert sel.card_type is None


def test_graduation_when_no_in_progress_module():
    """Every module completed → graduation; coach emits a friendly
    "what next?" coach-text rather than a card."""
    outline = _outline([_module(status="completed")])
    sel = selector.select_next_card(outline=outline, module_digest={})
    assert sel.graduation is True
    assert sel.card_type is None


def test_graduation_when_outline_missing():
    """Permissive on a missing outline — graduate rather than
    crashing. The coach should never reach this path because the
    outline is designed before the first card, but be defensive."""
    sel = selector.select_next_card(outline=None, module_digest={})
    assert sel.graduation is True


# ──────────────────────────────────────────────────────────────────
# Card_type rotation
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("count,expected", [
    (0, CARD_VOCAB),
    (1, CARD_CLOZE),
    (2, CARD_TRANSLATION),
    (3, CARD_MULTIPLE_CHOICE),
    (len(selector.EXERCISE_TYPE_ROTATION), CARD_VOCAB),   # wraps round
])
def test_pick_exercise_type_round_robin(count, expected):
    assert selector.pick_exercise_type(count) == expected
