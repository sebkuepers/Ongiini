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

def test_drill_phase_starts_when_all_lessons_taught_and_story_done():
    """Once every lesson topic in the module has at least
    TARGET_LESSONS_PER_TOPIC lessons AND the module's single story has
    been emitted, drill begins on the first practice topic. The
    rotation now starts at slot 0 which is RECOGNITION
    (multiple_choice) — the input→output gradient from Track E."""
    outline = _outline([_module()])
    digest = {"mod-1": {
        "topics_taught": {"l1": 1, "l2": 1},
        "topics_drilled": {},
        "stories_emitted": 1,
    }}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    assert sel.phase == "drill"
    assert sel.card_type == selector.EXERCISE_TYPE_ROTATION[0]
    assert sel.topic_id == "p1"


def test_drill_first_exercise_per_topic_is_recognition():
    """The 0th drill of a module uses the first card_type in the
    rotation. Track E makes that recognition (multiple_choice)."""
    outline = _outline([_module(lesson_topics=0, practice_topics=2)])
    digest = {"mod-1": {"topics_taught": {}, "topics_drilled": {}}}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    assert sel.card_type == selector.EXERCISE_TYPE_ROTATION[0]
    # Documented intent: first drill is recognition, not production.
    assert sel.card_type == CARD_MULTIPLE_CHOICE


def test_drill_second_exercise_in_module_is_next_in_rotation():
    """The 1st drill in the module uses rotation slot 1 — proving the
    round-robin. The rotation index is the module-level total of
    drills, not per-topic."""
    outline = _outline([_module(lesson_topics=0, practice_topics=2)])
    digest = {"mod-1": {
        "topics_taught": {},
        "topics_drilled": {"p1": 1},
    }}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    assert sel.topic_id == "p1"
    assert sel.card_type == selector.EXERCISE_TYPE_ROTATION[1]


def test_drill_rotates_per_module_not_per_topic():
    """Per-module rotation: drilling 4 practice topics in a row
    should cycle through 4 different card_types, not return the
    first type every time. Each new topic gets the NEXT type in
    the rotation."""
    outline = _outline([_module(lesson_topics=0, practice_topics=4)])
    rot = selector.EXERCISE_TYPE_ROTATION
    sel = selector.select_next_card(
        outline=outline,
        module_digest={"mod-1": {"topics_taught": {}, "topics_drilled": {}}},
    )
    assert sel.topic_id == "p1" and sel.card_type == rot[0]

    sel = selector.select_next_card(
        outline=outline,
        module_digest={"mod-1": {
            "topics_taught": {},
            "topics_drilled": {"p1": 2},
        }},
    )
    assert sel.topic_id == "p2"
    assert sel.card_type == rot[2]   # rotation slot 2

    sel = selector.select_next_card(
        outline=outline,
        module_digest={"mod-1": {
            "topics_taught": {},
            "topics_drilled": {"p1": 2, "p2": 2},
        }},
    )
    assert sel.topic_id == "p3"
    assert sel.card_type == rot[4]


def test_drill_advances_to_next_practice_topic_after_quota():
    """When p1's drill quota is met, drill moves to p2. Per-module
    rotation means p2's first drill is rotation[2], not rotation[0]."""
    outline = _outline([_module(lesson_topics=0, practice_topics=2)])
    digest = {"mod-1": {
        "topics_taught": {},
        "topics_drilled": {"p1": selector.TARGET_DRILLS_PER_PRACTICE_TOPIC},
    }}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    assert sel.topic_id == "p2"
    assert sel.card_type == selector.EXERCISE_TYPE_ROTATION[2]


# ──────────────────────────────────────────────────────────────────
# Recycle phase
# ──────────────────────────────────────────────────────────────────

def test_recycle_picks_taught_lesson_topic_with_fewest_drills():
    """When all practice topics have hit their drill quota, the
    selector recycles: drill a taught lesson topic. Tie-break is
    outline order; pick the topic with the fewest recycled drills
    so the recycling spaces evenly. The recycle cap is now the
    same as TARGET_DRILLS_PER_PRACTICE_TOPIC (was 2× before).
    ``stories_emitted: 1`` skips the new story phase that fires
    between teach and drill."""
    outline = _outline([_module()])
    digest = {"mod-1": {
        "topics_taught": {"l1": 1, "l2": 1},
        "topics_drilled": {
            "p1": selector.TARGET_DRILLS_PER_PRACTICE_TOPIC,
            "p2": selector.TARGET_DRILLS_PER_PRACTICE_TOPIC,
            # Neither lesson topic has been recycled yet; pick l1 by
            # outline order tie-break.
        },
        "stories_emitted": 1,
    }}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    assert sel.phase == "recycle"
    assert sel.topic_id == "l1"


# ──────────────────────────────────────────────────────────────────
# Advance / graduation
# ──────────────────────────────────────────────────────────────────

def test_advance_first_when_module_fully_drilled_and_recycled():
    """When even recycling has hit its cap, the selector tells the
    coach to advance the module. ``stories_emitted: 1`` skips the
    new story phase."""
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
        "stories_emitted": 1,
    }}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    assert sel.advance_first is True
    assert sel.card_type is None


# ──────────────────────────────────────────────────────────────────
# Story phase — comprehensible input (Track A)
# ──────────────────────────────────────────────────────────────────

def test_story_phase_fires_after_first_lesson_topic_taught():
    """The comprehensible-input track: once the first lesson topic is
    taught, the next selector emission is a STORY card — NOT a drill.
    This is the load-bearing pacing change for Track A."""
    from ongiini.learning.db import CARD_STORY
    outline = _outline([_module()])
    digest = {"mod-1": {
        # First lesson topic taught; second one not yet.
        "topics_taught": {"l1": 1},
        "topics_drilled": {},
        "stories_emitted": 0,
    }}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    # NOTE: with l2 still untaught the teach phase wins over story —
    # story only fires once every lesson topic has at least one lesson.
    assert sel.phase == "teach"
    assert sel.topic_id == "l2"

    # Now teach l2 too; story should fire next.
    digest["mod-1"]["topics_taught"] = {"l1": 1, "l2": 1}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    assert sel.phase == "story"
    assert sel.card_type == CARD_STORY
    # Story binds to the first lesson topic (situational anchor).
    assert sel.topic_id == "l1"


def test_story_phase_emits_exactly_once_per_module():
    """After the story has been emitted (stories_emitted: 1), the
    selector advances to drill — never a second story in the same
    module."""
    outline = _outline([_module()])
    digest = {"mod-1": {
        "topics_taught": {"l1": 1, "l2": 1},
        "topics_drilled": {},
        "stories_emitted": 1,
    }}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    assert sel.phase == "drill"
    # First drill is rotation[0] — see Track E pedagogical sequencing.
    assert sel.card_type == selector.EXERCISE_TYPE_ROTATION[0]


def test_story_phase_skipped_when_module_has_no_lesson_topics():
    """A module with only practice topics has no anchor for a story
    (no lesson taught yet for the story to ground in). Skip story,
    go straight to drill."""
    outline = _outline([_module(lesson_topics=0, practice_topics=2)])
    digest = {"mod-1": {
        "topics_taught": {}, "topics_drilled": {}, "stories_emitted": 0,
    }}
    sel = selector.select_next_card(outline=outline, module_digest=digest)
    assert sel.phase == "drill"
    assert sel.card_type == selector.EXERCISE_TYPE_ROTATION[0]


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

@pytest.mark.parametrize("count", [0, 1, 2, 3, len(selector.EXERCISE_TYPE_ROTATION)])
def test_pick_exercise_type_round_robin(count):
    """Round-robin against the canonical rotation tuple. The
    pedagogically-ordered (Track E) rotation reads:
    multiple_choice → vocab → cloze → translation → grammar → dialogue
    then wraps. Asserting against indices keeps the test stable when
    we tune the order again."""
    rot = selector.EXERCISE_TYPE_ROTATION
    expected = rot[count % len(rot)]
    assert selector.pick_exercise_type(count) == expected
