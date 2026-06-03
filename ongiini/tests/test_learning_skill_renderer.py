"""Tests for the per-language skill renderer.

Locks in:
- Placeholder substitution actually happens (no <<…>> markers leak).
- Per-target anchor gets included in the rendered text.
- Invalid pairs (same source/target, unknown code) raise.
- Cache works (template/anchor files read once).
"""
from __future__ import annotations

import pytest

from ongiini.learning import skill_renderer as sr


def test_render_substitutes_target_and_source_names():
    out = sr.render_skill_for_pair(source="english", target="afrikaans")
    # Display names replace the placeholders.
    assert "Afrikaans" in out
    assert "English" in out
    # No placeholder markers leak through.
    assert "<<TARGET_LANGUAGE>>" not in out
    assert "<<SOURCE_LANGUAGE>>" not in out
    assert "<<TARGET_ANCHOR>>" not in out


def test_render_includes_target_anchor_content():
    """The Afrikaans anchor mentions 'baie dankie' and 'Goeie môre'.
    Different target = different anchor visible in the rendered text."""
    af = sr.render_skill_for_pair(source="english", target="afrikaans")
    de = sr.render_skill_for_pair(source="english", target="german")
    en = sr.render_skill_for_pair(source="afrikaans", target="english")
    assert "Baie dankie" in af or "baie dankie" in af.lower()
    assert "Guten Morgen" in de or "Vielen Dank" in de
    # English anchor has 'Pleased to meet you' / 'Tell me about yourself'.
    assert "Tell me about yourself" in en


def test_render_swaps_source_in_lesson_card_shape():
    """The lesson card JSON shape references SOURCE_LANGUAGE for the
    teach-step body language — confirm the right name appears in that
    section regardless of target."""
    out = sr.render_skill_for_pair(source="german", target="afrikaans")
    # The concept step body says "short sentences in German" after
    # placeholder substitution.
    assert "sentences in German" in out


def test_unsupported_source_raises():
    with pytest.raises(ValueError, match="unsupported source_language"):
        sr.render_skill_for_pair(source="klingon", target="afrikaans")


def test_unsupported_target_raises():
    with pytest.raises(ValueError, match="unsupported target_language"):
        sr.render_skill_for_pair(source="english", target="esperanto")


def test_same_source_and_target_rejected():
    """Learning English-from-English is nonsensical; reject at the
    boundary so no nonsensical curriculum is ever designed."""
    with pytest.raises(ValueError, match="must differ"):
        sr.render_skill_for_pair(source="english", target="english")


def test_validate_language_pair_accepts_all_supported_pairs():
    """All 6 forward pairs (3 targets x 2 non-self sources) must
    validate cleanly."""
    for src in sr.SUPPORTED_LANGUAGES:
        for tgt in sr.SUPPORTED_LANGUAGES:
            if src == tgt:
                continue
            sr.validate_language_pair(src, tgt)   # must not raise


def test_supported_languages_set():
    """Lock the supported set so adding a new language is a deliberate
    change (anchor file + this set + UI)."""
    assert set(sr.SUPPORTED_LANGUAGES) == {"afrikaans", "english", "german"}


def test_display_names_match_supported_languages():
    """Every supported language has a display name — otherwise the
    rendered prompt would have an empty token."""
    for code in sr.SUPPORTED_LANGUAGES:
        assert code in sr.LANGUAGE_DISPLAY
        assert sr.LANGUAGE_DISPLAY[code]      # non-empty


def test_anchor_file_uses_target_not_source():
    """The anchor is keyed to the TARGET language — same source could
    be paired with different targets and each must show that target's
    anchor."""
    eng_to_af = sr.render_skill_for_pair(source="english", target="afrikaans")
    eng_to_de = sr.render_skill_for_pair(source="english", target="german")
    # Afrikaans-specific token shouldn't appear in the German render.
    assert "Goeie môre" not in eng_to_de
    assert "Guten Morgen" not in eng_to_af


def test_template_is_cached(monkeypatch):
    """Confirm the lru_cache means we don't re-read the template on
    every call — important under per-turn load."""
    sr.clear_cache()
    sr.render_skill_for_pair(source="english", target="afrikaans")
    sr.render_skill_for_pair(source="german", target="afrikaans")
    sr.render_skill_for_pair(source="afrikaans", target="english")
    info = sr._core_template.cache_info()
    # 3 calls but only one miss on the template.
    assert info.misses == 1
