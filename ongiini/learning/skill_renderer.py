"""Renderer for the language-pair learn skill.

Multi-language overhaul (Afrikaans · English · German):
The old single ``learning-afrikaans/SKILL.md`` is gone. In its place,
this module composes the skill text on demand from:

  * ``skills/_core.md.tmpl`` — language-agnostic JSON shapes, grading
    rubric, card-type definitions, personalisation guidance, "what to
    avoid" rules. Single source of truth for the rules.
  * ``skills/anchors/<target>.md`` — per-target-language anchor
    vocabulary table + cultural notes. Three files for three
    languages; adding a 4th = one new anchor file.

Render via :func:`render_skill_for_pair` for a given (source, target)
language pair. The output is plain text the API layer embeds in the
LLM system prompt (same shape as the old SKILL.md string — drop-in
replacement).

Placeholder syntax: ``<<TARGET_LANGUAGE>>``, ``<<SOURCE_LANGUAGE>>``,
``<<TARGET_ANCHOR>>``. Angle-bracket-doubled instead of ``{...}`` so
JSON code blocks inside the template don't collide with ``str.format``.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("ongiini.learning.skill_renderer")


# ──────────────────────────────────────────────────────────────────
# Supported language set + display names
# ──────────────────────────────────────────────────────────────────

LANG_AFRIKAANS = "afrikaans"
LANG_ENGLISH = "english"
LANG_GERMAN = "german"

SUPPORTED_LANGUAGES: tuple[str, ...] = (LANG_AFRIKAANS, LANG_ENGLISH, LANG_GERMAN)

# Display names — used in prompts, drawer titles, and the off-topic
# redirect. Stored separately from the canonical short codes so we can
# keep the codes lowercase + ASCII without losing presentation.
LANGUAGE_DISPLAY: dict[str, str] = {
    LANG_AFRIKAANS: "Afrikaans",
    LANG_ENGLISH: "English",
    LANG_GERMAN: "German",
}


def is_supported_language(code: str | None) -> bool:
    """True if ``code`` is one of the canonical short codes."""
    return isinstance(code, str) and code in SUPPORTED_LANGUAGES


def validate_language_pair(source: str | None, target: str | None) -> None:
    """Raise ``ValueError`` if either code is unknown or the pair is
    invalid (source == target). Used at every place a learner can
    introduce a language pair — the API request handlers + store
    helpers — so a bad pair fails at the boundary, not deep inside a
    prompt builder."""
    if not is_supported_language(source):
        raise ValueError(
            f"unsupported source_language: {source!r}; "
            f"must be one of {SUPPORTED_LANGUAGES}"
        )
    if not is_supported_language(target):
        raise ValueError(
            f"unsupported target_language: {target!r}; "
            f"must be one of {SUPPORTED_LANGUAGES}"
        )
    if source == target:
        raise ValueError(
            f"source_language and target_language must differ; "
            f"both were {source!r}"
        )


# ──────────────────────────────────────────────────────────────────
# Template + anchor loading
# ──────────────────────────────────────────────────────────────────

_SKILLS_DIR = Path(__file__).resolve().parent / "skills"
_CORE_TEMPLATE_PATH = _SKILLS_DIR / "_core.md.tmpl"
_ANCHORS_DIR = _SKILLS_DIR / "anchors"


@lru_cache(maxsize=1)
def _core_template() -> str:
    """Read the language-agnostic core template once at first call.
    Cached for process lifetime — the file doesn't change at runtime."""
    return _CORE_TEMPLATE_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=len(SUPPORTED_LANGUAGES))
def _anchor_for(language: str) -> str:
    """Read the per-target anchor file. Cached per-language so we
    don't re-read on every turn."""
    path = _ANCHORS_DIR / f"{language}.md"
    if not path.exists():
        log.warning("skill_renderer: missing anchor for %s at %s", language, path)
        return "## Anchor vocabulary\n\n(no anchor data for this language)"
    return path.read_text(encoding="utf-8").strip()


# ──────────────────────────────────────────────────────────────────
# Public render API
# ──────────────────────────────────────────────────────────────────

def render_skill_for_pair(*, source: str, target: str) -> str:
    """Return the full skill text rendered for one (source, target)
    language pair. Validates the pair before rendering — callers can
    rely on the result being non-empty + properly substituted.

    The result is the exact string the API layer used to read from
    the old SKILL.md file, so downstream prompt builders (curriculum
    / cards / grading / coach.question_handler) don't need to change."""
    validate_language_pair(source, target)
    template = _core_template()
    src_name = LANGUAGE_DISPLAY[source]
    tgt_name = LANGUAGE_DISPLAY[target]
    anchor = _anchor_for(target)
    rendered = (
        template
        .replace("<<TARGET_LANGUAGE>>", tgt_name)
        .replace("<<SOURCE_LANGUAGE>>", src_name)
        .replace("<<TARGET_ANCHOR>>", anchor)
    )
    return rendered


def clear_cache() -> None:
    """Test helper — wipe the lru_caches so a temp-file template can
    be reloaded between tests."""
    _core_template.cache_clear()
    _anchor_for.cache_clear()
