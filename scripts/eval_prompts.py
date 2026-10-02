"""Prompt templates for Ongiini-Eval-OW baselines — one place for every runner.

`ongiini-eval-ow-v1-zeroshot` is the template published in the concept
paper (arXiv:2609.31727, Appendix C) and the default for submissions.
It is the only template used for leaderboard rows.

`ongiini-eval-ow-v0-legacy` is the shorter template the May 2026 v0.1
baselines (Claude Opus 4.7, Gemma 4 26B) were generated with. Kept so
those outputs stay reproducible; not a leaderboard protocol.
"""
from __future__ import annotations

PAPER_ZEROSHOT_ID = "ongiini-eval-ow-v1-zeroshot"
PAPER_ZEROSHOT = """\
You are a professional translator translating English to {dialect} (an \
Oshiwambo language spoken in northern Namibia). Translate the following \
English sentence into natural, fluent {dialect}.

Only output the translation. Do not include explanations, glosses, or \
commentary.

English: {source}
{dialect}:"""

LEGACY_ID = "ongiini-eval-ow-v0-legacy"
LEGACY = """\
Translate the following English text into {dialect}.

Output ONLY the translation in {dialect}. Do not include the original \
English, any explanation, quotation marks, or romanisation notes.

English: {source}

{dialect}:"""

# Prompt-robustness variants (docs/eval-protocol.md §5). Diagnostic only —
# the leaderboard uses PAPER_ZEROSHOT. R23 is the zero-shot prompt of
# Robinson et al. (WMT 2023, after Gao et al. 2023); MIN is the bare
# completion format of AfroBench's first MT prompt.
R23_ID = "robustness-r23-zeroshot"
R23 = """\
This is an English to {dialect} translation, please provide the {dialect} \
translation for this sentence. Do not provide any explanations or text apart \
from the translation.
English: {source}
{dialect}:"""

MIN_ID = "robustness-min-zeroshot"
MIN = """\
English: {source}
{dialect}:"""

DIALECT_NAMES = {"oshindonga": "Oshindonga", "oshikwanyama": "Oshikwanyama"}

TEMPLATES = {PAPER_ZEROSHOT_ID: PAPER_ZEROSHOT, LEGACY_ID: LEGACY, R23_ID: R23, MIN_ID: MIN}

# Prompt screening (docs/eval-protocol.md §9, exploratory, 2026-10-02): each
# variant changes exactly one property of the paper prompt. Used to choose
# which properties the robustness check tests — never to pick the
# leaderboard prompt, which stays PAPER_ZEROSHOT.
_PAPER_NO_ROLE = PAPER_ZEROSHOT.replace(
    "You are a professional translator translating English to {dialect} (an "
    "Oshiwambo language spoken in northern Namibia). Translate", "Translate").replace(
    "sentence into natural, fluent {dialect}.",
    "sentence into natural, fluent {dialect} (an Oshiwambo language spoken in northern Namibia).")
SCREEN = {
    "screen-no-role": _PAPER_NO_ROLE,
    "screen-no-description": PAPER_ZEROSHOT.replace(
        " (an Oshiwambo language spoken in northern Namibia)", ""),
    "screen-iso-name": PAPER_ZEROSHOT,          # rendered with NAME_VARIANTS below
    # the umbrella name already says "Oshiwambo", so the description would repeat it
    "screen-umbrella-name": PAPER_ZEROSHOT.replace(
        " (an Oshiwambo language spoken in northern Namibia)", " (spoken in northern Namibia)"),
    "screen-no-format-rule": PAPER_ZEROSHOT.replace(
        "Only output the translation. Do not include explanations, glosses, or "
        "commentary.\n\n", ""),
}
NAME_VARIANTS = {
    "screen-iso-name": {"oshindonga": "Ndonga", "oshikwanyama": "Kwanyama"},
    "screen-umbrella-name": {"oshindonga": "Oshiwambo (Oshindonga dialect)",
                             "oshikwanyama": "Oshiwambo (Oshikwanyama dialect)"},
}
TEMPLATES |= SCREEN


def render(template_id: str, dialect: str, source: str) -> str:
    name = NAME_VARIANTS.get(template_id, DIALECT_NAMES)[dialect]
    return TEMPLATES[template_id].format(dialect=name, source=source)
