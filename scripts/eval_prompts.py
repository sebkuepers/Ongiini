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


def render(template_id: str, dialect: str, source: str) -> str:
    return TEMPLATES[template_id].format(dialect=DIALECT_NAMES[dialect], source=source)
