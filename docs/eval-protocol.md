# Evaluation protocol — Ongiini-Eval-OW automatic scoring (v1)

Fixed on 2026-10-01, **before** the first full scoring run. It fills in what
the concept paper (arXiv:2609.31727, `docs/oshiwambo-eval-concept.md`)
leaves open and records where we deviate from it. Changes after this date
are listed at the end with a date and a reason. Implemented by
`scripts/run_evaluation.py` and the scripts it calls.

## 1. Metrics

| Role | Metric | Exact setting | Why |
|---|---|---|---|
| **Primary** | chrF++, corpus level | `sacrebleu.CHRF(word_order=2)` → signature `nrefs:1\|case:mixed\|eff:yes\|nc:6\|nw:2\|space:no\|version:2.4.3` | Paper §4.3; NLLB (2022), AfroBench (2025), AmericasNLP (2021) rank on chrF/chrF++; chrF++ correlates better with human judgement than COMET-22 on African languages (SSA-MTE, EMNLP 2025: 0.50 vs 0.37 Spearman). |
| Secondary | spBLEU | `sacrebleu.BLEU(tokenize="flores200")` | Paper promises "SentencePiece BLEU" (§4.3); FLORES-200 convention. The flores200 SentencePiece model never saw Oshiwambo, so it splits words finely and spBLEU behaves close to a character metric. |
| Robustness | chrF (no word n-grams) | `CHRF(word_order=0)` | Word boundaries in written Oshiwambo vary between writers; shows whether word n-grams drive any ranking. |
| Learned (exploratory) | SSA-COMET-MTL | `McGill-NLP/ssa-comet-mtl`, source + reference | Its encoder (afro-xlmr-large-76L) has Kwanyama in pretraining; no human-scored Oshiwambo data. Used **only to compare systems**, never as an absolute quality score, and only alongside the language-ID check (COMET-style metrics barely penalise a closely related wrong language — Zouhar et al., WMT 2024). |
| Appendix | COMET-22 | `Unbabel/wmt22-comet-da` | Paper §4.3 names it; reported as a "language not covered" reference point only. Reference-free variants (CometKiwi, QE) are not used. |

Versions are pinned in `requirements-eval.txt`. Every result file carries the
sacrebleu signatures and a fingerprint (SHA-256) of the reference set.

## 2. Text handling

One normalisation for hypotheses **and** references: Unicode NFC, every run
of whitespace (including line breaks) → one space, trim. For hypotheses
only, the same cleaning for every system: remove one leading
`Oshindonga:` / `Oshikwanyama:` label and surrounding quotation marks. No
other surface rewriting (paper §4.2).

## 3. Outputs that are not translations

Reported per system and dialect, never hidden:

- **empty** output;
- **derailed** (`retry_derailed_baselines.is_derailed`: longer than
  3 × max(len(source), 20) characters, or meta-commentary such as "wait",
  "let me", "translation"), first attempt and final;
- **English copy**: chrF++ of the output against the English source > 60;
- **language** per output from GlotLID v3: Oshindonga, Oshikwanyama, other
  Oshiwambo-adjacent (Herero, Kwangali, …), Tswana, Swahili, English, other.
  Calibrated first on the human references (§4).

All of them stay in the score with whatever chrF++ gives them. A second
variant scores outputs in a foreign language (not Oshindonga/Oshikwanyama)
as 0.

**Retry rule.** Derailed outputs are re-asked with unchanged decoding, up to
5 attempts, every attempt logged; the final attempt is scored and the
first-attempt rate is reported.

**Reasoning models** (GLM 5.3, DeepSeek V4.1 reasoner, Qwen 3.8 27B) are
scored in their shipped configuration: reasoning at the provider default,
`max_tokens = 4096`. Outputs that come back empty because hidden reasoning
used up the budget stay empty and are reported as a failure mode. Two
diagnostics sit beside them, never in the main ranking:

- **reasoning off** for every model that offers both modes (DeepSeek chat,
  GLM 5.3, Qwen 3.8 27B): does thinking help translation into these
  languages?
- **16k redo**: empty outputs regenerated once with `max_tokens = 16384`.
  Complete for DeepSeek V4.1 reasoner (`deepseek-v4.1-reasoner-redo16k`);
  for GLM 5.3 it was stopped after 502 items, of which 441 came back empty
  again, and is reported as that count only.

## 4. Language identification

GlotLID v3 (`cis-lmu/glotlid`, fastText) labels every output and every
reference. Before using it on system outputs we measure on the references
how often Oshindonga is labelled `ndo_Latn` and Oshikwanyama `kua_Latn`. If
either is below 85 %, dialect confusion is reported only as "other
Oshiwambo" vs "foreign language", with the calibration numbers.

## 5. Splits and what is reported

- References exist today for ids 1–423 (295 development, 128 blind). Scores
  are computed on items with a reference; when the remaining references
  arrive the same command re-scores everything into a new results folder.
- **Development split**: headline per dialect, all breakdowns, item-level
  scores.
- **Blind split**: internal only, the headline chrF++ / spBLEU per dialect
  with confidence interval; **no** breakdowns or item-level scores (paper
  §3.3, submissions README).
- No combined score across the two dialects (paper §4.5).
- With n = 295 / 128 (later 420 / 180) confidence intervals are wide;
  published system rankings need ~2,500 segments to be stable (Adjovi et
  al. 2026). Small differences are therefore reported as ties.

## 6. Statistics

- 95 % confidence intervals: **paired** bootstrap, 1,000 resamples, seed 42,
  the same resampled items for every system, percentile interval.
- Pairwise significance: paired approximate randomisation, 10,000 trials,
  seed 42, two-sided, Holm correction over all pairs within a dialect,
  α = 0.05.
- **Significance clusters** (WMT style): systems are sorted by chrF++; a new
  cluster starts at the first system that is significantly worse than every
  system of the current cluster.
- Scores are shown with one decimal.

## 7. Breakdowns (development split)

Per phenomenon tag (an item counts in each of its tags), length bucket,
domain and provenance, each with n and a bootstrap interval. Cells with
n < 15 are marked "small sample".

## 8. Reference points

- English copied unchanged;
- a wrong sentence: each item paired with another item's reference
  (derangement, never itself; seed 42);
- NLLB-200 3.3B and MADLAD-400 3B into **Tswana** (closest supported
  relative; zero-shot transfer, not an Oshiwambo score);
- MADLAD-400 3B with its own **Kwanyama** tag `<2kj>` (a real vocabulary
  token, verified 2026-10-01; in a 3-sentence test it returned English or
  Swahili). Scored as native for Oshikwanyama and as transfer for
  Oshindonga, which MADLAD lacks.

## 9. Prompt robustness (diagnostic)

The leaderboard uses the paper's single zero-shot template
(`ongiini-eval-ow-v1-zeroshot`). As a robustness check, six systems of
different strength and origin (Claude Opus 5, Gemini 3.1 Pro, GPT-6 Astra,
DeepSeek V4.1 chat, Gemma 4 26B, Mistral Medium 3.5) are also run on the
development items with references using two more templates
(`scripts/eval_prompts.py`): `robustness-r23-zeroshot` (Robinson et al.
2023 / Gao et al. 2023) and `robustness-min-zeroshot` (bare completion,
AfroBench prompt 1). Reported: chrF++ per template with intervals, and
Kendall τ between the system rankings.

## 10. System similarity (exploratory)

Pairwise chrF++ between system outputs (symmetric mean), per dialect,
development split; the same restricted to items both systems got wrong
(chrF++ vs reference < 30) and compared with the similarity expected from
their closeness to the reference; hierarchical clustering. Read as a hint
about shared sources or strategies, not as evidence about training data.

## 11. Deviations from the concept paper

| Paper says | We do | Reason |
|---|---|---|
| Chinese/European frontier models via their native APIs | via OpenRouter (Muse Spark via Meta's API) | one billing path; providers behind OpenRouter may differ in quantisation — recorded per run |
| Open-weight models run on the DGX Spark | via OpenRouter | the Spark serves production; a Spark run of Gemma 4 follows separately |
| Mistral Large 3 | dropped | throttled to ~50 items/h; 62 of 600 done. Europe is covered by Mistral Medium 3.5 (frontier) and Mistral Small 4 (open-weight) |
| DeepSeek-R1-Distill-Llama-70B | not run | needs the Spark |
| "Sentence-piece BLEU" (unspecified) | spBLEU with `flores200` | FLORES-200 convention |
| COMET-22 tertiary | SSA-COMET-MTL exploratory, COMET-22 in the appendix | see §1 |
| MADLAD "no Oshiwambo language code" | MADLAD has `<2kj>` (Kwanyama), run as such | verified in the tokenizer; output is English/Swahili |
| No retry rule | §3 | derailment; reasoning models at their default, with reasoning-off and 16k-redo diagnostics |
| MT fallback `auto` (markdown) vs Tswana (LaTeX) | Tswana, plus MADLAD `kj` | see §8 |
| OkaLM in the main comparison | appendix diagnostic only (zero-shot and 5-shot), no ranking | base models that do not follow a translation instruction: zero-shot chrF++ 8–15 on Oshikwanyama dev, partly English copies; a zero-shot score measures instruction following, not Kwanyama ability. The 8B zero-shot run stops at 375/600 |

## Changes

- 2026-10-01, before any scores were read: reasoning models are scored at
  their default with 4096 tokens and no redo; the 16k redo becomes a
  diagnostic, and reasoning-off runs are added for GLM 5.3 and Qwen 3.8 27B.
  Reason: the 16k redo rescued only ~12 % of GLM's empty outputs at
  several times the cost, applying it to some systems and not others would
  mix configurations, and the literature finds little gain from thinking
  for direct translation (Li, Ji & Tiedemann 2025, arXiv:2510.06471).
- 2026-10-01, before any scores of these systems were read: added a
  mid-size open-weight class — gpt-oss-120b, Qwen 3.5 122B-A10B, Nemotron 3
  Super 120B-A12B — and Gemma 4 31B, all at their default configuration.
  Reason: the tested open models were either ~25–30B or ≥ 700B; ~120B
  models fit the DGX Spark at 4 bit and are the realistic self-hosted
  alternative.
- 2026-10-02: exploratory prompt screening before the robustness check
  (scripts/prompt_screening.py): five one-property variants of the paper
  prompt (no role, no language description, ISO names Ndonga/Kwanyama,
  umbrella name "Oshiwambo (… dialect)", no format rule) plus MIN, R23 and
  the May legacy prompt, on Gemma 4 26B, Qwen 3.5 122B and DeepSeek V4.1
  (both reasoning off), 100 development items. It selects which properties
  the robustness check tests; the leaderboard prompt stays the paper prompt.
  Reason: Gemma 4 26B scored ~5 chrF++ higher with the May prompt on the
  Spark than with the paper prompt via the API.
- 2026-10-02: added two dedicated English→Ndonga MT systems, run locally
  with beam 4 like NLLB: Helsinki-NLP opus-mt-en-ng (public) and Meyabase's
  fine-tune of it (private weights, shared with us; results unpublished
  until Meyabase agrees). Both translate into Ndonga only; Oshikwanyama is
  scored as transfer. Reason: the concept paper's claim that no dedicated
  MT system covers Oshiwambo missed opus-mt-en-ng.
