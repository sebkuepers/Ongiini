# The Ongiini-Eval-OW Benchmark

### A concept paper for the planned benchmarking of machine translation and large language models on Oshindonga and Oshikwanyama

**Sebastian Küpers**
*Common Intelligence Foundation (programme of the Ongiini AI project,*
[*https://ongiini.ai*](https://ongiini.ai)*) · Namibia*
*Corresponding author:* [*hi@ongiini.ai*](mailto:hi@ongiini.ai)

**Keywords:** machine translation · low-resource languages ·
benchmarking · African NLP · Oshiwambo · Oshindonga · Oshikwanyama ·
Bantu languages · evaluation · large language models

**Version 1.0 (concept paper) · September 2026 · Paper licence: CC-BY 4.0**

> **Citing this paper.** Until peer-reviewed publication, please cite
> as: *Küpers, S. (2026). The Ongiini-Eval-OW Benchmark: A concept
> paper for the planned benchmarking of machine translation and large
> language models on Oshindonga and Oshikwanyama. arXiv preprint
> [arXiv:2609.31727](https://arxiv.org/abs/2609.31727).*

---

## Abstract

Oshiwambo — a cluster of mutually intelligible Bantu languages
spoken by upwards of one million people across northern Namibia and
southern Angola and the home language of roughly half of Namibian
households — has, to our knowledge, no published machine-translation
evaluation benchmark. Major commercial translation services (Google
Translate, DeepL, Microsoft Translator), open multilingual MT models
(NLLB-200, MADLAD-400), and the open Masakhane checkpoint
collection all lack coverage of either of the two standardised
dialects, Oshindonga and Oshikwanyama. Prior work exists in the
form of the WON / "Writing Our Narratives" participatory training
corpus [Nekoto et al., 2022], but no benchmark for system
evaluation. This concept paper announces **Ongiini-Eval-OW**, a
planned 600-item English ↔ Oshindonga / English ↔ Oshikwanyama
evaluation benchmark with native-speaker reference translations from
two independent translators, a 30-item inter-translator agreement
set with published variance metrics, an 11-tag phenomenon-tagged
stratification (each tag carrying ≥30 items), a deterministic blind
split, and a reproducible scoring protocol over chrF++, BLEU, and
COMET-22 plus a focused human-evaluation round. We document the
empirical coverage gap, the dataset composition and methodology, the
intended launch-leaderboard model matrix across American, European,
and Chinese frontier and open-weight systems, and the contribution
pipeline. The dataset is targeted for first public release at Q4
2026 alongside an updated version of this paper; this v1.0 concept
paper announces the design and the call for participation. Data and
code will be released under CC-BY-4.0 (data) and MIT (code).

---

## 1. Introduction

Roughly one in two Namibian households uses an Oshiwambo dialect as
their primary home language [Namibia Statistics Agency, 2011 census;
The Namibian, 2023 census]. The cluster includes eight mutually
intelligible Bantu varieties of which two — Oshindonga and
Oshikwanyama — have standardised written forms, established
orthographies, school curricula, and broadcast presence. Together
they are spoken by approximately one to one and a half million
people across northern Namibia and southern Angola, with
Oshikwanyama particularly numerous in Angola's Cunene Province.
For a language community of this size, the absence of any
public machine-translation evaluation benchmark is striking.

The absence is not theoretical. We have directly inspected the
tokenizer of NLLB-200, the README and supported-language list of
MADLAD-400, and the public Masakhane checkpoint repositories, and
found no Oshiwambo coverage in any of them. We have additionally
consulted the public language lists of Google Translate, DeepL, and
Microsoft Translator, and found Oshiwambo absent from each. Frontier
large language models — Claude, GPT, Gemini, Llama, Qwen — do
produce Oshiwambo-like output when prompted, but the quality has
not been measured by any benchmark and is therefore neither
defensible nor falsifiable. Prior published parallel data for
Oshiwambo amounts, to our knowledge, to the WON / "Writing Our
Narratives" corpus by [Nekoto et al., 2022] — a ~5,500-sentence
Oshindonga → English participatory training corpus that has been
used for fine-tuning experiments but not as a standardised
evaluation benchmark.

This paper announces **Ongiini-Eval-OW**, a planned 600-item
evaluation benchmark for English ↔ Oshindonga and English ↔
Oshikwanyama machine translation and large-language-model output.
The contribution is fourfold:

1. **A reference-quality bilingual evaluation set.** 600 English
   source items paired with native-speaker reference translations
   into both standardised dialects, contributed by two independent
   translators (Kaarina Shoozi and Elizabeth Hamukwaya).
   A 30-item subset is translated by both translators independently
   so that inter-translator agreement can be published as a variance
   measurement, rather than asserted as a uniform reference of
   uncertain quality.
2. **Phenomenon-tagged stratification with usable per-slice
   power.** Eleven phenomenon tags target known low-resource MT
   failure modes (negation, noun-class agreement, code-switching,
   politeness register, polysemy, multi-sentence cohesion, etc.),
   with each tag carrying at least 30 items so that per-slice
   scores in the 180-item blind split (30 %) carry roughly nine
   items per slice — a working minimum for slice-level comparison.
3. **Deployment-derived source items.** Roughly 30 % of the source
   items are paraphrased from production user queries to the
   Ongiini AI WhatsApp assistant, scrubbed of PII and capped at
   three derived items per user. The result reflects the actual
   distribution of questions that Namibian users ask digital
   assistants in their language, not an encyclopaedic projection of
   it.
4. **A reproducible scoring protocol.** chrF++, BLEU, and COMET-22
   computed automatically over a published JSONL submission format,
   supplemented by a 50-item human-evaluation round with adequacy
   and fluency ratings and published Krippendorff's α.

This is a **concept paper for the planned v1.0 release**. An
internal v0.1 build at 423 items exists and has been used to
validate the pipeline end-to-end; the 600-item v1.0 build described
in §3 is in production at the time of writing, with first public
release of the dataset and an updated version of this paper
targeted for Q4 2026. We publish this concept paper in advance to
solicit model submissions, native-speaker reference review, and
phenomenon-coverage contributions during the build phase — the
patterns and the contact paths are detailed in §6.

The remainder of the paper is organised as follows. §2 documents
the empirical coverage gap and motivates the work. §3 specifies the
dataset composition, stratification, splits, schema, safety
provisions, and the inter-translator agreement methodology. §4
specifies the benchmark protocol — the launch model matrix, the
prompting and metrics conventions, the human-evaluation round, and
the published-score discipline. §5 covers reproducibility. §6
consolidates the call for participation, the roadmap, and contact
details. §7 lists known limitations and anticipated reviewer
critiques. §8 is the ethics statement. §9 contains acknowledgments,
§10 the contributions statement, §11 the data and code availability
statement, and §12 the references. Appendices A–E follow.

---

## 2. Background — the Oshiwambo translation gap

### 2.1 The languages

"Oshiwambo" refers to a cluster of eight mutually intelligible Bantu
dialects (Guthrie zone R.20; Maho 2009 lists the cluster as R.21–24).
Two of the eight have standardised written forms and dominate by
written-text presence and L1 census share:

| Dialect | ISO 639-1 | ISO 639-3 | Guthrie | Primary regions |
|---|---|---|---|---|
| Oshikwanyama | `kj` | `kua` | R.21 | Ohangwena and Omusati regions (Namibia); Cunene Province (Angola) |
| Oshindonga | `ng` | `ndo` | R.22 | Oshana and Oshikoto regions (Namibia) |

Of the two, Oshikwanyama is the more numerous in both Namibia and
Angola. At the 2011 Namibian Population and Housing Census,
Oshikwanyama was the home language of 21.2 % of households and
Oshindonga 15.1 %. The 2023 Namibian Census reports Aakwanyama
(712,165 — 23.6 % of Namibians) as the largest ethnic group and
Aandonga (311,211 — 10.3 %) as the second largest. Ethnologue and
Wikipedia cite roughly one million Kwanyama speakers in Cunene
Province, Angola, as of 2024.

Both standardised dialects are agglutinating Bantu languages with
rich morphology: active noun-class agreement chains, tonal phonology
(not orthographically marked in everyday writing), and productive
verbal derivation. Both are written in the Latin alphabet with a
small set of regular orthographic conventions.

### 2.2 The coverage gap

We audited the major translation systems for Oshiwambo coverage in
May 2026. For open systems we inspected tokenizer and repository
artefacts directly; for closed/commercial systems we consulted the
publicly published language list.

| System | Claims Oshiwambo | What we checked, and what we found |
|---|---|---|
| **NLLB-200** (Meta) | No | Inspected `special_tokens_map.json` on Hugging Face; no Oshiwambo language code present |
| **Madlad-400** (Google) | No | Inspected model README and supported-language list; no Oshiwambo language code present |
| **Masakhane** open checkpoints | No | Surveyed the public Masakhane MT repositories; no Oshiwambo checkpoint located |
| **Google Translate** (web) | No | Consulted public language list; Oshiwambo not offered as a translation option |
| **DeepL** | No | Consulted public language list; Oshiwambo not offered as a translation option |
| **Microsoft Translator** | No | Consulted public language list; Oshiwambo not offered as a translation option |
| **Aya-23** (Cohere) | No | Cohere documents Aya-23 as covering 23 languages, none of which are African; Oshiwambo not among them |
| **Meyabase Translate** ([meyabase.com](https://www.meyabase.com/)) | Yes (English <-> Oshindonga) | Namibian NMT effort led by Axel Mukwena; ~70k-pair corpus per the project website; the only Oshiwambo-specific NMT system we are aware of |
| **OpenAI / Anthropic / Google frontier LLMs** | Best-effort, no formal claim | Produce plausible Oshiwambo output on prompt; quality un-benchmarked until now |

Frontier LLMs do produce Oshiwambo output. Whether that output is
accurate, fluent, register-appropriate, or substantively useful for
real-world communication is an empirical question — and it is the
empirical question this benchmark is designed to answer.

### 2.3 Why this matters

Roughly one in two Namibian households speaks an Oshiwambo dialect at
home (49 %, 2011 census). Public services — health information,
education, government communication, broadcasting — increasingly
assume access to digital text. When the translation layer is absent,
the entire Oshiwambo-speaking population — over a million across
Namibia and Angola — is structurally excluded from the digital
infrastructure that other linguistic communities benefit from. This
is not a niche research question; it is a digital-inclusion question.

### 2.4 What changes when a benchmark exists

- A *measurable* baseline against which any team's MT or LLM advance
  can be cited.
- An empirical pressure point on the vendors whose language-coverage
  decisions are otherwise opaque.
- A scaffold for follow-on work on adjacent Namibian languages
  (Otjiherero, Khoekhoegowab, Rukwangali, Silozi) that today face
  the same coverage gap.
- A repeatable methodology — phenomenon-tagged, register-balanced,
  blind-split — that other low-resource language communities can
  adapt without re-deriving the design principles from scratch.

---

## 3. The Ongiini-Eval-OW dataset

This section describes the dataset as designed for first public
release (v1.0), drawing on the internal design rationale at
[`docs/eval-set-design.md`](./eval-set-design.md) and the canonical
dataset README at [`data/oshiwambo_eval/README.md`](../data/oshiwambo_eval/README.md).
An internal 423-item v0.1 build that exercises the same pipeline
already exists; the 600-item v1.0 build described below is in
production.

### 3.1 Composition

The v1.0 dataset is being built to six hundred (600) English source
items, each paired with reference translations in both Oshindonga
and Oshikwanyama, drawn from four provenance streams:

| Source | Items | Description |
|---|---|---|
| `v1_retained` | 150 | Phrasebook-style items, curated down from a larger v1 set; longer constructions preferred over greetings. |
| `mined_paraphrased` | 180 | **Inspired by** production WhatsApp conversation logs: PII-scrubbed, then paraphrased into clean English while preserving register and intent. **The published English source is not verbatim user text** — the label is honest about this. Capped at three derived items per user to prevent dominance. |
| `crafted` | 210 | Phenomenon-tagged items authored to ensure each phenomenon slice carries at least 30 items (negation, code-switching, noun-class agreement, politeness register, etc.). |
| `formal_drafted` | 60 | Government, health, education, legal, and Namibian news/broadcast register items, drafted to anchor the benchmark in institutional language. |
| **Total** | **600** | |

The composition revision from v0.1 (423 items) was deliberate: the
earlier dataset's per-phenomenon slices were too thin to support
meaningful per-slice statistical comparison in the blind split, and
the chat/conversational register dominated more than was
methodologically honest. v0.2 fixes both: every phenomenon carries
30+ items, formal/community/religious registers are bulked up, and
the `mined_paraphrased` label replaces the misleading `real_mined`.

### 3.2 Stratification

Items are stratified on three axes to allow per-slice scoring:

**Length buckets** (intentionally chosen to reverse v1's
short-phrase skew; longer items stress agglutinating morphology):

- 25 % short (1–6 words) — greetings, acknowledgements, real-traffic signals
- 50 % medium (7–18 words) — full user questions, single-turn replies (the bulk)
- 25 % long (19+ words) — multi-clause replies where Bantu morphology stresses the model

**Domain mix** (rebalanced from v0.1 to reduce chat skew):

- 38 % conversational chat
- 22 % phenomenon-tagged challenge items
- 22 % formal / institutional
- 12 % community (family / village / community organising)
- 6 % religious

**Phenomenon tags** (≥30 items per tag — calibrated so each tag has
~9 items in the 180-item blind split, the threshold below which
per-slice chrF++ comparisons become statistically uninformative;
items can carry multiple tags):

| Tag | Items | Tests |
|---|---|---|
| `numbers_dates` | 40 | Currency (N$), dates, times, phone numbers, IDs |
| `named_entities` | 30 | Namibian places, ministries, common Namibian names |
| `tense_aspect` | 30 | Perfect / recent past / habitual (Bantu makes finer distinctions than English) |
| `negation` | 30 | Single, double, scope ambiguity — historically the #1 MT failure mode |
| `code_switch` | 30 | English loanwords embedded in Oshiwambo |
| `pronoun_coreference` | 30 | Ambiguous antecedents -> Bantu noun-class pronouns force disambiguation |
| `idiom_nonliteral` | 30 | English idioms; translator matches local idiom or paraphrases |
| `politeness_register` | 30 | Tate / Meme / Kuku honorifics; elder / peer / child address |
| `noun_class_agreement` | 30 | Concord chains across subject prefix -> verb -> object marker -> adjective |
| `polysemy` | 30 | Context-dependent lexical choice ("bank", "right", "school") |
| `multi_sentence` | 30 | 2–4 sentence mini-paragraphs testing discourse cohesion |

### 3.3 Splits

Every item gets run through every system being evaluated — the
split is **not** about which items to test against, it is about
which score is reported as the headline. The 600 items are tagged
into two subsets:

- **Development set** (420 items, `in_blind_split = false`) — use
  these for prompt engineering, error analysis, fine-tune training,
  or anything else that might involve looking at items and
  iterating in response. Whatever you do to your system can be
  shaped by the dev items.
- **Blind set** (180 items, 30 %, `in_blind_split = true`) — items
  you commit not to look at while building your system. Deterministic
  `seed = 42`, stratified by phenomenon × length × domain so each
  slice retains roughly its full-set proportion.

Both subsets will be translated and published at v1.0 release; you
will be able to compute scores on either or on the full 600. The
convention is:

| Use | Which items? |
|---|---|
| Running a system to see what it outputs | All 600 |
| Per-phenomenon / per-length / per-domain diagnostics | All 600 |
| Tuning a prompt or fine-tuning a model | Development set only |
| **Headline leaderboard number, publishable claim** | **chrF++ on the blind set** |

Why hold a subset back at all, given the data is published?

1. *Prevent prompt-tuning leakage.* If everyone iterates against the
   full set, prompts implicitly fit those exact items and the
   headline score becomes inflated. The blind set is the portion
   nobody is allowed to look at while building, so it remains a
   clean probe.
2. *Slow training-data contamination.* Once references are
   published they will be crawled and eventually appear in some
   model's training corpus. At launch no system has seen the blind
   set because the dataset does not exist publicly yet; well-behaved
   teams undertake not to train on it later.
3. *Match the standard MT-eval convention* used by FLORES, WMT, and
   MAFAND-MT, so reviewers and downstream users read the numbers as
   they expect to.

The blind split was sized at 30 % (rather than the conventional
20 %) so that each per-phenomenon slice in the blind set carries
~9 items — a working minimum for slice-level comparison, even if
slice scores remain interpretive rather than headline (§4.5).

### 3.4 Schema

| Column | Type | Description |
|---|---|---|
| `id` | int | Stable 1..600 |
| `length_bucket` | enum | `S` / `M` / `L` |
| `domain` | enum | `chat` / `formal` / `religious` / `community` / `challenge` |
| `phenomenon_tags` | string | Semicolon-separated tags |
| `provenance` | enum | `v1_retained` / `mined_paraphrased` / `crafted` / `formal_drafted` |
| `english` | string | Source sentence |
| `oshindonga_reference` | string | Native-speaker reference translation (primary translator) |
| `oshikwanyama_reference` | string | Native-speaker reference translation (primary translator) |
| `oshindonga_reference_alt` | string | Alternate reference from the second translator on the 30-item overlap set; empty otherwise |
| `oshikwanyama_reference_alt` | string | Alternate reference from the second translator on the 30-item overlap set; empty otherwise |
| `oshindonga_translator_notes` | string | Optional translator commentary |
| `oshikwanyama_translator_notes` | string | Optional translator commentary |
| `in_blind_split` | bool | 180 items marked true (30 %, seed 42) |
| `in_agreement_set` | bool | 30 items per dialect marked true; both translators provide independent references |

The full schema reference, with allowed values and validation rules,
appears in Appendix B.

### 3.5 Safety and provenance

- **PII scrub at mining time** — phone numbers, ID numbers,
  addresses, and named individuals are stripped from source text
  before any human reviewer sees it.
- **Mined items are paraphrased, not copied** — the published English
  source is a paraphrase of the user input, preserving register and
  intent without leaking real-user text. This is reflected in the
  `mined_paraphrased` provenance label and is documented honestly
  rather than presented as raw "real-mined" data.
- **No machine-translated items shown to the human translator** —
  the translators work from English source only, with no automated
  proposal, to preserve reference independence.
- **Per-user cap of three items** — prevents any single user's
  speech patterns dominating the dataset.

### 3.6 Inter-translator agreement

Kaarina Shoozi and Elizabeth Hamukwaya will both independently
translate a designated **30-item overlap set per dialect**
(`in_agreement_set = true`). Both translations will be published in
`oshindonga_reference` (primary) and `oshindonga_reference_alt`
(second translator), analogously for Oshikwanyama. From this
overlap we will compute and publish:

- **Character-level chrF++ self-similarity** between the two
  translations of each item, reported as mean ± SD per dialect.
- **A qualitative disagreement count** (lexical / morphological /
  pragmatic differences) coded by the dataset language coordinator.
- **A diff sample** of representative disagreements in the
  Acknowledgments and methodology appendix.

We deliberately do **not** assert one translator's reference is
"correct" and the other "alternate"; both are valid references from
fluent native speakers. The agreement set is a public window into
the reference variance, not an adjudication exercise.

---

## 4. Benchmark protocol

This section specifies the experiment. External teams should be able
to read §4.1–§4.4 and the submission specification at
[`data/oshiwambo_eval/submissions/README.md`](../data/oshiwambo_eval/submissions/README.md)
and produce a valid leaderboard entry without contacting us.

### 4.1 Launch model matrix

To keep the launch claims honest, we separate **what we will run
ourselves** from **what we invite external teams to submit** (§6.1).
"Run ourselves" means we have working access today — either a paid
API endpoint or a model that fits on our NVIDIA DGX Spark (128 GB
unified memory, ~273 GB/s bandwidth). Anything else is welcome via
the submission pipeline; we will not pre-commit to running it.

The launch leaderboard will cover at least one system per category
and is geographically balanced across American, European, and
Chinese state-of-the-art models.

**Frontier proprietary LLMs (we run via API).**

- *American.* Claude Opus 5 (Anthropic); GPT-6 Astra (OpenAI);
  Gemini 3.1 Pro (Google DeepMind).
- *Chinese.* DeepSeek V4.1 — both `deepseek-chat` (non-thinking) and
  `deepseek-reasoner` (thinking) — via the DeepSeek API; Kimi K3
  (Moonshot) via the Moonshot API; GLM-5.3 (Z.ai / Zhipu) via the
  Z.ai API.
- *European.* Mistral Large 3 (675B MoE, 41B active) and Mistral
  Medium 3.5 (128B dense), via the Mistral API.

**Open-weight LLMs (we run on Spark; quantised where needed).**

- *American.* Gemma 4 26B MoE (Google DeepMind, 3.8B active —
  comfortable fit on Spark); Llama 4 Scout (Meta, 109B MoE / 17B
  active — fits at FP4 but slow at single-stream decode). Llama 4
  is Meta's last open-weight release; their frontier model is now
  closed (see Muse Spark callout below).
- *European.* Mistral Small 4 (24B dense, Apache 2.0 — easy fit on
  Spark).
- *Chinese.* Qwen 3.8 27B (Alibaba, Apache 2.0 — easy fit);
  DeepSeek-R1-Distill-Llama-70B (quantised) for a smaller-footprint
  reasoning baseline.

Running both API and on-Spark families lets us report whether
sovereign on-device inference is viable for a Namibian deployment,
not only what the largest cloud frontier model produces.

**Dedicated machine-translation systems (open research models, we
run on Spark).** **NLLB-200** (Meta) and **Madlad-400** (Google) do
not list Oshiwambo as a supported target, but both have been trained
on multiple other Bantu languages from the same broad family —
NLLB-200 includes Zulu (`zul_Latn`), Tswana (`tsn_Latn`), Xhosa
(`xho_Latn`), Swahili (`swh_Latn`), Sotho (`sot_Latn`), Venda
(`ven_Latn`), and others; Madlad-400 covers 419 languages with
similarly broad Bantu coverage. We run each system with the
closest-related-language target code (Tswana for the southwest
Bantu zone) and report what comes out. This is **explicitly a
zero-shot-transfer measurement**, not a Oshiwambo translation
score: we expect low chrF++. The interesting datum is whether
genuine Oshiwambo vocabulary or morphology leaks through, or
whether the output collapses to the fallback language.

We exclude the consumer / commercial translation APIs (Google
Translate, DeepL, Microsoft Translator) and Aya-23 from the launch
matrix. None of them lists Oshiwambo as a target, and unlike the
open research models above we cannot deliberately probe their
zero-shot behaviour with a specific Bantu fallback code; they would
either refuse or pick a fallback we don't control. The coverage
gap is already documented in §2.2; spending compute to confirm it
twice does not add information.

**Specialist systems (via external submission).** **Meyabase
Translate** (Axel Mukwena; English <-> Oshindonga) — the only
Oshiwambo-specific NMT system we are aware of, built on a ~70k-pair
corpus and the closest peer effort to ours. We do not have access
to run Meyabase ourselves; we expect the Meyabase team to run their
system against the dataset and submit results via the submission
pipeline (§6.1). Any Masakhane checkpoint that emerges with
Oshiwambo coverage is welcomed via the same path.

**Out of scope at launch.** Models requiring partnership-grade
quotas, region-locked deployments we cannot access from Namibia, or
on-premise hardware beyond a single DGX Spark are not pre-committed.
They are warmly welcomed via the submission pipeline (§6.1) and
will be added to the leaderboard with attribution as teams submit.

#### Meta Muse Spark

Meta's closed-weight frontier model **Muse Spark** (Meta
Superintelligence Labs, first released April 2026) deserves a
specific call-out. In informal hand-testing of a small number of
prompts through the meta.ai consumer interface, Muse Spark
**appeared to us to produce noticeably better Oshindonga and
Oshikwanyama than the other systems we have informally probed**,
including several frontier LLMs. This is a striking and unexpected
signal — a system with no documented Oshiwambo training claim looks
qualitatively strong — but the observation is anecdotal: a handful
of prompts, judged by us, with no metric and no held-out set.
Measuring it properly is precisely what this benchmark is for.

At the time this benchmark was designed, Muse Spark had no public
API, no on-premise option and no open weights, and the only way to
interact with it was the consumer-facing meta.ai web interface —
which made a reproducible 600-item run impossible. That objection
has since been removed: with the release of **Muse Spark 1.1** in
July 2026, Meta opened the Meta Model API to developers in public
preview, providing exactly the programmatic access the benchmark
requires.

**We therefore commit to evaluating Muse Spark 1.1 as part of the
launch leaderboard.** At the time of writing we do not have API
access — the public preview opened to United States developers
first, with a waitlist for broader availability. Should access not
be in place by first public release, Muse Spark moves to the
submission pipeline (§6) on the
same terms as every other system we cannot run ourselves, and we
would welcome a submission from Meta directly.

### 4.2 Prompting protocol

To keep comparisons apples-to-apples:

- **LLMs** are evaluated with a single, public, zero-shot prompt
  template (Appendix C). No in-context examples — this protects
  fairness across systems with uneven few-shot support and matches
  the regime in which an end-user would invoke the model.
- **MT systems** are called through native APIs with `target_lang`
  set to the dialect ISO code where supported, falling back to
  `auto` with the source line for systems that do not support
  Oshiwambo at all.
- **Pre/post-processing** is identical across systems: whitespace
  normalisation, line-by-line input, no surface form rewriting. Any
  necessary per-system differences (e.g. an API requiring a specific
  termination token) are documented in Appendix C.

### 4.3 Automated metrics (every item)

- **chrF++ — primary.** Character-F-score with word-boundary
  weighting. Robust on character-rich, agglutinating Bantu
  morphology where token-level BLEU under-rewards near-misses.
  Well-established in FLORES-200 and AfricaNLP work.
- **BLEU — secondary.** Sentence-piece BLEU, included for legacy
  comparability with older MT literature and existing African-
  language work.
- **COMET-22 — tertiary.** Neural quality estimation with the
  English-side reference. **Reported with an explicit caveat**:
  COMET-22's training data does not include Oshiwambo, so the
  score is suggestive, not definitive — interpret as a
  source-conditioned plausibility signal rather than a
  language-aware quality score.

All three are computed by a public script (provided in
`scripts/`) that consumes the submitted model outputs in JSONL form
plus the reference TSV. Outputs are reported as overall scores plus
per-slice matrices: per-phenomenon × per-length × per-domain ×
per-split.

### 4.4 Human evaluation (50-item stratified sample)

Automated metrics are necessary but not sufficient on low-resource
languages. We pair them with a focused human-eval round on a
50-item stratified sample, mirroring the dataset's overall
phenomenon, length, and domain proportions.

**Raters.** 2–3 native-speaker raters per dialect. Recruitment is
planned from Namibian universities, broadcasters, and partner
organisations; specific partnerships will be confirmed and named in
the published methodology at the time of the first human-eval round.
Rater demographics (dialect, region, age band, profession) are
reported in aggregate; individual identities are protected.

**Ratings.** Two per item per rater:

- **Adequacy** (1–5) — does the translation preserve the meaning of
  the English source?
- **Fluency** (1–5) — does the translation read naturally to a
  fluent speaker, independent of the source?

**Inter-Annotator Agreement (IAA).** Reported using Krippendorff's α
on the ordinal scale. Items where raters disagree by ≥2 points are
adjudicated by a third reader, with adjudication notes published in
the leaderboard appendix.

**Rater materials** — rubric, anchor examples, reference cards — are
published alongside the leaderboard so the human-eval methodology
is itself reproducible.

### 4.5 Reporting matrix

The published leaderboard is sliceable along these axes:

- **Per system × per dialect.** Oshindonga and Oshikwanyama scores
  are reported separately; aggregated scores hide dialect-specific
  failures.
- **Per metric.** chrF++, BLEU, COMET-22, human-adequacy mean,
  human-fluency mean.
- **Per slice.** Phenomenon × length bucket × domain × split.

**Headline scores: chrF++ on the blind 180-item split.** Everything
else is interpretive. This convention is the publishable claim;
slice-level scores are tools for understanding where each system
breaks.

#### Statistical power and what counts as a publishable score

We are explicit about what the numbers do and do not support:

- **Headline blind-set scores** (N = 180) are sufficient for paired
  bootstrap confidence intervals at the per-system level. As a
  general guide from the chrF++ literature on similarly-sized test
  sets, differences smaller than ~3 chrF++ points are often
  statistically indistinguishable; we publish CIs explicitly so
  comparisons can be made honestly rather than relying on point
  estimates.
- **Per-phenomenon blind-set scores** (typically ~9 items per slice)
  are interpretive. They are useful for spotting *where* a system
  struggles (e.g. noun-class agreement, multi-sentence cohesion)
  but will not be reported as headline rankings. The published
  leaderboard will present per-slice cells with explicit CIs and a
  footnote flagging small-N slices.
- **Per-system per-dialect** scores aggregate to the full blind set
  for both dialects (180 items each) and are reliable.
- **Item-level outputs** for every system are published alongside the
  scores. Anyone wanting to run their own significance test —
  paired bootstrap, sign test, Wilcoxon — has the raw material.

This framing is conservative on purpose. We would rather under-claim
on slice-level rankings than ship a per-phenomenon leaderboard cell
that a reviewer can dismantle on power grounds.

---

## 5. Reproducibility

- **Dataset publication target.** HuggingFace dataset at
  `CommonIntelligenceFoundation/ongiini-oshiwambo-mt-eval`. The
  scaffolding (schema, README, license) matches the target. The
  dataset will be released under CC-BY-4.0 at first public
  publication.
- **Scripts.** MIT-licensed at
  <https://github.com/sebkuepers/Ongiini/tree/main/scripts>. The
  pipeline scripts (`mine_eval_candidates.py`,
  `curate_mined_candidates.py`, `build_eval_v2.py`,
  `fill_baseline_translations.py`, `export_eval_set.py`) have been
  executed end-to-end on the internal v0.1 build and will be
  adapted for the v1.0 composition.
- **Baselines as reference.** Pre-computed Claude Opus 4.7 and
  Gemma 4 26B outputs on the v0.1 build exist internally in
  `data/oshiwambo_eval/data/baselines/` and validate the pipeline.
  Baselines will be re-computed against the v1.0 dataset (with
  Claude Opus 5 and Gemma 4 26B) before first public release and
  will form the first rows of the published leaderboard.
- **Citation.** Citation File Format manifest at
  <https://github.com/sebkuepers/Ongiini/blob/main/data/oshiwambo_eval/CITATION.cff>
  renders as BibTeX, APA, and Zenodo metadata automatically. A
  Zenodo DOI will be minted at first formal publication.
- **Versioning.** The first public dataset release will be pinned as
  `v1.0`. Schema-compatible updates will be minor (v1.1);
  schema-breaking changes will be major (v2). All releases will be
  archived on the dataset repository.

---

## 6. Call for participation and roadmap

Three contribution paths invite the wider community to populate the
benchmark and strengthen the reference quality. Each has a defined
pipeline so the cost-to-contribute is bounded.

### 6.1 Submit a model

The most-impact, lowest-friction path. Run your system against the
600 English source items, format the outputs as JSONL per the
schema at
[`data/oshiwambo_eval/submissions/schema.json`](https://github.com/sebkuepers/Ongiini/blob/main/data/oshiwambo_eval/submissions/schema.json),
and open a pull request adding the submission under
`data/oshiwambo_eval/submissions/<model-id>/` with a one-page
model card. We validate the JSON, run chrF++, BLEU, and COMET-22
on every item, publish the per-slice matrices, and add the system
to the leaderboard with attribution. Detailed instructions and an
end-to-end example are in
[`data/oshiwambo_eval/submissions/README.md`](https://github.com/sebkuepers/Ongiini/blob/main/data/oshiwambo_eval/submissions/README.md).
For submissions made before the academic-paper cutoff (Q1 2027),
the submitting team is offered co-authorship on the eventual
publication.

### 6.2 Review the reference translations

The dataset will ship with two translators and a 30-item
inter-translator agreement set (§3.6). Broader native-speaker review
beyond those two strengthens the reference further and surfaces
dialectal variation that should itself be documented. Review a
subset of the translations at your chosen size (even ten items
helps); submit alternates or flag disagreements via review forms
linked on the dataset repository at first public release; optionally
provide rater demographics (dialect, region) so aggregated rater
context can appear in the published methodology. Accepted alternates
will be integrated into a minor dataset version (v1.1 etc.) and
reviewers credited in the Acknowledgments.

### 6.3 Propose phenomena and items

Phenomenon coverage in v1.0 is balanced for the constructions we
know to test. Under-represented areas include Namibian-Afrikaans
code-switching, proverbs, tone-affecting honorifics, and discourse
markers. Propose new items with the required tags (length bucket,
domain, phenomenon list) and proposed reference translations;
optionally include a one-paragraph linguistic note explaining what
the item tests. Submit via the contribution template at
`data/oshiwambo_eval/contributions/`. Accepted items will enter a
future dataset version (v1.x for schema-compatible additions, v2
for schema changes); contributors are credited in the release notes.

### 6.4 Roadmap

| When | Milestone |
|---|---|
| **Q3 2026** | Concept paper v1.0 (this paper) published to arXiv as cs.CL. The 423-item internal v0.1 build fully translated into both dialects by both reference translators, validating the translation pipeline end-to-end at production scale. **Soliciting model submissions for the launch leaderboard.** |
| **Q4 2026** | 600-item source set locked; existing 423-item internal v0.1 build superseded. New crafted + formal items authored; second Spark mining run executed. Translator handoff including the 30-item agreement set. Claude Opus 5 + Gemma 4 26B baselines computed. **Dataset v1.0 released** alongside the **first public leaderboard** — chrF++, BLEU, COMET-22 across the launch model matrix; inter-translator agreement published. |
| **Q1 2027** | Human-evaluation round 1 results published with Krippendorff's α. Concept paper v2 submitted to arXiv with finalised numbers. |
| **Q2 2027** | Academic paper submission. Target venues (in order of preference): AfricaNLP Workshop; ACL / EMNLP main track; LREC. |

The roadmap will be held against a publicly visible status section
of the dataset README so partners and sponsors can verify progress.

### 6.5 Contact

- **GitHub.** Issues, pull requests, and discussion at
  <https://github.com/sebkuepers/Ongiini> (the
  `data/oshiwambo_eval/` directory).
- **Email.** [hi@ongiini.ai](mailto:hi@ongiini.ai) for partnership
  conversations, review requests, or submission questions.
- **The Ongiini AI project.** <https://ongiini.ai>.

---

## 7. Limitations

The benchmark is honest about its constraints. We surface them
explicitly so reviewers do not need to.

- **Two translators, not three or more.** Kaarina Shoozi and
  Elizabeth Hamukwaya cover the full dataset, with a 30-item
  agreement-set overlap from which inter-translator agreement is
  reported (§3.6). This is stronger than the single-translator
  reference common in low-resource MT eval but weaker than a 3+
  rater panel; the translation-review contribution path (§6.2)
  remains open as the explicit mitigation.
- **No back-translation validation.** A more rigorous protocol would
  back-translate references into English via a third translator
  blind to the source. Budget was directed toward broader
  phenomenon coverage and the inter-translator agreement set
  instead; future versions may add back-translation on a sampled
  basis.
- **Register bias.** Conversational chat remains the largest single
  domain at 38 % (rebalanced down from 47 % in v0.1).
  Long-form discourse, literary, and scientific registers are
  under-represented; the benchmark is for sentence-level MT and
  short-paragraph cohesion only.
- **N = 600 is mid-size.** Compared to FLORES-200 (3,001 per
  language) or MAFAND-MT (5,000+ per pair), Ongiini-Eval-OW is
  smaller. We deliberately optimised for **density of useful items
  in the deployment register and phenomenon coverage** over raw N;
  the contribution pipeline (§6.3) is the path to growth.
- **No audio.** The benchmark is text-only. Pronunciation, tone, and
  prosody are not directly tested. This is a real gap for
  Oshiwambo, which is a tonal oral vernacular as much as a written
  language.
- **English-source bias.** Dialect <-> dialect translation is out of
  scope. Third-language pivots (e.g. Oshiwambo <-> Portuguese, for
  Angolan speakers) are not tested.
- **No long discourse.** The longest items are 19+-word
  multi-sentence paragraphs; document-level coherence is not
  benchmarked.
- **Adjacent Namibian languages are explicitly out of scope.**
  Otjiherero (`hz` / `her`), Khoekhoegowab (`naq` — covering the
  Nama and Damara dialect continuum), Rukwangali (`kwn`), and Silozi
  (`loz`) face the same coverage gap; they are future work but **not
  part of any commitment this benchmark makes**.
- **Pre-release status.** This is a concept paper for the planned
  v1.0 release; an internal v0.1 build at 423 items exists and has
  been used to validate the pipeline, but the full 600-item v1.0
  composition is in production at the time of writing. Final
  numbers (inter-translator agreement, recomputed Claude Opus 5
  and Gemma 4 26B baselines, populated leaderboard) will appear in
  the arXiv v2 of this paper to be submitted at first public
  dataset release (Q4 2026).

#### Anticipated reviewer critiques and our responses

We expect peer reviewers to raise the following objections; we
surface them here rather than wait for them in review:

- *"N = 600 is small for an MT benchmark."* We agree. Our defence is
  that every phenomenon slice carries ≥30 items and the blind split
  is 30 % to keep per-slice power meaningful; that the contribution
  pipeline is designed for growth; and that for the specific
  deployment we serve (a free WhatsApp AI assistant in Namibia)
  benchmark *register* matters more than benchmark *volume*.
- *"Mined items are paraphrased, so they're not really
  deployment-derived."* We agree; the provenance label
  `mined_paraphrased` is honest about this. They are inspired by
  real-traffic distributions, not raw user text. They remain the
  closest publishable proxy to the actual deployment surface
  available, because verbatim user text cannot be released for
  privacy reasons.
- *"Per-phenomenon scores on ~9-item blind slices are noisy."* We
  agree, and we report them with explicit confidence intervals and
  a footnote flagging small-N slices; we do not present per-slice
  scores as headline rankings (§4.5).

---

## 8. Ethics statement

The Ongiini-Eval-OW benchmark is built on the following ethical
posture, made explicit so the reader can verify it rather than
trust it.

**Consent and PII.** The 180 source items derived from production
WhatsApp queries to the Ongiini AI assistant
([https://ongiini.ai](https://ongiini.ai)) are **paraphrased**, not
verbatim. Each source conversation is PII-scrubbed (phone numbers,
ID numbers, addresses, personal names) before any human reviewer
sees it, then rewritten into clean English by the dataset team
preserving register and intent. The published English source is not
attributable to any individual user, and verbatim user text is
never released. Source users have agreed to the privacy policy at
<https://ongiini.ai/privacy/>, which permits derived
non-attributable use of aggregated signals for the explicit purpose
of improving the assistant. A per-user cap of three derived items
prevents any single user's speech patterns dominating the dataset.

**Translator labour and credit.** Both reference translators
(Kaarina Shoozi and Elizabeth Hamukwaya) are compensated for their
work at market rates appropriate to professional Namibian
translation work; their names appear in the Acknowledgments and in
the dataset's CITATION.cff with their consent.

**Reference variance, not adjudication.** The 30-item
inter-translator agreement set publishes both translators' work
side-by-side. We do not assert one translator's reference is
"correct" and the other "alternate"; both are valid references from
fluent native speakers, and the agreement set is a public window
into the reference variance rather than an adjudication exercise.

**Open licence as anti-extraction posture.** The dataset is
released under CC-BY-4.0, the code under MIT, and this paper under
CC-BY 4.0. Anyone improving on Oshiwambo translation through use of
the benchmark must credit the artefact and the named translators —
a deliberate counterweight to the historical pattern of
low-resource language work being absorbed into proprietary systems
without attribution.

---

## 9. Acknowledgments

The benchmark is built collaboratively. Specific acknowledgments
will be expanded with each release; the current contributors are:

- **Kaarina Shoozi** and **Elizabeth Hamukwaya** — reference
  translators for both Oshindonga and Oshikwanyama.
- The Ongiini AI team at the Common Intelligence Foundation.
- The authors and maintainers of the publicly available Oshiwambo
  reference materials we consulted while designing this benchmark,
  including *Hai ti! A Beginner's Guide to Oshikwanyama*
  [Crane et al., 2004] and the Omniglot Oshiwambo phrasebook.
- **Meyabase** ([meyabase.com](https://www.meyabase.com/),
  [github.com/meyabase](https://github.com/meyabase)) — the
  Namibian Oshiwambo MT project led by Axel Mukwena. Meyabase's
  ~70,000-pair English ↔ Oshindonga corpus and Neural Machine
  Translation tool is the closest peer effort to ours; we hope the
  Meyabase team will submit their system against this benchmark via
  the submission pipeline at first public release. We are grateful
  for their public pioneering work.
- **Nekoto, Kreutzer, Rajab, Ochieng & Abbott** — for *Participatory
  Translations of Oshiwambo* [Nekoto et al., 2022]. The WON
  ("Writing Our Narratives") corpus — 5,419 Oshindonga → English
  sentences in the AfricaNLP version, ~7,500 in the EMNLP version
  — is the earliest published Oshindonga ↔ English parallel corpus
  we are aware of. Their participatory methodology and their honest
  documentation of the prior digitised-data landscape (including
  that the only OPUS-listed Oshikwanyama corpus is in fact
  mislabelled German) directly influenced our approach.

---

## 10. Contributions

Sebastian Küpers designed the benchmark composition, authored this
paper, built the pipeline, and runs the benchmark infrastructure.
Kaarina Shoozi and Elizabeth Hamukwaya provide the reference
translations for both Oshindonga and Oshikwanyama. The Meyabase
project led by Axel Mukwena and the WON corpus by Nekoto et al.
2022 are acknowledged as prior art that shaped the design.

---

## 11. Data and code availability

The Ongiini-Eval-OW v1.0 dataset is targeted for first public
release in Q4 2026 at
<https://huggingface.co/datasets/CommonIntelligenceFoundation/ongiini-oshiwambo-mt-eval>
under CC-BY-4.0. This arXiv submission is the **concept paper**
announcing the planned release; an updated version of the paper
will be submitted as arXiv v2 at first public release with
finalised numbers and the live HuggingFace URL. Pipeline scripts
(`mine_eval_candidates.py`, `curate_mined_candidates.py`,
`build_eval_v2.py`, `fill_baseline_translations.py`,
`export_eval_set.py`) are MIT-licensed and available at
<https://github.com/sebkuepers/Ongiini/tree/main/scripts>.
Pre-publication access to the v0.1 internal build (423 items, used
to validate the pipeline end-to-end) is available to prospective
reviewers and submitters on request to
[hi@ongiini.ai](mailto:hi@ongiini.ai). The CITATION.cff manifest at
<https://github.com/sebkuepers/Ongiini/blob/main/data/oshiwambo_eval/CITATION.cff>
renders as BibTeX and APA automatically; a Zenodo DOI will be
minted at first formal publication.

---

## 12. References

Adelani, D. I., Alabi, J. O., Fan, A., Kreutzer, J., Shen, X.,
Reid, M., Ruder, S., et al. (2022). A Few Thousand Translations
Go a Long Way! Leveraging Pre-trained Models for African News
Translation (MAFAND-MT). *Proceedings of NAACL 2022*, 3053–3070.

Anthropic (2026). *Claude Opus 5 model card.* San Francisco, CA.
<https://www.anthropic.com/news/claude-opus-5>.

Aryabumi, V., Dang, J., Talupuru, D., Dash, S., Cairuz, D., Lin, H.,
Venkitesh, B., et al. (2024). *Aya 23: Open Weight Releases to
Further Multilingual Progress.* arXiv:2405.15032.

Costa-jussà, M. R., Cross, J., Çelebi, O., Elbayad, M., Heafield,
K., Heffernan, K., Kalbassi, E., et al. (2022). *No Language Left
Behind: Scaling Human-Centered Machine Translation (NLLB-200).*
arXiv:2207.04672.

Crane, T. M., Lindgren-Streicher, K., and Wingo, A. (2004).
*Hai ti! A Beginner's Guide to Oshikwanyama.* Peace Corps Namibia.
CC-BY-SA. <https://wingolog.org/pub/hai-ti/hai-ti.pdf>.

DeepSeek AI (2026). *DeepSeek V4.1.* API documentation,
<https://api-docs.deepseek.com/updates>.

Federmann, C., Kocmi, T., and Xin, Y. (2022). *NTREX-128 — News
Test References for MT Evaluation of 128 Languages.* WMT 2022.

Google DeepMind (2026). *Gemini 3.* Blog post,
<https://blog.google/products/gemini/gemini-3/>.

Google DeepMind (2026). *Gemma 4 model card.*
<https://ai.google.dev/gemma/docs/core/model_card_4>.

Goyal, N., Gao, C., Chaudhary, V., Chen, P.-J., Wenzek, G., Ju, D.,
Krishnan, S., Ranzato, M., Guzmán, F., and Fan, A. (2022). The
FLORES-101 Evaluation Benchmark for Low-Resource and Multilingual
Machine Translation. *TACL* 10:522–538.

Hossain, M. M., Anastasopoulos, A., Blanco, E., and Palmer, A.
(2020). It's not a Non-Issue: Negation as a Source of Error in
Machine Translation. *Findings of EMNLP 2020*, 3869–3885.

Joshi, P., Santy, S., Budhiraja, A., Bali, K., and Choudhury, M.
(2020). The State and Fate of Linguistic Diversity and Inclusion
in the NLP World. *Proceedings of ACL 2020*, 6282–6293.

Krippendorff, K. (2004). *Content Analysis: An Introduction to Its
Methodology.* 2nd ed. Sage Publications.

Kudugunta, S., Caswell, I., Zhang, B., Garcia, X., Choquette-Choo,
C. A., Lee, K., Xin, D., Kusupati, A., et al. (2023). *MADLAD-400:
A Multilingual And Document-Level Large Audited Dataset.*
NeurIPS 2023 Datasets and Benchmarks.

Maho, J. F. (2009). *NUGL Online: The Online Version of the New
Updated Guthrie List, a Referential Classification of the Bantu
Languages.*

Mistral AI (2026). *Mistral Medium 3.5.*
<https://huggingface.co/mistralai/Mistral-Medium-3.5-128B>.

Mistral AI (2025). *Introducing Mistral 3 — Mistral Large 3,
Medium 3, Small 3.* <https://mistral.ai/news/mistral-3/>.

Moonshot AI (2026). *Kimi K3: Open Frontier Intelligence.*
<https://github.com/MoonshotAI/Kimi-K3>.

Namibia Statistics Agency (2024). *2023 Population and Housing
Census — Main Report.* Windhoek.
<https://nsa.org.na/document/2023-population-and-housing-census-main-report/>.

Nekoto, W. O., Kreutzer, J., Rajab, J., Ochieng, M., and Abbott, J.
(2022). Participatory Translations of Oshiwambo: Towards Culture
Preservation with Language Technology. *AfricaNLP Workshop at
ICLR 2022.* Extended at the NLP for Positive Impact workshop,
EMNLP 2022.

OpenAI (2026). *Introducing GPT-6 Astra.*
<https://openai.com/index/gpt-6/>.

Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. (2002). BLEU:
A Method for Automatic Evaluation of Machine Translation.
*Proceedings of ACL 2002*, 311–318.

Popović, M. (2017). chrF++: Words Helping Character n-grams.
*Proceedings of WMT 2017*, 612–618.

Rei, R., De Souza, J. G. C., Alves, D., Zerva, C., Farinha, A. C.,
Glushkova, T., Lavie, A., Coheur, L., and Martins, A. F. T. (2022).
COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task.
*Proceedings of WMT 2022.*

Touvron, H., et al. (2025). *The Llama 4 herd: Scout, Maverick,
Behemoth.* Meta AI blog,
<https://ai.meta.com/blog/llama-4-multimodal-intelligence/>.

Wang, J., Adelani, D. I., Agrawal, S., Rei, R., Briakou, E., Carpuat,
M., He, X., et al. (2024). AfriMTE and AfriCOMET: Enhancing COMET
to Embrace Under-resourced African Languages. *Proceedings of NAACL
2024,* 5435–5454.

Yang, A., et al. (2025). *Qwen 3.* Qwen GitHub release,
<https://github.com/QwenLM/Qwen3>.

Z.ai / Zhipu AI (2026). *GLM-5.3.* HuggingFace model release,
<https://huggingface.co/zai-org/GLM-5.3>.

---

# Appendices

## Appendix A — Phenomenon tag definitions

The eleven phenomenon tags are intended to be empirically
distinguishable failure modes for translation systems. Each
definition is followed by 2–3 worked examples drawn from the
dataset (English source only; references in the dataset itself).

**`negation`** — sentences whose meaning hinges on a negation
operator. Includes single (`I don't have it`), double (`I never
said I wouldn't go`), and scope-ambiguous (`Not everyone came`)
constructions. The historical #1 MT failure mode (Hossain et al.
2020).

**`numbers_dates`** — sentences containing numerals, monetary
amounts (Namibian dollar, `N$`), dates, times, phone numbers, or ID
numbers. Tests the system's ability to preserve numerical content
verbatim while translating surrounding context.

**`named_entities`** — proper nouns: Namibian places (Windhoek,
Oshakati), ministries, agencies, and common Namibian personal
names. Tests entity preservation across translation.

**`tense_aspect`** — Bantu languages make finer aspectual
distinctions than English (perfect vs recent past vs habitual vs
remote past). Sentences here test whether systems collapse the
distinctions or preserve them.

**`code_switch`** — English loanwords embedded in Oshiwambo, or
the reverse. Common in real WhatsApp register: "WhatsApp", "ID",
"grant", "Ministry" used in otherwise-Oshiwambo sentences.

**`pronoun_coreference`** — sentences with ambiguous antecedents
that Bantu noun-class pronouns force the translator to
disambiguate (e.g. *"she didn't"* requires the noun-class agreement
of the implied subject).

**`idiom_nonliteral`** — English idioms ("hit the books", "raining
cats and dogs") whose literal translation produces a non-idiomatic
Oshiwambo sentence. Tests whether systems propose a local idiom or
a literal paraphrase.

**`politeness_register`** — honorifics (Tate / Meme / Kuku),
elder/peer/child address forms, register-marking that has no
direct surface-form equivalent in English.

**`noun_class_agreement`** — concord chains across subject prefix
-> verb -> object marker -> adjective. Bantu-specific; systems
trained primarily on Indo-European data tend to fail here.

**`polysemy`** — English words whose Oshiwambo lexeme depends on
context: "bank" (river vs financial); "right" (correct vs
political); "school" (educational institution vs fish school).

**`multi_sentence`** — 2–4 sentence mini-paragraphs that test
discourse cohesion across sentence boundaries (pronoun reference,
tense consistency, topic chaining).

---

## Appendix B — Schema reference

Full schema for `data/oshiwambo_eval/data/eval_set.tsv`:

```
id                                : integer  : 1..600, stable
length_bucket                     : enum     : {S, M, L}
domain                            : enum     : {chat, formal, religious,
                                                community, challenge}
phenomenon_tags                   : string   : semicolon-separated subset of
                                               the 11 tags defined in
                                               Appendix A
provenance                        : enum     : {v1_retained,
                                                mined_paraphrased,
                                                crafted, formal_drafted}
english                           : string   : source sentence
oshindonga_reference              : string   : native-speaker reference
                                               (primary translator)
oshikwanyama_reference            : string   : native-speaker reference
                                               (primary translator)
oshindonga_reference_alt          : string   : alternate reference from
                                               second translator on the
                                               30-item agreement set
oshikwanyama_reference_alt        : string   : alternate reference from
                                               second translator on the
                                               30-item agreement set
oshindonga_translator_notes       : string   : optional commentary
oshikwanyama_translator_notes     : string   : optional commentary
in_blind_split                    : boolean  : 180 items (30%) marked true
                                               via stratified deterministic
                                               seed 42
in_agreement_set                  : boolean  : 30 items per dialect with
                                               independent dual references
```

Items are also published as JSONL at
`data/oshiwambo_eval/data/eval_set.jsonl` and as plain-text parallel
files at `data/oshiwambo_eval/data/{en,oshindonga,oshikwanyama}.txt`
(one item per line; line N corresponds to id N).

The blind and development subsets are published as separate JSONL
files (`blind_split.jsonl`, `development_split.jsonl`) for
participants who want to operate strictly within one split.

---

## Appendix C — Prompting templates

**LLM zero-shot template** (used for all frontier and open-weight
LLMs):

```
You are a professional translator translating English to {DIALECT}
(an Oshiwambo language spoken in northern Namibia). Translate the
following English sentence into natural, fluent {DIALECT}.

Only output the translation. Do not include explanations, glosses,
or commentary.

English: {SOURCE}
{DIALECT}:
```

Substitutions: `{DIALECT}` is one of `Oshindonga` or `Oshikwanyama`;
`{SOURCE}` is the English item. No system-prompt variation across
LLMs; same template for all.

**MT system templates** — system-specific because of API
heterogeneity. Documented per-system in the submission appendix at
publication time.

---

## Appendix D — Human-eval rater rubric

**Adequacy** (1–5): "Does the translation preserve the meaning of
the English source?"

| Score | Anchor |
|---|---|
| 5 | All meaning preserved; no information lost or added |
| 4 | Most meaning preserved; minor information missing or shifted |
| 3 | Core meaning preserved; significant information missing or distorted |
| 2 | Some meaning preserved; major elements wrong |
| 1 | Meaning is wrong, missing, or unrelated to the source |

**Fluency** (1–5): "Does the translation read naturally to a
fluent speaker?" (Rate independently of the source.)

| Score | Anchor |
|---|---|
| 5 | Natural, native-sounding sentence |
| 4 | Mostly natural; one or two minor awkward choices |
| 3 | Understandable but noticeably non-native or awkward |
| 2 | Difficult to understand; several errors |
| 1 | Incomprehensible or ungrammatical |

Raters receive anchor examples for each score before the rating
round. Adjudication by a third reader is invoked when raters
disagree by ≥2 points on either dimension.

---

## Appendix E — Contributor code of conduct

Contributors to Ongiini-Eval-OW agree to:

- **Respect for the languages and their speakers.** Contributions
  treat Oshindonga and Oshikwanyama as the living, evolving
  languages of millions of speakers — not as resources to be
  extracted.
- **Attribution and consent.** Reference translations and reviewer
  contributions are credited at the contributor's discretion;
  anonymous contribution is supported.
- **No machine-translated content in submissions.** Reference
  translations and reviewer alternates are native-speaker work, not
  MT post-edits.
- **Open licence.** All accepted contributions are released under
  CC-BY-4.0 (data) or MIT (code), consistent with the existing
  dataset and scripts.
- **Respectful engagement.** Discussions, code reviews, and dataset
  reviews are conducted with patience and care.

---

*This concept paper is versioned with the dataset. Version 1.0
(this version, September 2026) announces the design of the planned
Ongiini-Eval-OW v1.0 dataset; an internal v0.1 build at 423 items
exists and has been used to validate the pipeline end-to-end. An
updated arXiv v2 of this paper will be submitted at first public
release of the dataset (planned Q4 2026). Suggestions and
corrections are welcomed as pull requests against the source
markdown at*
[*github.com/sebkuepers/Ongiini*](https://github.com/sebkuepers/Ongiini)
*or by email to* [*hi@ongiini.ai*](mailto:hi@ongiini.ai).
