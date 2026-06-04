# The Ongiini-Eval-OW Benchmark

### A concept paper on benchmarking large language models and machine-translation systems for Oshindonga and Oshikwanyama

**Ongiini AI · Common Intelligence Foundation · Namibia**

> **DRAFT — v0.3, 4 June 2026.** *Internal working draft. Not yet for
> external distribution.* This revision rewrites every status,
> verification, and timeline claim in the document so that nothing is
> stated as "done" or "verified" beyond what we have evidence for. The
> composition (600 items, per-phenomenon power, inter-translator
> agreement) remains as planned in §3. Direct comments and corrections
> to Sebastian Küpers.

---

## 1. Executive summary

**What.** Ongiini-Eval-OW is, to our knowledge, the first published
evaluation benchmark for machine-translation and large-language-model
output on the Oshiwambo language pair — English <-> **Oshindonga** and
English <-> **Oshikwanyama** — with native-speaker reference
translations from two independent translators, a deterministic blind
split with stratified per-phenomenon coverage, and a reproducible
scoring protocol. Prior Oshiwambo parallel data exists (notably the
WON / "Writing Our Narratives" participatory corpus of 5,419
Oshindonga <-> English sentences by Nekoto et al., AfricaNLP 2022,
acknowledged below) but has been used as training material, not as a
standardised benchmark for system evaluation. Ongiini-Eval-OW
consists of six hundred (600) parallel items, with every phenomenon
slice carrying at least 30 items so per-slice scores carry
statistical signal.

**Why.** Oshiwambo is spoken by upwards of one million people across
northern Namibia and southern Angola — the largest indigenous language
cluster in Namibia and the home language of roughly half of Namibian
households (49 %, 2011 census). As of mid-2026, **no major commercial
or open-source machine-translation system supports either dialect**.
We have directly inspected the tokenizer of NLLB-200, the README of
Madlad-400, and the public Masakhane checkpoint repositories and
found no Oshiwambo coverage in any of them; we have additionally
checked the public language lists of Google Translate, DeepL, and
Microsoft Translator and found Oshiwambo absent. The absence of an
evaluation benchmark is part of why the absence persists: without
measurement, no team can claim progress, and no funder can justify a
fine-tune. This work creates the conditions.

**Who.** Ongiini AI — a Namibian non-profit AI assistant project under
the Common Intelligence Foundation — co-developed the dataset with
two native-speaker translators of both Oshindonga and Oshikwanyama,
**Kaarina Shoozi** and **Elizabeth Hamukwaya**, drawing on production
WhatsApp conversation data, original phenomenon-targeted crafting,
and register-balanced formal items.

**Status.** Composition is finalised; an internal 423-item v0.1 build
exists with phenomenon tags and a deterministic split, and it has
been used to validate the pipeline end-to-end with Claude Opus 4.7
and Gemma 4 26B as reference baselines. The 600-item v1.0 dataset is
now being built to the composition described in §3, and both
translators are contracted for the work. First public release of
both the dataset and the concept paper is targeted for Q3 2026 (see
§7 Roadmap).

**The ask.** Three concrete contribution paths, detailed in §6:

1. **Run your model against the benchmark.** We will validate your
   submission, compute the metrics, and add your model to the
   leaderboard with attribution.
2. **Help review the reference translations.** Native speakers
   cross-validating the reference data make the benchmark stronger.
3. **Propose phenomena and items.** Under-represented patterns —
   proverbs, tones, Afrikaans code-switching — are explicit gaps we
   want to close in future releases.

If your work touches African-language NLP, machine translation,
multilingual LLM evaluation, or low-resource translation policy:
**this paper is the invitation to engage with us before the formal
academic submission later this year**.

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

Six hundred (600) English source items, each paired with reference
translations in both Oshindonga and Oshikwanyama, drawn from four
provenance streams:

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

Both subsets are translated and published; you can compute scores
on either or on the full 600. The convention is:

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

Kaarina Shoozi and Elizabeth Hamukwaya both independently translate
a designated **30-item overlap set per dialect** (`in_agreement_set
= true`). Both translations are published in
`oshindonga_reference` (primary) and `oshindonga_reference_alt`
(second translator), analogously for Oshikwanyama. From this overlap
we compute and publish:

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

- *American.* Claude Opus 4.8 (Anthropic); GPT-5.5 (OpenAI);
  Gemini 3.1 Pro (Google DeepMind).
- *Chinese.* DeepSeek V3.1 — both `deepseek-chat` (non-thinking) and
  `deepseek-reasoner` (thinking) — via the DeepSeek API; Kimi K2.6
  (Moonshot, 1T-parameter MoE with 32B active) via the Moonshot API;
  GLM-5 (Z.ai / Zhipu, 744B MoE with 40B active) via the Z.ai API.
- *European.* Mistral Large 3 (675B MoE, 41B active) and Mistral
  Medium 3.1, via the Mistral API.

**Open-weight LLMs (we run on Spark; quantised where needed).**

- *American.* Gemma 4 26B MoE (Google DeepMind, 3.8B active —
  comfortable fit on Spark); Llama 4 Scout (Meta, 109B MoE / 17B
  active — fits at FP4 but slow at single-stream decode). Llama 4
  is Meta's last open-weight release; their frontier model is now
  closed (see Muse Spark callout below).
- *European.* Mistral Small 3.1 (24B dense, Apache 2.0 — easy fit on
  Spark).
- *Chinese.* Qwen 3 30B-A3B (Alibaba, 30B MoE with 3B active — easy
  fit) and Qwen 3 235B-A22B (235B MoE with 22B active — borderline
  fit at FP4; we will attempt and report constraints honestly);
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

#### Notable absence: Meta Muse Spark

Meta's closed-weight frontier model **Muse Spark** (Meta
Superintelligence Labs, April 2026) deserves a specific call-out.
In informal hand-testing of a small number of prompts through the
meta.ai consumer interface, Muse Spark **appeared to us to produce
noticeably better Oshindonga and Oshikwanyama than the other systems
we have informally probed**, including several frontier LLMs. This
is a striking and unexpected signal — a system with no documented
Oshiwambo training claim looks qualitatively strong — but the
observation is anecdotal: a handful of prompts, judged by us, with
no metric and no held-out set.

We are nonetheless **not including Muse Spark in the launch
leaderboard**. It has no public API; no on-premise option; no open
weights; and at the time of writing the only way to interact with it
is the consumer-facing meta.ai web/app interface. Running a 600-item
benchmark through a consumer chat UI by hand is neither reproducible
nor responsibly extrapolatable to a published score, and we will
not present a number that we cannot stand behind methodologically.

We flag the apparent capability here so that readers, partners, and
Meta itself are aware that this dataset is ready to measure Muse
Spark properly the moment an API or programmatic access is made
available. Until then, the observation is anecdotal and labelled as
such.

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
  but should not be reported as headline rankings. The leaderboard
  presents per-slice cells with explicit CIs and a footnote
  flagging small-N slices.
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
  `CommonIntelligenceFoundation/ongiini-oshiwambo-mt-eval` (target
  name; the scaffolding matches). CC-BY-4.0 licence on first
  publication; full schema in the README.
- **Scripts.** MIT-licensed in the Ongiini GitHub repository under
  [`scripts/`](../scripts/). The pipeline scripts —
  `mine_eval_candidates.py`, `curate_mined_candidates.py`,
  `build_eval_v2.py`, `fill_baseline_translations.py`,
  `export_eval_set.py` — have been executed end-to-end on the
  internal v0.1 build and will be adapted as needed for the v1.0
  build.
- **Baselines as reference.** Pre-computed Claude Opus 4.7 and Gemma
  4 26B outputs on the v0.1 build are available in
  [`data/oshiwambo_eval/data/baselines/`](../data/oshiwambo_eval/data/baselines/),
  MIT-licensed. These will be recomputed against the v1.0 dataset
  before first public release and will form the first two rows of
  the published leaderboard.
- **Citation.** Citation File Format manifest at
  [`data/oshiwambo_eval/CITATION.cff`](../data/oshiwambo_eval/CITATION.cff)
  — renders as BibTeX, APA, and Zenodo metadata automatically.
  Zenodo DOI to be issued at first formal publication.
- **Versioning.** The first public dataset release will be pinned as
  `v1.0`. Schema-compatible updates will be minor (v1.1);
  schema-breaking changes will be major (v2). All releases will be
  archived on the dataset repository.

---

## 6. How to contribute

Three concrete contribution paths. Each has a defined pipeline so
the cost-to-contribute is bounded and the benefit is durable.

### 6.1 Plug your model in

Most-impact, lowest-friction path. We have wired a JSONL submission
format; you produce outputs, we run the metrics.

**What you do:**

1. Run your model against the English source items in
   [`data/oshiwambo_eval/data/en.txt`](../data/oshiwambo_eval/data/en.txt),
   producing one translation per line in each of the two dialects.
2. Format the output as JSONL matching the schema at
   [`data/oshiwambo_eval/submissions/schema.json`](../data/oshiwambo_eval/submissions/schema.json).
   One submission per (model × dialect); two files per model.
3. Open a pull request adding your submission to
   `data/oshiwambo_eval/submissions/<model-id>/` along with a
   one-page model card describing the system and the inference
   conditions.

**What we do:**

- Validate the submission against the JSON schema.
- Run the automated metrics (chrF++, BLEU, COMET-22) on every item
  and publish the per-slice matrices.
- Add your model to the leaderboard with attribution.
- For submissions before the academic-paper cutoff (Q1 2027), the
  submitting team is offered co-authorship on the eventual
  publication.

Detailed instructions and an end-to-end example at
[`data/oshiwambo_eval/submissions/README.md`](../data/oshiwambo_eval/submissions/README.md).

### 6.2 Review the reference translations

The dataset ships with two translators and a 30-item inter-translator
agreement set (§3.6). Broader native-speaker review beyond those
two strengthens the reference further and surfaces dialectal
variation that should itself be documented.

**What you do:**

- Review a subset of the published translations (you choose the
  size — even ten items helps).
- Submit alternate translations or flag disagreements via a review
  form (one form per dialect; links forthcoming on the dataset
  repository).
- Optionally provide rater demographics (dialect, region) so
  aggregated rater context appears in the published methodology.

**What we do:**

- Aggregate the review submissions.
- Publish a minor dataset version (e.g. v1.1) integrating alternate
  translations as supplementary references.
- Credit reviewers in the dataset Acknowledgments and the academic
  paper.

### 6.3 Add phenomena and items

Phenomenon coverage in v1.0 is balanced for the constructions we
know to test; under-represented areas include Namibian-Afrikaans
code-switching, proverbs, tone-affecting honorifics, and discourse
markers.

**What you do:**

- Propose new items with the required tags (length bucket, domain,
  phenomenon list) and proposed reference translations.
- Optionally include a one-paragraph linguistic note explaining
  what the item tests.
- Submit via the contribution template at
  `data/oshiwambo_eval/contributions/`.

**What we do:**

- Review proposed items with the language coordinator and the
  reference translators.
- Accepted items enter a future dataset version (v1.x for
  schema-compatible additions, v2 for schema changes).
- Contributors credited in the release notes.

---

## 7. Roadmap

Concrete milestones, with the controlled steps separated from the
translator-dependent ones:

| When | Milestone |
|---|---|
| **Now (Q2 2026)** | Dataset rebuild to 600 items in progress; existing 423-item internal v0.1 build retired. New crafted + formal items being drafted; second Spark mining run scheduled. |
| **Q3 2026** | 600-item dataset translated by both Kaarina Shoozi and Elizabeth Hamukwaya, including the 30-item agreement set. Claude + Gemma 4 26B baselines re-computed. Concept paper v1.0 published alongside dataset v1.0. **Soliciting model submissions for the launch leaderboard.** |
| **Q4 2026** | First public leaderboard released — chrF++, BLEU, COMET-22 across the launch model matrix; inter-translator agreement published. |
| **Q1 2027** | Human-evaluation round 1 results published with Krippendorff's α. |
| **Q2 2027** | Academic paper submission. Target venues (in order of preference): AfricaNLP Workshop; ACL / EMNLP main track; LREC. |

The roadmap is held against a publicly visible status page on the
dataset repository so partners and sponsors can verify progress.

---

## 8. Limitations

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

## 9. Acknowledgments and citation

The benchmark is built collaboratively. Specific acknowledgments
will be expanded with each release; the current contributors are:

- **Kaarina Shoozi** and **Elizabeth Hamukwaya** — reference
  translators for both Oshindonga and Oshikwanyama.
- Native-speaker reviewers (per-release acknowledgments).
- The Ongiini AI team at the Common Intelligence Foundation.
- The authors and maintainers of the publicly available Oshiwambo
  reference materials we consulted while designing this benchmark,
  including *Hai ti! A Beginner's Guide to Oshikwanyama* (Crane,
  Lindgren-Streicher & Wingo 2004, CC-BY-SA) and the Omniglot
  Oshiwambo phrasebook.
- **Meyabase** ([meyabase.com](https://www.meyabase.com/),
  [github.com/meyabase](https://github.com/meyabase)) — the
  Namibian Oshiwambo MT project led by Axel Mukwena. Meyabase's
  ~70,000-pair English <-> Oshindonga corpus and Neural Machine
  Translation tool is the closest peer effort to ours; we hope the
  Meyabase team will submit their system against this benchmark via
  the submission pipeline at first public release. We are grateful
  for their public pioneering work.
- **Nekoto, Kreutzer, Rajab, Ochieng & Abbott** — for *Participatory
  Translations of Oshiwambo: Towards Culture Preservation with
  Language Technology* (AfricaNLP at ICLR 2022; extended at the
  NLP for Positive Impact workshop, EMNLP 2022). Their **WON**
  ("Writing Our Narratives") corpus — 5,419 Oshindonga -> English
  sentences in the AfricaNLP version, ~7,500 in the EMNLP version
  — produced by an eleven-participant, eight-day paid workshop with
  Oshindonga speakers in Namibia, is the earliest published
  Oshindonga <-> English parallel corpus we are aware of. Affiliations
  span Masakhane NLP, Google Research, University of the
  Witwatersrand, Microsoft Africa Research Institute, and Retro
  Rabbit. Their participatory methodology and their honest
  documentation of the prior digitised-data landscape (including
  that the only OPUS-listed Oshikwanyama corpus is in fact
  mislabelled German) was a direct influence on our approach.

**Citation.** The full citation manifest is at
[`data/oshiwambo_eval/CITATION.cff`](../data/oshiwambo_eval/CITATION.cff)
and renders as BibTeX, APA, and Zenodo automatically. A
non-canonical short form for working papers:

> *Ongiini AI (2026). Ongiini-Eval-OW v1.0: an evaluation set for
> machine translation and large-language-model output on Oshindonga
> and Oshikwanyama. Common Intelligence Foundation.
> https://huggingface.co/datasets/CommonIntelligenceFoundation/ongiini-oshiwambo-mt-eval*

---

## 10. Get in touch

For model submissions, review requests, or partnership conversations:

- **GitHub.** [`Ongiini/data/oshiwambo_eval/`](../data/oshiwambo_eval/) — issues, pull requests, discussions.
- **Email.** Contact details published on the dataset repository.
- **Common Intelligence Foundation.** [Foundation website link forthcoming]

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

*This document is versioned with the dataset. Concept paper
v0.3-draft (4 June 2026) is an internal working draft describing the
planned 600-item composition; the existing 423-item internal build
will be superseded. The first public release will be cut as concept
paper v1.0 alongside dataset v1.0 at the end of Q3 2026, once both
translators have completed the new items and the agreement set.
Suggestions and corrections are welcomed as pull requests against
this file.*
