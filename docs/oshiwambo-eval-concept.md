# The Ongiini-Eval-OW Benchmark

### A concept paper on benchmarking large language models and machine-translation systems for Oshindonga and Oshikwanyama

**Ongiini AI · Common Intelligence Foundation · Namibia**
*Version 1.0 · June 2026 · Distribution: external research teams, regional partners, sponsors, contributors*

---

## 1. Executive summary

**What.** Ongiini-Eval-OW is the first public benchmark for evaluating
machine-translation and large-language-model output on the Oshiwambo
language pair: English ↔ **Oshindonga** and English ↔ **Oshikwanyama**.
Four hundred and twenty-three (423) parallel items, native-speaker
references, deterministic blind split, and a fully reproducible
evaluation protocol.

**Why.** Oshiwambo is spoken by roughly 1.5 million people across
northern Namibia and southern Angola. As of mid-2026, **no major
commercial or open-source machine-translation system supports either
dialect**. We have empirically verified that NLLB-200, Madlad-400,
Masakhane checkpoints, Google Translate, and DeepL all lack Oshiwambo
coverage. The absence of an evaluation benchmark is part of why the
absence persists: without measurement, no team can claim progress, and
no funder can justify a fine-tune. This work creates the conditions.

**Who.** Ongiini AI — a Namibian non-profit AI assistant project under
the Common Intelligence Foundation — co-developed the dataset with a
contracted native-speaker translator (anonymised in this document
pending publication consent), drawing on production WhatsApp
conversation data, original phenomenon-targeted crafting, and
register-balanced formal items.

**Status.** Dataset complete, tagged, and stratified. Pre-computed
baseline outputs from Claude (Anthropic) and Gemma 4 26B (Google
DeepMind) published. Human reference translations integration is the
last blocking step before the first public leaderboard.

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

"Oshiwambo" refers to a cluster of mutually intelligible Bantu
languages (ISO codes `kj`, `ng`; Guthrie zone R.20). Two dialects
dominate by L1 speaker count and written-text presence:

| Dialect | ISO | Speakers (≈) | Primary regions |
|---|---|---|---|
| Oshindonga | `ng` | 600,000 | North-central Namibia (Oshana, Ohangwena, Omusati) |
| Oshikwanyama | `kj` | 900,000 | North-western Namibia and southern Angola |

Both are agglutinating Bantu languages with rich morphology: active
noun-class agreement chains, tonal phonology (not orthographically
marked in everyday writing), and productive verbal derivation. Both
are written in the Latin alphabet with a small set of regular
orthographic conventions.

### 2.2 The coverage gap, empirically

We audited the major translation systems against an Oshiwambo
input in May 2026:

| System | Claims Oshiwambo | Verified output |
|---|---|---|
| **NLLB-200** (Meta) | No | No Oshiwambo language code in `special_tokens_map.json` |
| **Madlad-400** (Google) | No | No Oshiwambo language code |
| **Masakhane** open checkpoints | No | No publicly available Oshiwambo checkpoint |
| **Google Translate** (web) | No | No language option |
| **DeepL** | No | No language option |
| **Microsoft Translator** | No | No language option |
| **Aya-23** (Cohere) | Listed African coverage | Not Oshiwambo |
| **OpenAI / Anthropic / Google frontier LLMs** | Best-effort, no formal claim | Produces plausible Oshiwambo; quality un-benchmarked until now |

Frontier LLMs do produce Oshiwambo output. Whether that output is
accurate, fluent, register-appropriate, or substantively useful for
real-world communication is an empirical question — and it is the
empirical question this benchmark is designed to answer.

### 2.3 Why this matters

Roughly one in two Namibians speaks an Oshiwambo language at home.
Public services — health information, education, government
communication, broadcasting — increasingly assume access to digital
text. When the translation layer is absent, the entire population of
~1.5M speakers is structurally excluded from the digital
infrastructure that other linguistic communities benefit from. This
is not a niche research question; it is a digital-inclusion question.

### 2.4 What changes when a benchmark exists

- A *measurable* baseline against which any team's MT or LLM advance
  can be cited.
- An empirical pressure point on the vendors whose language-coverage
  decisions are otherwise opaque.
- A scaffold for follow-on work on adjacent Namibian languages
  (Herero, Khoekhoegowab, Damara/Nama) that today face the same
  invisibility.
- A repeatable methodology — phenomenon-tagged, register-balanced,
  blind-split — that other low-resource language communities can
  adapt without re-deriving the design principles from scratch.

---

## 3. The Ongiini-Eval-OW v2 dataset

This section describes the dataset as published, drawing on the
internal design rationale at [`docs/eval-set-design.md`](./eval-set-design.md)
and the canonical dataset README at
[`data/oshiwambo_eval/README.md`](../data/oshiwambo_eval/README.md).

### 3.1 Composition

Four hundred and twenty-three (423) English source items, each
paired with reference translations in both Oshindonga and
Oshikwanyama, drawn from four provenance streams:

| Source | Items | Description |
|---|---|---|
| `v1_retained` | 150 | Phrasebook-style items, curated down from a larger v1 set; longer constructions preferred over greetings. |
| `real_mined` | 143 | Sampled from production WhatsApp conversation logs, PII-scrubbed and rewritten so no verbatim user text is published. Capped at three items per user to prevent dominance. |
| `crafted` | 110 | Phenomenon-tagged items authored to ensure coverage of specific test conditions (negation, code-switching, noun-class agreement). |
| `formal_drafted` | 20 | Government, health, and education register items, drafted to anchor the benchmark in institutional language. |
| **Total** | **423** | |

### 3.2 Stratification

Items are stratified on three axes to allow per-slice scoring:

**Length buckets** (intentionally chosen to reverse v1's
short-phrase skew; longer items stress agglutinating morphology):

- 24 % short (1–6 words) — greetings, acknowledgements, real-traffic signals
- 49 % medium (7–18 words) — full user questions, single-turn replies (the bulk)
- 27 % long (19+ words) — multi-clause replies where Bantu morphology stresses the model

**Domain mix:**

- 47 % conversational chat
- 22 % phenomenon-tagged challenge items
- 20 % formal / institutional
- 8 % community (family / village / community organising)
- 3 % religious

**Phenomenon tags** (≥10 items per tag for statistical power; items
can carry multiple tags):

| Tag | Items | Tests |
|---|---|---|
| `numbers_dates` | 40 | Currency (N$), dates, times, phone numbers, IDs |
| `named_entities` | 24 | Namibian places, ministries, common Namibian names |
| `tense_aspect` | 23 | Perfect / recent past / habitual (Bantu makes finer distinctions than English) |
| `negation` | 22 | Single, double, scope ambiguity — historically the #1 MT failure mode |
| `code_switch` | 17 | English loanwords embedded in Oshiwambo |
| `pronoun_coreference` | 15 | Ambiguous antecedents → Bantu noun-class pronouns force disambiguation |
| `idiom_nonliteral` | 12 | English idioms; translator matches local idiom or paraphrases |
| `politeness_register` | 12 | Tate / Meme / Kuku honorifics; elder / peer / child address |
| `noun_class_agreement` | 10 | Concord chains across subject prefix → verb → object marker → adjective |
| `polysemy` | 10 | Context-dependent lexical choice ("bank", "right", "school") |
| `multi_sentence` | 10 | 2–4 sentence mini-paragraphs testing discourse cohesion |

### 3.3 Splits

- **Full set** (423 items) — for tuning, prompt development, error
  analysis.
- **Development set** (339 items) — `in_blind_split = false`. Use
  this for any iteration that affects what your model will look like.
- **Blind set** (84 items, 20 %) — `in_blind_split = true`,
  deterministic `seed = 42`. **Headline scores are reported only on
  the blind set.** This is the reproducibility convention; the
  development set is for honest iteration, the blind set is for
  honest reporting.

### 3.4 Schema

| Column | Type | Description |
|---|---|---|
| `id` | int | Stable 1..423 |
| `length_bucket` | enum | `S` / `M` / `L` |
| `domain` | enum | `chat` / `formal` / `religious` / `community` / `challenge` |
| `phenomenon_tags` | string | Semicolon-separated tags |
| `provenance` | enum | `v1_retained` / `real_mined` / `crafted` / `formal_drafted` |
| `english` | string | Source sentence |
| `oshindonga_reference` | string | Native-speaker reference translation |
| `oshikwanyama_reference` | string | Native-speaker reference translation |
| `oshindonga_translator_notes` | string | Optional translator commentary |
| `oshikwanyama_translator_notes` | string | Optional translator commentary |
| `in_blind_split` | bool | 84 items marked true (20 %, seed 42) |

The full schema reference, with allowed values and validation rules,
appears in Appendix B.

### 3.5 Safety and provenance

- **PII scrub at mining time** — phone numbers, ID numbers,
  addresses, and named individuals are stripped from source text
  before any human reviewer sees it.
- **Mined items are rewritten, not copied** — the published English
  source is a paraphrase of the user input, preserving the linguistic
  phenomenon without leaking real-user text.
- **No machine-translated items shown to the human translator** —
  the translator works from English source only, with no automated
  proposal, to preserve reference independence.
- **Per-user cap of three items** — prevents any single user's
  speech patterns dominating the dataset.

---

## 4. Benchmark protocol

This section specifies the experiment. External teams should be able
to read §4.1–§4.4 and the submission specification at
[`data/oshiwambo_eval/submissions/README.md`](../data/oshiwambo_eval/submissions/README.md)
and produce a valid leaderboard entry without contacting us.

### 4.1 Launch model matrix

The launch leaderboard will cover at least one system per category.
The leaderboard is open to additions thereafter.

**Frontier proprietary LLMs.** Claude Opus 4.7 (Anthropic); GPT-5
class (OpenAI); Gemini 2.5 Pro (Google DeepMind). Accessible via
public APIs.

**Open-weight LLMs.** Llama 4 (Meta — 405B if accessible, 70B
otherwise); Mistral Large; Qwen 3 235B; Gemma 4 26B. Runnable
on-premise; tests whether sovereign deployment is viable.

**Dedicated machine-translation systems.** NLLB-200 (Meta);
Madlad-400 (Google); Google Translate (commercial API); DeepL. We
document the systems as "best effort even when Oshiwambo is not a
supported target," because **the failure mode itself is data** —
empty outputs, language fallbacks, or English-passthrough are
recorded explicitly.

**Smaller / specialist systems.** Aya-23 / Aya-Vision (Cohere); any
Masakhane checkpoint that emerges; MT560-trained baselines. Anchors
what specialist work in the African-language space has achieved.

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

**Raters.** 2–3 native-speaker raters per dialect. Raters are
recruited from Namibian universities, broadcasters, and partner
organisations. Rater demographics (dialect, region, age band,
profession) are reported in aggregate; individual identities are
protected.

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

**Headline scores: chrF++ on the blind 84-item split.** Everything
else is interpretive. This convention is the publishable claim;
slice-level scores are tools for understanding where each system
breaks.

---

## 5. Reproducibility

- **Dataset publication.** HuggingFace dataset at
  `Ongiini/oshiwambo-eval-v2` (target name; matches the
  scaffolding). CC-BY-4.0 licence; full schema in the README.
- **Scripts.** MIT-licensed in the Ongiini GitHub repository under
  [`scripts/`](../scripts/). The five pipeline scripts —
  `mine_eval_candidates.py`, `curate_mined_candidates.py`,
  `build_eval_v2.py`, `fill_baseline_translations.py`,
  `export_eval_set.py` — are production-ready and have been
  executed end-to-end on the released dataset.
- **Baselines as reference.** Pre-computed Claude and Gemma 4 26B
  outputs are published alongside the dataset in
  [`data/oshiwambo_eval/data/baselines/`](../data/oshiwambo_eval/data/baselines/),
  MIT-licensed. They are a reference for participants; they are also
  the first two rows of the leaderboard.
- **Citation.** Citation File Format manifest at
  [`data/oshiwambo_eval/CITATION.cff`](../data/oshiwambo_eval/CITATION.cff)
  — renders as BibTeX, APA, and Zenodo metadata automatically.
  Zenodo DOI issued at first formal publication.
- **Versioning.** The dataset is pinned as `v2.0` at first
  publication. Schema-compatible updates are minor (v2.1);
  schema-breaking changes are major (v3). All releases are
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

Single-translator reference data is a known limitation; broader
native-speaker review strengthens the reference and surfaces
dialectal disagreements that should themselves be documented.

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
- Publish a minor dataset version (v2.1) integrating alternate
  translations as supplementary references.
- Credit reviewers in the dataset Acknowledgments and the academic
  paper.

### 6.3 Add phenomena and items

Phenomenon coverage in v2 is balanced for the constructions we know
to test; under-represented areas include Namibian-Afrikaans
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
  reference translator.
- Accepted items enter a future dataset version (v2.x for
  schema-compatible additions, v3 for schema changes).
- Contributors credited in the release notes.

---

## 7. Roadmap

Concrete milestones, with the controlled steps separated from the
translator-dependent ones:

| When | Milestone |
|---|---|
| **Now (Q2 2026)** | Dataset complete, Claude + Gemma 4 26B baselines published, concept paper distributed. **Soliciting model submissions for the launch leaderboard.** |
| **Q3 2026** | Native-speaker reference translations integrated. First human-eval round commissioned. |
| **Q4 2026** | First public leaderboard released — chrF++, BLEU, COMET-22 across the launch model matrix. |
| **Q1 2027** | Human-evaluation round 1 results published with IAA. |
| **Q2 2027** | Academic paper submission. Target venues (in order of preference): AfricaNLP Workshop; ACL / EMNLP main track; LREC. |

The roadmap is held against a publicly visible status page on the
dataset repository so partners and sponsors can verify progress.

---

## 8. Limitations

The benchmark is honest about its constraints. We surface them
explicitly so reviewers do not need to.

- **One primary translator per dialect.** No back-translation; no
  dual-annotator gold standard for reference quality. The
  translation-review contribution path (§6.2) is the explicit
  mitigation, but the reference is **single-source for v2**.
- **Register bias.** Conversational chat and formal institutional
  registers dominate. Long-form discourse, legal text, and literary
  register are under-represented.
- **No audio.** The benchmark is text-only. Pronunciation, tone, and
  prosody are not directly tested.
- **English-source bias.** Dialect ↔ dialect translation is out of
  scope. Third-language pivots (e.g. Oshiwambo ↔ Portuguese, for
  Angolan speakers) are not tested.
- **No long discourse.** The longest items are 19+-word
  multi-sentence paragraphs; document-level coherence is not
  benchmarked.
- **Adjacent Namibian languages are explicitly out of scope.**
  Herero (`hz`), Khoekhoegowab (`naq`), Damara/Nama, Rukwangali,
  Silozi face the same coverage gap; they are future work but **not
  part of any commitment this benchmark makes**.

---

## 9. Acknowledgments and citation

The benchmark is built collaboratively. Specific acknowledgments
will be expanded with each release; the current contributors are:

- The reference translator (anonymised pending consent for
  publication credit).
- Native-speaker reviewers (per-release acknowledgments).
- The Ongiini AI team at the Common Intelligence Foundation.
- The Oshiwambo Skill contributors — drawing on *Hai ti! A
  Beginner's Guide to Oshikwanyama* (Crane, Lindgren-Streicher &
  Wingo 2004, CC-BY-SA 2.0); the Omniglot Oshiwambo phrasebook; and
  the MT560 / jw.org-derived parallel corpora for vocabulary
  attestation.

**Citation.** The full citation manifest is at
[`data/oshiwambo_eval/CITATION.cff`](../data/oshiwambo_eval/CITATION.cff)
and renders as BibTeX, APA, and Zenodo automatically. A
non-canonical short form for working papers:

> *Ongiini AI (2026). Ongiini-Eval-OW v2: an evaluation set for
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
→ verb → object marker → adjective. Bantu-specific; systems
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
id                            : integer        : 1..423, stable
length_bucket                 : enum           : {S, M, L}
domain                        : enum           : {chat, formal, religious,
                                                  community, challenge}
phenomenon_tags               : string         : semicolon-separated subset of
                                                 the 11 tags defined in
                                                 Appendix A
provenance                    : enum           : {v1_retained, real_mined,
                                                  crafted, formal_drafted}
english                       : string         : source sentence
oshindonga_reference          : string         : native-speaker reference
oshikwanyama_reference        : string         : native-speaker reference
oshindonga_translator_notes   : string         : optional commentary
oshikwanyama_translator_notes : string         : optional commentary
in_blind_split                : boolean        : 84 items (20%) marked true
                                                 via deterministic seed 42
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

*This document is versioned with the dataset. Concept paper v1.0
corresponds to dataset v2.0. Suggestions and corrections are
welcomed as pull requests against this file.*
