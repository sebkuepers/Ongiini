---
language:
  - en
  - kj   # Oshikwanyama (ISO 639-1)
  - ng   # Oshindonga (ISO 639-1)
license: cc-by-4.0
task_categories:
  - translation
task_ids:
  - text-translation
pretty_name: Ongiini AI Oshindonga + Oshikwanyama MT Evaluation Set
size_categories:
  - n<1K
multilinguality:
  - multilingual
language_bcp47:
  - en
  - ng-NA      # Oshindonga as spoken in Namibia
  - kj-NA      # Oshikwanyama as spoken in Namibia
tags:
  - machine-translation
  - low-resource-languages
  - african-languages
  - oshiwambo
  - namibia
  - llm-evaluation
  - benchmark
configs:
  - config_name: default
    data_files:
      - split: full
        path: data/eval_set.tsv
      - split: blind
        path: data/blind_split.jsonl
      - split: development
        path: data/development_split.jsonl
---

# Ongiini-Eval-OW — Oshindonga + Oshikwanyama MT Evaluation Set

An English → **Oshindonga** and English → **Oshikwanyama** evaluation set for machine translation and large language models. Both are written standards of Oshiwambo, the home language of roughly half of Namibian households, and both are absent from FLORES-200, MAFAND-MT, NLLB-200, MADLAD-400, Aya-23 and the major commercial translation services as of 2026.

The design is described in the concept paper: **[arXiv:2609.31727](https://arxiv.org/abs/2609.31727)**. To our knowledge this is the first evaluation benchmark for these languages. An earlier Oshindonga ↔ English training corpus exists ("Participatory Translations of Oshiwambo", AfricaNLP 2022) but is not built for evaluation.

> **Status — pre-release (September 2026).**
> The 600 English sources of v1.0 are public in [`../oshiwambo_eval_v3.tsv`](../oshiwambo_eval_v3.tsv), so systems can be prepared now. **Reference translations are not published yet**: the 423 v0.1 references are complete and held back, and the 177 items added in v1.0 are with the translator. Dataset, references and a public leaderboard are planned for release on Hugging Face at the end of 2026. The files under `data/` below are the v0.1 package with empty reference columns.

## Why this dataset exists

Generative AI for African languages is improving fast, but Namibian languages — Oshindonga, Oshikwanyama, Otjiherero, Khoekhoegowab, Rukwangali, Silozi — remain almost entirely uncovered. Models trained on web-scraped text produce fluent-looking but fabricated output in these languages: invented words, broken grammar, wrong meaning. There has been no public way to measure how badly. This dataset provides one.

The sources follow the register that [Ongiini AI](https://ongiini.ai), a free AI assistant for people in Namibia, sees in practice — questions about jobs, school, health, government services, family and faith — plus items crafted to probe the phenomena where translation systems fail.

## Composition (v1.0)

| Source | v0.1 | v1.0 | Description |
|---|---|---|---|
| `v1_retained` | 150 | 150 | Phrasebook-style items from the first draft, curated towards longer constructions |
| `mined_paraphrased` | 143 | 180 | Inspired by production conversations, then fully **paraphrased** by the dataset team. No verbatim user text, no names, numbers or other identifying details; at most three items per user |
| `crafted` | 110 | 210 | Written to probe 11 phenomena, at least 30 items each |
| `formal_drafted` | 20 | 60 | Notices, letters and announcements in the register of Namibian government, clinics, schools, councils and utilities |
| **Total** | **423** | **600** | |

**Source style.** The English is plain second-language English, as Namibians write it on WhatsApp; the difficulty is meant to sit in the target language, not in the English. Formal items keep authentic Namibian officialese. Crafted and formal items were drafted with LLM assistance; the translator flags any source that sounds unnatural before translating it.

**Length** (v1.0): 24 % short (1–6 words), 51 % medium (7–18), 25 % long (19+).
**Domain** (v1.0): 36 % chat, 30 % challenge, 23 % formal, 8 % community, 3 % religious.

## Phenomenon coverage (v1.0)

Items can carry several tags. A tag is only assigned where that phenomenon is the real translation difficulty of the item.

| Tag | Items | What it probes |
|---|---|---|
| `negation` | 38 | Single and double negation, scope |
| `numbers_dates` | 58 | Currency (N$), dates, times, quantities |
| `named_entities` | 39 | Namibian places, ministries, institutions |
| `tense_aspect` | 34 | Recent past, perfect, habitual, progressive — Oshiwambo marks finer distinctions than English |
| `code_switch` | 31 | English loanwords normally kept in Oshiwambo (WhatsApp, PDF, NSFAF, airtime, portal) |
| `pronoun_coreference` | 30 | Antecedents that must be resolved to pick the Oshiwambo pronoun |
| `politeness_register` | 36 | Tate / Meme / Kuku honorifics, elder, peer and child address, official address |
| `idiom_nonliteral` | 30 | Figurative language. v0.1 has 12 native-speaker idioms; v1.0 adds everyday figurative expressions ("my phone is dead", "money is tight") |
| `noun_class_agreement` | 31 | Concord across subject prefix, verb, object marker, adjective, numeral, demonstrative |
| `polysemy` | 30 | Ambiguous English words — bank, right, charge, light, match, cell — where context decides the lexeme |
| `multi_sentence` | 36 | Two to four sentences — discourse cohesion, pronouns and tense across sentence boundaries |

## Schema

`data/eval_set.tsv` (and `eval_set.jsonl`):

| Column | Type | Description |
|---|---|---|
| `id` | int | Stable identifier — 1..423 in v0.1, 1..600 in v1.0 |
| `length_bucket` | enum | `S` (≤6 words), `M` (7–18), `L` (19+) |
| `domain` | enum | `chat`, `formal`, `religious`, `community`, `challenge` |
| `phenomenon_tags` | str | Semicolon-separated tags, e.g. `negation;numbers_dates` |
| `provenance` | enum | `v1_retained`, `mined_paraphrased` (`real_mined` in v0.1), `crafted`, `formal_drafted` |
| `english` | str | English source |
| `oshindonga_reference` | str | Oshindonga reference (empty until release) |
| `oshikwanyama_reference` | str | Oshikwanyama reference (empty until release) |
| `oshindonga_translator_notes` | str | Translator notes — mostly on loanwords kept on purpose |
| `oshikwanyama_translator_notes` | str | Translator notes |
| `in_blind_split` | bool | True for the held-back 20 % |

Parallel plaintext files (`data/en.txt`, `data/oshindonga.txt`, `data/oshikwanyama.txt`, line N = id N) are provided for tools such as sacrebleu, fairseq and sentencepiece.

## Splits

A fixed 20 % of items form the **blind** split (v0.1: 84 of 423; v1.0: 119 of 600). Every item keeps its split when the set grows — v1.0 only adds items.

- `development` — use freely for prompt design and system development.
- `blind` — use only for final reporting; do not look at blind items while tuning.
- Report headline, per-phenomenon and per-length scores on the blind split.

## Baselines and scoring

Scores are corpus chrF++ (primary) and BLEU from sacrebleu, plus a **derailment rate**: the share of outputs that run away (looping text to the token limit) or narrate instead of translating. Scoring script: [`scripts/score_eval_baselines.py`](../../scripts/score_eval_baselines.py).

`data/baselines/claude.jsonl` holds Claude Opus 4.7 zero-shot outputs for v0.1. Gemma 4 26B outputs exist internally; Gemma derailed on 31 % of first attempts on v0.1, and derailed outputs are regenerated with unchanged decoding (up to five attempts, every attempt logged, first-pass rate reported alongside). Baselines for all 600 items and further systems come with the v1.0 release. They are system outputs, **not** alternative references.

To submit a system, see [`submissions/`](submissions/) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Methodology

1. Sources from the four streams are assembled, tagged and checked before any translation starts: exact duplicates and tag floors by script, near-duplicates against earlier items by review, length and domain mix.
2. The translator sees **only the English**, in randomised order, in a phone-editable Word document — no machine translations, so the references are independent, not post-edits.
3. Returned documents are imported by script, translator notes are kept, and irregular entries are resolved by hand and logged.
4. Design rationale and the literature behind it: [`docs/design.md`](docs/design.md) and the [concept paper](https://arxiv.org/abs/2609.31727).

## Provenance and ethics

Items marked `mined_paraphrased` are inspired by the topics and register of real messages to Ongiini AI, never copied from them. Candidates were drawn only from users who had not objected to research use, filtered for personal data, reviewed by hand, and then rewritten in full by the dataset team — generalising names, places, ages, quantities and other details that could point to a person. **No verbatim user content is in this dataset.** Ongiini AI's [privacy policy](https://ongiini.ai/privacy/) covers derived, non-attributable use for improving the service.

## Known limitations

1. **One shared reference per item and dialect.** v0.1 references were produced collaboratively by two translators; there is no second independent reference, so chrF++ measures closeness to one phrasing, and inter-translator agreement is not yet measured (planned as a 30-item agreement set).
2. **Regional variation.** Both dialects vary regionally; the references reflect the varieties of the translators from northern Namibia.
3. **No back-translation check.**
4. **English as the source.** Items are authored in English and translated into Oshiwambo, not the other way round, which can favour English structures.
5. **Conversational register dominates**, matching the deployment; formal and multi-sentence slices only partly compensate. No document-level items.
6. **Short items are noisy.** chrF++ and BLEU are unstable on 1–6 word segments; per-slice results for short-heavy phenomena should be read with care.

## Citation

```bibtex
@dataset{ongiini_oshiwambo_mt_eval_2026,
  author       = {Shoozi, Kaarina and Hamukwaya, Elizabeth and
                  Küpers, Sebastian},
  title        = {Ongiini AI Oshindonga + Oshikwanyama Machine
                  Translation Evaluation Set},
  year         = 2026,
  publisher    = {Common Intelligence Foundation},
  version      = {1.0.0},
  license      = {CC-BY-4.0},
  url          = {https://huggingface.co/datasets/CommonIntelligenceFoundation/ongiini-oshiwambo-mt-eval},
  doi          = {[pending Zenodo registration at publication]}
}
```

To cite the benchmark design, cite the concept paper:

```bibtex
@misc{kuepers2026ongiinievalow,
  author        = {Küpers, Sebastian},
  title         = {The {Ongiini-Eval-OW} Benchmark: A Concept Paper for the
                   Planned Benchmarking of Machine Translation and Large
                   Language Models on {Oshindonga} and {Oshikwanyama}},
  year          = 2026,
  eprint        = {2609.31727},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2609.31727},
  url           = {https://arxiv.org/abs/2609.31727}
}
```

See [`CITATION.cff`](CITATION.cff) for additional formats (Citation File Format, used by GitHub and Zenodo automatically).

## License

This dataset is released under **Creative Commons Attribution 4.0 International (CC-BY-4.0)**. You may share, adapt, and use commercially, with attribution.

See [`LICENSE`](LICENSE) for full text.

## Contact

- **About the eval set**: open an issue at [github.com/sebkuepers/Ongiini](https://github.com/sebkuepers/Ongiini)
- **About Ongiini AI** (the AI assistant this eval set is built for): [https://ongiini.ai](https://ongiini.ai)
- **About the Common Intelligence Foundation**: [https://common-intelligence.org](https://common-intelligence.org)

## Acknowledgements

- **Kaarina Shoozi** and **Elizabeth Hamukwaya** — for the reference translations into both Oshindonga and Oshikwanyama. This dataset doesn't exist without your work.
- The MT-eval literature that shaped our methodology — FLORES-200 (Goyal et al.), NTREX-128 (Federmann et al.), MAFAND-MT (Adelani et al.), AfriCOMET (Wang et al.), AfroBench (2025), ACES challenge sets (Amrhein et al.), and the chat-MT work by Farinha et al. (TACL 2024).
- The real Namibian users of Ongiini AI whose conversational patterns shaped the source distribution. (Their messages are not in this dataset; their distribution is.)
