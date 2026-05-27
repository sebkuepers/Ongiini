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

# Ongiini AI Oshindonga + Oshikwanyama Machine Translation Evaluation Set

A 423-item evaluation set for machine translation between **English ↔ Oshindonga and English ↔ Oshikwanyama** — two Namibian Bantu languages of the Oshiwambo cluster that are absent from FLORES-200, MAFAND-MT, NLLB, and every major commercial translation API (Google, Azure, Amazon, DeepL, Cohere) as of May 2026.

This is, to our knowledge, the **first published MT evaluation set for these languages**.

## Quick use

```python
from datasets import load_dataset

ds = load_dataset("CommonIntelligenceFoundation/ongiini-oshiwambo-mt-eval")

for row in ds["full"]:
    en   = row["english"]
    odg  = row["oshindonga_reference"]
    okw  = row["oshikwanyama_reference"]
    # Compare your system's outputs against odg / okw
```

For reproducible benchmarking, report scores on the `blind` split only (see [Splits](#splits) below).

## Why this dataset exists

Generative AI for African languages is rapidly improving but Namibian indigenous languages — Oshindonga, Oshikwanyama, Otjiherero, Khoekhoegowab, Rukwangali, Silozi — remain almost entirely uncovered. Foundation models trained on web-scraped corpora generate fluent-looking but **fabricated** output in these languages: invented words, broken grammar, hallucinated meaning.

There is no public way to measure how badly. This dataset provides one.

The 423 source items are sampled from the actual register that a free WhatsApp-based AI helper for Namibians sees in production — daily questions about jobs, school, health, government services, family, religion — plus deliberately crafted items probing specific linguistic phenomena where MT models systematically fail (negation, noun-class agreement, code-switching, idioms, pronoun coreference, polysemy, multi-sentence cohesion).

Reference translations are by **[Translator name pending consent]**, a native Oshindonga and Oshikwanyama speaker from northern Namibia.

## Composition

The 423 items come from four deliberately balanced sources:

| Source | Items | Description |
|---|---|---|
| Retained from v1 | 150 | First-pass authored items (English source by the dataset team), curated to drop trivials and prefer longer constructions |
| Real WhatsApp-mined | 143 | Stratified sample from a production WhatsApp AI helper's logs, PII-scrubbed, then **paraphrased** into clean natural English while preserving register and intent. No verbatim user content. |
| Crafted challenge subset | 110 | Authored to probe 11 specific linguistic phenomena (see [Phenomenon coverage](#phenomenon-coverage)). Each phenomenon has ≥10 items so per-slice scores are statistically meaningful. |
| Formal / institutional | 20 | Longer items in the register of Namibian government, health, school, and bank communications |

### Length distribution

Designed against FLORES-200 / NTREX-128 / Europarl conventions (mean ~21 words/sentence; deliberately exclude very-short fragments where BLEU/chrF/COMET are noisy):

- **24% short** (1–6 words) — greetings, acks, intent triggers (kept as real-traffic signal)
- **49% medium** (7–18 words) — full user questions, single-turn replies (the meat)
- **27% long** (19+ words) — multi-clause replies and instructions where Bantu morphology stresses the model

### Domain mix

- 47% conversational chat (WhatsApp helper register)
- 22% phenomenon-tagged challenge items
- 20% formal / institutional
- 8% community (family / village / community organising)
- 3% religious

## Phenomenon coverage

Each phenomenon has ≥10 items so per-slice scoring is statistically meaningful. Items can carry multiple phenomenon tags (e.g. "She didn't bring the 12 forms by Friday" → `negation;numbers_dates;tense_aspect`).

| Tag | Items | What it probes |
|---|---|---|
| `negation` | 22 | Single, double, scope ambiguity — #1 MT failure mode (Hossain et al. 2020) |
| `numbers_dates` | 40 | Currency (N$), dates, times, phone numbers, IDs |
| `named_entities` | 24 | Namibian places, ministries, common Namibian names |
| `tense_aspect` | 23 | Perfect vs recent past vs habitual — Bantu makes finer distinctions than EN |
| `code_switch` | 17 | EN loanwords embedded in Oshiwambo (WhatsApp, ID, grant, Ministry) |
| `pronoun_coreference` | 15 | Ambiguous antecedents → Bantu noun-class pronouns force disambiguation |
| `idiom_nonliteral` | 12 | EN idioms (translator either matches local idiom or paraphrases) |
| `politeness_register` | 12 | Tate/Meme/Kuku honorifics, elder/peer/child address forms |
| `noun_class_agreement` | 10 | Chains across subject prefix → verb → object marker → adjective concord (Bantu-specific) |
| `polysemy` | 10 | "bank", "right", "school" — context decides the Oshindonga lexeme |
| `multi_sentence` | 10 | 2–4 sentence mini-paragraphs testing discourse cohesion |

## Schema

The TSV at `data/eval_set.tsv` has these columns:

| Column | Type | Description |
|---|---|---|
| `id` | int | Stable identifier, 1..423 |
| `length_bucket` | enum | `S` (≤6 words), `M` (7–18), `L` (19+) |
| `domain` | enum | `chat`, `formal`, `religious`, `community`, `challenge` |
| `phenomenon_tags` | str | Semicolon-separated phenomenon tags, e.g. `negation;numbers_dates` |
| `provenance` | enum | `v1_retained`, `real_mined`, `crafted`, `formal_drafted` |
| `english` | str | The EN source sentence |
| `oshindonga_reference` | str | Gold-standard Oshindonga translation |
| `oshikwanyama_reference` | str | Gold-standard Oshikwanyama translation |
| `oshindonga_translator_notes` | str | Optional notes (register / dialect / cultural) |
| `oshikwanyama_translator_notes` | str | Optional notes |
| `in_blind_split` | bool | True for the held-back 20% (see [Splits](#splits)) |

Plaintext per-language files are also provided at `data/en.txt`, `data/oshindonga.txt`, `data/oshikwanyama.txt` — one segment per line, indexed by line number = `id` (so `line 1` of all three files is the same item). Use these for tools that expect parallel plaintext (mosesdecoder, fairseq, sentencepiece).

## Splits

The 423 items are tagged with a **`blind` split flag** that holds back 84 items (≈20%) chosen by deterministic random sampling (seed=42).

- **`full` split**: all 423 items. Use for development, prompt tuning, exploratory analysis.
- **`development` split**: 339 items where `in_blind_split=false`. Use for any model-development work.
- **`blind` split**: 84 items where `in_blind_split=true`. **Use this split only for final reporting.** Per ACL Reproducibility Checklist conventions, do not look at blind-split items during prompt engineering or model selection.

We recommend reporting:
- Headline scores (chrF, BLEU, COMET, manual 1–5) on the **`blind` split only**
- Per-phenomenon scores on the **`blind` split only**
- Per-length-bucket scores on the **`blind` split only**

Using the `full` split is fine for system development but should not be reported as a benchmark score.

## Methodology

The full design rationale — length distribution research, phenomenon coverage justification, source-mix arguments, literature review — is at [`docs/design.md`](docs/design.md).

Quick summary:
1. **Source items** (143 real-mined + 110 crafted + 20 formal + 150 retained) were assembled and PII-scrubbed before any translation work began.
2. The **translator was shown only English source items** in randomised order via a phone-friendly Word document. No machine translations (Claude, Gemma, NLLB) were shown — translator's work is unbiased reference, not error-correction.
3. **Validation pass** before sending to translator: each phenomenon tag ≥10 items, length distribution within ±8% of target, domain mix within ±10% of target, no duplicate English strings, no digit-leak PII heuristic flags.
4. After translation: **back-import → spot-check 10 random items per language → compute baseline machine translations (Claude, Gemma 4 26B)** for downstream comparison.

## Provenance and ethics

**Real-mined items have been paraphrased.** They are *inspired by* the topical distribution of real WhatsApp queries to Ongiini AI, but the English source you see is rewritten — never the user's verbatim text. We did this in two passes:

1. Aggressive PII-scrub (regex for emails / phones / IDs / specific village + business + church names + specific personal names) eliminated items with identifying information.
2. **Full rewrite pass**: every retained real-mined item was rewritten by the dataset team into clean natural English while preserving intent, register, and length bucket. The rewriting also removes residual user voice that could be attributed to individuals.

The Common Intelligence Foundation operates Ongiini AI under a privacy policy that permits derived non-attributable use of aggregate signals for the explicit purpose of improving the helper. See [https://ongiini.ai/privacy/](https://ongiini.ai/privacy/).

**No verbatim user content from production is present in this dataset.**

## Known limitations

1. **One translator, one dialect.** Oshindonga and Oshikwanyama have regional variation. The reference translations represent the variety spoken by our translator. Document this when reporting: "Claude scored X on \[Translator]-Oshindonga reference".

2. **No back-translation verification.** A more rigorous protocol would have a second translator back-translate each reference to English. Budget went into language depth (two languages) instead of dual-annotator validation. Future versions may add this.

3. **Conversational register dominates.** This matches the deployment surface (a WhatsApp helper) but may underrepresent formal-document and literary registers. The `formal` and `multi_sentence` slices partially compensate.

4. **No long-form text.** Multi-sentence items are 2–4 sentences. Document-level translation (paragraphs, full letters) is not tested here. Future versions could add a `document_level` slice.

5. **PRELIMINARY: Claude and Gemma 4 26B baselines are included as a convenience for benchmarking, but were generated by the dataset team — they are NOT alternative gold references.** Treat them strictly as system outputs to score against the translator's reference.

## Citation

```bibtex
@dataset{ongiini_oshiwambo_mt_eval_2026,
  author       = {[Translator family name], [Translator first name] and
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

See [`CITATION.cff`](CITATION.cff) for additional formats (Citation File Format, used by GitHub and Zenodo automatically).

## License

This dataset is released under **Creative Commons Attribution 4.0 International (CC-BY-4.0)**. You may share, adapt, and use commercially, with attribution.

See [`LICENSE`](LICENSE) for full text.

## Contact

- **About the eval set**: open an issue at [github.com/sebkuepers/Ongiini](https://github.com/sebkuepers/Ongiini)
- **About Ongiini AI** (the WhatsApp helper this eval set is built for): [https://ongiini.ai](https://ongiini.ai)
- **About the Common Intelligence Foundation**: [pending — link when foundation site is live]

## Acknowledgements

- **[Translator name pending consent]** — for the reference translations into both Oshindonga and Oshikwanyama. This dataset doesn't exist without your work.
- The MT-eval literature that shaped our methodology — FLORES-200 (Goyal et al.), NTREX-128 (Federmann et al.), MAFAND-MT (Adelani et al.), AfriCOMET (Wang et al.), AfroBench (2025), ACES challenge sets (Amrhein et al.), and the chat-MT work by Farinha et al. (TACL 2024).
- The real Namibian users of Ongiini AI whose conversational patterns shaped the source distribution. (Their messages are not in this dataset; their distribution is.)
