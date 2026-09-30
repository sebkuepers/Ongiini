# Ongiini AI

> *Ongiini* — "How are you?" in Oshiwambo.

[![arXiv](https://img.shields.io/badge/arXiv-2609.31727-b31b1b.svg)](https://arxiv.org/abs/2609.31727)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](data/oshiwambo_eval/LICENSE)

**Ongiini AI** is a free AI assistant for people in Namibia, used mainly
through WhatsApp. It answers everyday questions about school, jobs, health,
government services, farming and family life, in English and Afrikaans. It
runs an open-weight model (Gemma 4 26B) on a single NVIDIA DGX Spark instead
of a commercial API, so every conversation stays on one machine we control.

This repository contains the whole project: the production service, the
chat-agent framework it is built on, the website, and **Ongiini-Eval-OW**, a
machine-translation benchmark for Oshindonga and Oshikwanyama described in our
[concept paper on arXiv](https://arxiv.org/abs/2609.31727).

Ongiini AI is the first project of the Common Intelligence Foundation, a
non-profit being established in Estonia. Until the registration is complete,
the service is operated by Sebastian Küpers in a personal, non-profit
capacity. The pilot hardware currently runs in Germany.

---

## Ongiini-Eval-OW — the benchmark

Oshiwambo is the home language of roughly half of Namibian households, yet its
two written standards, **Oshindonga** and **Oshikwanyama**, are missing from
FLORES-200, NLLB-200, MADLAD-400 and the major commercial translation services.
There was no public way to measure how well any system translates into them.
Ongiini-Eval-OW is an English → Oshindonga / Oshikwanyama evaluation set with
native-speaker references, built from the register Ongiini AI actually sees.

| | |
|---|---|
| **Paper** | Küpers (2026), *The Ongiini-Eval-OW Benchmark* — [arXiv:2609.31727](https://arxiv.org/abs/2609.31727) · [markdown version](docs/oshiwambo-eval-concept.md) · [LaTeX source](docs/concept-paper-latex/) |
| **Size** | 600 English source items for v1.0, each to be translated into both dialects |
| **Composition** | 150 retained phrasebook items · 180 paraphrased from production conversations · 210 crafted for 11 linguistic phenomena · 60 formal / institutional |
| **Phenomena** | negation, noun-class agreement, pronoun coreference, tense/aspect, numbers and dates, named entities, code-switching, politeness register, figurative language, polysemy, multi-sentence cohesion — at least 30 items each |
| **Splits** | 80 % development / 20 % blind (seeded, fixed per item) |
| **Metrics** | chrF++ (primary) and BLEU via sacrebleu, a derailment rate for runaway outputs, and a planned human evaluation of adequacy and fluency |
| **References** | Kaarina Shoozi and Elizabeth Hamukwaya, native speakers from northern Namibia |

### Status

| | State |
|---|---|
| **v0.1** — 423 items | References complete in both dialects, held back until the v1.0 release |
| **v1.0** — 600 items | English sources final and public ([`data/oshiwambo_eval_v3.tsv`](data/oshiwambo_eval_v3.tsv)); the 177 new items are with the translator |
| **Release** | Dataset, references and a public leaderboard on Hugging Face (CC BY 4.0), planned for the end of 2026 |

The English sources are public now so that anyone can prepare a submission.
Reference translations are not in this repository; they are released together
with v1.0. Until then, [`data/oshiwambo_eval/`](data/oshiwambo_eval/) holds the
dataset card, the submission schema and the contribution guide.

### Where things are

| Path | Contents |
|---|---|
| [`data/oshiwambo_eval/`](data/oshiwambo_eval/) | Dataset card, licence, citation file, [submission format](data/oshiwambo_eval/submissions/), [contribution guide](data/oshiwambo_eval/CONTRIBUTING.md), [design notes](data/oshiwambo_eval/docs/design.md) |
| [`data/oshiwambo_eval_v3.tsv`](data/oshiwambo_eval_v3.tsv) | The 600 v1.0 English sources with length bucket, domain, phenomenon tags, provenance and split |
| `data/oshiwambo_eval_v*_seeds.md` | The crafted, formal and paraphrased-mined source items, with their tags |
| [`scripts/build_eval_v3.py`](scripts/build_eval_v3.py) | Builds the 600-item set from the frozen v0.1 set plus the v3 seeds, and checks duplicates and tag coverage |
| [`scripts/generate_oshiwambo_eval_doc.py`](scripts/generate_oshiwambo_eval_doc.py) · [`import_eval_translations.py`](scripts/import_eval_translations.py) | Translator hand-off: a phone-editable Word document out, references back in |
| [`scripts/fill_baseline_translations.py`](scripts/fill_baseline_translations.py) · [`retry_derailed_baselines.py`](scripts/retry_derailed_baselines.py) | Zero-shot baseline translations (Claude via API, Gemma via vLLM); regenerates derailed outputs with unchanged decoding and logs every attempt |
| [`scripts/score_eval_baselines.py`](scripts/score_eval_baselines.py) | chrF++ / BLEU / derailment scoring, by split, dialect, length, domain and phenomenon |

Scoring a system once the references are available:

```sh
pip install sacrebleu==2.4.3
python3 scripts/score_eval_baselines.py \
    --refs path/to/eval_set.jsonl \
    --system mymodel=mymodel_oshindonga.jsonl,mymodel_oshikwanyama.jsonl
```

System outputs follow the [submission schema](data/oshiwambo_eval/submissions/schema.json):
one JSON line per item and dialect, with `id`, `dialect`, `model_id`,
`prompt_template_id`, `translation` and `timestamp`. The
[submission guide](data/oshiwambo_eval/submissions/README.md) has the
zero-shot prompt and the reporting conventions.

---

## The assistant

```
WhatsApp Cloud API ──▶ Cloudflare Tunnel ──▶ FastAPI webhook (Docker)
                                                 │
                                                 ├─▶ Gemma 4 26B A4B (NVFP4) on vLLM — text and images
                                                 ├─▶ faster-whisper on CPU — voice notes
                                                 ├─▶ Tavily — web search
                                                 ├─▶ short-term JSON history + mem0 long-term facts (local qdrant)
                                                 └─▶ aggregate statistics for ongiini.ai/statistics
```

Every message goes through the same loop: a Gemma-based classifier decides
what kind of turn it is and how much work it needs, a policy table maps that
to a turn shape, and a typed step executor runs it with tool calls. The loop
comes from **Owela**, a small framework written for this setting: open-weight
models on self-hosted inference engines, answering through messenger apps
that show one plain-text reply and no UI.

| Path | Contents |
|---|---|
| [`owela/`](owela/) | The framework — router → policy → step executor, protocols for model, transport, memory and hooks. No product code. [README](owela/README.md) |
| [`ongiini/`](ongiini/) | The application — Gemma adapter, WhatsApp transport, memory, tools, system prompt, statistics, the Learn app backend. [README](ongiini/README.md) |
| [`website/`](website/) · [`functions/`](functions/) | ongiini.ai — landing page, web chat, Learn, contribution page, statistics (Cloudflare Pages) |
| [`deploy/`](deploy/) | DGX Spark host scripts (vLLM restart, network watchdog) |
| [`docs/`](docs/) | [Operator manual](docs/operations.md), [statistics framework](docs/statistics.md), [webhook resilience](docs/webhook-resilience.md), usage analyses |
| [`scripts/`](scripts/) | Benchmark pipeline, product-knowledge builder, analysis tools |

Running it yourself needs a machine that can serve Gemma 4 with vLLM, a
WhatsApp Cloud API number and a Tavily key. The full recipe — model flags,
container stack, tunnel and Meta configuration — is in the
[operator manual](docs/operations.md).

```sh
cp .env.example .env                         # WhatsApp, Tavily and vLLM settings
docker compose up -d --build webhook         # the service
python3 -m pytest owela/tests ongiini/tests  # unit tests, no live stack needed
```

---

## Privacy

People bring real problems to Ongiini AI, so the system is built to keep as
little as possible:

- **No message content in logs.** Usage and trace logs record counts, names,
  lengths and timings only.
- **Scrubbed before storage.** Email addresses, mobile phone numbers, IBANs,
  card and national-ID numbers are replaced with placeholders before anything
  is written to disk or to long-term memory. The model sees the raw text;
  storage does not.
- **No images or audio kept.** Photos are described and voice notes are
  transcribed, then the bytes are discarded.
- **Nothing leaves the machine.** Memory, the vector store and the model all
  run on the same computer. Users can ask what is remembered about them and
  have it deleted from inside the chat.
- **The benchmark contains no user messages.** Items derived from production
  conversations are full paraphrases written by the dataset team, with names,
  numbers and other identifying details removed.

[SECURITY.md](SECURITY.md) describes the controls in detail, and
[docs/statistics.md](docs/statistics.md) explains how the public statistics
page stays aggregate-only.

## Responsible AI

Ongiini AI is a limited-risk AI system under the EU AI Act. Every new user is
told up front that they are talking to an AI; the assistant is instructed to
refer medical, legal, financial and safety questions to qualified people, and
to search the web for anything time-sensitive. The full statement, including
known limitations of the model, is in the
[operator manual](docs/operations.md#ai-literacy-and-eu-ai-act-compliance).

---

## Licence

- **Code** — MIT, see [LICENSE](LICENSE).
- **Ongiini-Eval-OW dataset** — CC BY 4.0, see [data/oshiwambo_eval/LICENSE](data/oshiwambo_eval/LICENSE).
- **Concept paper** — CC BY 4.0.

## Citation

If you use the benchmark, please cite the concept paper:

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

The dataset itself has its own [citation file](data/oshiwambo_eval/CITATION.cff).

## Contact

[ongiini.ai](https://ongiini.ai) · [hi@ongiini.ai](mailto:hi@ongiini.ai) ·
[issues](https://github.com/sebkuepers/Ongiini/issues) for the code and the benchmark ·
[common-intelligence.org](https://common-intelligence.org) for the foundation
