# Contributing to Ongiini-Eval-OW

Three contribution paths, each with a defined pipeline. Pick the one
that fits the kind of contribution you have to offer.

For the full benchmark spec, see the [**concept paper**](../../docs/oshiwambo-eval-concept.md).
For the schema and composition, see the [dataset README](./README.md).

---

## 1. Submit a model — for NLP/MT research teams

Most-impact, lowest-friction path. You run your model against the
eval set; we run the metrics; your row appears on the leaderboard.

→ **Detailed spec:** [`submissions/README.md`](./submissions/README.md)
→ **JSON schema:** [`submissions/schema.json`](./submissions/schema.json)

What you produce: two JSONL files (one per dialect) plus a one-page
model card.

What we do: validate the submission, compute chrF++ / BLEU /
COMET-22, publish per-slice matrices, attribute on the leaderboard,
invite co-authorship on the eventual paper if submitted before the
Q1 2027 cutoff.

---

## 2. Review the reference translations — for native speakers

Single-translator reference data is a known limitation of v2.
Broader native-speaker review strengthens the reference set and
surfaces dialect-specific disagreements that should themselves be
documented.

What you do:

- Review whichever subset of the published translations you have
  capacity for — even ten items is useful.
- Submit alternates or flag disagreements via a review form. Forms
  will be linked here at publication time; until then, open an
  issue on the dataset repository and we will share the form
  directly.
- Optionally include rater demographics (dialect, region) so
  aggregated reviewer context can be reported.

What we do:

- Aggregate review submissions.
- Publish a minor dataset version (v2.1) integrating alternate
  translations as supplementary references.
- Credit reviewers in the Acknowledgments and the eventual paper.

---

## 3. Propose phenomena and items — for linguists and educators

Phenomenon coverage in v2 is balanced for the constructions we
already know to test. Under-represented areas include
Namibian-Afrikaans code-switching, proverbs, tone-affecting
honorifics, and discourse markers — these are the kinds of items we
want most.

What you do:

- Submit proposed items with the required tags (`length_bucket`,
  `domain`, `phenomenon_tags`) and proposed reference
  translations.
- Optionally include a one-paragraph linguistic note explaining
  what the item tests and why it matters.
- Use the contribution template — link forthcoming under
  `contributions/`. Until that lands, open an issue on the dataset
  repository with the proposal inline.

What we do:

- Review proposed items with the dataset language coordinator and
  the reference translator.
- Accepted items enter a future dataset version (v2.x for
  schema-compatible additions, v3 for schema changes).
- Contributors are credited in the release notes.

---

## Code of conduct

Contributors agree to:

- **Treat Oshindonga and Oshikwanyama as living languages**, not
  resources to be extracted. Contributions should respect the
  speakers and the contexts in which the languages are used.
- **Cite or anonymise honestly.** Reference translations and
  reviewer contributions are credited at the contributor's
  discretion; anonymous contribution is supported.
- **No machine-translated content in reference submissions.**
  Native-speaker work only.
- **Open licence.** All accepted contributions are released under
  CC-BY-4.0 (data) or MIT (code), matching the existing dataset
  and scripts.
- **Engage respectfully.** Discussions, code reviews, and dataset
  reviews are conducted with patience and care, especially across
  language and cultural lines.

---

## Questions

Open an issue on the dataset repository. Whatever the question, we
would rather answer it than have it block your contribution.
