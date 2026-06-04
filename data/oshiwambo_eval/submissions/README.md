# Model submissions

This directory holds external model submissions to the Ongiini-Eval-OW
benchmark. See the [**concept paper**](../../../docs/oshiwambo-eval-concept.md)
for the full benchmark spec and the [dataset README](../README.md) for
schema details.

This file is the **technical contract** for plugging your model in. If
you read this end-to-end and follow the steps, your submission will
be valid.

---

## What you submit

For each model you want to enter into the leaderboard:

1. **One JSONL file per dialect.** Two files total per model:
   - `<model-id>_oshindonga.jsonl`
   - `<model-id>_oshikwanyama.jsonl`
2. **A `model_card.md`.** One-page description of your system,
   inference conditions, and any deviations from the default
   prompting protocol.

Place both under
`data/oshiwambo_eval/submissions/<model-id>/`. Submit via pull
request, or email if you don't have a GitHub account (contact
details in the dataset README).

---

## JSONL format

One JSON object per line. One line per item in the eval set. Order
matches `data/en.txt` (one item per line, line N = id N).

### Required fields

```json
{
  "id":                    1,
  "dialect":               "oshindonga",
  "model_id":              "anthropic/claude-opus-4-7",
  "prompt_template_id":    "ongiini-eval-ow-v1-zeroshot",
  "translation":           "Wa lalapo!",
  "timestamp":             "2026-06-04T12:34:56Z"
}
```

| Field | Type | Description |
|---|---|---|
| `id` | int | Eval-set item ID, 1..423 (matches the dataset). |
| `dialect` | enum | `"oshindonga"` or `"oshikwanyama"`. One file per dialect. |
| `model_id` | string | Stable model identifier. Recommended format: `<vendor>/<model-name>[:version]`. Examples: `anthropic/claude-opus-4-7`, `meta/llama-4-405b`, `google/madlad400-7b-mt`. |
| `prompt_template_id` | string | Identifier for the prompt template used. Default: `ongiini-eval-ow-v1-zeroshot` (Appendix C of the concept paper). |
| `translation` | string | Your model's translation. Pass-through (English left untranslated) is allowed; empty string is allowed. Both are recorded as-is. |
| `timestamp` | string | ISO-8601 UTC timestamp of inference. |

### Optional fields

```json
{
  "id":                    1,
  "dialect":               "oshindonga",
  "model_id":              "anthropic/claude-opus-4-7",
  "prompt_template_id":    "ongiini-eval-ow-v1-zeroshot",
  "translation":           "Wa lalapo!",
  "timestamp":             "2026-06-04T12:34:56Z",
  "model_version":         "20260415",
  "inference_compute":     {"hardware": "8x A100", "wall_time_ms": 421},
  "seed":                  42,
  "raw_response":          "Wa lalapo!"
}
```

| Field | Type | Description |
|---|---|---|
| `model_version` | string | Vendor build / checkpoint identifier. |
| `inference_compute` | object | Optional inference details. Free-form; `hardware` + `wall_time_ms` are suggested keys. |
| `seed` | int | Random seed used during decoding (where applicable). |
| `raw_response` | string | The full model response before any post-processing. Useful if you applied trimming or formatting. |

---

## Validation

Run the JSON-schema validator before opening a pull request:

```bash
pip install jsonschema
jsonschema -i data/oshiwambo_eval/submissions/<model-id>/<file>.jsonl \
           data/oshiwambo_eval/submissions/schema.json
```

A valid submission is exactly 423 lines per file, with every `id`
field appearing exactly once, in the order matching `en.txt`. The
JSON Schema enforces field shapes; the count and ordering are
checked at intake.

---

## Model card

Place `model_card.md` alongside the JSONL files. Suggested
sections:

```markdown
# <model-id>

## System description
One paragraph: what the model is, who built it, the architecture
in one sentence (encoder-decoder MT? decoder-only LLM? what scale?).

## Training data and language coverage
Does the model formally claim Oshiwambo coverage? If yes, cite the
source. If no, note this honestly — it is informative for the
leaderboard reader.

## Inference conditions
- Hardware
- Sampling parameters (temperature, top_p, beam size)
- Whether system or user prompts were customised
- Any deviation from the default prompting protocol in
  Appendix C of the concept paper

## Notes
Anything else a reader of the leaderboard should know.
```

The model card is published alongside the leaderboard row so
readers can interpret the score.

---

## What we do with your submission

1. **Schema validation.** Run the JSON-schema check. If it fails,
   we open a comment on the PR with the validation output so you
   can fix and resubmit.
2. **Metric computation.** We run chrF++, BLEU, and COMET-22
   against the reference translations on every item, plus the
   per-slice matrices (per-phenomenon, per-length bucket,
   per-domain, blind vs development split).
3. **Sanity check.** A native-speaker reviewer spot-checks ~10
   items to catch obvious pipeline bugs (empty outputs, wrong
   language, encoding issues).
4. **Leaderboard update.** Your model is added to the public
   leaderboard with attribution and a link to the model card.
5. **Co-authorship invitation.** For submissions before the
   academic-paper cutoff (Q1 2027), submitting teams are offered
   co-authorship on the eventual publication.

---

## Worked example

Suppose Acme AI wants to submit their `acme/wambo-v1` translator.

1. Run inference. For each English item in `data/en.txt`:
   ```python
   for idx, source in enumerate(open("en.txt"), start=1):
       for dialect in ("oshindonga", "oshikwanyama"):
           translation = acme_translate(source.strip(), target=dialect)
           write_jsonl(dialect, {
               "id": idx,
               "dialect": dialect,
               "model_id": "acme/wambo-v1",
               "prompt_template_id": "ongiini-eval-ow-v1-zeroshot",
               "translation": translation,
               "timestamp": utcnow().isoformat() + "Z",
           })
   ```
2. Validate:
   ```bash
   jsonschema -i submissions/acme--wambo-v1/acme--wambo-v1_oshindonga.jsonl \
              submissions/schema.json
   ```
3. Write `model_card.md`. Two paragraphs is fine.
4. Open a PR:
   ```
   submissions/
   └── acme--wambo-v1/
       ├── acme--wambo-v1_oshindonga.jsonl
       ├── acme--wambo-v1_oshikwanyama.jsonl
       └── model_card.md
   ```

That's the complete process. The path from "inference is finished"
to "you're on the leaderboard" is one pull request.

---

## Open questions

If anything in this spec is unclear, please open an issue on the
dataset repository. Misunderstandings here ripple into incomparable
leaderboard entries; we'd rather answer a question than fix a bad
submission.
