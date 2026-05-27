# Oshindonga + Oshikwanyama eval set — design rationale

**Status:** v2 design (May 2026). Replaces v1 (`~/Desktop/ongiini-oshindonga-eval-v1.tsv`) which is being deprecated before reaching the translator.

This document captures *why* we made the choices we did building v2.
Future-Sebastian and future-Claude reading this in six months will not
remember the trade-offs we considered, so they're written down here.

---

## Why this eval set exists at all

The foundation has promised `ongiini.ai` will support Oshindonga + Oshikwanyama "via a translation layer that converts to and from English under the hood, while keeping Gemma 4 as the brain."

Before we build that layer, we need to **measure how well our candidate translators (Claude API, Gemma 4 26B, eventually NLLB and others) actually translate into these two Namibian languages**. None of them were specifically trained on Oshiwambo. The bot's previous attempt at direct Oshiwambo generation fabricated fake-Bantu word-salad — exactly the failure mode an eval set must surface.

The eval set is a **measurement instrument**, not training data. It needs to:

1. Match the distribution Ongiini AI actually serves (WhatsApp helper for Namibians on daily questions about jobs, school, health, family).
2. Have linguistic phenomenon coverage so we can score *per slice* — "how does Gemma handle negation?" vs "how does Claude handle code-switching?" — and find specific failure modes instead of getting a single noisy aggregate score.
3. Hold back a portion as a blind test split. Items we tune prompts against MUST NOT be the items we report final numbers on.

---

## Why v1 was inadequate

v1 had 200 items, all human-authored by Claude (me), categorised into 14 topic buckets (greeting, ack, cv_jobs, education, etc.). Reviewing in the Word-doc layout, Sebastian noticed many phrases looked too simple:

```
v1 length distribution:
  short  (1-6 words):  142  (71%)
  medium (7-18 words):  49  (24%)
  long   (19+ words):    0  ( 0%)
```

This is closer to a **phrasebook** than an MT eval set. The MT literature confirms:

- **FLORES-200, NTREX-128, Europarl** all average ~21 words/sentence and deliberately exclude very short fragments because **BLEU/chrF/COMET are noisy on <5-token segments** and don't reward subtle morphology.
- A 1-word translation ("Yes." → "Eeno.") doesn't measurably separate a good translator from a bad one. Bantu noun-class agreement, verb-extension chains, and tense/aspect marking only stress the model on longer multi-word sentences.

v1 also had **zero coverage** of the phenomena where MT models silently fail:
- Negation (the #1 known MT failure mode per ACES challenge sets)
- Code-switching (EN loanwords inside Oshindonga — pervasive in real use)
- Noun-class agreement chains (Bantu-specific)
- Tense/aspect distinctions finer than English
- Pronoun coreference with ambiguous antecedents
- Polysemy ("bank", "right", "school" — context decides)
- Idiom / non-literal expression
- Politeness register (elder / peer / child address — strong Oshiwambo norms)
- Multi-sentence cohesion

If we shipped v1 to the translator and used it to score Claude vs Gemma, we'd see surface-level differences only — and would miss the actual failure modes that matter to users.

---

## v2 composition decisions

### Size: ~400 items (~150 retained + ~150 mined + ~80 crafted + ~20 formal)

Published guidance (Koehn 2004; Graham et al. on DA significance):
- ~300 segments is the floor for separating mid- vs low-tier systems.
- 1.0 BLEU difference needs ~1000 sentences; 2.0 BLEU is reliable at ~300.
- For low-resource languages, **more items >> more references** (Zouhar et al. 2024).

A skilled translator does ~250–400 words/hour on careful reference work. At ~20 segments/hour and a mean of ~14 words, **~400 items × 2 languages = ~40 translator-hours** = ~€400 at Elizabeth's rate. That's enough for statistical significance on real quality gaps and lets us slice per-phenomenon (~10–20 items per tag).

We deliberately did NOT chase 1000 items. Past that, translator quality collapses and our gold standard becomes unreliable — defeating the whole purpose.

### Length distribution: 20% short / 50% medium / 30% long

Reversed from v1's 71/24/0 skew. The medium bucket (7–18 words) does most of the work — that's where real WhatsApp questions and replies live. Long items (19+ words) probe Bantu morphology and discourse coherence. Short items (1–6 words) stay as ~15% because they're real traffic — greetings, acks, intent triggers — but no longer dominate.

### Domain mix: 60% chat / 15% formal / 10% religious / 10% community + idiom / 5% multi-sentence

Strong bias toward **conversational WhatsApp-style** language because that is the actual deployment domain. Farinha et al. (2024 TACL) showed news-trained MT metrics under-correlate with human judgment on chat by 15–25% — eval-set domain must match deployment domain.

We include some formal/institutional text so the eval can detect when a model **only** does casual well — we'll need that headroom when Ongiini expands to government-services explainers. Religious + family registers are over-weighted vs. their natural frequency because they're high cultural-fluency signals: a model that fails on "Kalunga e li pamwe naye" reveals more than one that aces "the meeting is at 9".

### Source mix: 60% real-mined / 40% crafted

Both, with bias toward real. The crafted subset covers phenomena that wouldn't show up naturally in any reasonable sample of 5,000 messages (you can't sample your way to good idiom/negation/noun-class coverage). FLORES + ACES follow the same split for the same reason.

**Real-mined**: stratified sample from the production `usage.log` + per-user JSONs on Spark. PII-scrubbed (`ongiini.pii.sanitize`). Per-user capped at 3 candidates so chatty users don't dominate. Sebastian's manual review filters out proper-noun PII (church names, village names, personal names) the regex scrubber can't catch.

**Crafted**: I author the 80-item challenge subset and 20-item formal subset. Items are tagged with phenomenon labels before they go to the translator, so when results come back we get per-phenomenon scores for free.

### Both Oshindonga AND Oshikwanyama in one pass

Same EN sources → Elizabeth translates each into both languages. Doubles her hours (~40h total) but gives us **two complete eval sets from one curation effort**. Comparing them post-hoc tells us where the two related languages diverge (lexicon vs. morphology vs. grammar), and where translators (Claude / Gemma / NLLB) handle them differently.

### Blind test split: 20%

A deterministic random 20% (80 items) is marked `in_blind_split=true`. These items must **never be used during prompt tuning**. Headline scores in any future report are computed only on this held-out split. Without this discipline, we'd over-fit our prompts to the eval set itself.

---

## Tagging schema

Each v2 item carries:

```
length_bucket      S | M | L  (computed from word count)
domain             chat | formal | religious | community | challenge
phenomenon_tags    semicolon-separated list (multi-tag allowed)
provenance         v1_retained | real_mined | crafted | formal_drafted
in_blind_split     bool — held-back 20%
```

Multi-tag is essential. Real items naturally hit multiple phenomena: "She didn't bring the 12 forms by Friday" simultaneously tests `negation`, `numbers_dates`, `tense_aspect`, and (depending on context) `pronoun_coreference`.

---

## What we deliberately did NOT do

- **Use only crafted items** — they wouldn't reflect real distribution; we'd measure the model against our imagination, not against users.
- **Use only real items** — we'd miss the phenomena (negation, idiom) that don't naturally appear in any small random sample.
- **Have Elizabeth author EN source items in addition to translating** — separation of concerns matters for measurement validity. Her hours go into careful translation, not into item authoring.
- **Try to cover all Namibian languages in v2** — Otjiherero, Khoekhoegowab, Rukwangali, Silozi all need their own eval sets. v3+.
- **Run the eval automatically in CI** — useful later; this round just creates the dataset.
- **Score Claude vs Gemma now** — that's a separate task after Elizabeth returns. We'll compute chrF / BLEU automatically and add a manual 1–5 score per item, per language, per system.

---

## Known limitations

1. **One translator = one dialect**. Oshindonga has regional variation; Elizabeth's translations represent her variety. We accept this rather than try to span dialects with a single annotator. Document this when reporting results — "Claude scored 3.4/5 on Elizabeth-Oshindonga" is the honest framing.

2. **No back-translation check**. A more rigorous protocol would have a second translator back-translate Elizabeth's output to English and check meaning preservation. Out of scope for this round; budget went into depth (two languages) instead of dual-annotator validation.

3. **Mined items may carry register bias**. Real production messages skew toward help-seeking (people message Ongiini when they need something). We don't have a representative sample of unsolicited casual chat. The crafted challenge subset partially compensates.

4. **No long-form (paragraph) test**. Multi-sentence items are 2–4 sentences each. Real documents (CV, formal letter) can be much longer. v3 could add a small "document-level" bucket if relevant.

---

## Sources

The research that informed these decisions is preserved at
`~/.claude/plans/swirling-toasting-bunny-agent-a3c00cab48fb94b4e.md`
(the Phase-1 research summary from May 2026), with citations to:

- FLORES-200 README + Goyal et al. (FLORES-101)
- NTREX-128 (Federmann et al., SUMEval 2022)
- MAFAND-MT (Masakhane / Adelani et al.)
- AfriMTE & AfriCOMET (Wang et al. 2023)
- SSA-COMET (2025)
- AfroBench (ACL 2025)
- ACES challenge sets (Amrhein et al. 2022)
- "Is Context Helpful for Chat Translation Evaluation?" (Farinha et al., TACL 2024)
- Negation as MT error source (2020)
- Quality and Quantity of MT References (Zouhar et al. 2024)
- Significance tests for MT (Koehn 2004)
