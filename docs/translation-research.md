# Translation layer — research notes

**Status:** Research only. Not implemented. Revisit when ready to design the actual integration.

The foundation has promised on `ongiini.ai` that Oshiwambo (and later Otjiherero, Damara-Nama) will be supported "via a translation layer that converts to and from English under the hood, while keeping Gemma 4 as the brain." This document is the research that answers the narrowed question Sebastian asked first: **where would the translations actually come from?**

Implementation architecture — where in the request pipeline translation sits, latency, mem0 hooks, tool-output wrapping — is out of scope. We come back to it later.

---

## Why this came up

A Namibian user from Kavango East wrote to Ongiini in Oshiwambo (a proverb / riddle about ancestry: *"Hambelela nyokokulu ngeno nyoko inadalwa otashiti ngiini?"*) and bounced when we couldn't help them. We then tested **Gemma 4 directly** against the same phrase. It:

- Mis-identified the language as **Otjiherero** (it's Oshiwambo)
- Fabricated greetings in fake-Bantu (*"Mwaa, o mwa mbe mbe?"*, *"Ondu ku mbe mbe"* — both word salad)
- Invented a wrong meaning (*"lost or disturbed"*) for *otashiti ngiini* (actually means roughly *"how does it say / what does it mean"*)

The strict EN/AF redirect we ship today **loses these users honestly**; letting Gemma try would lose them more harmfully. The current behaviour is the right floor; the question is how to go above it.

---

## The landscape — what we surveyed

### Commercial translation APIs — none cover Oshiwambo

| Provider | Lang count | Oshiwambo? | Notes |
|---|---|---|---|
| Google Translate | ~130 | ✗ | Open community request to add Oshiwambo, unanswered |
| Microsoft Azure Translator | ~130 | ✗ | Same profile as Google |
| Amazon Translate | ~75 | ✗ | Narrower coverage |
| DeepL | ~29 | ✗ | Europe-heavy; smallest list |
| **Vulavula** (Lelapa AI) | ~4 | ✗ | South-African startup behind InkubaLM. EN / Afrikaans / isiZulu / Sesotho today, "more sub-Saharan coming". **Most aligned commercial roadmap; not shipping Oshiwambo yet.** |
| **Khaya** (Ghana NLP) | ~10 | ✗ | Translation + ASR + TTS API. Ghana / East Africa focus (Twi, Ewe, Ga, Yoruba, Dagbani, Kikuyu, Luo). Excellent work, wrong region. |
| TranslatorMind & similar wrapper sites | "unlimited" | "supports" but ✗ | Marketing pages claiming Oshiwambo support; underneath they call a frontier LLM (GPT-4 / Claude / Gemini). No API of their own. Disclaimer: *"may not be perfect, use as a guide."* |

### African voice-AI / language-AI startups (also no Oshiwambo)

| Startup | Lang focus | API? | Notes |
|---|---|---|---|
| **Spitch AI** (Lagos) | Yoruba, Hausa, Igbo, Amharic | Yes — pay-as-you-go ASR + TTS | West / Horn of Africa focus; no Namibia roadmap |
| **Intron Health** (Nigeria) | Nigerian medical/legal | Yes | Healthcare-specific; unrelated coverage |
| **Awarri** (Nigeria) | 5 low-resource Nigerian | Government-backed LLM in development | Not a commercial product yet |
| **NKENNEAi** (Nigeria) | Tonal languages | API in development | $1M NSF Phase II; Oshiwambo not tonal |
| **Cohere Aya / Aya Expanse** | 23 (Expanse) / 70+ (Tiny Aya) | Yes via Cohere API | Broad multilingual; Oshiwambo absent from documented support lists |

Pattern: **strong African-language coverage is being built out for West, East, and South-African languages first. Namibian indigenous languages are notably absent from every commercial offering.**

### Open-source multilingual MT models

| Model | Lang count | Oshiwambo? | Notes |
|---|---|---|---|
| Meta **NLLB-200** | 200 | ✗ | Confirmed via the FLORES-200 README — Ndonga, Kwanyama, Herero, Khoekhoegowab, Lozi, Rukwangali are **all absent**. 55 African languages covered, none Namibian. |
| Meta NLLB-200 Distilled | same | ✗ | Smaller variant; same language list |
| Lelapa AI **InkubaLM** | 5 | ✗ | Swahili, Yoruba, IsiXhosa, Hausa, isiZulu only |
| Helsinki **OPUS-MT** | many | ✗ | No Oshiwambo↔EN pair published |
| Google **MADLAD-400** | 400+ | partial / unverified | Includes many very-low-resource languages; specific Namibian-set status not confirmed |

### Research projects and specialised corpora

This is where the actual raw material is.

| Project | What it is | Why it matters |
|---|---|---|
| **"Participatory Translations of Oshiwambo"** (Microsoft Research / AfricaNLP@ICLR 2022, updated since) | **~7,500** Oshindonga↔English parallel sentences. Largest Oshiwambo parallel corpus to date. Community-translated. | Smallest viable training set for fine-tuning. Mission-aligned (community-led, not extractive). |
| **Meyabase / Osheng** ([github.com/meyabase/oshiwambo](https://github.com/meyabase/oshiwambo)) | Active Namibian developer initiative. Oshindonga + Oshikwanyama corpus build. Sources: dictionaries, Bible texts, online articles. Most repos private. Contact: `axel@meyabase.com` | **Most aligned partnership opportunity** — Namibian project working on exactly this. Worth a direct outreach. |
| **Bible translations** | Full Bibles in Oshindonga, Oshikwanyama, Otjiherero, Khoekhoegowab, Rukwangali, Lozi | Standard low-resource MT fallback. ~30k aligned verses/language. Formal register, not conversational. |
| **Wikipedia** | Tiny Oshindonga and Oshikwanyama editions | Useful as pretraining augmentation; tiny alone. |
| **Lelapa AI / Masakhane / NUST / UNAM ecosystem** | African NLP research community | Natural partners if we want to fund or collaborate on building this. |
| **Translate4Africa, Anglopremier** | Commercial human translators | Out of scope for real-time MT. Useful for one-off curated content and as evaluation references. |

### Whisper coverage for inbound voice

Whisper-large-v3-turbo (used in `audio.py`) claims 99 languages. **None of Oshiwambo, Otjiherero, Khoekhoegowab, Lozi, Rukwangali, Thimbukushu are in the 99.** A voice note in any of them will be mis-transcribed — likely tagged as Xhosa, Zulu, or Swahili. Voice support requires its own ASR story, not just MT.

---

## Honest verdict

**There is no off-the-shelf path in 2026.** After an exhaustive sweep of mainstream MT APIs, African-focused commercial APIs, frontier-LLM-wrapper services, and open-source multilingual MT models, **not one ships Oshiwambo today** — nor any other Namibian indigenous language.

Vulavula has the most-aligned commercial roadmap commitment but isn't there yet. Everyone else is building elsewhere first.

The only available raw material is:

- ~7,500 Oshindonga↔English sentences (Microsoft Research participatory corpus)
- The Meyabase / Osheng GitHub project
- Religious / Bible corpora (formal register, ~30k aligned verses per language)
- Wikipedia (tiny)
- A research community (Masakhane, Lelapa, NUST/UNAM) working on this

Building Oshiwambo translation for Ongiini is a **real engineering project**, not a config change. It cannot be solved by signing up to a vendor.

---

## Four paths forward, ranked

### Path A — Fine-tune NLLB-200 on the Microsoft corpus, run on the Spark (recommended)

1. Start from **NLLB-200-distilled-600M** — small enough to run alongside Gemma 4 on the Spark without GPU contention
2. Fine-tune on the Microsoft ~7.5k Oshindonga↔EN corpus, augmented with Bible parallel data and Wikipedia
3. Serve Oshindonga↔EN as a separate FastAPI route on the same box
4. Call it before Gemma when Whisper or a separate language detector flags non-EN/AF input

**Pros:** stays on-prem (matches "no US cloud"), open weights, no licensing, realistic quality (BLEU 12-25 published for similar low-resource fine-tunes — usable with light correction).

**Cons:** ML engineering effort (1-2 weeks focused work). BLEU 12-25 isn't great — informal / proverbial speech will trip it. May need permission for downstream use of the corpus.

**Cost:** zero ongoing; one-time training run; ongoing inference on existing hardware.

### Path B — Few-shot retrieval prompting via Gemma + the corpus

1. Embed the ~7.5k Oshindonga↔EN pairs in the existing qdrant
2. On inbound Oshiwambo, retrieve the 10 most similar parallel examples
3. Prompt Gemma: *"Translate this Oshindonga sentence to English. Examples: [retrieved pairs]. Translate: [user message]"*

**Pros:** no new model, no training infra, reuses qdrant + Gemma, ships in days.

**Cons:** quality bounded by Gemma's prior (poor — confirmed by the test above). Each request becomes 10-shot — high prompt-token cost. Confabulation risk doesn't fully go away.

**Verdict:** good prototype, bad production architecture.

### Path C — Partnership + participatory continuation

1. Reach out to the Microsoft Research / Lelapa AI / NUST team behind the participatory work
2. Partner with a Namibian organisation (NUST, UNAM, teachers' association) to keep collecting parallel data
3. Combine Path A's technical track with this ongoing data-collection track
4. Position Ongiini's `/statistics` page and research-data flows (Privacy Policy Section 7) as a contribution back to the community effort

**Pros:** mission-aligned, compounds over time (every interaction can contribute, with consent), attracts funding and academic legitimacy.

**Cons:** slow, depends on people not code, doesn't ship this week.

### Path D — Bridge to a frontier LLM API as a stopgap

Route Oshiwambo / Otjiherero / Damara-Nama inputs through **Claude / GPT-4 / Gemini as a translator only** — not as the reply engine. The model receives the user's message in the local language and translates to English; Gemma 4 answers in English; the same frontier model translates the answer back. Gemma stays the brain; the frontier model is a thin translation membrane.

Quality test (Claude, on real Oshiwambo phrases sent to Ongiini in May 2026): correctly identified Oshiwambo vs Otjiherero, parsed morphology of a complex proverb, gave meaningful literal translations with appropriate hedging. **Meaningfully better than Gemma 4** at this task — but **not native-speaker grade**. Formal/written text translates well; slang, dialects, code-switching, and live idioms will trip it. Acceptable for "knowledgeable friend on WhatsApp" use cases with the right disclosure.

**Pros:**
- Solves the "we lose every indigenous-language user at hello" problem **today**, not in 6+ weeks
- Zero ML engineering effort — just a per-language routing rule
- Operationally proven APIs, no infra work
- Doesn't preclude Path A — build Path A in parallel, migrate when it's ready
- Quality is a real step-change over Gemma-alone (the actual alternative today)

**Cons:**
- **Principles trade-off.** Claude's API runs on AWS; GPT-4 on Azure; Gemini on Google Cloud. All three are US-cloud. The foundation site explicitly says *"No US cloud provider sits in the pipeline."* Routing user inputs through any of them as a translation layer walks that back. Has to be disclosed in the Privacy Policy AND on the page, or the on-page promise becomes dishonest.
- **Per-token cost.** Frontier-LLM API rates at modest user volume work out to ~low single-digit dollars per active user per month for the translation round-trip. Not a financial blocker, but a recurring obligation rather than zero.
- **Vendor lock-in risk.** If Anthropic / OpenAI / Google deprecate a model, change pricing, or shut down the API, the translation layer breaks or becomes uneconomic. Path A is invariant to all three.
- **Quality ceiling.** Frontier LLMs miss dialect-specific idioms and modern colloquial speech, same as Path A would. Just at a slightly higher floor — on someone else's servers.

**Verdict on Path D:** This is a real option that has to be **explicitly considered and either accepted or declined**, not ignored. Two defensible stances:

- *"We'd rather lose Namibian indigenous-language users for 6 months than walk back the no-US-cloud promise."* — **mission-purity.** Keeps the foundation's identity coherent. Costs an unknown number of would-be users in the interim.
- *"We'd rather serve those users imperfectly via Claude/GPT-4 now, disclose it transparently, and migrate to local Path A as soon as it works."* — **user-first.** Costs a temporary deviation from the no-US-cloud promise and the rhetorical clarity that comes with it.

Either choice is honest as long as it's a deliberate choice, not a default.

### Combining A + D — the strongest hybrid

A and D aren't mutually exclusive. Four ways they layer cleanly:

**1. Phased rollout** (the obvious one).
Path D ships in week 1 — today's Namibian-indigenous-language users get served. Path A builds in parallel over 2–6 weeks. When A's BLEU is acceptable, routing switches from D to A. The US-cloud dependency becomes **time-bounded and transparently disclosed** — a documented temporary detour, not a permanent retreat from the no-US-cloud promise. The page disclosure can honestly read: *"While we build a local Namibian-language translation layer, we temporarily route Oshiwambo / Otjiherero / Damara-Nama through Anthropic / OpenAI / Google for translation only. Replies are still generated locally. We expect to switch to fully-local by [DATE]."*

**2. Training-time support** (the underappreciated cluster).
Beyond runtime translation, D can play several roles in *building* Path A — each independently valuable, all stackable. This is where D's leverage is highest:

  **2a. Synthetic parallel-corpus generation.** D translates **large monolingual or parallel resources** (full Oshiwambo Bible, Wikipedia, news archives, UN parallel documents) to produce a synthetic Oshiwambo↔English corpus orders of magnitude larger than the ~7.5k-sentence Microsoft baseline. A is fine-tuned on the combined real + synthetic corpus. Standard low-resource MT technique ("knowledge distillation from a teacher model"). Typical lift: 5–10 BLEU. Cost: one-time ~few-hundred-dollar API spend.

  **2b. Back-translation augmentation.** Standard MT-data-augmentation trick. Start with **monolingual Oshiwambo text** (Bible, Wikipedia, NUST publications, online forums). D translates to English. Each translation produces a new (Oshiwambo, English) training pair without needing pre-aligned data. Effectively doubles the training set per piece of monolingual Oshiwambo we can find.

  **2c. Corpus cleaning.** The Microsoft 7.5k corpus has known noise — misalignments, dialect mixing, OCR errors from scanned sources. D verifies each pair: *"Is this Oshindonga sentence a valid translation of this English sentence?"* Bad pairs get filtered out. A smaller but cleaner training set typically beats a noisy larger one.

  **2d. Curriculum + difficulty grading.** Sort training examples by complexity (greetings → simple declaratives → complex morphology → idioms → proverbs). Curriculum-learning: train on easy first, harder progressively. D rates difficulty per example. Speeds convergence and improves final quality on hard cases.

  **2e. Test-set construction.** Build a thoughtful held-out evaluation covering distinct domains (everyday speech, healthcare, schoolwork, government forms, religious register, dialect variations, slang). D generates challenging test sentences. Catches A's blind spots before real users do.

  **2f. Active-learning loop** (post-ship). A translates production-style inputs; D grades the output (*"good / acceptable / wrong, plus why"*); low-graded outputs feed back as new training examples with D's correction. Continuous improvement without manual annotation. Compounds over the months after A's first version ships.

  **2g. Dialect bridging.** The Microsoft corpus is Oshindonga only. Oshikwanyama is the other major Wambo dialect. D converts Oshindonga sentences → Oshikwanyama (with hedging) to seed A's training data for the second dialect. Quality risky but better than no Kwanyama coverage.

Layer them as compute and time allow. **(2a) + (2b) + (2c) probably triple the effective training-data quality vs the bare 7.5k corpus.** (2f) compounds over the months after A ships.

**Honest caveats on D-as-training-source:**
- **Quality cap.** A model fine-tuned on D's translations can't easily be *better* than D on the same data — it can only match D's quality. If D's Oshiwambo is mediocre on some construction, A will inherit that ceiling.
- **Bias propagation.** Any systematic error D makes (wrong idiom, wrong dialect default) gets baked into A. A native-speaker QA pass on the synthetic data before training catches a lot of this.
- **Native verification still needed.** The gold standard remains the participatory community translation work (Path C). Synthetic data accelerates A's progress; it doesn't replace the underlying need for real Oshiwambo speakers verifying the training set. Path C complements 2a–2g rather than competing with them.

**3. Quality fallback** (post-A).
Even after A is the production translator, use D as a **runtime fallback** for inputs A handles poorly — long-form text, complex morphology, sentences A flags as low-confidence. A handles 95%+ of traffic on-prem; D handles the edge cases. Minimises but doesn't zero the US-cloud dependency, on a clearly bounded slice of traffic.

**4. Evaluation oracle** (offline only).
Use D as the **benchmark** for evaluating A's quality over time, never on the hot path. Periodically translate held-out test sets through both; the divergence is a signal for what to fine-tune next. Frontier-model touches the offline pipeline only — no production user traffic, no US-cloud dependency at serve time.

These layer cleanly. **A + D-phased (#1) is the minimum viable hybrid.** **A + D-phased + D-distillation (#1 + #2) is probably the strongest practical path** — solves the immediate user problem, accelerates Path A's quality with a one-time synthetic-data investment, then graduates to local-only serving. The principles trade-off becomes much more defensible because the US-cloud detour is explicitly temporary and replaced by a tangible local capability that's being actively built.

### Recommendation

**A + D phased, augmented by D-distillation, with C running alongside.**

- **Week 1**: ship D for Oshiwambo / Otjiherero / Damara-Nama. Disclose transparently on the page and in the Privacy Policy. Stop losing indigenous-language users at hello.
- **Weeks 1–2**: data preparation in parallel.
  - **2a** — D translates the full Oshiwambo Bible + Wikipedia + any accessible parallel corpora into a synthetic Oshiwambo↔English dataset
  - **2b** — D back-translates monolingual Oshiwambo text (NUST corpus, online articles) to expand further
  - **2c** — D verifies the existing ~7.5k Microsoft pairs, dropping the misaligned / noisy ones
  - **2e** — D drafts a held-out test set covering ~6 register/domain buckets
  - **2g** — D drafts a first Oshikwanyama synthetic corpus from the Oshindonga set
- **Weeks 2–4**: build Path A. Fine-tune NLLB-200-distilled-600M on the combined real + synthetic corpus, ideally with curriculum learning per **2d**. BLEU iteration against the test set built in **2e**.
- **Week ~4–6**: when A's BLEU passes the bar (≥15 conservatively for Oshindonga), switch the primary routing from D to A. Update the page disclosure to remove the US-cloud caveat. Keep D as offline evaluation oracle (#4) plus optional low-confidence runtime fallback (#3).
- **Post-launch**: enable the **2f** active-learning loop. Every production translation gets graded by D; the lowest-graded outputs become next-cycle training examples. A's quality compounds without manual annotation.
- **Throughout**: Path C — partnership with the Microsoft Research participatory team, Lelapa AI, NUST / UNAM, and Meyabase (contact `axel@meyabase.com`). Slow burn, mission-aligned, compounds over years. Crucially: **community-verified data overrides D-generated data wherever it exists** — the synthetic corpus is scaffolding, not a substitute.

Path B (few-shot Gemma + retrieval) is no longer needed once D is on — D is strictly better for the same prototype use case at similar cost-to-ship.

Other Namibian languages (Otjiherero, Khoekhoegowab, Lozi, Rukwangali, Thimbukushu) follow the same template. Quality at launch will be lower per language since corpora are smaller, but the framework generalises — and D bridges the quality gap especially well for languages where Path A's training data is thinnest.

Other Namibian languages (Otjiherero, Khoekhoegowab, Lozi, Rukwangali, Thimbukushu) follow the same template with even less corpus data — quality at launch will be lower per language, but the framework generalises.

---

## Open decisions for when we revisit

1. **Oshindonga first, or all Wambo dialects?** The Microsoft corpus is Oshindonga-only. Build for Kwanyama in parallel or after?
2. **GPU budget on the Spark.** NLLB-200-distilled-600M needs ~2.5 GB GPU RAM. Comfortable but non-zero alongside Gemma 4 26B.
3. **Reach out to Lelapa / Microsoft Research / Meyabase?** That's the difference between Path C being a real thing vs theoretical.
4. **Transparent vs visible translation UX.** Should the user see the translated-from-English reply only, or both their original-language version and the English? Visibility builds trust but is a different design.

---

## Concrete next steps when we come back to this

1. Pull the Microsoft Oshiwambo parallel corpus — confirm licence, format, exact count, dialect breakdown
2. Test Whisper-large-v3-turbo against 5 real Oshiwambo voice notes — confirm the mis-identification hypothesis; decide whether voice support waits for a dedicated ASR
3. Quick fine-tune of NLLB-200-distilled-600M on the corpus — measure BLEU on a held-out split; if ≥15, Path A is viable; if <10, more data first
4. One-paragraph outreach email to Lelapa AI, the Microsoft Research participatory team, and `axel@meyabase.com` — no commitment, just opening the door

---

## Sources

- [200 languages within a single AI model — Meta AI (NLLB-200)](https://ai.meta.com/blog/nllb-200-high-quality-machine-translation/)
- [FLORES-200 language list — facebookresearch/flores](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)
- [Participatory Translations of Oshiwambo — Microsoft Research](https://www.microsoft.com/en-us/research/publication/participatory-translations-of-oshiwambo-towards-sustainable-culture-preservation-with-language-technology/)
- [Meyabase Oshiwambo repository](https://github.com/meyabase/oshiwambo)
- [InkubaLM — Lelapa AI](https://lelapa.ai/inkubalm-a-small-language-model-for-low-resource-african-languages/)
- [Vulavula — Lelapa AI products](https://lelapa.ai/products/vulavula/)
- [Khaya — Ghana NLP](https://translation.ghananlp.org/)
- [Spitch AI](https://spitch.app/)
- [Cohere Aya Expanse documentation](https://docs.cohere.com/docs/aya-expanse)
- [Google Translate community request for Oshiwambo](https://support.google.com/translate/thread/100001385/)
- [Oshikwanyama Dictionary on Google Play](https://play.google.com/store/apps/details?id=com.ad123.oshikwanyamadictionary)
- [Translate4Africa Oshiwambo human-translation services](https://translate4africa.com/languages/oshiwambo-translation-services/)
- [Anglopremier Oshiwambo translation services](https://www.anglopremier.com/en/translations-namibia/oshiwambo/)
- [New Oshiwambo Bible translation (12-year, N$25M project) — Namibia Economist](https://economist.com.na/22037/headlines/new-oshiwambo-bible-translation-will-take-12-years-and-n25-million-to-complete/)
