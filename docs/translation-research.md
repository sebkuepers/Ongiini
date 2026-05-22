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

## Three paths forward, ranked

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

### Recommendation

**A + C in parallel.** A is the engineering track that produces usable Oshindonga↔EN MT in ~2 weeks. C is the relationship track that runs forever. B is a useful same-week prototype but shouldn't be the production architecture.

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
