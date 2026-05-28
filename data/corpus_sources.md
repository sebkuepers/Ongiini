# Parallel & monolingual corpus sources for Oshindonga + Oshikwanyama

A living catalog of identified text sources for building a parallel /
monolingual corpus to support machine translation, language modeling,
and other NLP work on Namibian indigenous languages.

**Maintained by:** Common Intelligence Foundation
**Started:** 2026-05-28
**License of this document:** CC-BY-4.0

---

## Status legend

- ✅ Confirmed parallel + accessible + ready to harvest
- 📥 Confirmed available, harvest still pending
- ⏳ Identified, needs outreach for permission
- 🔍 Needs investigation (URL/contact known, content not yet verified)
- ⚠️ Has caveats (lower quality, partial coverage, licensing complications)
- ❌ Checked, not parallel / not usable
- ➕ Add new entries above this line as discovered

---

## TIER 1: Government & institutional parallel publications

These are the highest-value "free" sources because they're already
translated by professional translators and (mostly) have permissive
licenses for research use.

### 1.1 Namibian Constitution — Oshiwambo (Oshikwanyama dialect)

- **Status:** 📥
- **Publisher:** Konrad-Adenauer-Stiftung (KAS) Foundation Office Namibia & Angola
- **Year:** 2017
- **Languages:** English (original) + Oshikwanyama (translation)
- **URL (announcement):** https://www.kas.de/en/web/namibia/veranstaltungsberichte/detail/-/content/kas-uebersetzt-namibische-verfassung-auf-oshiwambo
- **URL (Oshiwambo version):** https://www.kas.de/en/web/namibia/single-title/-/content/the-constitution-of-the-republic-of-namibia-oshiwambo-version
- **URL (English version, LAC):** https://www.lac.org.na/laws/annoSTAT/Namibian%20Constitution.pdf
- **URL (English ILO NATLEX):** https://natlex.ilo.org/dyn/natlex2/natlex2/files/download/9565/NAM9565%202.pdf
- **Estimated volume:** ~148 articles + preamble × ~10-15 sentences ≈ **1,500-2,000 parallel pairs**
- **License:** KAS Foundation document; permits research/educational use with attribution. Confirm via outreach.
- **Domain:** Legal/constitutional. Complementary to news + WhatsApp.
- **Quality:** Professional translation, high quality
- **Next steps:** Download both PDFs, build extractor + sentence-aligner, email KAS Namibia office for explicit permission letter

### 1.2 NIED — National Institute for Educational Development curriculum

- **Status:** 📥
- **Publisher:** Ministry of Education, Arts and Culture (MoEAC) / NIED
- **Languages:** English (master) + Oshindonga + Oshikwanyama (translations of curriculum)
- **URL (NIED main):** http://www.nied.edu.na/
- **Specific docs identified:**
  - Pre-Primary Syllabus Oshindonga (2014): http://www.nied.edu.na/assets/documents/02Syllabuses/01PrePrimary/04Oldsyllabuses(Upto2014)/Pre_Primary_Syllabus_Oshindonga_Version_2014.pdf
  - Pre-Primary Syllabus Oshikwanyama: https://www.yumpu.com/xx/document/view/9022881/oshikwanyama-pre-primary-syllabus-nied
  - First Language Syllabus Grades 1-3 Oshindonga (2015): http://www.nied.edu.na/assets/documents/02Syllabuses/02JuniorPrimary/01Syllabuses/06Oshindonga/JP_Syllabuses_FL(Oshindonga)_Mar.2015.pdf
  - 2022 Textbook Catalogue (Senior Primary): http://www.nied.edu.na/assets/documents/02Syllabuses/07Textbookcatalogue/2022_Textbook_Catalogue_for_Senior_Primary_Phase.pdf
  - National Curriculum for Basic Education 2016: https://www.nied.edu.na/assets/documents/05Policies/NationalCurriculumGuide/National_Curriculum_Basic_Education_2016.pdf
- **Estimated volume:** Syllabuses + textbooks combined ≈ **500-2,000 parallel pairs** (varies by which textbooks have direct parallels)
- **License:** Government-owned. Generally usable for non-commercial research; confirm with NIED legal contact
- **Domain:** Education-formal. Complementary domain coverage.
- **Quality:** Professional, edited; pedagogical register
- **Caveats:** Textbooks may have figures/images that don't translate well; focus on text-only sections
- **Next steps:** Download all available syllabus PDFs + email NIED for textbook catalog; specifically request the "Nongonona Elaka" Oshikwanyama series

### 1.3 Electoral Commission of Namibia (ECN) — voter education

- **Status:** ⏳
- **Publisher:** ECN
- **Languages:** English + Oshiwambo + 4 other indigenous languages
- **URL:** https://www.ecn.na/voter-education/
- **Note:** Confirmed multi-language voter education materials in Oshiwambo, Rukwangali, San, Setswana, Silozi, Tjiherero
- **Estimated volume:** ~100-300 pairs per election cycle (low volume, but high frequency of use → high quality translations)
- **License:** Government public-domain-ish
- **Domain:** Civic/instructional
- **Next steps:** Email ECN comms team for digital copies of all voter education materials; check 2014/2019/2024 election archives

### 1.4 Ministry of Health and Social Services — COVID-19 + ongoing health communications

- **Status:** 🔍
- **Publisher:** MoHSS (Ministry of Health and Social Services)
- **URL:** https://mhss.gov.na/
- **Confirmed (via WHO Africa report):** "Communication materials were produced and translated in local languages for radio, television and print media"
- **Estimated volume:** Highly variable. Probably 100-500 pairs of print materials.
- **Caveats:** Most translation went to RADIO (audio) which doesn't help text MT. Need to verify what's in print/text form.
- **Next steps:** Email MoHSS communications office; check ReliefWeb + WHO Africa for shared print materials

### 1.5 Office of the President & other ministerial publications

- **Status:** 🔍
- **Notes:** Some major state announcements (Independence Day speeches, national addresses) get translated for radio broadcast and occasionally published. Volume unknown without investigation.
- **Next steps:** Check archives of major state speeches, also Namibia Gazette

---

## TIER 2: Religious texts — high-volume parallel

### 2.1 Christian Bible — Oshindonga (Ombiimbeli Ondjapuki)

- **Status:** ⏳
- **Publisher:** Bible Society of Namibia (Bibel-Liga Namibia) + various missionary presses
- **Languages:** Oshindonga + numerous English translations exist
- **Estimated volume:** Full Bible ≈ **31,000 verses**. Approximately 31,000 parallel pairs.
- **Quality:** Professionally translated, theologically reviewed, well-edited
- **License:** Modern translations are copyrighted; older translations (>50 years) may be public domain. Some research-friendly licenses exist.
- **Notes:**
  - The JW300 corpus (used widely in academic MT) is Watch Tower (Jehovah's Witnesses) publications, NOT mainstream Bible — separate from Bible Society
  - Modern Bible Society Oshindonga editions: 1991 (revised 2002)
  - **AfricaNLP papers have already used Oshiwambo Bible** — see academic source #4.1 below
- **Next steps:** Contact Bible Society of Namibia for research-use permission. Check open online Bible APIs (BibleGateway, GetBible, etc.) for already-aligned versions.

### 2.2 Christian Bible — Oshikwanyama

- **Status:** ⏳
- **Same as 2.1 but for the Oshikwanyama variant**
- **Estimated volume:** ~31,000 verses
- **Notes:** Oshikwanyama Bible (Ombiimbeli Yotembele) — modern editions ~1986+

### 2.3 Watch Tower (Jehovah's Witnesses) publications — JW300 corpus

- **Status:** ✅
- **Notes:** Already extracted as the JW300 corpus, widely used in MT research
- **URL:** https://opus.nlpl.eu/JW300.php (Helsinki-NLP OPUS)
- **Estimated volume for Oshindonga + Oshikwanyama:** Tens of thousands of parallel sentences (need to verify exact volume for these two specifically)
- **Quality:** Professional translation by JW — religious-doctrinal register, very consistent style
- **License:** Open for research use via OPUS Helsinki distribution
- **Caveats:** Heavy religious-doctrinal register; may bias models toward religious vocabulary. Probably some Oshiwambo is included but verify which dialect.
- **Next steps:** Download from OPUS, filter for ODG + OKW, check volume

---

## TIER 3: Monolingual sources for community-translation pipeline

These don't have English parallels but are valuable for:
- Continued pretraining of language models
- Back-translation (community translators produce EN side)
- Sentence-piece tokenizer training
- Language identification model training

### 3.1 The Namibian — Oshiwambo section

- **Status:** ⏳ (need editor permission)
- **Publisher:** Namibia Media Holdings / The Namibian newspaper
- **URL (section):** https://www.namibian.com.na/category/oshiwambo/
- **Languages:** Oshiwambo (mixed Oshindonga / Oshikwanyama / standard Wambo)
- **Publishing frequency:** ~3-5 articles per day
- **Estimated archive:** 3-5 years × ~1,500 articles/year = **4,500-7,500 articles**
- **Estimated pairs IF community-translated:** ~12 pairs/article × 700-1,000 selected articles = **8,000-12,000 parallel pairs**
- **Quality:** Edited, native-speaker journalism. High quality.
- **License:** Copyright protected. **Outreach needed.** See email template in `/docs/corpus_outreach_templates.md` (to be created).
- **Domain:** News (politics, business, sports, culture, community)
- **Next steps:**
  1. Email editor at The Namibian for explicit permission
  2. If granted: build respectful scraper (rate-limited, robots.txt-respecting)
  3. Curate ~700-1,000 articles for community translation pipeline

### 3.2 New Era — Indigenous language sections

- **Status:** 🔍
- **Publisher:** New Era Publication Corporation (government-owned)
- **URL:** https://neweralive.na/
- **Note:** Government-owned newspaper, likely also has Oshiwambo / other indigenous-language sections. Government ownership may simplify licensing.
- **Next steps:** Visit website, check for indigenous-language sections, document URLs

### 3.3 Namibian Broadcasting Corporation (NBC) — indigenous language radio scripts

- **Status:** 🔍
- **Publisher:** NBC (Namibian Broadcasting Corporation, government-owned)
- **URL:** https://www.nbcnamibia.com/
- **Note:** Broadcasts in 10+ Namibian languages. Scripts/transcripts of broadcasts could be a TEXT source even though primary medium is audio.
- **Estimated volume:** Unknown, potentially very large if archived scripts exist
- **License:** Government-owned, but practical access uncertain
- **Next steps:** Contact NBC indigenous-language programming department; ask about archived scripts

### 3.4 University of Namibia (UNAM) — academic dissertations & research

- **Status:** 🔍
- **Publisher:** UNAM Library + various academic publishers
- **Note:** Linguistics, education, African studies dissertations sometimes include Oshindonga/Oshikwanyama text. Some may include parallel translations as appendices.
- **Next steps:** Contact UNAM Library; search UNAM research repository

---

## TIER 4: Academic & NGO existing corpora

### 4.1 AfricaNLP / Microsoft Research — Participatory Translations of Oshiwambo

- **Status:** ✅ (small) / 📥 (extract more)
- **Paper:** https://www.microsoft.com/en-us/research/wp-content/uploads/2022/04/participatory_translations_of_.pdf
- **Open review:** https://openreview.net/pdf?id=BFbg59zVUZc
- **Year:** 2022
- **Note:** Existing academic effort on Oshiwambo translation. Methodology + possible dataset to build on.
- **Next steps:** Contact authors; ask if dataset is available; check methodology for reuse

### 4.2 MAFAND-MT (Masakhane African MT) — sister-language transfer

- **Status:** ✅
- **URL:** https://github.com/masakhane-io/lafand-mt
- **Note:** 16 African languages. **NOT including Oshindonga/Oshikwanyama**, but includes related Bantu sister languages (Chichewa, Shona, isiZulu, Sotho, etc.) that may help transfer learning.
- **License:** Open
- **Use case:** Pretrain or fine-tune on sister Bantu languages first, then transfer to Oshindonga/Oshikwanyama

### 4.3 FLORES-200

- **Status:** ✅
- **URL:** https://github.com/facebookresearch/flores
- **Note:** 200 languages, **NOT including Oshindonga/Oshikwanyama directly**, but useful for tooling + comparable evaluation methodology
- **Use case:** Reference methodology; our eval set follows FLORES conventions

### 4.4 Helsinki-NLP OPUS — multilingual MT data

- **Status:** 📥
- **URL:** https://opus.nlpl.eu/
- **Note:** Open repository for sentence-aligned bilingual texts. Contains JW300 (mentioned in 2.3) and possibly other Oshiwambo sources.
- **Next steps:** Browse OPUS for any "Oshiwambo", "Oshindonga", "Oshikwanyama", "ndo", "kua" tagged corpora

### 4.5 OPUS Tatoeba — community-contributed sentence pairs

- **Status:** 🔍
- **URL:** https://tatoeba.org/eng/
- **Note:** Crowdsourced sentence pairs. Check for ANY Oshindonga/Oshikwanyama contributions. Likely very few but worth checking.

### 4.6 Common Voice (Mozilla) — speech corpus

- **Status:** ❌ (audio only, not text)
- **URL:** https://commonvoice.mozilla.org/
- **Note:** No text corpus, but Common Voice has audio recordings. For future ASR work.

### 4.7 Lelapa AI (Vulavula)

- **Status:** ❌ (no Oshiwambo coverage)
- **URL:** https://lelapa.ai/
- **Note:** South African startup doing African MT. NO Oshindonga/Oshikwanyama coverage as of May 2026. Potential future commercial partner for distributing our model.

---

## TIER 5: Speculative / hard to verify

### 5.1 Religious materials beyond mainstream Bible

- Catholic liturgical translations
- Lutheran World Federation publications (Lutheran church is very active in Owamboland)
- Adventist publications

### 5.2 Government Gazette — Namibian state legal publications

- **URL:** https://gazettes.africa/gazettes/na (third party)
- **Note:** Most gazette content is English. Could check if any laws/regulations have been parallel-published in Indigenous languages.

### 5.3 Translator-publisher textbooks

- Macmillan Namibia, Pearson Namibia, Namibia Publishing House (NPH)
- These produce school textbooks in Indigenous languages
- Commercial, copyright-protected, but research-license possibly negotiable

### 5.4 LDLT (Living Dictionaries) / Endangered Languages

- Living Tongues Institute for Endangered Languages
- ELDP (Endangered Languages Documentation Programme)
- Possibly archived Oshindonga/Oshikwanyama lexical data + sample texts

---

## Outreach log

Use this section to log every outreach email and response, so we know
who has been contacted and what was said.

```
[2026-XX-XX]  to: editor@namibian.com.na
              re:  Permission to use Oshiwambo section content for non-commercial
                   academic / language-preservation research
              status:  pending
              follow-up:  if no response in 2 weeks, resend with cc to deputy editor

[2026-XX-XX]  to: namibia@kas.de
              re:  Permission letter for Oshikwanyama Constitution PDF use in research dataset
              status:  pending

[2026-XX-XX]  to: <NIED contact tbd>
              re:  Catalog of Oshindonga/Oshikwanyama curriculum + research-use permission
              status:  pending

[etc.]
```

---

## Maintenance notes

- Add new sources ABOVE the speculative tier as discovered
- Update status whenever progress made (✅ confirmed, 📥 harvested, ⏳ awaiting permission, etc.)
- Log every outreach in the outreach log
- Periodically (every 2-4 weeks) review and prune dead leads
- Cross-reference with progress in `data/oshiwambo_eval/` (the published eval set work)
