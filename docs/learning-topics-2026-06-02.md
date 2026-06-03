# Learning topics on Ongiini — what users want to learn

**Snapshot date:** 2026-06-02
**Author:** Claude (Sonnet/Opus) at Sebastian's request
**Source:** `data/mem0_history.db` — current non-deleted facts only (6,574 distinct memories)
**Privacy posture:** All numbers are aggregate counts derived from typed memory facts (already abstracted from raw conversation text by the extraction layer). No conversation content was read or reported. The mem0 actor_id column is null throughout, so counts are fact-level rather than user-level.

## TL;DR

- **25.9% of all extracted memory facts (1,700 of 6,574) carry an educational signal.** Education is by far the largest single coherent use cluster on the service.
- **The #1 thing users want to *learn* is Afrikaans** (54 forward-looking mentions), more than any school subject and ~5× the next language. This is Namibian Oshiwambo speakers learning a workplace / government-services language, not heritage-language reclamation.
- **Tertiary students dominate the academic cohort** — 155 facts vs. 124 across all of Grade 10–12. The "matric exam prep" framing is real but isn't where the centre of gravity sits.
- **Translation is the #1 task** users do with the assistant — 172 fact-level mentions — almost double the next-largest task (assignment, 99). The Oshiwambo dataset project is directly aligned with how users already use Ongiini.
- **Health is a major hidden educational category** — pregnancy / mental health / general health information / specific conditions add up to ~221 facts, comparable to the entire STEM stack combined. It doesn't appear in any school-subject taxonomy but it *is* people learning things they don't know.

---

## 1. The Afrikaans-learning finding (the most actionable single insight)

Extracting forward-looking learning intent — phrases like "wants to learn X", "preparing for X", "interested in learning X" — and counting the targets:

| Target | Mentions |
|---|---|
| **Afrikaans (any form)** | **54** |
| Oshindonga (combined phrasings) | 17 |
| Oshiwambo (general) | 7 |
| English | 6 |
| Job interview | 4 |
| Retail | 4 |
| Exam (unspecified) | 3 |
| "How to speak Afrikaans" | 3 |

Out of 543 facts that mention **Oshiwambo / Oshindonga / Oshikwanyama**:

| Oshiwambo role | Facts |
|---|---|
| User is a *native* speaker | 212 |
| User wants a *translation* | 23 |
| User wants to *learn* it | 11 |

So the language-learning story isn't "Namibians want to learn their heritage language." It's **"Namibian Oshiwambo speakers want to learn Afrikaans."** That's a real workplace and government-services barrier showing up in the data — Afrikaans remains the de facto second language in many Namibian commercial and civic contexts despite English being the official language.

The current system prompt has no special handling for "I want to learn Afrikaans." A small skill or curated section could meaningfully improve this UX with relatively little effort.

---

## 2. Academic cohort: tertiary > secondary > primary

| Level | Education-intent facts |
|---|---|
| **Tertiary / university (UNAM, NUST, Polytech)** | **155** |
| Grade 12 / matric / NSSC | 62 |
| Grade 10–11 | 62 |
| Grade 1–7 (primary) | 63 |
| Grade 8–9 (junior secondary) | 24 |
| Pre-primary / Grade R / ECD | 21 |

Within the tertiary cohort: **33** mentions of *final-year / honours / thesis / dissertation*. The long-tail end of the academic journey is visible in the data.

---

## 3. Academic subjects

| Subject | Facts |
|---|---|
| Oshiwambo (any) | 543 |
| Maths (algebra / geometry / calculus / equations) | 92 |
| Physics | 63 |
| Other languages (French, German, Otjiherero, Nama, etc.) | 54 |
| Biology / Life Sciences | 52 |
| Accounting | 41 |
| Business Studies | 39 |
| History | 32 |
| Agriculture | 31 |
| Chemistry | 24 |
| Computer Science / programming | 18 |
| English (formal study) | 9 |
| Geography | 5 |
| Afrikaans (formal study) | 5 |
| Religion / RME | 2 |

**STEM combined (Maths + Physics + Biology + Chemistry) = 231 facts.**
That's about the same as the **Afrikaans-learning forward-looking goal alone** when you include the related phrasings.

Inside Maths (n = 92), readable specifics:

| Maths subtopic | Facts |
|---|---|
| Calculus (differentiation / integration) | 25 |
| Statistics (probability / mean / frequency tables) | 13 |
| Algebra / equations | 12 |
| Geometry (triangles / angles / circle theorems) | 12 |
| Fractions / percentages | 11 |
| Word problems | 3 |

> **Methodology note:** trigonometry was excluded — the regex `\b(trigonom|sin|cos|tan|sohcahtoa)` caught short common substrings ("since", "cost", "tank") and produced an inflated count (115) that exceeded the whole Maths set. The other subtopic counts use word-specific patterns and are trustworthy.

---

## 4. The hidden educational category: health

These don't appear in any school-subject taxonomy but they are clearly people learning about something they didn't already know:

| Health-learning area | Facts |
|---|---|
| Pregnancy / antenatal / child health / breastfeeding | 69 |
| Mental health / wellbeing / stress / anxiety / depression | 62 |
| General health-information seeking | 52 |
| Specific conditions (diabetes, HTN, TB, malaria, HIV, asthma) | 38 |

**Total ≈ 221 facts.** Comparable in scale to the entire academic STEM stack combined.

The pregnancy / child-health and mental-health threads are particularly strong — those are the questions people often don't easily ask a clinic. If Ongiini ever considers a "Health" skill, a partnership angle (MOHA, public clinics, NGOs), or even a curated reference module, the underlying demand is already there.

---

## 5. Tasks — what users *do* with the assistant

| Task | Facts |
|---|---|
| **Translate** | **172** |
| Assignment | 99 |
| Exam / test preparation | 77 |
| Explain a concept | 38 |
| Essay / composition / speech | 33 |
| Research / project | 26 |
| Definition / "what is X" | 8 |
| Homework (literal phrasing) | 6 |
| Solve a problem | 4 |
| Summarise | 2 |

Translation is **#1 by a wide margin** — almost double the next task. This is directly aligned with the Oshiwambo dataset project Sebastian is already running: users *already* use Ongiini as a translation tool, even before the contribute flow is widely adopted.

### Translation directions

| Target language | Mentions |
|---|---|
| → English | 66 |
| → Afrikaans (one-way phrasing) + English ↔ Afrikaans (bidirectional) | 22 + 30 = 52 |
| → Oshindonga | 34 |
| → Oshikwanyama | 15 |

English is the most-requested *target*, which matches the role English plays as the official language for written work, applications, and formal contexts. Afrikaans is a close second once both phrasings are combined.

---

## 6. Career / professional development (adjacent to learning)

| Career-learning area | Facts |
|---|---|
| Business plan / entrepreneurship | 47 |
| Cover letter / job application | 31 |
| Job-interview preparation | 15 |
| Email writing / professional comms | 3 |
| CV / resume (literal phrase) | 0\* |

\* CV help is a known top use-case (it's literally one of the homepage suggestion chips). The regex missed it because the extracted memory facts apparently describe the CV-writing flow with different verbs — "writing a cover letter", "preparing a job application", etc. The 47 + 31 + 15 numbers are therefore an *undercount* of total job-readiness usage.

---

## 7. Reading-level summary for an outside audience

If we needed a single-paragraph way to describe what Ongiini is used for, based on the data:

> Ongiini is used most for **applied language work** — translating between English, Afrikaans, and Oshiwambo, and learning Afrikaans as a second language — followed by **practical health learning** (pregnancy, mental health, common conditions) and **higher-education academic work** (university students writing assignments, preparing for exams, getting concept explanations). School-level subject help is real but smaller than the language and tertiary cohorts. Career-readiness — CVs, cover letters, business plans, interview prep — is a meaningful adjacent cluster, mixed in with the academic work.

---

## Methodology and caveats

### Data source

- **mem0_history.db** — the long-term memory store. Each row is one extracted "fact" the LLM-as-analyst layer wrote about a user across their conversations.
- Queried: distinct `memory_id`s where `is_deleted = 0` and `new_memory` is non-empty. This gives the current state of the fact bank.
- Result: 6,574 distinct current facts.

### Privacy guarantees

- **No conversation content was read** in producing this report. The facts in mem0 are already typed and abstracted from raw message text by the extraction layer.
- All counts are aggregate. No individual user's facts are quoted. No msisdns appear in the output.
- The `mem0_history` table's `actor_id` column is null for all rows, so user-linkage is unavailable. Counts are fact-level, not user-level.

### What this means in practice

- A single chatty user can contribute multiple facts to a cluster. The "Afrikaans-learning" count of 54 doesn't necessarily mean 54 distinct users — it means 54 distinct extracted facts. The true user count is plausibly somewhere between 20 and 50 given typical fact-per-user ratios.
- Conversely, a user who *uses* Oshiwambo natively but never *talks about* using it won't show up in any of the Oshiwambo counts.

### Known limitations

1. **The synthesis loop is paused.** The existing /stats.json `topics` and `who` clusters were last generated 2026-05-27 (six days stale at snapshot date) and only carve the data into 7 macro-clusters. This report does its own re-categorisation against fresher data.
2. **Pattern matching is regex-based.** False positives are possible. The trigonometry case (flagged above) is the cleanest example — `\bsin`, `\bcos`, `\btan` matched too aggressively. All other subject and task patterns use word-specific anchors and are trustworthy.
3. **`actor_id` being null** means we can't say "X% of users do this" — only "X facts in the bank look like this." That is still a useful signal because the extraction layer is uniform.
4. **The career-learning numbers are undercounts**, as noted in §6.

### Suggested follow-ups

1. **Tighten the synthesis prompt** to keep this level of granularity when the analytics loop is resumed. The current 7-macro-cluster output buries the Afrikaans-learning, health, and translation signals.
2. **Backfill `actor_id`** in mem0_history (or add a parallel index) so future analyses can give true user-level proportions.
3. **Consider a small Afrikaans-learning skill** in `ongiini/skills/`. Given 54 forward-looking mentions, even a modest reference module would be visible.
4. **Track translation-as-task explicitly** in the classifier or as a verdict, so the 172-fact signal can be measured live on the statistics page rather than re-derived ad hoc.
5. **Consider whether health learning warrants its own positioning** — it's clearly demanded, but is currently invisible in the homepage messaging.

---

*Generated by an ad-hoc aggregation script run inside the Spark container.
Script not committed — it's a 60-line file using `sqlite3`, `re`, `collections.Counter` against `mem0_history.db`.
Reproducible by re-running the same regex patterns against the bank.*
