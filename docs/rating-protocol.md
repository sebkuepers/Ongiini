# Rating protocol — reference check (round 1)

Fixed before any real ratings are collected. Changes after data
collection starts are listed at the end with a date and a reason.

## Goal

Check the quality of the human reference translations of the
Ongiini-Eval-OW development split (Oshindonga and Oshikwanyama), using
volunteer native speakers who rate on their phones. Two questions:

1. What share of the references is acceptable (not wrong)?
2. Are the references at least as good as a strong machine translation
   (Claude Opus 5) on the same sentences, judged by the same person?

The blind split is never shown to raters.

## Task

One English sentence and one translation per screen. Question: **"Does
it say the same as the English?"**

- ✓ **Yes**: same meaning
- ~ **Almost**: a small detail is off
- ✗ **No**: different meaning, or something is missing

After ~ or ✗, raters can optionally tag the issue (word choice, spelling
or grammar, sounds unnatural, other dialect) and suggest a better
translation. "I can't judge this one" takes a reason.

Raters never learn whether a translation is the reference, a machine
output or a control.

Main measure: **not wrong** (Yes or Almost) vs **wrong** (No).
Secondary measure: Yes vs not Yes.

Onboarding has two practice items: references with a planted error. The
rater must answer ✗ to continue. Rounds are 5 items each.

## Items (per dialect)

| group | n | purpose |
|---|---|---|
| random references | 45 | unbiased acceptance estimate (drawn outside the flagged top) |
| flagged references | 25, 2 raters each | most severe back-translation flags (Claude Opus 5, blind back-translation, "major") |
| Claude Opus 5 outputs | 30 (20 random + 10 flagged sentences) | paired with the same rater's judgment of the reference, ≥ 5 tasks later |
| Gemma 4 26B outputs | 10 | discrimination check, rated by a different person than the reference |
| planted errors | 12 | reference with a changed number or a dropped final clause or sentence; expected ✗ |
| wrong-sentence controls | 3 | the reference of another sentence; expected ✗ |
| repeats | ≤ 5 per rater | same item again ≥ 15 tasks later; consistency |

Assignment uses a weighted round robin over the groups, so every session
mixes all of them. Items are built by `scripts/build_rating_tasks.py`
with seed 42.

## Raters

- **Recruitment.** Volunteers open ongiini.ai/rate, tap "Start on
  WhatsApp" and send one message to Ongiini AI. The assistant replies
  with a personal link. A few early testers got links from the admin.
- **Identity.** One rater per WhatsApp number. We store only a salted
  one-way hash of the number, so ratings are pseudonymous.
- **Self-report** at the first visit: dialect(s) rated, and whether it is
  their first language (yes, or "no, but I speak it well").
- **Exclusion.** The reference translator is blocked and cannot rate.
- **Consent and pay.** Raters are unpaid volunteers. Consent and data
  handling are covered by ongiini.ai/privacy, "Translation ratings".

## Qualification rule (main numbers)

A rater counts in the main numbers if:

- their first language is **yes**, and
- they caught (✗) **at least 2/3 of the planted errors they saw, having
  seen at least 2**.

Results for **all raters** are always reported alongside. Raters are
never removed from the data, only filtered in the analysis.

## Analysis (`scripts/analyze_ratings.py`)

Only first ratings count for the estimates. Repeats are used for
consistency only. "Can't judge" answers are excluded and counted.

1. **Acceptance rate** with Wilson 95 % CI. The random sample is reported
   separately from the flagged top. A stratified whole-set estimate
   weights the flagged top by its share of the development references.
2. **Reference vs Claude Opus 5**, paired within rater: exact McNemar on
   wrong / not wrong, plus the difference with a 95 % CI. The claim is
   "not worse than", with a margin of −10 points.
3. **Discrimination.** Planted errors must score clearly below the
   references (non-overlapping CIs). If they do not, no claim about
   reference quality is made.
4. **Flagged references.**
   - 2× wrong → confirmed error, goes on the correction list with the
     suggestions
   - split → a third rater
   - 2× not wrong → cleared
5. **Rater quality.** Planted and control hit rates, repeat consistency,
   time per task, can't-judge reasons.

## Changes

- 2026-10-01: question changed from "Is this a good translation?" to
  "Does it say the same as the English?", rounds of 5, and practice must
  be answered correctly. Reason: the pilot rater rated all 70 items ✓ in
  about 1 s each, including all planted errors. Pilot data were discarded.
