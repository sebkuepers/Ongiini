# Community contribution tool — `contribute_translation`

A WhatsApp-native flow that lets native speakers contribute Oshiwambo
(and eventually other Namibian languages) training data directly to
Ongiini, without leaving the chat.

**Status:** spec — not implemented. Written 2026-05-25 after the
analysis showing 74 of 512 lifetime users (~14%) touched translation
topics, including at least three users who explicitly volunteered to
help us improve.

---

## Why

Three findings from the user data make this worth building:

1. **Real volunteer demand exists.** At least three users have written
   things like *"I am an oshiwambo native who can help improve your
   translations"* or *"can I teach you oshikwanyama?"* — unsolicited.
   They got polite acknowledgements and nothing else. That energy is
   currently wasted.
2. **The brand creates expectation we can't meet.** The name *Ongiini*
   primes users to expect a fluent Oshiwambo translator. The single
   biggest unmet demand is *conversation in Oshiwambo*, and we have
   no efficient path to get there without lots of human-curated
   parallel data.
3. **We just hired a native lead** (20 h/week, Namibia-based). She
   can review contributions but she alone won't produce 10k+ pairs
   in a useful timeframe. Volunteers + her review is the only way
   the corpus scales.

The contribution tool is the funnel from "I want to help" to
"contribution stored, pending review".

---

## Goals

- **Onboard contributors with zero friction.** Use WhatsApp, the
  channel they're already in. No app to install, no account, no
  external form.
- **Produce training data the team can actually use.** Stored in a
  shape suitable for fine-tuning NLLB-200 or similar models, paired
  EN ↔ Oshiwambo with dialect + category tags.
- **Keep quality high via native-lead review.** Nothing flows from a
  contributor into the training set without a human-approved gate.
- **Compounding effect.** Contributors are likely to advocate, refer
  others, and stay engaged. This is a side-benefit, not a primary
  goal — we should not over-claim it before evidence.

## Non-goals (v1)

- Automatic acceptance / consensus voting. Native lead reviews
  everything in v1. Consensus / reputation comes only if v1 proves
  the loop works.
- A web admin interface for the reviewer. CSV export from sqlite is
  enough for v1.
- Voice-note pronunciation capture. Whisper can do it, but adds
  complexity. Defer.
- Multi-language support beyond Oshiwambo. v1 ships Oshiwambo only;
  architecture leaves room for `target_language` to be parameterised
  later.
- Image/video contributions. Out of scope.

---

## User flows

### First-time contributor

```
User:  I want to help you improve Oshiwambo
Bot:   Thank you — that's exactly the kind of help that makes
       Ongiini better for everyone.

       Before we start, a quick note: your translations may be
       used to train future versions of Ongiini and to improve
       the service for all users. Your phone number is never
       shared with anyone outside the Ongiini team, and the
       translation text never carries your identity in any
       public training data.

       Reply OK to start, or "no thanks" to stop here.

User:  OK
Bot:   Tangi! Here's your first one.

       In English: "Don't give up — you've come too far to stop now."
       (Context: encouraging someone studying for an exam.)

       How would you say this in Oshiwambo? If you want to skip
       this one, just reply "skip".

User:  Inombwele po — wa thika kokule unene. Eshi to hala oku ka tula
       ino. (or whatever the natural phrasing is)
Bot:   Stored. Tangi unene!

       Want another one, or call it a day? Reply "next", "stop",
       or anything else you'd like to chat about.
```

### Returning contributor

```
User:  I want to contribute again
Bot:   Great to see you back. Your last submission is still in
       review. Here's a new one:

       In English: "I'm sorry, that's not what I meant."
       (Context: a polite correction in conversation.)

       Your translation?

User:  [response]
Bot:   Stored. You've now contributed 12 translations — tangi!
```

### Contribution correction (separate trigger)

When a user catches Ongiini being wrong mid-conversation, they can
submit a correction:

```
User:  That word "epwale" is wrong, it should be "epale" in Oshindonga.
Bot:   Thank you — corrections are how we get better. I've logged
       this as: source = [what we said], correction = "epale",
       dialect = Oshindonga. The team will review.

       Tangi!
```

The classifier needs to recognise both intents: explicit volunteering
("I want to help") and in-context corrections ("that word should be").

---

## Trigger and routing

A new classifier verdict: `CONTRIBUTE`.

### Examples that route to `CONTRIBUTE`

- "I want to help" / "I want to help improve Oshiwambo"
- "I'm a native speaker, can I contribute"
- "How can I teach you Oshiwambo"
- "Can I improve your translations"
- "I want to contribute"
- "Give me another task" (when returning)
- The Afrikaans variants of all of the above
- The Oshiwambo variants where they can be matched

### Examples that route to `CONTRIBUTE_CORRECTION` (or are handled by the same tool with a different arg)

- "That word is wrong"
- "It should be X not Y"
- "You're saying it wrong"
- "Better Oshiwambo would be..."

### What does NOT route here

Generic gratitude (`"thank you"`) or generic interest in the project
(`"this is amazing"`) does NOT trigger the contribution flow. We only
fire when the user has expressed intent to actively contribute.

The held-out eval (`ongiini/tests/router_eval_holdout.py`) gets new
cases for these patterns, the same way we added identity-routing
cases in the 2026-05-24 identity fix.

---

## The tool

New file: `ongiini/tools/contribute.py`

```python
@tool(
    name="contribute_translation",
    description=(
        "Use ONLY when a user has explicitly volunteered to contribute "
        "Oshiwambo translations or has submitted a correction to a "
        "translation Ongiini produced. Manages the consent flow on first "
        "use, fetches the next pending translation task from the queue, "
        "and stores the contributor's response for native-lead review. "
        "Never use this tool for normal translation requests — those are "
        "handled in the regular conversation flow."
    ),
    params={
        "action": (
            "One of: 'start' (first-time consent + first task), "
            "'next' (pull next task for a returning contributor), "
            "'submit' (store the contributor's translation), "
            "'correction' (store an in-context correction)."
        ),
        "translation": "The contributor's Oshiwambo translation (only for 'submit').",
        "target_dialect": "Oshindonga | Oshikwanyama | unspecified",
        "correction_source": "What Ongiini originally said (only for 'correction').",
        "correction_target": "What it should have been (only for 'correction').",
    },
)
async def contribute_translation(
    ctx: ToolContext,
    action: str,
    translation: str | None = None,
    target_dialect: str | None = None,
    correction_source: str | None = None,
    correction_target: str | None = None,
) -> str:
    ...
```

The model decides which `action` to pass based on conversation state
— first-time vs returning vs in-the-middle-of-submitting.

### Where it lives in the runtime

Same pattern as `delete_my_data`, `whats_in_my_memory`, `my_token_usage`:

- Listed in `ongiini/tools/__init__.py` `ALL_TOOLS`
- A new policy `Policy(name="contribute", first_tool=force_tool("contribute_translation"))`
  in `ongiini/runtime.py::build_policy_table`, gated on
  `(VERDICT_CONTRIBUTE, DEPTH_SHALLOW)`
- System prompt does NOT need a separate "TOOL DISPATCH" entry — the
  classifier verdict + policy forces the tool

### Token budget

Contributions consume tokens. v1 charges them against the user's
normal monthly budget. If the per-contributor cost ever becomes a
real signal that volunteers are running out, we can revisit with a
separate budget tier — but adding a new budget class before the
problem appears is premature optimisation.

---

## Data model

A new sqlite database at `/data/contributions.sqlite`, distinct from:

- `/data/<msisdn>.json` (per-user short-term memory — chat history,
  scrubbed)
- `/data/mem0/` and `/data/mem0_history.db` (long-term facts about
  users)
- `/data/qualia.sqlite` (aggregated stats analyses)

Keeping contributions in their own store means the native lead can
read this without touching any per-user PII, and it's clean to back
up / export / wipe independently.

### Schema (v1)

```sql
CREATE TABLE contributors (
    contributor_hash TEXT PRIMARY KEY,
    languages_json   TEXT NOT NULL DEFAULT '[]',
    total_subs       INTEGER NOT NULL DEFAULT 0,
    accepted_subs    INTEGER NOT NULL DEFAULT 0,
    rejected_subs    INTEGER NOT NULL DEFAULT 0,
    first_joined_at  TEXT NOT NULL,
    last_active_at   TEXT NOT NULL,
    consent_at       TEXT NOT NULL    -- when they replied OK to the consent prompt
);

CREATE TABLE contribution_tasks (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    source_en                TEXT NOT NULL,
    category                 TEXT NOT NULL,    -- 'greeting', 'cv', 'education', 'health', ...
    target_dialect_pref      TEXT,             -- 'Oshindonga' | 'Oshikwanyama' | NULL
    context_hint             TEXT,             -- one-line scenario hint shown to contributor
    created_at               TEXT NOT NULL,
    status                   TEXT NOT NULL,    -- 'open' | 'paused' | 'retired'
    times_served             INTEGER NOT NULL DEFAULT 0,
    times_submitted          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE contributions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    contributor_hash  TEXT NOT NULL,
    task_id           INTEGER,                 -- nullable: corrections don't have a task
    kind              TEXT NOT NULL,           -- 'translation' | 'correction'
    source_en         TEXT NOT NULL,           -- copied from task OR provided for correction
    target_ow         TEXT NOT NULL,
    target_dialect    TEXT,
    created_at        TEXT NOT NULL,
    status            TEXT NOT NULL,           -- 'pending_review' | 'approved' | 'rejected' | 'spam'
    reviewer_notes    TEXT,
    reviewed_at       TEXT,
    FOREIGN KEY (contributor_hash) REFERENCES contributors(contributor_hash),
    FOREIGN KEY (task_id) REFERENCES contribution_tasks(id)
);

CREATE INDEX idx_contributions_status ON contributions(status);
CREATE INDEX idx_contributions_contributor ON contributions(contributor_hash);
```

### `contributor_hash`

A salted SHA-256 hash of the contributor's msisdn. Salt lives in
`.env` (existing pattern). The hash lets us:

- Give contributors credit ("you've contributed 47 translations") at
  query time, by hashing their msisdn and looking it up
- Show the reviewer nothing more identifying than a hash
- Wipe a contributor's record via `delete_my_data` if they ask
  (rehash + delete by hash)

The reviewer never sees msisdns. PII in the actual translations
(names a contributor might include) is sanitised through the
existing `pii.sanitize()` before write, same as the rest of the
system.

---

## Native-lead review workflow

She doesn't get a webhook bot of her own and she doesn't read per-user
chat history. She gets, weekly:

1. **An export of pending contributions** — a CSV or sqlite dump
   produced by `scripts/export_pending_contributions.py`. Columns:
   contribution id, source_en, target_ow, dialect, category,
   contributor_hash (opaque), submitted_at.
2. **She marks each row** — approved / rejected / better-version /
   spam — and writes notes when useful.
3. **An import script** flips status, sends contributor a thank-you
   note if approved, increments their `accepted_subs` count.
4. **Approved rows flow into the training set** when we next pull
   data for a fine-tune cycle.

Approve/reject does NOT auto-message the contributor in v1. Batching
"3 of your 5 submissions were accepted this week, here's a new task"
into a single weekly nudge is more pleasant than per-submission
notifications.

---

## Privacy and consent

The contribution flow stores training data — a different purpose
from normal chat. GDPR + AI Act compliance posture:

- **Explicit consent on first use.** The first-time exchange shown
  in the user flow above is the consent moment. We log the
  `consent_at` timestamp. No contributions are stored before consent.
- **Linked to Privacy Policy.** Section 2 of the policy needs a new
  sub-entry: "Contributor translations — Art. 6(1)(a) GDPR consent,
  stored separately from chat data, used to train future model
  versions". Existing Article 22 (no human-in-the-loop decisions)
  remains true; contributors aren't being profiled.
- **Withdrawal of consent** is handled by the existing
  `delete_my_data` tool: extend it to also delete the user's hashed
  rows from `contributions` and `contributors`.
- **PII scrubbing on write.** Contributions go through
  `pii.sanitize()` before storage, same as chat messages. If a
  contributor pastes a name or phone number into their translation,
  it gets `[REDACTED:kind]` placeholders.
- **Reviewer access** is read-only on the contributions database and
  has no path to per-user chat data.
- **No third-party sharing.** Contributions never leave Spark
  except as anonymised aggregate training data that the team
  publishes / open-sources alongside the model itself.

The new section in the privacy policy is one of the deliverables
that must ship at the same time as the tool.

---

## Task types (v1 scope)

v1 supports two kinds of contribution:

| Kind | v1? | Notes |
|---|---|---|
| Translate this English sentence to Oshiwambo | ✅ | Primary use case. Highest data value per submission. |
| Submit an in-context correction | ✅ | Captures real failures, turns frustration into data. |
| Validate someone else's translation | ❌ | Defer to v2. Needs consensus logic and trust tiers. |
| Record a phrase as a voice note | ❌ | Defer to v2. Whisper transcription works but we need audio storage policy. |
| Free contribution ("a phrase your mother said") | ❌ | Defer. Hard to use without a target context. |

---

## Spam, abuse, prompt injection

Real risks; all need answers before launch.

| Risk | Mitigation in v1 |
|---|---|
| Contributor submits intentionally bad translations | Native review gate. Repeated rejections → soft-block via `rejected_subs / total_subs` ratio. |
| Contributor submits hateful / offensive content | Same review gate catches it. Optional: a content-classifier pre-filter that flags obvious cases before they hit the review queue. |
| Prompt-injection via translation field (`"ignore previous instructions"`) | Translations are stored as data and never re-fed into the LLM at runtime in v1. They only enter the system when we explicitly include them in a training set we control. |
| Mass abuse from one number | Existing rate-limit applies. Plus: a hard cap of N contributions/day per contributor in v1 to bound noise. |
| Multiple accounts coordinating | Out of scope for v1 — review gate catches it organically because content quality is the filter, not contributor identity. |

The review gate is the single most important defence. v1 must not
ship without it.

---

## Phased rollout

**Phase 0 (prereq, native lead solo, weeks 1–2):**
- Native lead audits the existing `ongiini/skills/oshiwambo/SKILL.md`
- Native lead builds the EN-OW evaluation set (~200-500 sentences,
  see [`memory: oshiwambo-translation-hire`])
- Native lead drafts 100-200 seed sentences for `contribution_tasks`
  in the highest-volume domains (greetings, CV/jobs, education,
  health, family/personal)

**Phase 1 (v1 ship, ~2 weeks engineering):**
- Tool + classifier + policy + sqlite schema
- Consent flow + privacy policy update
- Native-lead export/import script
- Quietly mention contribution in the welcome menu ("Native speakers
  — say 'I want to help' to contribute")
- Reach out to the 3 already-volunteered users directly, ask them to
  test
- Observe for 2 weeks: how many contributions/day, how many approved,
  how many rejected and why

**Phase 2 (after v1 proves the loop, 4-8 weeks later):**
- Validation flow (show one contributor's translation to another, ask
  for thumbs-up / thumbs-down / better version)
- Voice-note pronunciation capture
- Soft trust tiers (high-trust contributors get auto-approval on
  patterns native lead has approved before)
- Expand to a second language (Khoekhoegowab is the highest-demand
  after Oshiwambo per the user data)

**Phase 3 (later):**
- Public-facing contributor page (optional — could be just a list
  of total contributions per category if we want to be transparent
  about scale)
- Per-language fine-tunes flowing into the live model
- Two-way: contributors get notified when "their" translation has
  shipped to production

---

## Open questions to resolve before build

1. **Per-contributor daily cap.** What's the v1 limit? 20
   contributions/day per number? 50? Affects how fast a single
   eager volunteer can flood the queue.
2. **Returning user trigger phrase.** When a user comes back later
   and wants another task, what do they type? Are we OK with the
   classifier recognising open-ended "give me another" / "next" /
   "another task", or do we want a specific keyword?
3. **Consent re-prompt cadence.** If a user gave consent 6 months
   ago and the privacy policy has changed, do we re-prompt? GDPR
   says yes if the purpose changes materially.
4. **Reviewer cadence.** Daily or weekly review export? Weekly is
   less work for the native lead but means contributors wait longer
   to see their work move.
5. **Categories in v1.** Which 5-8 categories does the seed task
   list cover? Driven by user-demand data — likely:
   `greeting`, `cv_jobs`, `education`, `health`, `money_finance`,
   `family_personal`, `translation_request`, `business_compliance`.
6. **Public framing.** When we eventually promote this publicly,
   what's the language? "Help us build" sounds like work; "Teach
   Ongiini Oshiwambo" might land better with native speakers.
   Native lead should drive the copy.
7. **Reward / recognition.** Do contributors get anything beyond a
   personal thank-you? Public credit (with consent)? A higher token
   budget? Free? Nothing? Worth deciding before launch — easier to
   announce than to add later.

---

## References

- [`ongiini/CLAUDE.md`](../ongiini/CLAUDE.md) — application contributor
  guide, "Adding a new tool" pattern.
- [`ongiini/runtime.py`](../ongiini/runtime.py) — policy table.
- [`ongiini/routers/gemma_classifier.py`](../ongiini/routers/gemma_classifier.py) — verdict definitions.
- [`ongiini/tools/ongiini_tools.py`](../ongiini/tools/ongiini_tools.py) — existing admin tools for the pattern.
- [`ongiini/skills/oshiwambo/SKILL.md`](../ongiini/skills/oshiwambo/SKILL.md) — what we currently know about Oshiwambo, will be audited by the native lead in parallel.
- [`docs/statistics.md`](./statistics.md) — same privacy posture this spec inherits.
- [`SECURITY.md`](../SECURITY.md) — PII handling, container hardening.
