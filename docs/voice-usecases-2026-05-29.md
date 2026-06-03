# Voice-note use-case analysis — 2026-05-29

Companion to [`photo-usecases-2026-05-29.md`](./photo-usecases-2026-05-29.md).
Built from the 96 user-turns where `[voice note]` appears in any user's
per-user memory file, across 43 distinct users. Each turn carries the
Whisper transcript that was generated at write time (audio bytes
themselves are never persisted, per PII contract).

**Method.** `scripts/_extract_voice_turns.py` walks `/data/*.json`,
extracts every voice-note turn plus surrounding context (previous user
message, previous assistant reply, bot's reply). Categorisation is
heuristic — priority-ordered keyword rules in
`scripts/_categorize_voice.py`. The first matching rule wins per turn.

**Privacy note.** Same as photos — hashed msisdns only (first 6 chars
SHA-256). Whisper transcripts are stored in the per-user memory file
and treated as text from there on, scrubbed for PII like any other
content.

---

## Distribution

| Category | Turns | % | Users | Avg words/voice |
|---|---:|---:|---:|---:|
| 💔 Emotional / relationship | **14** | **14.6%** | 8 | **41** |
| 🗣️ Native-language attempt (Whisper struggles) | 14 | 14.6% | 9 | 11 |
| 🎓 Schoolwork dictation | 12 | 12.5% | 9 | 34 |
| ❓ Information / factual question | 10 | 10.4% | 5 | 26 |
| 🙏 Religious / spiritual | 7 | 7.3% | **2** | **63** |
| 🌍 Translation request | 6 | 6.2% | 6 | 12 |
| ❌ Whisper transcription failed (non-EN/AF audio) | 4 | 4.2% | 4 | 12 |
| 😡 Complaint / frustration | 4 | 4.2% | 3 | 19 |
| 🇬🇧 Language practice (English speaking) | 3 | 3.1% | 2 | 22 |
| 💻 Tech question | 2 | 2.1% | 2 | 36 |
| 📊 Business / marketing | 2 | 2.1% | 2 | 15 |
| misc / not classified | 18 | 18.8% | 13 | 35 |

---

## Patterns — why do users reach for voice?

The data points to **five distinct triggers** for switching from typing
to voice. Each one has a recognisable signature in the transcript.

### 1. Emotional intensity → voice (14.6%, avg 41 words, often 100+)

The single biggest voice-driver isn't a topic — it's an emotional
register. When users get vulnerable, frustrated, or want to express
something nuanced about a relationship, they switch to voice.

**The clearest pattern**: one user (`...d9951b`) sent seven voice notes
in a row, all between 250–800 words, working through a difficult moment
with his girlfriend (he had been away in the field, lost contact, was
trying to apologise). Each voice note got longer as he poured out more
detail — typing this out would have taken him 20+ minutes per turn.
Voice made it possible.

Sample transcripts (truncated):
- *"i want you to write for me a good text a good text that i want to to
  text my girlfriend it is been a while now without texting with her..."*
- *"write for me a letter that apologizes to my ex in Afrikaans, write a
  letter in Afrikaans about my ex."*
- *"No, I actually want you to send her a text saying that yesterday was
  kind of messed up. I couldn't keep on with the chat..."*

Implication: **voice is the emotional pressure-release channel**.
People who would never type 200 words about their relationship
breakdown will gladly dictate 600.

### 2. Native-language attempts (14.6%) — voice as the bypass

When users want to speak Oshiwambo / Damara / Umbundu, voice is the
*natural* keyboard. Typing those languages on a WhatsApp keyboard is
painful; speaking them is effortless.

**The problem**: Whisper doesn't handle Bantu / Khoisan languages
well. The transcripts come out as either:
- Plausible-looking Latin gibberish that resembles Oshiwambo phonemes
  (*"Andi ti andi buru kuni nga sikeko koko e panji no mati Notify..."*)
- Or Whisper guessing the wrong language entirely and outputting
  Icelandic / Korean / Arabic strings (*"Það hæða hú röngu er enn jökú
  þá. Þið nokku, okkú þóttar og bísnesa ginn."*)

Either way, the bot reads "I don't understand, please use English or
Afrikaans" and the user's intent is lost.

Implication: **this is our single biggest voice-quality leak**. Users
are giving us the most authentic Oshiwambo we will ever get — and we
are dropping it. Worth thinking about whether to capture these
voice-bytes (with consent) as a future training corpus *before* they
get destroyed by Whisper's misclassification. Or fine-tune Whisper on
Oshiwambo audio.

### 3. Schoolwork dictation (12.5%, avg 34 words)

Students reach for voice when explaining their project context, asking
multi-clause academic questions, or describing what they did in the
lab. It's the channel for "I have a complicated thing to set up before
I get to my actual question."

Sample:
- *"okay i'm a student doing agriculture at unum and i was given a
  project to do welling castration and then identification and then
  after doing the project i was instructed to write a report on what i
  have done..."* (60 words)
- *"If you are given the value of Z is at 1% confidence level, how do
  you get it on the student positive Z table or student negative Z
  table?"* (29 words)

Typing this with one thumb on a phone keyboard is much slower than
speaking it. The longer + more contextual the academic ask, the more
likely it comes as voice.

### 4. Religious / spiritual content (7.3% — only 2 users — but 63
words/voice avg)

Two users dominate this category, but they generate **very long**
voice notes — songs, prayer chants, requests for prayer-service
structuring. One user (`...d9951b` again) recorded a 478-character
worship chant ("*Lord, we declare your victory, we declare your
victory...*") to ask the bot to help structure a Sunday service.

Implication: **religious/spiritual content seems uniquely voice-
native**. Songs, declarations, chants need to be heard to be felt.
Typing them out feels weird; speaking them feels natural. Small group
of users but high-intensity engagement.

### 5. Frustration → voice (4.2%, all very short)

When users are angry, they swear in voice. Three different users had
voice notes that the bot needed to apologise to, including:
- *"No, no, no, no. Fuck you for saying Meteor AI sells my data. Fuck
  you, okay? Fuck you."*
- *"and also you're not bold like meta ai i can see that because you
  are taking quite long to respond..."*

Different from typed complaints — voice complaints land harder, are
more raw, less filtered.

---

## Quantitative signatures

**Length distribution.** Voice transcripts skew much longer than
typical typed messages:

| Word count | Voice notes | % |
|---|---:|---:|
| 1–5 | 18 | 18.8% |
| 6–15 | 26 | 27.1% |
| 16–30 | 24 | 25.0% |
| 31–60 | 14 | 14.6% |
| 61–120 | 11 | 11.5% |
| 120+ | 3 | 3.1% |

29% of voice notes are 31+ words — that's a sentence count typed users
almost never reach. Voice is how people send *long* messages.

**Position in conversation.** Only 1 of 96 voice notes is the user's
first turn ever (`is_first_turn=True`). Users almost always type-greet
first, then switch to voice once the conversation has context.
Suggests: people don't *start* with voice, they *escalate* to voice
when typing feels insufficient.

**Concentration.** 43 distinct users vs 164 for photos — voice is a
narrower-base behaviour, but the users who do it, do it *a lot*. One
user has 7 voice notes; another has 9; a third has 5. Voice-heavy
users dictate at 5–10× the rate of voice-light users.

---

## Implications

- **The Oshiwambo voice corpus is the single biggest opportunity.**
  ~14% of voice messages are users trying to speak their native
  language, and Whisper destroys them on the way through. If we can
  capture-and-keep those bytes (with explicit consent + opt-in) before
  they hit the transcript, we have a high-quality OW speech corpus
  worth far more than the text translations we're collecting now.
  Talk to Elizabeth + the contributors community about this.
- **Voice = emotional release valve.** Don't optimise voice handling
  only for "transcribe + give factual answer." A meaningful share of
  voice users are venting / processing / asking for relationship help.
  The bot's empathetic register matters more for voice replies than
  for typed replies.
- **Voice = long-form dictation tool.** Average 34 words, max 800+.
  Users use voice when they have *a lot* to say. The bot's reply
  needs to actually engage with the full content — summarising voice
  into a one-line ack is a guaranteed disappointment.
- **First-turn voice is essentially zero.** No need to design a
  "first-time voice user" onboarding — users always type first. Voice
  is an escalation behaviour, not an entry behaviour.
- **Whisper language-detection is too eager.** It generates plausible
  text from unsupported audio rather than admitting it doesn't know
  the language. Worth investigating whether we can detect "Whisper
  confidence too low" and fall back to "please type instead" rather
  than acting on the garbled transcript.

---

## Reproduce

```sh
# On Spark, extract
docker exec ongiini-webhook python3 /data/_extract_voice_turns.py

# Pull locally
scp spark-tail:/data/voice_turns.tsv /tmp/voice_turns.tsv

# Categorise + count
python3 /tmp/_categorize_voice.py
```

Both scripts live in the repo under `scripts/` (matching names without
the leading underscore).
