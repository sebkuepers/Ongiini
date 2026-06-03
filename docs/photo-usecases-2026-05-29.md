# Photo use-case analysis — 2026-05-29

Update of the original early-stage photo analysis. Built from the
487 user-turns where `[image attached]` appears in any user's
per-user memory file, across 164 distinct users.

**Method.** `scripts/_extract_photo_turns.py` walks `/data/*.json` on
Spark, extracts every photo turn + the immediately following assistant
reply + the preceding user message (for context). Categorisation is
heuristic — priority-ordered keyword rules in
`scripts/_categorize_photos.py` against the bot reply + caption + prev
user message. First match wins, so each turn lands in exactly one
bucket.

**Privacy note.** All numbers are hashed-msisdn (first 6 chars
SHA-256). No raw msisdns, no image bytes — only the bot's textual
description of what it saw, which is what the memory layer persists by
contract.

---

## Distribution

| Category | Turns | % | Unique users |
|---|---:|---:|---:|
| 🎓 School / Homework | **197** | **40.5%** | 75 |
| 📄 CV / Job applications | **85** | **17.5%** | 36 |
| 📊 Business / Marketing | **58** | **11.9%** | 31 |
| 🏠 Rental property documentation | 18 | 3.7% | 1 (power user) |
| 🌍 Translation / language | 14 | 2.9% | 13 |
| 🔍 Product identification | 13 | 2.7% | 7 |
| 💬 Personal / relationships | 9 | 1.8% | 8 |
| 💻 Tech support (PDF, Excel, apps) | 8 | 1.6% | 6 |
| 🍰 Recipes / cookbook | 7 | 1.4% | 2 |
| 🏥 Health / medical | 5 | 1.0% | 5 |
| 🏛️ Gov / tender documents | 5 | 1.0% | 1 |
| 🌾 Agriculture | 4 | 0.8% | 4 |
| 🎨 Photo edit (refused, can't do) | 4 | 0.8% | 4 |
| 🙏 Religious / spiritual | 3 | 0.6% | 2 |
| misc / not classified | 57 | 11.7% | ~25 |

---

## Movement since the early-stage analysis

The early analysis (when photos were still single-digit per day) had a
much more conversational tone — "what is this?", "do you know this
food?". Today's distribution shows photos are now **task-driven**:
users have a concrete outcome in mind and the photo is the artefact
they need help acting on.

### 1. School/Homework is the killer use case (40.5%)

Not "what is X" but concrete tasks:

- Cambridge / NSSC math + physics exam papers — "answer + explain how"
- Lesson plans for student teachers, photographed page-by-page
- Research methodology + literature reviews (NUST / IUM / UNAM
  students working on theses)
- Course outlines with "make me exam notes" / "develop study schedule"
- WhatsApp / online quiz photos — "which letter is right"

**Power-user pattern.** One user (`...97beba`) sent **10 photos of a
single Cambridge mathematics paper** in one session and walked through
every question one-by-one with Ongiini as tutor.

### 2. CV / Job is more sophisticated than expected (17.5%)

Not just "make me a CV" but layered review work:

- Existing CV photo + "what should I change" → bot gives structured
  feedback on professional summary, work history phrasing, layout
- Certificate photos (KAYEC, NIMT, Rössing Foundation, Ministry
  training programs) → "what does this add to my CV"
- ID card photos → bot correctly warns about PII sharing
- Cover letters written specifically for the job advert the user
  uploaded
- **Notable**: several users photographed the *own Ongiini job
  advert* and asked if it was legit before applying.

### 3. Business owners use Ongiini as creative director (~12%)

Direct + spillover from "misc" probably ~17% total:

- Hair-bundle seller with pricing lists (Vietnamese raw hair)
- Food / braai photos for flyer + WhatsApp status copy
- Crochet products for branding feedback ("Monika" — high-quality
  product photography)
- Restaurant menus ("Omwandi Eatery" with traditional + modern food)
- Sneaker / perfume seller doing "in stock" posts
- Logo design feedback (driving school, podcast)

The bot consistently positions itself as "I can't make the image, but
I can be your creative director" — exactly the right honest framing.

### 4. Real-world field utility — small in count, high in value

- Farmer: "Difence fungicide — can I spray it on tomatoes that I
  transplanted on 24 Feb 2026?" → bot reads label, gives
  application-rate guidance
- Patient: "Pain in my foot after a long walk" → bot recommends RICE
  method + see a doctor
- Construction company: tax + BIPA + Social Security certs uploaded
  one at a time → bot helps assemble the tender-readiness story

### 5. Genuinely surprising — not on our radar

- **Rental property documentation** (one user, 17 turns): User was
  photographing every room of a rental property for an incoming
  inspection. Ongiini coached them through wall checks, plumbing,
  ceiling damage, doorframe wear, every blind spot. This is a hi-value
  use case that emerged without us designing for it.
- **Relationship coaching**: users send WhatsApp conversation
  screenshots and ask "was my answer good?" — Ongiini acts as
  sounding board on emotional / interpersonal threads.
- **Scam checks**: users photograph Forex trading ads, "Legacy
  Builder Program", fake game-money videos and ask if the offer is
  real. Ongiini correctly flags these as MLM / clickbait / digital
  reselling traps.

---

## Implications

- **School/homework is the growth-driver.** 40% of photos + most
  user-messages are academic. Ongiini in Namibia is being shaped
  primarily as a study companion, not a general chatbot. Worth
  investing in math / physics / research-help quality.
- **CV / job help is the second pillar.** Directly monetisable
  utility, drives repeat engagement (users come back with each
  certificate / cover-letter iteration).
- **Image-edit refusals (4 turns):** users want photo-editing /
  generation we can't do. Possible feature gap to consider, or a
  deliberate "we're not that tool" limitation to keep.
- **Rental documentation emerged on its own.** Suggests the system
  prompt + memory layer can carry very specific, long-running
  documentation flows without us pre-designing them. Worth being
  careful before adding more structure that might break this kind of
  emergent capability.
- **Privacy enforcement is working as intended.** The PII contract
  catches ID-card uploads consistently; the bot warns the user and
  doesn't persist the sensitive details in mem0.

---

## Reproduce

```sh
# On Spark, extract
docker exec ongiini-webhook python3 /data/_extract_photo_turns.py

# Pull locally
scp spark-tail:/data/photo_turns.tsv /tmp/photo_turns.tsv

# Categorise + count
python3 /tmp/_categorize_photos.py
```

Both scripts live in the repo under `scripts/` (matching names without
the leading underscore).
