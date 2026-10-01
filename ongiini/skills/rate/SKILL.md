---
name: rate
description: Phrasing for the rate_link tool result — the user asked to help CHECK existing Oshiwambo translations on ongiini.ai/rate (not to translate; that is the contribute loop). The classifier forced the tool; you only write the reply.
load: on_demand
---

# Checking translations — reply phrasing

`rate_link` already ran. Write a short WhatsApp reply from its JSON. This
is about **checking** translations someone else made — never offer
sentences to translate here.

- `status: "ok"` — thank them in one line, then:
  - say what it is: on the page they see an English sentence and a
    translation and tap whether it says the same; about 5 sentences,
    2 minutes; they can stop any time
  - give the `url` exactly as returned, on its own line
  - say the link is just for them, please don't forward it
  - if `returning` is true: "Here's a fresh link — the old one no longer
    works." If `done` > 0, thank them for the `done` they already checked
- `status: "whatsapp_only"` — checking works through WhatsApp: message
  Ongiini AI on WhatsApp and ask to help check translations.
- `status: "blocked"` — say kindly that this check isn't open to them, no
  reasons, and offer help with anything else.
- `status: "error"` — apologise, the link couldn't be created right now,
  please try again later.

If they also want to translate sentences themselves, say they can ask
for that separately ("I want to help translate").
