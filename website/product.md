# Ongiini — Product Knowledge

_Auto-generated from `website/*.html` on 2026-05-23. Do not edit by hand: edit the source HTML and re-run `scripts/build_product_knowledge.py`. This file is consumed by the WhatsApp webhook's `lookup_ongiini_docs` tool so the assistant always answers questions about Ongiini itself from the same canonical copy that's on the website._

## Privacy summary (homepage cards)

Privacy

Your data. Your control.

What we keep

About 50 turns of back-and-forth so Ongiini can follow a real conversation, plus a few things it remembers so you don't have to repeat yourself — your preferred language, or a follow-up you've asked it to keep in mind.

What we don't

No selling. No advertising. No use of your conversations to train any AI we run. Personal data — email, ID, bank card — gets stripped out before anything is saved. Your stored conversation memory stays on the small computer we run ourselves.

See it. Delete it.

Ask "what do you remember about me?" any time to see what Ongiini has stored. Send "delete my data" to wipe it. Both phrases work in English and Afrikaans.

## Pricing and the monthly limit

Free, with a fair limit

Free for everyone — within a fair monthly limit.

One Million Tokens.

Per user · per month · resets on the 1st

That's about 500 conversations over the course of a month — plenty for normal use. We picked this number so the service stays sustainable on a single small computer. (What's a token? See the questions below.)

5 conversations a day, every day, for a month

your buffer — plenty more for the big days

## Why it's built this way

Commitment

Nothing hidden. Nothing sold.

Hardware in plain sight

It runs on one small computer — about the size of a thick book — sitting on a shelf in our office. Not a data centre. Not thousands of servers somewhere.

AI anyone can use

The AI that powers Ongiini is shared freely by its makers — anyone is allowed to use it, including us. That means nobody can flip a switch and take it away.

Built in the open

Every part of how Ongiini works is published online. Anyone can look, anyone can check, anyone can build their own version.

Your data isn't a product

Ongiini doesn't make money from your conversations. No ads, no profile-building, no training of any AI by us on what you say. OpenAI, Google, Meta — their business models are built on user data. This one isn't.

## How to support the project

Contribute

Help shape this.

Ongiini is the first project of the Common Intelligence Foundation. While the foundation is being formally established as a non-profit in Estonia, this project is privately funded and operated on a non-profit basis. That's part of the point: to show this can work without a giant company behind it. If Ongiini is useful to you, there are three good ways to help.

Use it. Push it. Tell us what broke.

Throw real questions at it. Find where it shines and where it falls over. Honest, specific feedback from people actually trying to use it is the best signal we get.

Tell someone

Most Namibians still don't know there's a free AI helper for everyday questions on WhatsApp. Sharing the number with one person who could use it is the biggest help anyone can give.

Open the code

The whole stack is on GitHub — webhook, prompts, deployment, everything. File an issue when something's wrong, send a pull request when you can fix it, or fork the project and run your own version for another country.

Open Ongiini on GitHub

File an issue

## Frequently asked questions

### What is Ongiini?

A free AI helper on WhatsApp. Send a question — about learning, work, health, daily life — and you get a useful answer back. Built on a simple idea: a tool this useful belongs to all of us, not just to whoever can afford a subscription.

### Why 'Ongiini'?

It's the everyday Oshiwambo greeting — literally 'How are you?', used the way English speakers use 'Hello'. It's the first word many Namibians hear and say each day. Naming the project after it felt right: the brand name is its own first message to you, said in a language of the country it serves.

### Why Namibia?

A few reasons that fit. Almost everyone uses WhatsApp on the phone they already have — nothing to install, no account to make. English and Afrikaans are widely spoken, and they're the two languages Gemma 4 handles well today. The country is small enough (~3 million people) that one DGX Spark can realistically serve a meaningful share of users on a generous one-million-token-per-month cap each. And the mix of urban and rural, multilingual, formal and informal use cases makes Namibia a real test of whether free AI on WhatsApp can be a genuine game-changer for everyday life — schoolwork, farming, contracts, health, small business — and not just a demo. If it proves itself here, it'll probably prove itself elsewhere too.

### Does it cost anything?

No. Each user gets one million free tokens per month, which covers about 500 normal conversations — easily more than typical daily use. The cap resets on the 1st of every month. WhatsApp's normal data charges from your mobile provider still apply, but Ongiini itself is free.

### What is a token?

It's how AI measures the text it reads and writes — roughly three-quarters of a word, including punctuation. "How are you?" is about 4 tokens. "My maize leaves are turning yellow" is about 9. One million tokens — your free monthly cap — works out to roughly 500 normal back-and-forth conversations over a month.

### What exactly counts toward my monthly tokens?

Every reply Ongiini sends you, including the tokens used to read any photo you attach. Plus a small amount for the behind-the-scenes work of keeping your long-term memory in sync after each message, and the occasional summary that compresses old turns of a long conversation. Reading existing memory back to you is free. Asking 'how many tokens have I used this month?' on WhatsApp gives you a breakdown across all three.

### Which languages can I use?

English and Afrikaans work well today. Adding Oshiwambo is something we're actively working on — the plan is a translation layer that converts to and from English under the hood, while keeping Gemma 4 as the brain. We don't want to promise a date until it's solid enough to use day-to-day.

### Can I send voice notes?

Yes. Press and hold the microphone button in WhatsApp, speak your question, release to send — exactly the same flow you'd use with a friend. English and Afrikaans both work; up to about 90 seconds per note. Ongiini transcribes the recording on the Spark itself using Whisper (running on CPU so it doesn't compete with the chat model for GPU). The transcript is treated exactly like a typed message — same memory, same tools, same English / Afrikaans language rule. Audio bytes are discarded right after transcription; nothing audio-shaped is ever stored. Replies still come back as text for now — voice replies are on the roadmap.

### What can I ask it?

Lots of the things you'd ask a knowledgeable friend. Explain a school topic, help you understand a contract or news article, translate, draft a CV or email, work through a maths problem, summarise a document, ask about a sick maize plant, a fever, a legal notice — learning and everyday questions of many kinds. You can also send a photo if it's easier to show than tell.

### Can I send photos?

Yes. Take a photo on WhatsApp like normal, add a question as the caption, and send. Ongiini can look at the picture and answer questions about it — useful for a sick plant, a strange rash, an ingredient label, a contract page you can't quite read, or a handwritten note. It works the same way in English and Afrikaans. The photo itself isn't kept — Ongiini reads it in memory and then discards the file. Only what it learned from the picture (something like 'shared photo of yellowing maize leaves') goes into long-term memory, and 'delete my data' wipes those learnings too.

### How accurate is it?

Sometimes very accurate. Sometimes confidently wrong — that's the nature of AI today. Treat its answers like advice from a knowledgeable friend who can also make mistakes. Never make decisions about your health, money or legal status based on Ongiini's answer alone. Verify with a doctor, lawyer or other qualified person.

### What exactly do you store about me?

Three things. (1) Roughly the last 50 turns of back-and-forth from your phone number so Ongiini can follow a real conversation — anything older gets compressed into a short summary line, and obvious personal data (email, ID, IBAN, card) is auto-scrubbed and replaced with placeholders before saving. (2) A small set of durable facts about you that Ongiini extracts across all our chats — things like 'lives in Oshakati', 'farms maize', 'prefers Afrikaans replies' — stored as a tiny vector database on the same computer. This is what lets it actually feel like talking to someone who remembers you between sessions. (3) A line per message in a usage log with token counts and a timestamp — no message content. Photos and voice notes never become a fourth thing — the file itself is processed in memory and discarded; only the textual learning from it (a short description of the picture, or the transcript of the voice note) flows into (1) and (2) above. Nothing else. No precise location, no contacts list, no Meta profile data. Want to see what's stored? Ask 'what do you remember about me?' on WhatsApp. Want it all gone? Send 'delete my data' — both work in English and Afrikaans, both wipe all three layers.

### Where is it physically running?

On a single NVIDIA DGX Spark — a personal AI supercomputer about the size of a small book (one petaflop of compute, 128 GB of unified memory), not a data centre. It's currently in Germany during the pilot phase. The goal is to move the hardware to Namibia once the service is sustainable. No US cloud provider is in the pipeline at any point.

### Why a German number, hosted in Germany?

Honest answer: convenience. Ongiini is the first project of the Common Intelligence Foundation, in pilot phase, and Germany is where the foundation's current operator is based and where the hardware sits. A German WhatsApp Business number on a German broadband line was the fastest way to get something real running and put it in front of users. The service itself is locked to Namibian numbers (+264) — that's the audience this is built for. If the pilot earns traction and becomes sustainable, the plan is to physically move the hardware to Namibia and switch to a Namibian number.

### Why WhatsApp?

Because almost everyone in Namibia already has it. No download, no sign-up, no learning curve. You write, it writes back.

## Privacy policy (full text)

How Ongiini handles your data — last updated 21 May 2026

**The short version.** Ongiini is a free AI helper on WhatsApp. To work, we receive your messages (via Meta) and your phone number, and we keep a small amount of information so the assistant can follow conversations. We do not sell your data, do not show you ads, and do not train anyone's AI on your conversations. We do publish aggregate, anonymous statistics about how the service is used — themes, professions, growth — to be transparent about our impact; individual conversations are never published (see Section 7).

You can see what we remember about you by sending *"what do you remember about me?"* on WhatsApp. You can delete it by sending *"delete my data"*. Both work in English and Afrikaans, any time.

Under the EU AI Act (Reg. 2024/1689), Ongiini is classified as a **limited-risk AI system** — a chatbot subject only to transparency obligations, not the stricter high-risk requirements. See Section 9 below.

### 1. Who is responsible

The controller of your personal data within the meaning of Article 4 (7) GDPR is:

Sebastian Küpers · Hibiskusweg 17b · 13089 Berlin · Germany · [sebastian.kuepers@gmail.com](mailto:sebastian.kuepers@gmail.com) · [+49 170 2372987](tel:+491702372987)

Ongiini is the first project of the [Common Intelligence Foundation](https://common-intelligence.org), currently being formally established as a non-profit foundation in Estonia. Until that registration is complete, the service is privately funded and operated on a non-profit basis. After registration, the foundation will become the controller; you will be informed of that change in this policy.

The service is provided without charge and is not operated for profit.

Our processing does not meet the thresholds in Art. 37 GDPR for the mandatory appointment of a Data Protection Officer. You can reach us with any privacy enquiry using the contact details above.

### 2. What we process, why, and on what legal basis

#### Your phone number and message content

When you message the Ongiini WhatsApp number, we receive your phone number (as your WhatsApp identifier) and the content of each message you send — text, images, and voice notes. Voice notes are downloaded from Meta and transcribed to text on our own computer using an open-source speech-to-text model (Whisper). The original audio bytes are not retained after transcription; only the text transcript is processed and stored, the same way as any other text message.

**Purpose:** to read your question and provide an answer — i.e. to deliver the service you requested by initiating the conversation.

**Legal basis:** Art. 6 (1) (b) GDPR — performance of pre-contractual measures requested by you. By sending a message, you ask the service to respond.

#### Short-term conversation memory

We keep approximately the last 50 turns of your conversation (each turn = one message from you plus one reply) in a local file on the computer that runs Ongiini, identified by your phone number. Once the stored history grows beyond about 70 entries, the oldest entries are condensed into a single short summary line ("Earlier in this conversation: …") and the most recent ~40 turns are kept verbatim. Before any message is written to disk, we automatically scrub obvious personal data patterns from the text (email addresses, IBANs, credit-card numbers and Namibian-format ID numbers are replaced with placeholders such as `[REDACTED:email]`).

**Purpose:** to allow the assistant to follow a conversation across messages (so you don't have to repeat context).

**Legal basis:** Art. 6 (1) (b) GDPR — necessary to provide the conversation service you requested.

#### Long-term memory ("mem0")

Across all your chats, the assistant extracts a small number of typed facts about you and stores them as short text fragments. Categories are: `[PROFILE]` (location, role, family), `[PREFERENCE]` (language, style), `[SITUATION]` (ongoing topics), `[COMMITMENT]` (follow-ups, reminders), `[QUOTE]` (verbatim phrasing the assistant might re-use), and `[EMOTION]` (recent state). These facts are stored as numerical embeddings in a local vector database (qdrant), identified by your phone number, and are retrieved by semantic similarity each time you write so the assistant can recall relevant context. mem0 also maintains a separate local SQLite database that records when each fact was added, updated or deleted — this contains the fact text and metadata, but no full message content beyond the extracted fact itself.

**Purpose:** to make the assistant useful over time — to remember that you farm maize, that you prefer Afrikaans replies, that you previously asked about a school topic, etc.

**Legal basis:** Art. 6 (1) (b) GDPR — provision of the personalised service you requested. You can delete all of this at any time (see Section 6).

#### Usage log

We keep one line per message in a usage log: your phone number (used as identifier), a timestamp, the message kind (text / image / audio), the number of input/output tokens used, and a flag indicating whether the assistant used web search for that turn. The log does not contain the content of any message.

Separately, a structural trace file records, per message: number of model calls, token counts per call, latency, finish reason, and tool-call names + payload lengths. It deliberately does not record message content, tool arguments verbatim, or tool results — only structural signals.

A short-term in-memory rate-limiter tracks message timestamps per phone number to detect bursts of activity. This state lives only in process memory and is lost whenever the service restarts; it is not written to disk.

**Purpose:** to monitor fair use of the free monthly token allowance, to detect abuse, and to keep the service operationally sustainable.

**Legal basis:** Art. 6 (1) (f) GDPR — legitimate interest in service sustainability and abuse prevention. The interest is proportionate because none of these records contain message content.

#### Website access logs (ongiini.ai)

The website is served via Cloudflare. For security and operational purposes, Cloudflare logs each request: source IP address, user-agent string, requested URL, timestamp, and HTTP status. Cloudflare also processes basic security signals (rate-limiting, bot detection).

The website does not set advertising cookies, does not load any analytics scripts, and uses your browser's `localStorage` only to remember your language preference (English / Afrikaans). This is technically necessary and does not require consent.

**Legal basis:** Art. 6 (1) (f) GDPR — legitimate interest in security and operability.

### 3. Who else sees your data (processors and third parties)

- **Meta Platforms Inc. (WhatsApp)** — to deliver messages between you and us via the WhatsApp Business API, Meta receives, processes and routes them. Meta retains messages in line with its own policies. We have a data-processing relationship with Meta but cannot control Meta's own processing of your data — that is governed by Meta's terms and privacy policy.
- **Tavily (Tavily Inc., a US-based provider)** — used only when the assistant decides to perform a web search for your question. In that case, the assistant's search query (which may reference what you wrote) is sent to Tavily's API, and Tavily returns search results. Per Tavily's published [privacy policy](https://www.tavily.com/privacy), Tavily *may* use search queries to improve its own service. If you do not want your search queries to be used in this way, you can object directly to Tavily at [support@tavily.com](mailto:support@tavily.com). The exact processing location is governed by Tavily's own privacy practices.
- **Cloudflare Inc.** — hosts the website (ongiini.ai). Cloudflare does not have access to your WhatsApp conversation or any data on our local infrastructure.

The AI model used (Google DeepMind's Gemma 4 26B) is an open-weight model that runs *locally* on hardware we operate. Google does not receive any of your data through the use of this model.

Voice-note transcription is also performed locally by an open-source speech-to-text model (Whisper / faster-whisper). No third party receives the audio.

**We do not sell, rent or share your data with anyone else. We do not use it for advertising. We do not use it to train any AI model.** We do, however, run aggregate analyses across stored data for transparency reporting and research — only aggregate, statistical results are produced and published; individual conversations are never published. See the "Research, analytics & transparency reporting" section below for details. One nuance: when web search is invoked, Tavily — as a separate company governed by its own terms — may use the search queries it receives from us to improve its own service (see the Tavily entry above). Ongiini itself trains no model on user data.

**Data Processing Agreements.** Our two primary processors operate under formal data-processing terms required by Art. 28(3) GDPR:

- Meta's WhatsApp Business API is governed by Meta's [Business Data Processing Terms](https://www.whatsapp.com/legal/business-data-processing-terms) and [Business Data Transfer Addendum](https://www.whatsapp.com/legal/business-data-transfer-addendum), incorporated by reference into the WhatsApp Business API terms we accepted.
- Cloudflare's [Customer Data Processing Addendum](https://www.cloudflare.com/cloudflare-customer-dpa/) is publicly published and automatically incorporated into our service terms.

Tavily does not currently publish a standalone Art. 28 GDPR Data Processing Agreement; its relationship with users is governed by its own published [privacy policy](https://www.tavily.com/privacy) and [terms of service](https://www.tavily.com/terms), including the SCC references for non-EU transfers cited there.

### 4. International data transfers

Meta and Tavily are based in the United States and process data there. Cloudflare also operates globally including in the United States. Transfers to the United States are based on the **EU–US Data Privacy Framework** (adequacy decision of the European Commission of 10 July 2023) where the recipient is certified under it, otherwise on the European Commission's **Standard Contractual Clauses (SCC)** with supplementary technical and organisational measures.

For transfers under the SCCs, the supplementary measures we rely on include: (a) **encryption in transit** (TLS) for all API calls to processors; (b) **data minimisation** — US-based processors only receive the data strictly necessary for their service (Meta receives WhatsApp message routing per its API; Tavily receives only the model's search query at the moment a search is performed; neither receives access to our local memory, mem0 store, or usage logs); (c) **no persistent storage of our memory data on the processor side** — short-term JSON memory, the mem0 vector store, the usage log and the trace log all live on our own computer.

The Ongiini computer itself, where short-term and long-term memory are stored, is currently physically located in Germany (EU). Once the service moves to Namibia, processing will take place in Namibia.

### 5. How long we keep your data

- **Short-term conversation memory:** approximately the last 14 turns, summarised thereafter. There is no hard time limit; older turns roll out as new ones come in.
- **Long-term memory ("mem0"):** the extracted facts (vector embeddings in qdrant) and their mutation history (local SQLite log) are stored indefinitely *until you ask us to delete them*, or until the conversation has been inactive for an unusually long time and we decide to prune the data set during routine maintenance. The "delete my data" command wipes both.
- **Usage log:** stored indefinitely, but contains no message content.
- **Trace file:** stored indefinitely. Structural signals only — no message content, no tool arguments, no tool results.
- **Cloudflare access logs for the website:** retained by Cloudflare per its default retention (typically around 7–30 days).

You can delete your conversation memory at any time by sending *"delete my data"* on WhatsApp (see Section 6).

### 6. Your rights

Under the GDPR, you have the right to:

- **Access** (Art. 15) — find out what we have stored. Send *"what do you remember about me?"* on WhatsApp at any time. For the full dataset including timestamps and the usage log, email us at the address in Section 1.
- **Rectification** (Art. 16) — correct anything inaccurate. The simplest way is to tell the assistant the correct fact and the memory will be updated. Alternatively, email us.
- **Erasure** (Art. 17) — delete your data. Send *"delete my data"* on WhatsApp. Your conversation memory (short-term and long-term) is wiped immediately. The usage log line referring to that message is retained as anonymous statistics (no content, no identifier link).
- **Restriction of processing** (Art. 18) — limit how we use your data. Email us.
- **Objection** (Art. 21) — object to processing based on legitimate interest. Email us; you can also simply stop using the service.
- **Data portability** (Art. 20) — receive an export of your data in a machine-readable format. Email us; we'll send you a JSON export of your mem0 facts and short-term memory.
- **Lodge a complaint** with a supervisory authority. Since we are based in Berlin, the relevant authority is the [Berliner Beauftragte für Datenschutz und Informationsfreiheit](https://www.datenschutz-berlin.de/). You may also complain to the supervisory authority in the country where you live.

### 7. Research, analytics & transparency reporting

Part of the Common Intelligence Foundation's mission is to understand how AI access changes the lives of underserved communities — and to share what we learn openly. To do that, we run aggregate analyses across the data we already store for service operation.

**The questions we ask the data.** Examples:

- How many people use Ongiini per day, week, month — and how is the user base growing?
- What share of conversations relate to health, education, agriculture, contracts, daily life, or other topics?
- What roles or professions appear in our user base (farmers, students, teachers, parents, etc.), as inferred from conversation context?
- Which languages are people using, and how is that distribution shifting?
- How often does the same user return, and how does engagement change over time?
- When do people use the service — time of day, day of week?

**Lawful basis.** Art. 6 (1) (f) GDPR — legitimate interest of the foundation in understanding the impact of the service it operates, in reporting transparently on its work, and in contributing to the wider field of research on AI access in underserved communities. Where results are published as scientific research, Art. 89 GDPR and § 27 of the German Federal Data Protection Act (*Bundesdatenschutzgesetz*, BDSG — the "Forschungsprivileg" or research privilege) also apply, with the safeguards described below. You may object to this processing at any time under Art. 21 GDPR — see "Your right to object" at the end of this section.

**What we publish.** Only aggregate statistical results — counts, percentages, charts, trends — that do not identify any individual. Such results may appear on the website (e.g. on a public `/statistics` page), in foundation reports, or in academic papers.

**What we never publish.**

- Individual conversation content, in full or in excerpt, where any reasonable risk of identifying a specific person exists.
- Phone numbers, narrow locations, names, or any other direct identifier.
- Quotes or examples that have not been independently reviewed and stripped of identifying detail.

**Safeguards.** The data analysed is the same data we already store for the service: short-term memory, mem0 facts, and the structural usage log (see Section 2). PII patterns are scrubbed before storage. Analysis is performed on our own infrastructure, by team members operating under a confidentiality obligation. Where a human-readable example is being considered for publication, it goes through a separate de-identification review before it can leave the foundation.

**Your right to object.** Because this processing relies on legitimate interest, you may object at any time under Art. 21 GDPR. Email us at the address in Section 1 with the subject line *"object to research processing"* and your phone number. We will mark your data as excluded from current and future aggregate analyses and publications. The service itself continues to work normally.

### 8. Automated processing and AI

Ongiini's replies are generated entirely automatically by an AI model — there is no human in the loop. The service does not make decisions that produce legal effects or similarly significantly affect you within the meaning of Article 22 GDPR; it provides information in response to your questions, and you decide what to do with that information.

AI-generated answers may be inaccurate, incomplete, or wrong. Do not rely on Ongiini's answers for medical, legal, financial, or other significant decisions. See the [Terms of Service](/terms/) for the full disclaimer.

### 9. EU AI Act classification

Under Regulation (EU) 2024/1689 (the AI Act), Ongiini is a chatbot — a **limited-risk AI system** subject to the transparency obligations of Article 50. It is not a high-risk AI system under Annex III, and it does not engage in any of the prohibited practices listed in Article 5.

We meet the Article 50 transparency obligations by clearly identifying every interaction as AI-mediated: in the WhatsApp Business profile, in the assistant's first reply to every new user, in the disclosures on this website, and in this policy. The disclosures are provided in plain language at the point of first interaction (Article 50(5)).

The underlying model (Google DeepMind's Gemma 4 26B) is a general-purpose AI model. Under Article 25 of the AI Act, we accept the provider responsibilities for the integrated Ongiini chatbot system that we build on top of it.

### 10. Children

Ongiini is intended for general users in Namibia and is not specifically directed at children. We do not knowingly process personal data of children under 16. If you believe a child has used the service, please contact us and we will remove the relevant data.

### 11. Security

Conversation data is stored on a single computer operated by us, in Germany, behind a firewall and accessible only via authenticated administrative access. We apply state-of-the-art technical and organisational measures (Art. 32 GDPR), but no system is perfectly secure. If we become aware of a personal-data breach affecting your rights, we will notify the supervisory authority within 72 hours (Art. 33 GDPR) and, where required, inform you directly (Art. 34 GDPR).

### 12. Changes to this policy

If we change this policy materially, we will update the date at the top and, for substantial changes (e.g. new categories of processing, new processors, change of controller upon foundation registration), notify users via the WhatsApp service or on the website.

Last updated: 21 May 2026. Effective immediately.

## Terms of service (full text)

Plain-language rules for using Ongiini — last updated 21 May 2026

**The short version.** Ongiini is a free AI helper on WhatsApp. By messaging the Ongiini number, you accept these terms.

Treat Ongiini's answers as a useful starting point — **not as professional advice**. For decisions about your health, your money, your legal situation, your safety, or anything else that really matters, always check with a qualified person.

We provide the service for free, as is, with no guarantees. We may suspend or stop it for anyone, at any time. Use at your own risk.

Under the EU AI Act (Reg. 2024/1689), Ongiini is classified as a **limited-risk AI system** — a chatbot subject only to transparency obligations, not the stricter high-risk requirements. See Section 12 below.

**A note on legal references.** These Terms are governed by **German law** (see Section 15). Citations like "§ 312g BGB" refer to the German Civil Code (*Bürgerliches Gesetzbuch*, BGB); other German statutes are spelled out on first mention.

### 1. Acceptance and scope

These Terms of Service ("Terms") govern your use of Ongiini, an AI helper accessible via WhatsApp at the number published on [ongiini.ai](https://ongiini.ai) and described on that website ("the Service"). These Terms and the [Privacy Policy](/privacy/) are continuously available on this website. The WhatsApp Business profile of the Service links to this website, giving you the opportunity to review the Terms before initiating use. By sending a message to the Service, you confirm that you have had that opportunity and that you accept these Terms. If you do not accept them, do not use the Service.

The Service is operated by Sebastian Küpers (Hibiskusweg 17b, 13089 Berlin, Germany) as the first project of the [Common Intelligence Foundation](https://common-intelligence.org), currently being formally established as a non-profit foundation in Estonia. The Service is provided **free of charge** and is **not operated for profit**. Once the foundation is registered, it will become the operator; we will update these Terms accordingly and notify you of the change.

### 2. What the Service is

Ongiini is an AI helper that receives your WhatsApp messages (text, photos, and voice notes), generates responses using an AI model, and replies in text. Voice notes are transcribed to text on our own computer using an open-source speech-to-text model; we do not send your audio to any third party. The Service is designed to help with everyday questions — for example school topics, farming questions, contract clauses, health information, government services, and similar.

The Service is **not a medical, legal, financial, psychological, or any other kind of professional service**, and it is **not operated by qualified professionals**. It is a computer program. See Section 6.

Access is restricted to phone numbers from Namibia (+264). The Service may decline to respond, or respond differently, based on country code or other technical signals.

### 3. Eligibility

You must be at least **16 years old** to use the Service. If you are between 16 and 18, you confirm that your use is permitted under the law of the country in which you live and, where required, by your parent or legal guardian.

By using the Service you confirm that you have the legal capacity to enter into these Terms.

### 4. Free service, no entitlement to availability

The Service is provided **free of charge**. There is no payment, no subscription, no commercial relationship. You acquire no entitlement to the Service being available, complete, accurate, or fit for any particular purpose. We may change, restrict, suspend or discontinue the Service in whole or in part at any time, with or without notice, and without compensation.

Each user has a monthly allowance of free tokens (a measure of message length and conversation depth) as published on the website. We monitor usage and may, at our discretion, rate-limit or block any user whose usage materially exceeds normal individual use.

Because the Service is provided free of charge as a gratuitous service, **no statutory right of withdrawal under § 312g of the German Civil Code (BGB) applies** — no fee-based distance contract is concluded. You may stop using the Service at any time and may delete your data at any time via the in-product commands described in the [Privacy Policy](/privacy/).

### 5. Acceptable use

You agree that, when using the Service, you will **not**:

- Use the Service in any way that violates applicable law (whether in Germany, Namibia, or any other relevant jurisdiction) or infringes the rights of any third party.
- Use the Service to harass, threaten, defame, defraud, stalk, or harm any person.
- Send content that is illegal, hateful, sexually exploitative of minors, or otherwise abusive.
- Use the Service to generate, promote, or distribute disinformation — including false or misleading claims about political figures, elections, public-health matters, or other consequential topics — or to mass-produce content for that purpose.
- Attempt to extract, copy, or reverse-engineer the Service's underlying instructions, system prompt, model weights, or infrastructure.
- Attempt to bypass safety mechanisms, country-code restrictions, or rate limits, or impersonate any other user.
- Use the Service to generate content that you then present as professional advice (medical, legal, financial, etc.) to a third party.
- Use automated tools, bots, or scripts to access the Service, except as expressly authorised by us.
- Resell, repackage, syndicate, or commercially redistribute the Service or its output — including republication of Ongiini's answers in any paid product, paid newsletter, paid course, or any context in which a fee is charged for access to the output — without our prior written permission. Non-commercial sharing of individual answers in personal or community contexts is fine.

We may suspend or block your access (by phone number or otherwise) at any time, without notice, if we reasonably believe you have breached these Terms.

### 6. Not professional advice — your decisions are your own

Ongiini is an AI. Its answers can be wrong, incomplete, or out of date. They may sound confident even when they are not. **Do not act on Ongiini's answers in any matter that materially affects your health, your money, your legal situation, your safety, or another person's life or rights — without consulting a qualified human professional** (a doctor, lawyer, accountant, financial adviser, agricultural extension officer, etc.).

Specifically:

- **Medical:** Ongiini is not a doctor and does not provide medical diagnosis or treatment. In an emergency, contact local emergency services.
- **Legal:** Ongiini is not a lawyer and does not provide legal advice or representation. Consult a licensed legal professional.
- **Financial:** Ongiini is not a financial adviser and does not provide investment, tax, or accounting advice. Consult a qualified adviser.
- **Safety / agriculture / mechanical:** Where the consequences of an error are serious (e.g. taking dangerous chemicals, operating machinery, deciding whether to seek emergency care for a child), confirm with someone qualified before acting.

You agree that you understand the above, that you will exercise your own judgement, and that you will **not rely on Ongiini's output as a substitute for professional advice** in any consequential matter.

### 7. No warranty — Service provided "as is"

To the maximum extent permitted by applicable law, the Service is provided **"as is" and "as available"**, without any warranty of any kind, whether express, implied, or statutory. We do not warrant that the Service will be accurate, complete, current, uninterrupted, error-free, secure, fit for a particular purpose, or that any defects will be corrected.

AI output is probabilistic and inherently fallible. The Service may produce content that is factually incorrect, internally inconsistent, biased, outdated, or otherwise unsuitable. The Service does not learn or improve from your specific conversations in a way that guarantees better answers over time.

### 8. Limitation of liability

Because the Service is provided **free of charge and without commercial purpose**, our liability is governed by the rules applicable to gratuitous services under German law (in particular § 521 BGB by analogy). Subject to the mandatory provisions described at the end of this Section, our liability is limited as follows:

- We are liable only for damages caused by **intent (Vorsatz)** or **gross negligence (grobe Fahrlässigkeit)** on our part or on the part of our legal representatives or vicarious agents.
- For damages caused by **simple negligence (einfache Fahrlässigkeit)**, our liability is excluded.
- In no event are we liable for indirect, incidental, consequential, special, or punitive damages, for lost profits, lost data, lost opportunities, or for damages arising from your reliance on the Service's output in matters described in Section 6 (medical, legal, financial, safety) where you did not seek professional advice.
- Subject to the mandatory provisions below, our aggregate liability arising out of or relating to your use of the Service is limited to **EUR 100** per claim and per calendar year.

**The above limitations do not apply**, and we remain fully liable, in cases of: (a) injury to life, body, or health caused by us; (b) liability under the German Product Liability Act (Produkthaftungsgesetz); (c) liability under any other mandatory statutory provision; (d) damages arising from the breach of a fundamental contractual obligation (Kardinalpflicht), in which case liability is limited to the typical, foreseeable damage.

### 9. Indemnification

You agree to indemnify and hold us harmless from and against any third-party claim, liability, damage, loss, or expense (including reasonable legal fees) arising out of or in connection with: (a) your breach of these Terms; (b) your use of the Service in a way that causes harm to another person or violates applicable law; or (c) your reliance on the Service's output in a manner inconsistent with Section 6.

### 10. Intellectual property

The Ongiini name, brand, website design, and code are owned by us (or, after registration, by the Common Intelligence Foundation). The Ongiini source code is published under an open-source licence as indicated in the GitHub repository — please refer to the repository for the applicable licence terms.

The output that the Service generates in response to your messages is not subject to copyright by us. You may use it freely, subject to applicable law and these Terms. You acknowledge that AI-generated content may be similar or identical for different users and that we make no claim of originality on your behalf.

You retain ownership of any content you submit to the Service. By submitting content, you grant us a non-exclusive, royalty-free, worldwide licence to process that content for the purpose of providing the Service, as described in our [Privacy Policy](/privacy/).

### 11. Privacy

Our processing of personal data is described in the [Privacy Policy](/privacy/), which forms part of these Terms.

### 12. EU AI Act notice

Ongiini is classified as a **limited-risk AI system** (a chatbot) under Regulation (EU) 2024/1689 (the AI Act). It is subject to the transparency obligations of Article 50 — you are entitled to know you are interacting with an AI rather than a human, and we ensure you do: in the WhatsApp Business profile, in the assistant's first reply to every new conversation, in these Terms, and in our [Privacy Policy](/privacy/). Ongiini is not a high-risk AI system under Annex III.

Specifically, Ongiini is not used for, and you must not use it for, any of the practices prohibited under **Article 5 of the AI Act**:

- (a) subliminal or purposefully manipulative techniques causing significant harm;
- (b) exploitation of vulnerabilities of specific groups (age, disability, social or economic situation) causing significant harm;
- (c) social scoring of natural persons leading to detrimental or unfavourable treatment;
- (d) risk assessment of natural persons to predict criminal offending based solely on profiling or personality traits;
- (e) untargeted scraping of facial images from the internet or CCTV footage to build or expand facial-recognition databases;
- (f) emotion inference in the workplace or in education institutions (except for narrowly defined medical or safety reasons);
- (g) biometric categorisation that infers sensitive characteristics (race, political opinion, religious belief, sexual orientation, etc.);
- (h) real-time remote biometric identification in publicly accessible spaces for law-enforcement purposes (outside the narrow exceptions in the AI Act).

You must also not use Ongiini's outputs as a component of any high-risk AI system listed in Annex III of the AI Act without conducting your own conformity assessment as required by the AI Act.

### 13. Suspension and termination

For **good cause** — including but not limited to material breach of these Terms, abusive use, illegal content, threats to the security or integrity of the Service, or imminent technical risk — we may suspend or terminate your access immediately, with or without prior notice (§ 314 BGB).

For termination **without good cause** (for example, if we decide to discontinue the Service in your region, retire the pilot, or change the country scope), we will give you at least **30 days' notice** through the WhatsApp Business profile, the website, or a direct message — where it is reasonably possible for us to reach you.

You may stop using the Service at any time. You may also delete all of your data at any time by sending *"delete my data"* on WhatsApp.

### 14. Changes to these Terms

We may amend these Terms at any time. The current version is published at this URL. Material changes will be highlighted by updating the date at the top of this page. Your continued use of the Service after a change constitutes acceptance of the amended Terms.

### 15. Governing law and jurisdiction

These Terms are governed by the laws of the **Federal Republic of Germany**, excluding the United Nations Convention on Contracts for the International Sale of Goods (CISG) and excluding the German rules on conflicts of laws. Mandatory consumer-protection provisions of the country in which you have your habitual residence remain unaffected.

The non-exclusive place of jurisdiction for disputes is **Berlin, Germany**, to the extent permitted by law.

For consumers in the EU: the European Commission's online dispute-resolution platform is available at [ec.europa.eu/consumers/odr](https://ec.europa.eu/consumers/odr/). We are not obliged and not willing to participate in dispute-resolution proceedings before a consumer arbitration board.

### 16. Severability

If any provision of these Terms is or becomes invalid or unenforceable, the validity of the remaining provisions is not affected. The invalid or unenforceable provision shall be replaced by a valid and enforceable provision that comes as close as possible to the economic intent of the original.

### 17. Contact

For questions about these Terms or about the Service, please refer to the contact information in the [Imprint](/imprint/).

Last updated: 21 May 2026. Effective immediately.

## Imprint (German § 5 DDG)

Legal information pursuant to German law

### Information pursuant to § 5 DDG

Sebastian Küpers Hibiskusweg 17b 13089 Berlin Germany
### Contact

Phone: [+49 170 2372987](tel:+491702372987) Email: [sebastian.kuepers@gmail.com](mailto:sebastian.kuepers@gmail.com)

### Responsible for content pursuant to § 18 (2) MStV

Sebastian Küpers Hibiskusweg 17b 13089 Berlin Germany
### About this site

Ongiini is the first project of the [Common Intelligence Foundation](https://common-intelligence.org), which is in the process of being formally established as a non-profit foundation in Estonia. Until that registration is complete, this website and the Ongiini service are operated by Sebastian Küpers in his personal capacity, on a non-profit basis.

Ongiini is not affiliated with WhatsApp Inc., Meta Platforms Inc. or any related entity. The trademarks WhatsApp and Meta belong to their respective owners.

### Liability for content

As a service provider, we are responsible for our own content on these pages under general law pursuant to § 7 (1) DDG. According to §§ 8 to 10 DDG, however, we as a service provider are not obligated to monitor transmitted or stored third-party information or to investigate circumstances that indicate illegal activity. Obligations to remove or block the use of information under general law remain unaffected. However, liability in this regard is only possible from the point in time at which knowledge of a specific legal violation becomes apparent. If we become aware of corresponding legal violations, we will remove the content in question immediately.

### Liability for links

Our offering contains links to external websites operated by third parties whose content we have no influence over. We therefore cannot accept any liability for this external content. The respective provider or operator of the linked pages is always responsible for their content. The linked pages were checked for possible legal violations at the time of linking. Illegal content was not recognisable at that time. A permanent inspection of the linked pages is not reasonable without concrete evidence of a legal violation. Upon becoming aware of legal violations, we will remove such links immediately.

### Copyright

The content and works on these pages created by the site operator are subject to German copyright law. Duplication, processing, distribution and any form of commercialisation of such material beyond the scope of copyright law require the written consent of the respective author or creator. Downloads and copies of this site are permitted only for private, non-commercial use. Insofar as content on this site was not created by the operator, the copyrights of third parties are respected. In particular, third-party content is identified as such. If you nevertheless become aware of a copyright infringement, please inform us accordingly. Upon becoming aware of legal violations, we will remove the content in question immediately.

Last updated: May 2026.
