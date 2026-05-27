#!/usr/bin/env python3
"""Rewrite the 145 mined items into clean natural WhatsApp-English.

Real user messages have typos, grammar errors, and fragmented phrasing
that would force Elizabeth to translate her *interpretation* of intent
rather than the literal source. That contaminates the eval signal —
we'd be measuring "does the model interpret the same way Elizabeth
does" instead of "does the model translate correctly".

Each rewrite preserves:
  - Intent / meaning (the real-WhatsApp signal)
  - Domain + length bucket
  - Casual register (informal but grammatically clean)
  - Code-switching where meaningful (e.g. "Tangi", "Khoekhoegowab")

Two IDs are REMOVED outright because regex missed them (curly-quote
or comma-separated name patterns):
  68  — personal name "Eino, surname Enkono"
  229 — specific kindergarten "Smiley's" (curly apostrophe)

Modifies data/eval_v2_real_candidates.tsv in place:
  - For approved rows: replace english with the cleaned version
  - For the 2 removed IDs: clear sebastian_approved
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

TSV = Path("/Users/sebkuepers/dev/Ongiini/data/eval_v2_real_candidates.tsv")


# id → rewritten EN (or "" to mark for removal)
REWRITES: dict[int, str] = {
    # ── chat ────────────────────────────────────────────────────
    4: "My day was rough. I was at work and it was a productive day, but now I'm a bit tired.",
    7: "I'm an HR student. I want to learn Afrikaans in case someone at my internship doesn't speak English well but speaks Afrikaans.",
    13: "Help me rewrite that CV. Add a small space in the corner for a photo and put page borders on it — it should be ATS-friendly.",
    23: "I want to cancel my funeral cover because the amount they'll pay out if someone passes won't even buy a coffin.",
    31: "I just want to understand the module so the exam will be easier — I want to see the whole picture of what it's about.",
    32: "Thanks. I'm a content creator and I want to sell my work, but I don't know where I can sell digital products.",
    33: "The thing is, I don't have a smartphone or a laptop, so what do I do? Right now I'm using a stranger's phone to ask for help.",
    34: "What careers are open to a NUST student doing a Bachelor of Science in general physics, biology, chemistry, and computer maths for natural sciences?",
    40: "The challenge is I don't know how to pronounce the word. It would be best if you sent me an audio recording of the vowels and consonants.",
    41: "Now can I send my CV or give you my details? I'm in Namibia — how would you submit it from here?",
    46: "I applied for NSFAF a few days ago and still haven't received a response. My exams are coming up and I'm behind on tuition fees.",
    48: "I have experience cleaning my aunt's house — I'm good at laundry, ironing and washing dishes. I'm looking for a full-time job.",
    51: "I want to improve my symbols for further studies, because some fields only accept students with certain points and symbols.",
    55: "If I give you a task to write a preliminary research report for me, can you do it without me getting expelled from NUST for plagiarism?",
    57: "I'm driving a truck but the cruise-control down-hill button isn't working and the retarder isn't working either — what should I use to go down the hill?",
    59: "Here's the situation: I'm doing research for my honours study, and the owner of the farm wants these chemicals to be registered.",
    60: "I came across the post on Facebook and decided to contact you, so I can learn Oshiwambo words straight from English.",
    62: "I just need a day-old chick feeder, and a veterinary supplier for the medicines. I'll take care of the chickens myself.",
    63: "I'd like to work as a remote WhatsApp agent in Namibia, helping with every new function update based on our local culture and environment.",
    64: "Yes, it changes, but I experience it more when I'm sleeping, standing for a long time, or sitting for more than an hour.",
    65: "My exams start with accounting on the 18th, quantitative methods on the 19th, economics on the 23rd, and business analytics on the 25th. Draw up an effective timetable for me.",
    66: "It's low down near my hips — a sharp stabbing pain, sometimes with diarrhoea, and it happens even when I'm resting. I've had this for a full year now.",
    67: "Help me rephrase this sentence: 'I won't be available tomorrow, so let's reschedule for when you're free too.'",
    68: "",  # REMOVE — personal name slipped through regex
    71: "I follow some pages on Instagram and I want to ask if they offer HR internships — how do I start the message in English but professionally?",
    73: "What are UNAM's requirements if I want to further my study with a bachelor's degree in nursing science? I'm currently an enrolled nurse.",
    75: "It's not a dry mouth — actually a lot of saliva, and no headache. It's the same whether I eat or not.",
    79: "How long does it take for prepared baby formula to spoil? I'd like to leave it ready before I go to work.",
    80: "I'm a medical student. Let's go through a general exam starting from the feet before we move to the systemic exam, and then the respiratory examination.",
    81: "Should I refer to the case study as being 'in the Zambezi region' or 'in the Zambezi region, Namibia'?",
    82: "The first document is on how to write a report, and the second contains the idea that the report should be about.",
    83: "She has no manners — she talks to anyone however she wants and always insults people with bad words.",
    85: "I understand English well, but I can't speak fluently and my pronunciation is very bad. What do I need to do?",
    88: "Write me a letter in Afrikaans that apologises to my ex.",
    93: "Will you remind me if I haven't finished my assignment, or will you wait for me to come back here?",
    95: "I have chemistry mock exams coming up on the 1st of June for my grade 11. I need help studying the essentials.",
    99: "Are there any Namibian agencies, besides NSFAF, that offer assistance for online masters degrees abroad?",
    101: "Yes, let's start with the introduction. How do I write it properly with in-text citations and everything in good order?",
    102: "I don't have a computer with me right now — can you keep the information here so I can copy it when I get to a PC?",
    103: "What does a typical rental contract look like? I want to help someone draw one up for her house.",
    104: "I have a story already — I just need to know where to start so I can make it watchable.",
    105: "How do I counsel a learner who is very slow at studying and doing work, and who has low self-esteem?",
    106: "I want you to help me prepare lesson plans for Agricultural Science grade 8 to 11.",
    109: "Give me a homestead plan for Oshindonga people, with a modern main house that includes elugo, ojugo, olupale, and everything else.",
    111: "I was looking for an app with a green logo or something — I want to write my learner's licence.",
    112: "So do I need to send you a picture of my CV so you can add some experience, or how does it work?",
    113: "I want to know — for people who wrote the mature-age entry last year, did they have to take a test?",
    115: "I teach communication with parents and class management. I'm looking for a GRN teaching job — make me a quick printable for it.",
    116: "How does one get started? Is it like a class, or can I only ask for help when I need it?",
    117: "Yeah, I can contribute. Let's start with basic Afrikaans first. By the way, can you help with relationship tips and that kind of thing?",
    118: "Okay, I want to start a group for helping students with maths so I can create a daily income.",
    119: "How about the kind that burns near the belly button, and on the surface of the stomach it's warm?",
    120: "Good morning. Can you help me write a financial assistance request letter and tell me what to include in it?",
    121: "Could you please help me with a proper flyer for my hair business? I'll provide what should be written on it.",
    122: "I'm looking for recent fitter-and-turner job postings in Namibia, around the north and other towns.",
    123: "Remember, I'm a woman and I'm really interested in how technology works. But because I'm poor, I can't afford it.",
    125: "Thanks for your reply — how can I start a small business if I don't have any capital to start with?",
    126: "Help me come up with notes I can give learners, and help me explain set notation.",
    127: "Yes, thanks. While I'm busy with my research, please help me draft a suggestion for the research proposal.",
    128: "Please tell me more about how to put an email on a CV as a clickable link.",
    129: "I want to study my MSc abroad. What steps should I take, and where can I find a fully-funded scholarship?",
    130: "NSFAF rejected me — apparently the course and institution aren't accredited by the NQA. What should I do?",
    131: "Will you be able to show me how to write a cover letter for an internship?",
    134: "Write a brief research report on the effects of overcrowding in the maternity ward on intrapartum care.",
    135: "Help me make four lesson plans, including an assessor's evaluation, a self-evaluation, and evidence of teaching.",
    136: "Create a junior-primary story for me. It should have the basic elements of a story.",
    139: "In maths, give me the common questions and answers most likely to be asked on the topic of calculus.",
    140: "I want you to help our salon move up to the next level and attract more customers.",
    144: "Help me with weight-gain tablets — I've been trying to gain weight for 5 years now with no change.",
    146: "Thanks for this. If this opportunity to become a pharmacist falls through, can I have a second chance?",
    147: "How do I wish my friend a happy birthday — she's really smart and turned 19 before me.",
    148: "I have 10 years of experience as a security guard. Can you add this to my CV?",
    151: "Make a clear note for a grade 8 learner, easy to understand for a learner who struggles with English.",
    154: "I want you to help me design my CV in a way that will get me employed quickly.",
    156: "Hello — could you please make grade 9 mathematics lesson plans for me, strictly according to the Namibian mathematics syllabus?",
    157: "Let's start on my CV. I want to apply for electrical work — I'm a level 3 certificate holder.",
    158: "I'm a Chokwe speaker. Last question — how do you study for the English NSSCO? I'm writing the exam today at 2pm.",
    159: "I'm a grade 9 dropout, but I worked as a waitress and a cleaner. My languages are Afrikaans and Khoekhoegowab.",
    160: "I only have a topic so far — can I drop a PDF here so you can continue from there?",
    164: "What kind of job can you get with a level 4 vocational qualification in solar equipment installation and maintenance?",
    166: "Alright, thanks. I just wanted to know. I'll knock on your door when I'm in need.",
    167: "I wanted you to give me well-summarised notes from my lecturers so I can study for the exam.",
    170: "Okay, I have back pain — the moment I lift heavy things from the ground, it hurts.",
    171: "Can you give me the first chapter of the grade 10 accounting textbook on the new curriculum, along with its notes?",
    172: "Not a story — just any book you have in mind that an actuarial science student would recognise.",
    175: "About the things you guys posted on Facebook — I wish I had a laptop so I could join.",
    176: "What's the best way to attract middle-class customers to your bar with a more relaxed vibe?",
    177: "I'm interested in doing both, step by step. Let's start with the scenario practice.",
    178: "I have an old CV — can I attach it so you can modernise it into a one-page version?",
    179: "Yes, I need your help with an updated CV. Should I share my information so you can help me?",
    # ── formal ──────────────────────────────────────────────────
    198: "Thank you. Under Namibian tax law, can property owners with rental properties claim depreciation as a deductible expense from their gross income, in addition to other related expenses?",
    199: "The subject is computerised accounting. I don't understand balance sheets, the cash book, or bank reconciliation — I don't know how to put them together.",
    203: "I need sample interview questions and answers for a Senior Hydrogeologist position at the Ministry of Agriculture, Water and Land Reform. Tell me everything I need to know.",
    204: "Yes, I need your assistance with my school finances. If you're willing to help, I can provide all my school information.",
    205: "I'll send you the one I have. Please upgrade it for me, and help me with possible written-interview questions for a teaching post from pre-primary up to grade 4.",
    210: "Thanks again — I think I'll first try NSFAF and the legal aid clinic to see how they can assist me. Thanks again for your information and suggestions.",
    212: "I'm a qualified but unemployed teacher who is struggling to find a job in Namibia. Please help me with oral and written interview questions.",
    214: "I have an interview for a Policy Analyst position at the Office of the Prime Minister. I just want to prepare properly and get it right.",
    216: "How do I write a letter to my governor asking for school financial assistance in a professional way? Please give me an example using simple English.",
    218: "Yes, my CV. I've been applying for a long time without success at the Ministry of Education and the health services. Please help me — I'm really struggling.",
    221: "You're right. Could you please provide me with a PDF of all the possible formulas we use in microeconomics?",
    222: "I want to create a budget to show the bank so I can get a loan for better equipment.",
    223: "Could you please give me the meaning of communication, how important it is, and what relationships have to do with it?",
    224: "Examine the influence of ICT integration on the academic performance of grade 10 learners at a selected school.",
    225: "Help me write a short, professional email to send when I attach these files to my application.",
    227: "How is your evening going? I'd like to chat during the daytime so you can help me with my school work.",
    229: "",  # REMOVE — specific kindergarten with curly-apostrophe (regex missed)
    230: "Help me with a research proposal on 'Exploring the Role of ICT in Leadership Practice in Remote Schools'.",
    231: "Please give me a full university-level summary on property, plant and equipment, and explain with examples.",
    232: "How do I describe myself for an interview for a Chief Administrative Officer position in the Government of Namibia?",
    234: "Alright, I understand you — I have a certificate in cleaning agents and 15 years of experience.",
    235: "I'm in secondary school doing AS-level — Maths, English, Physics, and Chemistry. I love the medical industry.",
    237: "Thank you. I have a grade 9 certificate and I want a job — which jobs can I apply for?",
    238: "I'm a language teacher, teaching English and Afrikaans, with more than 15 years of experience.",
    239: "My career goals: since I'm a teacher, I'd like to grow professionally through studying and getting promoted.",
    240: "What jobs can I quickly get in Namibia with a grade 12 certificate and 18 points?",
    242: "I want to rewrite my English grade 12 exam on the Namibian syllabus. I finished school in 2000.",
    244: "Do you know how to format a document so that AI is not detected?",
    246: "I'm working as a government nurse — I want to move to the private sector.",
    248: "Could you please rewrite the things in the document I sent you?",
    250: "I have proof of payment — I want to start with the August intake.",
    251: "Can you tell me more about the Minister of Urban and Rural Development?",
    253: "Could you send me a summary of the Kavango East biology grade 10 syllabus?",
    254: "Yes — please give me 18 business ideas that school learners can start.",
    255: "Someone who made me cry: my best friend from school.",
    # ── religious ───────────────────────────────────────────────
    256: "When she said 'God is peace in the storm and peace in the fire', it encouraged me — even in hard times, God's peace speaks louder in my life.",
    257: "Thanksgiving, asking for forgiveness, inviting the Holy Spirit, spiritual warfare prayers, thanking God for answering us, declarations, and worship.",
    259: "I want to understand the scripture in Matthew 11:12 — I want to understand it really well.",
    261: "Hi — please give me a Bible verse I can send to my big sister.",
    262: "What is the word for 'prayer' in Afrikaans?",
    # ── community ───────────────────────────────────────────────
    266: "That's life — never mind. As long as the sun rises and sets, and you eat your food surrounded by the love of your wife and kids, that's all that matters.",
    267: "Clever — infiltrating a country using their dialect with a technology they'll never fully comprehend. Something tells me there's an element of German linguistics embedded in this, because there's a brother-sister correlation between Afrikaans and German.",
    268: "It's a situation about my son — I willingly gave him to my grandmother to raise, and now I have to decide whether he should come back or stay.",
    269: "Wow, that's great information. Please draft a proposal to the Ministry of Fisheries requesting approval for a community fish-farming project in our village.",
    271: "How do I ease the stress caused by my younger sister, who is getting married? Do I really have to attend her wedding?",
    275: "My wife wants to leave me, but I really love her and don't know what to do.",
    277: "Wow, good to hear that. When school starts, will you help me with my daughter's homework?",
    278: "It's a community engagement event with the Minister of Education — I'll raise those two topics there.",
    279: "The story should take place in a village, and it should be a new one — something that's never happened before.",
    280: "I'll be the director of the community engagement event tomorrow — help me with how to start.",
    281: "Please redraft the proposal. The purpose of the project is women's empowerment in our village.",
    282: "I want to produce a nice song about my late twin brother.",
    283: "A young woman who is interested in poultry and goats at the village level.",
    284: "How do I plan a great date with my wife in Swakopmund?",
    286: "I want a village-style 2-bedroom house design without an indoor toilet.",
}


def main() -> int:
    with TSV.open() as f:
        cols = f.readline().rstrip("\n").split("\t")
    with TSV.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    print(f"loaded {len(rows)} candidates from {TSV.name}", file=sys.stderr)

    expected_approved = {r["id"] for r in rows if r.get("sebastian_approved") == "y"}
    rewrite_keys = {str(k) for k in REWRITES.keys()}
    missing = expected_approved - rewrite_keys
    extra = rewrite_keys - expected_approved
    if missing:
        print(f"ERROR: approved IDs missing from REWRITES: {sorted(missing)[:10]}",
              file=sys.stderr)
        return 1
    if extra:
        print(f"WARN: REWRITES has IDs not approved: {sorted(extra)[:10]}",
              file=sys.stderr)

    rewritten = 0
    removed = 0
    for row in rows:
        if row.get("sebastian_approved") != "y":
            continue
        rid = int(row["id"])
        if rid not in REWRITES:
            continue
        new_en = REWRITES[rid]
        if new_en == "":
            # Mark for removal
            row["sebastian_approved"] = ""
            removed += 1
        else:
            row["english"] = new_en
            row["word_count"] = str(len(new_en.split()))
            rewritten += 1

    with TSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nrewrote {rewritten} approved items", file=sys.stderr)
    print(f"removed {removed} items (regex missed PII)", file=sys.stderr)
    final = sum(1 for r in rows if r.get("sebastian_approved") == "y")
    print(f"final approved: {final}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
