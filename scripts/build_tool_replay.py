#!/usr/bin/env python3
"""Native tool-calling replay: keep the adapter calling tools like Gemma 4 (paper 3).

T2 kept GSM8K, IFEval, images and safety, but called Ongiini AI's tools much less
often than the base model (tool_suite.py, 2026-10-08: fetch_url 5/24 vs 20/24,
web_search 56/78 vs 74/78). Cause: the training mix had no example of Gemma 4's
native tool format — the replay's tool prompts asked for JSON in plain text — so
every training row taught "answer directly".

This builds self-distilled rows in the native format: a system prompt with tool
declarations (rendered by the chat template, as vLLM renders them), a user request,
and the base 12B's own first move — a tool call or a direct answer, whichever it
chooses (greedy). The share of calls is therefore Gemma 4's, not ours. System
prompts, tool names and requests are synthetic and differ from tool_suite.py and
the production prompt, so the tool suite stays a transfer test.

Rows are pre-rendered {"prompt", "completion"} strings (train_lora.py --chat-format
rendered passes them through): the completion is the raw generated text up to and
including the stop token (<turn|> after an answer, <|tool_response> after a call).

    python3 scripts/build_tool_replay.py --out data/private/corpus/replay_tools_v1/replay.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
TEMPLATE = Path("deploy/spark/tool_chat_template_gemma4.jinja")


def fn(name, desc, props=None, req=None):
    props = props or {}
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
        "required": req if req is not None else list(props)[:1]}}}


SEARCH = [fn("search_web", "Search the internet for current information, news, prices, contacts and local facts.", {"query": "search terms"}),
          fn("internet_search", "Look up up-to-date facts on the web. Use for anything that may have changed recently.", {"q": "the search query", "region": "optional country code"}),
          fn("search", "Web search. Returns the top results with short snippets.", {"query": "what to search for"})]
READ = [fn("read_webpage", "Download one web page and return its main text.", {"link": "the full URL"}),
        fn("open_url", "Open a URL and return the readable text of the page.", {"url": "page address"}),
        fn("get_page_text", "Fetch the text content of a web page the user mentions.", {"address": "http(s) URL"})]
PERSONAL = [fn("forget_me", "Permanently erase all stored conversation history and saved facts about the current user.", {}),
            fn("erase_user_data", "Delete everything the service has stored about this user.", {}),
            fn("show_saved_profile", "Return everything currently saved about this user (profile facts, recent history).", {}),
            fn("list_memories", "List the facts remembered about the user from earlier chats.", {}),
            fn("usage_stats", "Return this user's usage for the current month (messages, tokens, limit).", {}),
            fn("quota_status", "How much of the monthly allowance the user has used and what is left.", {})]
ABOUT = [fn("service_info", "Official facts about this assistant: who runs it, cost, privacy, data retention, languages, limits.", {"topic": "optional topic"}, []),
         fn("help_center", "Search the help pages of this service (pricing, privacy, how it works).", {"question": "the user's question"})]
OTHER = [fn("get_weather", "Weather forecast for a town.", {"town": "town name", "days": "number of days"}),
         fn("convert_currency", "Convert an amount between currencies at today's rate.", {"amount": "number", "from": "currency code", "to": "currency code"}, ["amount", "from", "to"]),
         fn("book_taxi", "Book a taxi.", {"pickup": "pickup point", "destination": "where to", "time": "when"}, ["pickup", "destination"]),
         fn("get_bus_times", "Bus departures between towns.", {"origin": "from", "destination": "to", "date": "travel date"}, ["origin", "destination"]),
         fn("report_outage", "Report a power or water outage to the town council.", {"town": "town", "kind": "power or water"}, ["town", "kind"]),
         fn("set_reminder", "Set a reminder for the user.", {"text": "what to remind", "when": "date and time"}, ["text", "when"]),
         fn("find_clinic", "Find clinics and hospitals near a town.", {"town": "town name"})]

SYSTEMS = [
    "You are Kaya, a friendly assistant for people in Namibia. Answer in the user's language, keep replies short "
    "and practical. Use your tools whenever a question depends on current or local information you cannot be sure of.",
    "You are the virtual assistant of a community radio station in northern Namibia. Be warm and clear. "
    "Use the search tool for news, prices and facts about places; never invent phone numbers or opening hours.",
    "You are a helpful assistant that chats on WhatsApp. Messages should be brief and easy to read on a phone. "
    "If the user shares a link, read it before answering. If they ask about their stored data, use the data tools.",
    "You are StudyBuddy, an assistant for learners and students. Explain things step by step. "
    "For admission dates, fees or anything that changes every year, search first.",
    "You are an AI assistant. Tools are available; call one when it is needed to answer correctly, "
    "otherwise answer directly.",
    "You are Tuli, an assistant for small business owners in Namibia. Give concrete, affordable advice. "
    "Look up current prices, regulations and contacts instead of guessing. Questions about this service itself "
    "(cost, privacy, who runs it) are answered from the service information tool.",
    "You help people in Windhoek with everyday questions: transport, services, health, shopping. "
    "Be kind and to the point. Check facts that may be outdated with a web search.",
    "You are a general-purpose assistant. Respond in the language of the user (English or Afrikaans). "
    "Respect privacy: when users want to see or delete their data, use the matching tool.",
]

TOWNS = ["Windhoek", "Oshakati", "Ondangwa", "Ongwediva", "Rundu", "Katima Mulilo", "Walvis Bay", "Swakopmund",
         "Keetmanshoop", "Otjiwarongo", "Gobabis", "Outapi", "Eenhana", "Tsumeb", "Grootfontein", "Lüderitz", "Opuwo", "Mariental"]
LOCAL = ["What are the opening hours of the home affairs office in {t}?", "Which pharmacies in {t} are open on Sunday?",
         "How much does a room at a guesthouse in {t} cost per night?", "Is there a driving school in {t}? What do lessons cost?",
         "Who is the mayor of {t} at the moment?", "Are there any vacancies for teachers in {t} right now?",
         "What is the phone number of the police station in {t}?", "Where can I buy solar panels in {t}?",
         "Is the water in {t} safe to drink this week?", "When is the next job fair in {t}?",
         "What does a taxi ride inside {t} cost now?", "Which banks in {t} offer student accounts?",
         "Hoe laat maak die biblioteek in {t} oop?", "Watter klinieke in {t} is oop oor die naweek?",
         "Is there load shedding planned for {t} this week?", "What events are happening in {t} this weekend?"]
NATIONAL = ["What is the current inflation rate in Namibia?", "Who won the last Namibian presidential election?",
            "What is the new fee for a Namibian ID card?", "Has the price of diesel gone up this month?",
            "What are the latest COVID or flu rules for schools?", "When do the next school holidays start?",
            "What is the current exchange rate between the Namibian dollar and the euro?",
            "Is there a drought emergency declared in the Kunene region?", "What is the pension amount for elderly people now?",
            "Which mobile network has the cheapest data bundles at the moment?", "What did parliament decide about the new land bill?",
            "Hoeveel kos 'n liter petrol tans in Namibië?", "How do I register a company with BIPA these days?"]
SITES = ["https://www.nampa.org/news", "https://www.observer24.com.na/", "https://www.mict.gov.na/",
         "https://www.mtc.com.na/offers", "https://www.nust.na/admissions", "https://www.cityofwindhoek.org.na/",
         "https://www.nfa.org.na/", "https://www.newera.com.na/", "https://www.allgemeine-zeitung.com.na/",
         "https://www.bon.com.na/", "https://www.nsfaf.na/news", "https://www.namcor.com.na/"]
URL_ASK = ["What does this page say? {u}", "Summarise {u} in three bullet points.", "Can you read {u} and tell me the main news?",
           "My friend sent me this: {u} — what is it about?", "Please check {u} and tell me if there is anything about deadlines.",
           "Gee my 'n kort opsomming van {u}"]
DELETE = ["Remove all the information you saved about me.", "I want you to forget me completely.", "Clear my data please.",
          "Delete my history and everything you know.", "Wis asseblief al my data uit."]
SHOW = ["What have you saved about me?", "Which information do you keep about me?", "Show me my profile.",
        "Do you know my name and where I live?", "Wat het jy van my gestoor?"]
USAGE = ["How much have I used this month?", "Did I reach my limit?", "How many messages do I have left?",
         "Show me my usage.", "Hoeveel het ek hierdie maand gebruik?"]
ABOUT_ASK = ["Does this service cost money?", "Who runs this assistant?", "Do you sell my data?",
             "How long do you keep my chats?", "Which languages can you speak?", "Is this an official government service?"]
OTHER_ASK = ["Remind me tomorrow at 7 to call the clinic.", "What will the weather be in {t} on Saturday?",
             "Convert 500 Namibian dollars to US dollars.", "Book me a taxi from the mall to {t} hospital at 6 pm.",
             "The power is out in {t} since last night.", "When is the next bus from {t} to Windhoek?",
             "Find a clinic near {t}."]
FOLLOW = [("I'm thinking of studying nursing.", "That's a great career! Do you want to know about where to study, costs or requirements?",
           "Where can I study it in {t} and what are the fees this year?"),
          ("My phone was stolen.", "I'm sorry, that's stressful. Do you want help with blocking your SIM or reporting it?",
           "Reporting it. Which police station in {t} should I go to and what are their hours?"),
          ("We want to have a wedding next year.", "Congratulations! Shall we plan the budget, the venue or the guest list?",
           "The venue. What venues are there in {t} and what do they charge?"),
          ("I want to get fit.", "Good decision! Do you prefer exercising at home or in a gym?",
           "At home. Give me a simple 20-minute routine.")]


def build_items(n: int, rng: random.Random) -> list[dict]:
    dolly = [r for r in load_dataset("databricks/databricks-dolly-15k", split="train")
             if not r.get("context") and len(r["instruction"]) < 400]
    rng.shuffle(dolly)
    items = []
    for k in range(n):
        kind = rng.choices(["local", "national", "url", "delete", "show", "usage", "about", "other", "follow", "general"],
                           weights=[14, 8, 10, 4, 4, 4, 5, 8, 6, 37])[0]
        t = rng.choice(TOWNS)
        if kind == "local":
            turns = [rng.choice(LOCAL).format(t=t)]
        elif kind == "national":
            turns = [rng.choice(NATIONAL)]
        elif kind == "url":
            turns = [rng.choice(URL_ASK).format(u=rng.choice(SITES))]
        elif kind in ("delete", "show", "usage", "about"):
            turns = [rng.choice({"delete": DELETE, "show": SHOW, "usage": USAGE, "about": ABOUT_ASK}[kind])]
        elif kind == "other":
            turns = [rng.choice(OTHER_ASK).format(t=t)]
        elif kind == "follow":
            turns = [x.format(t=t) for x in rng.choice(FOLLOW)]
        else:
            turns = [dolly[k % len(dolly)]["instruction"].strip()]
        tools = [rng.choice(SEARCH)] + ([rng.choice(READ)] if rng.random() < 0.8 else [])
        tools += rng.sample(PERSONAL, rng.choice([0, 2, 3, 4])) + rng.sample(ABOUT, rng.choice([0, 1]))
        tools += rng.sample(OTHER, rng.choice([0, 1, 2, 3]))
        rng.shuffle(tools)
        msgs = [{"role": "system", "content": rng.choice(SYSTEMS)}]
        msgs += [{"role": "user" if j % 2 == 0 else "assistant", "content": x} for j, x in enumerate(turns)]
        items.append({"messages": msgs, "tools": tools, "source": f"toolreplay:{kind}"})
    return items


def generate(items: list[dict], model_path: str, batch: int, max_new: int) -> None:
    import torch
    from train_lora import load_model
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    tpl = TEMPLATE.read_text()
    for it in items:
        it["prompt"] = tok.apply_chat_template(it["messages"], tools=it["tools"], tokenize=False,
                                               add_generation_prompt=True, chat_template=tpl)
    stops = [tok.convert_tokens_to_ids(t) for t in ("<turn|>", "<|tool_response>")]
    model = load_model(model_path, four_bit=False).eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i]["prompt"]))
    for b in range(0, len(order), batch):
        idx = order[b: b + batch]
        enc = tok([items[i]["prompt"] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=stops,
                               pad_token_id=tok.pad_token_id)
        for i, ids in zip(idx, g[:, enc["input_ids"].shape[1]:].tolist()):
            ids = [x for x in ids if x != tok.pad_token_id]
            cut = next((j for j, x in enumerate(ids) if x in stops), None)
            items[i]["finish"] = "length" if cut is None else "stop"
            items[i]["completion"] = tok.decode(ids if cut is None else ids[: cut + 1], skip_special_tokens=False)
        print(f"answered {min(b + batch, len(order))}/{len(order)}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/models/gemma-4-12b-it-bf16")
    ap.add_argument("-n", type=int, default=2400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--max-tokens", type=int, default=1400, help="prompt + completion")
    ap.add_argument("--keep-length", action="store_true", help="keep completions that hit --max-new (tiny tests only)")
    ap.add_argument("--seed", type=int, default=53)
    args = ap.parse_args(argv)
    items = build_items(args.n, random.Random(args.seed))
    print(f"prompts: {len(items)}", flush=True)
    generate(items, args.model, args.batch, args.max_new)
    tok = AutoTokenizer.from_pretrained(args.model)
    kept, why, calls = [], {"hit max_new": 0, "too long": 0, "empty": 0}, {}
    for it in items:
        if it["finish"] == "length" and not args.keep_length:
            why["hit max_new"] += 1
            continue
        if not it["completion"].strip():
            why["empty"] += 1
            continue
        if len(tok(it["prompt"] + it["completion"], add_special_tokens=False).input_ids) > args.max_tokens:
            why["too long"] += 1
            continue
        call = "<|tool_call>" in it["completion"]
        k = it["source"].split(":")[1]
        calls.setdefault(k, [0, 0])
        calls[k][0] += call
        calls[k][1] += 1
        kept.append({"prompt": it["prompt"], "completion": it["completion"], "source": it["source"], "tool_call": call})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    print(json.dumps({"kept": len(kept), "dropped": why, "tool_calls_by_kind": {k: f"{a}/{n}" for k, (a, n) in calls.items()}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
