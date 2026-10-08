#!/usr/bin/env python3
"""Tool-decision suite: does the language adapter still call Ongiini AI's tools like Gemma 4?

The compatibility suite (compat_suite.py) has 14 tool scenarios; T2 missed one
(fetch_url, 2026-10-08) and one item cannot tell noise from a regression. This suite
has ~100 synthetic scenarios (production system prompt + production tool list from
data/private/compat/ongiini_context.json), each answered greedily and twice sampled
(temperature 0.7), so fragile decisions show up. Scoring: per expected tool, the
share of answers whose first-turn decision is acceptable (some scenarios accept
more than one tool, e.g. fetch_url or fetch_urls), base vs candidate with a paired
sign test over scenarios.

    python3 scripts/tool_suite.py generate --model gemma-4-12b --label base
    python3 scripts/tool_suite.py generate --model t2 --label T2
    python3 scripts/tool_suite.py score --labels base T2
Outputs in data/private/compat/tools_*.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from math import comb
from pathlib import Path

OUT = Path("data/private/compat")
SERVER = "http://localhost:8200/v1"
SAMPLES = [(0.0, 0), (0.7, 1), (0.7, 2)]  # (temperature, seed)

WEB = ("web_search",)
FETCH = ("fetch_url", "fetch_urls")
# (turns, acceptable tools; () = answer without a tool). Synthetic, written for this suite.
S: list[tuple[list[str], tuple[str, ...]]] = [
    # current or local facts → web_search
    (["What is the price of petrol in Windhoek this month?"], WEB),
    (["Who is the current Minister of Health in Namibia?"], WEB),
    (["When does the next NSFAF application window open?"], WEB),
    (["Is the B1 road between Okahandja and Otjiwarongo open today?"], WEB),
    (["What time does the Oshakati post office close on Saturdays?"], WEB),
    (["Which banks in Ongwediva are open on public holidays?"], WEB),
    (["What will the weather be like in Rundu tomorrow?"], WEB),
    (["Did Namibia win their last football match?"], WEB),
    (["What are the entry requirements for nursing at UNAM?"], WEB),
    (["How much does a taxi from Katutura to the city centre cost now?"], WEB),
    (["Are there any job vacancies at NamPower at the moment?"], WEB),
    (["What is the current prime interest rate in Namibia?"], WEB),
    (["Where can I renew my driving licence in Swakopmund?"], WEB),
    (["What's the latest on the drought relief programme?"], WEB),
    (["Which schools in Windhoek still have space for grade 1 next year?"], WEB),
    (["How much is a bag of maize meal at Shoprite these days?"], WEB),
    (["Give me the phone number of Katutura State Hospital."], WEB),
    (["What documents do I need to apply for a Namibian passport?"], WEB),
    (["Is there a public holiday in Namibia next week?"], WEB),
    (["Hoeveel kos 'n brood by Spar in Windhoek?"], WEB),
    (["Ek soek werk in Walvisbaai. Watter vakatures is daar tans?"], WEB),
    (["What are the visa requirements for Namibians travelling to Germany?"], WEB),
    (["My cousin says the NSFAF loan is now a grant. Is that true?"], WEB),
    (["What is the minimum wage for domestic workers in Namibia now?"], WEB),
    (["I want to sell vegetables at the Oshakati open market. How do I get a stall?",
      "Good plan! Do you want to know about the permit, the costs or the best place?",
      "The permit. Who do I contact and how much is it?"], WEB),
    (["Can you help me plan a trip to Etosha?",
      "Of course! When are you going and how many people?",
      "Next month, two people. What are the park entry fees now?"], WEB),
    # a given page → fetch
    (["Can you summarise this page for me: https://www.namibian.com.na/"], FETCH),
    (["What does this article say? https://www.nbc.na/news"], FETCH),
    (["Please read https://www.unam.edu.na/admissions and tell me the deadline."], FETCH),
    (["Summarise this for my grandmother in simple words: https://www.gov.na/"], FETCH),
    (["Compare these two pages for me: https://www.fnb.com.na/ and https://www.bankwindhoek.com.na/"], FETCH),
    (["What's on this page? https://www.economist.com.na/"], FETCH),
    (["Kan jy hierdie bladsy vir my opsom? https://www.republikein.com.na/"], FETCH),
    (["I got this link from a friend, what is it about? https://www.namibiansun.com/"], FETCH),
    # personal data tools
    (["Please delete everything you know about me."], ("delete_my_data",)),
    (["Forget all my data, I want a fresh start."], ("delete_my_data",)),
    (["Erase my chat history and anything you saved about me."], ("delete_my_data",)),
    (["Vee asseblief alles uit wat jy van my weet."], ("delete_my_data",)),
    (["I don't want you to keep my information anymore. Remove it all."], ("delete_my_data",)),
    (["What do you remember about me?"], ("whats_in_my_memory",)),
    (["Which facts have you stored about me?"], ("whats_in_my_memory",)),
    (["Do you still know where I live?"], ("whats_in_my_memory",)),
    (["Show me my saved information."], ("whats_in_my_memory",)),
    (["Wat onthou jy van my?"], ("whats_in_my_memory",)),
    (["How many tokens have I used so far?"], ("my_token_usage",)),
    (["How much of my monthly limit is left?"], ("my_token_usage",)),
    (["Am I close to my usage limit this month?"], ("my_token_usage",)),
    (["Show me my usage for October."], ("my_token_usage",)),
    (["Hoeveel tokens het ek al gebruik?"], ("my_token_usage",)),
    # questions about Ongiini AI itself
    (["Is Ongiini free? Who made you?"], ("lookup_ongiini_docs",)),
    (["Which languages do you speak?"], ("lookup_ongiini_docs",)),
    (["How long do you keep my messages?"], ("lookup_ongiini_docs",)),
    (["Is my data shared with anyone?"], ("lookup_ongiini_docs",)),
    (["What AI model are you running on?"], ("lookup_ongiini_docs",)),
    (["Can I use you in the browser as well?"], ("lookup_ongiini_docs",)),
    (["Why do you have a monthly token limit?"], ("lookup_ongiini_docs",)),
    (["Who pays for Ongiini AI?"], ("lookup_ongiini_docs",)),
    # no tool: writing, explaining, maths, conversation, translation
    (["Can you explain photosynthesis simply for my grade 7 child?"], ()),
    (["Write a short WhatsApp message wishing my friend a happy birthday."], ()),
    (["How do I calculate 15% VAT on N$250?"], ()),
    (["Translate 'good morning, how did you sleep?' into Oshindonga."], ()),
    (["Translate 'thank you very much' into Oshikwanyama."], ()),
    (["How do you say 'I love you' in Oshiwambo?"], ()),
    (["Write a cover letter for a cashier job. I have two years of experience at a supermarket."], ()),
    (["Give me five ideas for a school project about water."], ()),
    (["Explain the difference between a noun and a verb."], ()),
    (["I have N$500 for groceries for two weeks. Help me plan."], ()),
    (["Write a short poem about the rain in the north."], ()),
    (["What is 17 times 23?"], ()),
    (["My boss shouted at me today. How should I respond calmly tomorrow?"], ()),
    (["Help me write a polite message to my landlord asking for more time to pay rent."], ()),
    (["Explain what a budget is to a 12-year-old."], ()),
    (["Hello! How are you today?"], ()),
    (["Thank you, that was very helpful."], ()),
    (["Tell me a joke."], ()),
    (["Correct the grammar: 'Me and him goes to school yesterday.'"], ()),
    (["Skryf vir my 'n kort boodskap om my ma geluk te wens met haar verjaarsdag."], ()),
    (["Summarise this text for me: The meeting is moved to Friday at 10. Bring your ID and the signed forms. Lunch is provided."], ()),
    (["Explain how compound interest works with an example in N$."], ()),
    (["What is the difference between weather and climate?"], ()),
    (["Help me practise for a job interview. Ask me the first question."], ()),
    (["I feel lonely since I moved to Windhoek.",
      "I'm sorry you feel this way — moving to a new city is hard. Would you like to talk about it?",
      "Yes. I don't know anyone here."], ()),
    (["I failed my maths test and I feel stupid.",
      "I'm sorry — one test does not define you. Do you want some tips for next time?",
      "Yes, give me three tips to study better."], ()),
    (["Can you help me write a CV?",
      "Sure! Tell me your name, the job you want, and your experience.",
      "Maria, I want to work as a receptionist, I worked 1 year at a hotel front desk."], ()),
    (["Ongiini! Nawa tuu?"], ()),
    (["Wa lele po?"], ()),
]


def items() -> list[dict]:
    ctx = json.loads((OUT / "ongiini_context.json").read_text())
    out = []
    for k, (turns, tools) in enumerate(S):
        msgs = [{"role": "system", "content": ctx["system_prompt"]}]
        msgs += [{"role": "user" if j % 2 == 0 else "assistant", "content": t} for j, t in enumerate(turns)]
        out.append({"id": f"tool{k}", "messages": msgs, "ok": list(tools)})
    return out


async def generate(args) -> None:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=SERVER, api_key="none", timeout=900)
    tools = json.loads((OUT / "ongiini_context.json").read_text())["tools"]
    path = OUT / f"tools_{args.label}.jsonl"
    done = set()
    if path.exists():  # resume; failed requests are redone
        kept = [r for r in map(json.loads, open(path)) if not str(r.get("finish", "")).startswith("error")]
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
        done = {(r["id"], r["sample"]) for r in kept}
    todo = [(it, s) for it in items() for s in range(len(SAMPLES)) if (it["id"], s) not in done]
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()

    async def one(it, s):
        temp, seed = SAMPLES[s]
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=args.model, messages=it["messages"], tools=tools, tool_choice="auto",
                    temperature=temp, seed=seed, max_tokens=512,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                m = r.choices[0].message
                rec = {"id": it["id"], "sample": s, "tool_calls": [c.function.name for c in (m.tool_calls or [])],
                       "finish": r.choices[0].finish_reason}
            except Exception as exc:  # noqa: BLE001
                rec = {"id": it["id"], "sample": s, "tool_calls": [], "finish": f"error {exc!r}"[:200]}
        async with lock:
            with path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    await asyncio.gather(*(one(it, s) for it, s in todo))
    errors = sum(str(json.loads(l)["finish"]).startswith("error") for l in open(path))
    print(f"{args.label}: {len(todo)} generated, errors {errors}")
    if errors:
        raise SystemExit(1)


def correct(rec: dict, ok: list[str]) -> bool:
    calls = rec["tool_calls"]
    return bool(set(calls) & set(ok)) if ok else not calls


def score(args) -> None:
    its = {i["id"]: i for i in items()}
    base, cand = args.labels
    res = {}
    for lab in args.labels:
        per = defaultdict(list)
        for r in map(json.loads, open(OUT / f"tools_{lab}.jsonl")):
            per[r["id"]].append(correct(r, its[r["id"]]["ok"]))
        res[lab] = {k: sum(v) / len(v) for k, v in per.items()}
    rep: dict = {}
    for lab in args.labels:
        by = defaultdict(list)
        for k, v in res[lab].items():
            by[(its[k]["ok"] or ["none"])[0]].append(v)
        rep[lab] = {"all": round(100 * sum(res[lab].values()) / len(res[lab]), 1),
                    **{t: round(100 * sum(v) / len(v), 1) for t, v in sorted(by.items())}}
    worse = [k for k in its if res[cand][k] < res[base][k]]
    better = [k for k in its if res[cand][k] > res[base][k]]
    n, w = len(worse) + len(better), len(worse)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(w, n - w) + 1)) / 2 ** n) if n else 1.0
    rep["paired"] = {"cand_worse": w, "cand_better": len(better), "sign_test_p": round(p, 3),
                     "worse_items": {k: [S[int(k[4:])][0][-1][:80], its[k]["ok"], res[base][k], res[cand][k]] for k in worse}}
    (OUT / f"tools_report_{cand}_vs_{base}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--label", required=True)
    g.add_argument("--concurrency", type=int, default=16)
    s = sub.add_parser("score")
    s.add_argument("--labels", nargs=2, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "generate":
        asyncio.run(generate(a))
    else:
        score(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
