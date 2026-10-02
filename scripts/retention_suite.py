#!/usr/bin/env python3
"""Retention suite: does an Oshiwambo LoRA adapter cost general abilities? (paper 3)

Runs the same automatic checks on the base model and on base + adapter:

1. English perplexity on 100 WikiText-2 test paragraphs (no generation).
2. Instruction following: 8 questions × 5 checkable format rules (IFEval-style).
3. Reasoning: 100 GSM8K test problems, exact final number.
4. Other languages: English→Afrikaans and →German on 100 FLORES-101 devtest
   sentences, chrF.
5. Tool use: 30 requests that need one of six tools, answered as JSON;
   valid JSON and the right tool.
6. English answers to 40 Ongiini-style questions, saved for a blind pairwise
   comparison by an LLM judge (scripts/retention_judge.py).

Test data: data/private/retention/ (downloaded once; see the paper-3 notes).
Results: data/private/experiments/retention/<label>.json (+ _answers.json).

    python3 scripts/retention_suite.py --model /models/gemma-4-12b-it-bf16 --label gemma-4-12b-base
    python3 scripts/retention_suite.py --model ... --adapter data/private/lora/A_parallel_ndo_12b_r16 --label gemma-4-12b-A
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lora import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/private/retention"
OUT = ROOT / "data/private/experiments/retention"

QUESTIONS = [
    "How can I renew my national ID card?",
    "What are early signs of dehydration in children?",
    "Give advice for planting mahangu at the start of the rainy season.",
    "How do I write a short cover letter for a job as a shop assistant?",
    "What should I do if my phone is stolen?",
    "Explain what interest on a loan means.",
    "How can I save water at home during a drought?",
    "What is a good way to prepare for a job interview?",
]
RULES = {
    "three_bullets": ("Answer in exactly 3 bullet points, each line starting with '- ', and nothing else.",
                      lambda t: sum(l.strip().startswith("- ") for l in t.splitlines()) == 3
                      and all(l.strip().startswith("- ") for l in t.splitlines() if l.strip())),
    "all_caps": ("Write your whole answer in capital letters only.",
                 lambda t: any(c.isalpha() for c in t) and t == t.upper()),
    "under_25_words": ("Answer in fewer than 25 words.", lambda t: 0 < len(t.split()) < 25),
    "json": ('Answer only with a JSON object with the keys "answer" and "confidence" (a number from 0 to 1).',
             lambda t: _json_ok(t)),
    "end_phrase": ("End your answer with the exact sentence: Hope this helps.",
                   lambda t: t.rstrip().endswith("Hope this helps.")),
}
TOOLS = {
    "get_weather": "current weather for a town (args: town)",
    "search_web": "search the internet (args: query)",
    "set_reminder": "set a reminder (args: time, text)",
    "convert_currency": "convert money (args: amount, from, to)",
    "find_clinic": "nearest clinic or hospital (args: town)",
    "translate": "translate text (args: text, target_language)",
}
TOOL_CASES = [
    ("Will it rain in Oshakati tomorrow?", "get_weather"), ("How hot is it in Windhoek right now?", "get_weather"),
    ("Is it windy in Swakopmund today?", "get_weather"), ("What's the weather like in Rundu?", "get_weather"),
    ("Who won the Namibian presidential election in 2024?", "search_web"),
    ("What are the latest NAMCOR fuel prices?", "search_web"),
    ("Find news about the Katima Mulilo bridge.", "search_web"), ("When does the UNAM semester start?", "search_web"),
    ("Remind me at 7pm to call my mother.", "set_reminder"), ("Set a reminder for 8 tomorrow morning: pay rent.", "set_reminder"),
    ("Remind me on Friday at noon to collect my ID.", "set_reminder"), ("Please remind me in an hour to take my pills.", "set_reminder"),
    ("How much is 500 Namibian dollars in euros?", "convert_currency"), ("Convert 100 US dollars to NAD.", "convert_currency"),
    ("What is 2000 rand in Botswana pula?", "convert_currency"), ("How many euros do I get for 1500 N$?", "convert_currency"),
    ("Where is the nearest clinic in Ondangwa?", "find_clinic"), ("I need a hospital in Eenhana.", "find_clinic"),
    ("Which clinic is close to Okahao?", "find_clinic"), ("Find a doctor near Opuwo.", "find_clinic"),
    ("Translate 'good morning' into Afrikaans.", "translate"), ("How do you say 'thank you' in Oshindonga?", "translate"),
    ("Translate this into German: The shop opens at nine.", "translate"), ("Say 'where is the bus station' in Portuguese.", "translate"),
    ("Is there a storm warning for Walvis Bay?", "get_weather"), ("Look up the opening hours of the Home Affairs office in Ongwediva.", "search_web"),
    ("Remind me tonight at 9 to lock the gate.", "set_reminder"), ("Change 300 euros into Namibian dollars.", "convert_currency"),
    ("Nearest hospital to Outapi please.", "find_clinic"), ("Translate 'I am hungry' into Oshikwanyama.", "translate"),
]
CHAT = QUESTIONS + [
    "My child has had diarrhoea for two days. What should I do?",
    "Write a polite message to my landlord asking for two more days to pay rent.",
    "How do I open a bank account in Namibia?",
    "What documents do I need to register a birth?",
    "Explain climate change to a ten-year-old.",
    "Give me a simple budget plan for a monthly income of N$4,000.",
    "How can I start a small vegetable garden?",
    "What is the difference between a virus and bacteria?",
    "How do I prepare for the grade 12 exams?",
    "Suggest three ways to make money from a small poultry farm.",
    "How should I store maize meal so it does not go bad?",
    "What should I do after a car accident?",
    "Write a short birthday message for my grandmother.",
    "How can I tell if a job offer on Facebook is a scam?",
    "What are the symptoms of malaria?",
    "How do I apply for a learner's licence?",
    "Explain how mobile money works.",
    "Give tips for staying safe when travelling by taxi at night.",
    "How can I improve my English writing?",
    "What are good foods for a pregnant woman?",
    "How do I treat a minor burn at home?",
    "What is HIV PrEP and who should use it?",
    "How do I write a complaint to my municipality about a broken water pipe?",
    "Explain what inflation is in simple words.",
    "How can farmers protect cattle from foot-and-mouth disease?",
    "What should I pack for a trip to Etosha?",
    "How do I calculate the area of my plot of land?",
    "Give me a short motivational message for a student who failed a test.",
    "What can I cook with beans, onions and rice?",
    "How do solar panels work?",
    "What is the best way to learn to type faster?",
    "How do I report domestic violence in Namibia?",
]


def _json_ok(t: str) -> bool:
    m = re.search(r"\{.*\}", t, re.S)
    try:
        o = json.loads(m.group(0)) if m else None
        return isinstance(o, dict) and "answer" in o and 0 <= float(o.get("confidence", -1)) <= 1
    except (ValueError, TypeError):
        return False


def generate(model, tok, prompts: list[str], max_new: int, batch: int = 16) -> list[str]:
    out = [""] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for b in range(0, len(order), batch):
        idx = order[b: b + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": prompts[i]}], tokenize=False,
                                         add_generation_prompt=True) for i in idx]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        for i, t in zip(idx, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            out[i] = t.strip()
    return out


def perplexity(model, tok, paras: list[str]) -> float:
    nll, n = 0.0, 0
    for p in paras:
        ids = tok(p, return_tensors="pt", truncation=True, max_length=512).input_ids.to(model.device)
        with torch.no_grad():
            loss = model(ids, labels=ids).loss.item()
        nll += loss * (ids.shape[1] - 1)
        n += ids.shape[1] - 1
    return round(math.exp(nll / n), 3)


def main(argv=None) -> int:
    import sacrebleu
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--label", required=True)
    args = ap.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = load_model(args.model, four_bit=False)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    res: dict = {"label": args.label, "adapter": args.adapter or None}

    res["en_perplexity"] = perplexity(model, tok, json.loads((DATA / "wikitext_100.json").read_text()))
    print("perplexity", res["en_perplexity"], flush=True)

    cases = [(q, name, rule) for q in QUESTIONS for name, rule in RULES.items()]
    ans = generate(model, tok, [f"{q}\n\n{r[0]}" for q, _, r in cases], 200)
    by_rule = {name: [] for name in RULES}
    for (_, name, rule), a in zip(cases, ans):
        by_rule[name].append(bool(rule[1](a)))
    res["instructions"] = {k: round(100 * sum(v) / len(v), 1) for k, v in by_rule.items()}
    res["instructions_all"] = round(100 * sum(sum(v) for v in by_rule.values()) / len(cases), 1)
    print("instructions", res["instructions_all"], flush=True)

    gsm = json.loads((DATA / "gsm8k_100.json").read_text())
    ans = generate(model, tok, [g["q"] + "\n\nSolve step by step, then end with a line 'Answer: <number>'." for g in gsm], 400)
    def final(t):
        m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", t) or re.findall(r"(-?\d[\d,]*\.?\d*)", t)
        return m[-1].replace(",", "").rstrip(".") if m else ""
    res["gsm8k"] = round(100 * sum(final(a) == g["a"] or (final(a) and g["a"] and float(final(a) or 0) == float(g["a"]))
                                   for a, g in zip(ans, gsm)) / len(gsm), 1)
    print("gsm8k", res["gsm8k"], flush=True)

    fl = json.loads((DATA / "flores_en_af_de_100.json").read_text())
    chrf = sacrebleu.metrics.CHRF()
    for code, lang in (("af", "Afrikaans"), ("de", "German")):
        hyp = generate(model, tok, [f"Translate into {lang}. Only output the translation.\n\nEnglish: {r['en']}" for r in fl], 200)
        hyp = [h.splitlines()[0] if h else "" for h in hyp]
        res[f"flores_en_{code}_chrf"] = round(chrf.corpus_score(hyp, [[r[code] for r in fl]]).score, 1)
    print("flores", res["flores_en_af_chrf"], res["flores_en_de_chrf"], flush=True)

    tool_desc = "\n".join(f"- {k}: {v}" for k, v in TOOLS.items())
    prompts = [f"You can call one of these tools:\n{tool_desc}\n\nUser request: {q}\n\n"
               'Reply only with JSON: {"tool": "<tool name>", "arguments": {...}}' for q, _ in TOOL_CASES]
    ans = generate(model, tok, prompts, 120)
    valid = right = 0
    for a, (_, want) in zip(ans, TOOL_CASES):
        m = re.search(r"\{.*\}", a, re.S)
        try:
            o = json.loads(m.group(0)) if m else None
        except ValueError:
            o = None
        if isinstance(o, dict) and isinstance(o.get("arguments", {}), dict):
            valid += 1
            right += o.get("tool") == want
    res["tools_valid_json"] = round(100 * valid / len(TOOL_CASES), 1)
    res["tools_right_tool"] = round(100 * right / len(TOOL_CASES), 1)
    print("tools", res["tools_valid_json"], res["tools_right_tool"], flush=True)

    chat = generate(model, tok, CHAT, 400)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.label}_answers.json").write_text(json.dumps(dict(zip(CHAT, chat)), ensure_ascii=False, indent=1))
    (OUT / f"{args.label}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
