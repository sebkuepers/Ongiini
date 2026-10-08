#!/usr/bin/env python3
"""Does the Viljoen dictionary help at inference time? (paper 3, experiment G2)

Gemma 4 12B, base and with an adapter (T2), translate the Ndonga dev split with
and without dictionary hints: the English words of the source are looked up in
the Viljoen/Amakali/Namuandi index (scripts/viljoen_lexicon.py; same matching as
experiment G1, scripts/exp_dictionary_prompting.glossary) and listed above the
sentence in the benchmark prompt. Generation exactly as eval_lora_generate.py.
Writes per-condition outputs and chrF++ with a paired bootstrap vs no-glossary.

    python3 scripts/exp_viljoen_glossary.py --model /models/gemma-4-12b-it-bf16 \\
        --adapter data/private/lora/T2_12b --out data/private/experiments/glossary_viljoen
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_prompts import DIALECT_NAMES, PAPER_ZEROSHOT_ID, render  # noqa: E402
from exp_dictionary_prompting import glossary  # noqa: E402
from train_lora import load_model  # noqa: E402

LEX = Path("data/private/corpora/dictionaries/viljoen-1984/lexicon_en_ow.json")


def with_glossary(dialect: str, english: str, gloss) -> str:
    base = render(PAPER_ZEROSHOT_ID, dialect, english)
    if not gloss:
        return base
    lines = "\n".join(f"- {en}: {ow}" + (f" ({note})" if note else "") for en, ow, note in gloss)
    block = (f"Dictionary entries from an English–{DIALECT_NAMES[dialect]} dictionary that may help. "
             "Use one only if its meaning fits this sentence, and inflect it as the sentence needs:\n"
             f"{lines}\n\n")
    return base.replace("English: ", block + "English: ", 1)


def generate(model, tok, prompts: list[str], batch: int) -> list[str]:
    out = [""] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for b in range(0, len(order), batch):
        idx = order[b: b + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": prompts[i]}], tokenize=False,
                                         add_generation_prompt=True) for i in idx]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        max_new = min(400, 20 + 3 * max(len(prompts[i].split("English: ")[-1].split()) for i in idx))
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        for i, t in zip(idx, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            out[i] = t.strip().split("\n")[0].strip()
    return out


def chrf(h, r):
    return sacrebleu.metrics.CHRF(word_order=2).corpus_score(h, [r]).score


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    lex = {k: [tuple(x) for x in v] for k, v in json.loads(LEX.read_text()).items()}
    ev = {json.loads(l)["id"]: json.loads(l) for l in open("data/private/export/data/eval_set.jsonl")}
    ids = json.load(open("data/private/export/data/referenced_ids.json"))
    src = {int(r["id"]): r["english"].strip() for r in csv.DictReader(open("data/oshiwambo_eval_v3.tsv"), delimiter="\t")}
    dev = [i for i in ids if ev[i].get("oshindonga_reference") and not ev[i]["in_blind_split"]]
    if args.limit:
        dev = dev[: args.limit]
    refs = [ev[i]["oshindonga_reference"] for i in dev]
    gloss = {i: glossary(src[i], lex) for i in dev}

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = load_model(args.model, four_bit=False)
    labels = [("base", None)]
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        labels.append(("adapter", args.adapter))
    model.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hyps = {}
    for name, adapter in labels:
        for cond in ("plain", "glossary"):
            prompts = [render(PAPER_ZEROSHOT_ID, "oshindonga", src[i]) if cond == "plain"
                       else with_glossary("oshindonga", src[i], gloss[i]) for i in dev]
            if adapter:
                h = generate(model, tok, prompts, args.batch)
            else:
                with model.disable_adapter() if args.adapter else torch.no_grad():
                    h = generate(model, tok, prompts, args.batch)
            hyps[(name, cond)] = h
            with (out / f"{name}_{cond}.jsonl").open("w") as f:
                for i, p, t in zip(dev, prompts, h):
                    f.write(json.dumps({"id": i, "prompt": p, "translation": t}, ensure_ascii=False) + "\n")
            print(f"{name} {cond}: chrF++ {chrf(h, refs):.1f}", flush=True)

    report = {"items": len(dev), "with_entries": sum(1 for i in dev if gloss[i]),
              "mean_entries": round(sum(len(gloss[i]) for i in dev) / len(dev), 2)}
    rng = random.Random(1)
    for name, _ in labels:
        a, b = hyps[(name, "glossary")], hyps[(name, "plain")]
        delta = chrf(a, refs) - chrf(b, refs)
        ds = []
        for _ in range(1000):
            s = [rng.randrange(len(dev)) for _ in dev]
            ds.append(chrf([a[k] for k in s], [refs[k] for k in s]) - chrf([b[k] for k in s], [refs[k] for k in s]))
        ds.sort()
        report[name] = {"plain": round(chrf(b, refs), 2), "glossary": round(chrf(a, refs), 2),
                        "delta": round(delta, 2), "ci95": [round(ds[25], 2), round(ds[974], 2)]}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
