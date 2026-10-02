#!/usr/bin/env python3
"""Zero-shot-transfer baselines with dedicated MT models (NLLB-200, MADLAD-400).

Neither model lists Oshiwambo as a target. As the concept paper (§ baselines,
"Dedicated machine-translation systems") specifies, we translate into the
closest-related supported language — Tswana — and score the output against
both Oshindonga and Oshikwanyama references. This measures whether genuine
Oshiwambo vocabulary or morphology leaks through; it is not an Oshiwambo
translation score.

  nllb     facebook/nllb-200-*      src eng_Latn → tgt tsn_Latn (forced BOS)
  madlad   google/madlad400-*-mt    input "<2tn> " + English

Whole items are translated as one input, like every other system. Beam
search (4 beams), no sampling — deterministic. Runs locally (Apple MPS or
CPU). Resumable: items already in the output files are skipped. Writes the
same submission-schema JSONL as the other baselines, one line per item and
dialect (the two dialect files carry the same text).

    ~/.venvs/ongiini-eval/bin/python scripts/run_mt_baseline.py --system nllb \\
        --model facebook/nllb-200-3.3B --label nllb-200-3.3b
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data/oshiwambo_eval_v3.tsv"           # public English sources
OUT_DIR = ROOT / "data/private/export/data/baselines"
DIALECTS = ("oshindonga", "oshikwanyama")
# Default target: Tswana (closest supported relative, paper §4.1). MADLAD also
# has a Kwanyama tag <2kj> (a real vocabulary token; verified 2026-10-01) —
# run it with --tgt kj; it is scored as native for Oshikwanyama and as
# transfer for Oshindonga, which MADLAD lacks.
DEFAULT_TGT = {"nllb": "tsn_Latn", "madlad": "tn", "marian": "ng"}
# marian: single-direction English→Ndonga models (Helsinki-NLP opus-mt-en-ng
# and Meyabase's fine-tunes of it) — no target tag; scored as native for
# Oshindonga and as transfer for Oshikwanyama.


def restore_untied_madlad(model, repo: str) -> None:
    """MADLAD-400 keeps separate input embeddings and output head
    (tie_word_embeddings=False). transformers 5 ties them on load anyway,
    and the model then emits digit / CJK garbage. Put both back from the
    checkpoint. (Verified: "<2de> I love pizza!" -> "Ich liebe Pizza!".)"""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    f = safe_open(hf_hub_download(repo, "model.safetensors"), "pt")
    emb, head = f.get_tensor("decoder.embed_tokens.weight"), f.get_tensor("lm_head.weight")
    model.shared = torch.nn.Embedding.from_pretrained(emb, freeze=True)
    model.encoder.embed_tokens = model.shared
    model.decoder.embed_tokens = model.shared
    model.lm_head = torch.nn.Linear(head.shape[1], head.shape[0], bias=False)
    model.lm_head.weight.data = head


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=("nllb", "madlad", "marian"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tgt", default="", help="Target code: NLLB e.g. tsn_Latn, MADLAD e.g. tn / kj")
    args = ap.parse_args(argv)

    with SOURCES.open() as f:
        items = [(int(r["id"]), r["english"].strip()) for r in csv.DictReader(f, delimiter="\t")]
    outs = {d: OUT_DIR / f"{args.label}_{d}.jsonl" for d in DIALECTS}
    done = set.intersection(*[{json.loads(l)["id"] for l in p.read_text().splitlines() if l}
                              if p.exists() else set() for p in outs.values()])
    todo = [it for it in items if it[0] not in done]
    if args.limit:
        todo = todo[: args.limit]
    todo.sort(key=lambda it: len(it[1]))                   # similar lengths per batch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"{args.label}: {len(todo)} items to translate on {device}", file=sys.stderr)
    if not todo:
        return 0

    tgt = args.tgt or DEFAULT_TGT[args.system]
    template_id = f"mt-zero-shot-{args.system}-{tgt}"
    if args.system == "nllb":
        tok = AutoTokenizer.from_pretrained(args.model, src_lang="eng_Latn")
        prefix, gen_extra = "", {"forced_bos_token_id": tok.convert_tokens_to_ids(tgt)}
    elif args.system == "marian":
        tok = AutoTokenizer.from_pretrained(args.model)
        prefix, gen_extra = "", {}
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        prefix, gen_extra = f"<2{tgt}> ", {}
        if len(tok.tokenize(f"<2{tgt}>")) > 2:          # ['▁', '<2xx>'] when the tag exists
            sys.exit(f"MADLAD has no language tag <2{tgt}>")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=torch.float32)
    if args.system == "madlad":
        restore_untied_madlad(model, args.model)
    model = model.to(device).eval()

    t0 = time.time()
    for b in range(0, len(todo), args.batch):
        batch = todo[b: b + args.batch]
        enc = tok([prefix + en for _, en in batch], return_tensors="pt", padding=True).to(device)
        max_new = min(512, int(enc["input_ids"].shape[1] * 2.5) + 20)
        with torch.no_grad():
            gen = model.generate(**enc, num_beams=args.beams, do_sample=False, max_new_tokens=max_new, **gen_extra)
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for (iid, _), text in zip(batch, texts):
            for d in DIALECTS:
                rec = {"id": iid, "dialect": d, "model_id": args.model,
                       "prompt_template_id": template_id, "translation": text.strip(),
                       "timestamp": stamp, "seed": None, "raw_response": text}
                with outs[d].open("a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n = b + len(batch)
        print(f"  {n}/{len(todo)}  {(time.time() - t0) / n:.1f}s/item", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
