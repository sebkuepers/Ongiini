#!/usr/bin/env python3
"""Training mix T4: data quality first (paper 3, goal chrF > 50 without forgetting).

Builds on the T1/T2 recipe (many templates + unique self-distilled replay) and adds:
- quality-selected back-translations: the best --bt-n of 200k by forced-decoding
  loss under adapter A (trained on human pairs only, never saw a back-translation;
  scripts/score_bt_pairs.py), after a length-ratio filter;
- tagged back-translation (Caswell et al. 2019): synthetic rows start with "<bt> "
  so the model can tell them from human translations; inference never uses the tag;
- human translations oversampled: LAC/ASSAR pairs and the Viljoen dictionary example
  sentences (eval-set overlap removed), --human-repeat times;
- dictionary hints (Viljoen index) in --glossary-share of the translation prompts;
- vocabulary tasks from the dictionary in both directions, excluding the words of the
  retention vocabulary test so that test stays meaningful;
- optionally native tool-call replay (--tool-replay), pre-rendered rows that keep
  the adapter calling tools like the base model (T2 lost that, tool_suite.py);
- optionally (T7, 2026-10-19) --consensus: English chat sentences translated by an ensemble of
  adapters, the consensus output as target (scripts/consensus_distill.py)
- optionally (T5, 2026-10-13) READING: --flip-n back-translations reversed, so the
  authentic corpus Oshindonga is the input and the teacher English the answer
  (T4 had only dictionary words in that direction; ow→en did not improve, eval_ow_en.py),
  and Kwanyama (--kua-bt, DeepSeek back-translations of the corpus's Kwanyama sentences;
  T4 had no Kwanyama row at all and Kwanyama reading dropped 4–6 chrF) in both directions.

    python3 scripts/build_sft_t4.py --out data/private/corpus/osheng_v1/sft_T4_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sft_templates import FEWSHOT, NAMES, SYSTEMS, ZERO  # noqa: E402
from eval_prompts import PAPER_ZEROSHOT_ID, render  # noqa: E402
from exp_dictionary_prompting import glossary  # noqa: E402

D = Path("data/private/corpus/osheng_v1")
DICT = Path("data/private/corpora/dictionaries/viljoen-1984")
VOCAB_ASK = ["What is the Oshindonga word for '{en}'?", "How do you say '{en}' in Oshindonga?",
             "Oshindonga word for: {en}", "Give me the Oshindonga for \"{en}\"."]
VOCAB_BACK = ["What does the Oshindonga word '{ow}' mean in English?", "Translate the Oshindonga word '{ow}' into English.",
              "Meaning of '{ow}' (Oshindonga)?"]


# Function words make noisy, space-wasting hints ("and: e / na", "that: kutya"):
# only content words get dictionary hints.
HINT_STOP = set("""and that this the with from have has had was were are been for not but you your they them their
there here what which when where who how why will would could should into onto than then also just only some any
very much many more most such other each every all both our out about over under after before because while""".split())


def grams(t: str, n: int = 6) -> set[str]:
    w = re.findall(r"\w+", t.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


FLIP_ASK = ["Translate into English: {s}", "What does this {L} sentence mean in English?\n\n{s}",
            "Translate the following {L} text into English.\n\n{s}", "{s}\n\nWhat does that mean in English?",
            "Please translate this from {L} into English:\n{s}", "{L}: {s}\nEnglish:"]


def render_flip(rng, ow: str, en: str, dialect: str) -> dict:
    """Reading row: authentic Oshiwambo in, English out (no BT tag — the input side is real text)."""
    msgs = [{"role": "user", "content": rng.choice(FLIP_ASK).format(L=rng.choice(NAMES[dialect]), s=ow)}]
    if rng.random() < 0.2:
        msgs = [{"role": "system", "content": rng.choice(SYSTEMS)}] + msgs
    return {"messages": msgs + [{"role": "assistant", "content": en}]}


def render_pair(rng, src: str, tgt: str, pool, lex, args, tag: str = "", dialect: str = "oshindonga") -> dict:
    """One translation row in a random template; optional dictionary hints and BT tag."""
    L = rng.choice(NAMES[dialect])
    hints = ""
    if dialect == "oshindonga" and rng.random() < args.glossary_share:  # the Viljoen index is Ndonga
        g = [e for e in glossary(src, lex) if e[0] not in HINT_STOP]
        if g:
            hints = ("Dictionary entries that may help (use one only if its meaning fits; inflect as needed):\n"
                     + "\n".join(f"- {en}: {ow}" + (f" ({note})" if note else "") for en, ow, note in g) + "\n\n")
    r = rng.random()
    if r < args.keep_share:
        user = render(PAPER_ZEROSHOT_ID, dialect, src)
        if hints:
            user = user.replace("English: ", hints + "English: ", 1)
        msgs = [{"role": "user", "content": user}]
    elif r < args.keep_share + args.fewshot_share and pool and dialect == "oshindonga":
        shots = "".join(f"English: {s}\n{L}: {t}\n\n" for s, t in rng.sample(pool, rng.choice([1, 2, 3])))
        msgs = [{"role": "user", "content": hints + FEWSHOT.format(L=L, shots=shots, s=src)}]
    else:
        msgs = [{"role": "user", "content": hints + rng.choice(ZERO).format(L=L, s=src)}]
        if rng.random() < 0.2:
            msgs = [{"role": "system", "content": rng.choice(SYSTEMS)}] + msgs
    if tag:
        msgs[-1]["content"] = tag + msgs[-1]["content"]
    return {"messages": msgs + [{"role": "assistant", "content": tgt}]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bt", default=str(D / "bt_ndo_dict.jsonl"))
    ap.add_argument("--bt-scores", default=str(D / "bt_scores_A.jsonl"))
    ap.add_argument("--bt-n", type=int, default=30000)
    ap.add_argument("--human-repeat", type=int, default=3)
    ap.add_argument("--vocab-n", type=int, default=3000)
    ap.add_argument("--keep-share", type=float, default=0.4)
    ap.add_argument("--fewshot-share", type=float, default=0.1)
    ap.add_argument("--glossary-share", type=float, default=0.2)
    ap.add_argument("--replay", nargs="*", default=["data/private/corpus/replay_v1/replay.jsonl",
                                                    "data/private/corpus/replay_v2/replay.jsonl",
                                                    "data/private/corpus/replay_v3/replay.jsonl"])
    ap.add_argument("--tool-replay", nargs="*", default=[],
                    help="pre-rendered native tool-call replay (build_tool_replay.py); rendered format only")
    ap.add_argument("--tool-n", type=int, default=0, help="use only the first N tool-replay rows (0 = all)")
    ap.add_argument("--flip-n", type=int, default=0, help="reversed BT rows (Oshindonga in, English out)")
    ap.add_argument("--kua-bt", default="", help="Kwanyama back-translations (backtranslate_corpus.py --dialect Oshikwanyama)")
    ap.add_argument("--kua-n", type=int, default=0, help="Kwanyama BT pairs; each is used en→kua (tagged) and kua→en")
    ap.add_argument("--kua-min-lid", type=float, default=0.7, help="LID confidence floor for Kwanyama sentences")
    ap.add_argument("--consensus", default="", help="consensus_distill.py select output (T7): ensemble-translated English")
    ap.add_argument("--consensus-n", type=int, default=0, help="best-agreement consensus pairs, en→Ndonga, untagged")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    lex = {k: [tuple(x) for x in v] for k, v in json.loads((DICT / "lexicon_en_ow.json").read_text()).items()}

    ev = [json.loads(l) for l in open("data/private/export/data/eval_set.jsonl")]
    evg = set().union(*(grams(r[k]) for r in ev for k in ("english", "oshindonga_reference", "oshikwanyama_reference") if r.get(k)))

    # human pairs: LAC/ASSAR (benchmark template rows) + dictionary example sentences
    human = []
    for l in open(D / "sft_A_parallel_ndo_train.jsonl"):
        m = json.loads(l)["messages"]
        s = re.search(r"English: (.*)\nOshindonga:\s*$", m[0]["content"], re.S)
        if s:
            human.append((s.group(1).strip(), m[-1]["content"].strip()))
    dict_ex = [(x["en"], x["ow"]) for x in map(json.loads, open(DICT / "examples.jsonl"))
               if not (grams(x["en"]) & evg or grams(x["ow"]) & evg)]
    human += dict_ex

    # back-translations: length filter, then best by adapter-A loss
    bt = [json.loads(l) for l in open(args.bt)]
    scores = {json.loads(l)["row"]: json.loads(l)["nll"] for l in open(args.bt_scores)} if args.bt_n else {}
    def ok(r):
        a, b = len(r["text"].split()), len((r.get("en") or "").split())
        return a >= 3 and b >= 3 and 0.5 <= a / b <= 3 and not (grams(r["text"]) & evg)
    cand = sorted((i for i, r in enumerate(bt) if ok(r) and i in scores), key=lambda i: scores[i])
    chosen = [bt[i] for i in cand[: args.bt_n]]
    pool = human + [(r["en"], r["text"]) for r in chosen[:5000]]

    rows = []
    for _ in range(args.human_repeat):
        rows += [render_pair(rng, s, t, pool, lex, args) for s, t in human]
    rows += [render_pair(rng, r["en"].strip(), r["text"].strip(), pool, lex, args, tag="<bt> ") for r in chosen]
    # reading: the best-scored BT pairs reversed (beyond --bt-n first, so the two directions use different sentences)
    flips = [bt[i] for i in cand[args.bt_n: args.bt_n + args.flip_n]]
    if len(flips) < args.flip_n:
        flips += [bt[i] for i in cand[: args.flip_n - len(flips)]]
    rows += [render_flip(rng, r["text"].strip(), r["en"].strip(), "oshindonga") for r in flips]
    kua = []
    if args.kua_bt and args.kua_n:
        kua = [r for r in map(json.loads, open(args.kua_bt)) if r.get("en") and r.get("lid_prob", 0) >= args.kua_min_lid]
        kua = [r for r in kua if ok(r)]
        rng.shuffle(kua)
        kua = kua[: args.kua_n]
        rows += [render_pair(rng, r["en"].strip(), r["text"].strip(), pool, lex, args, tag="<bt> ", dialect="oshikwanyama") for r in kua]
        rows += [render_flip(rng, r["text"].strip(), r["en"].strip(), "oshikwanyama") for r in kua]

    cons = []
    if args.consensus and args.consensus_n:  # sorted by agreement: the most reliable targets first
        cons = [r for r in map(json.loads, open(args.consensus)) if not (grams(r["en"]) & evg or grams(r["text"]) & evg)]
        cons = cons[: args.consensus_n]
        rows += [render_pair(rng, r["en"], r["text"], pool, lex, args) for r in cons]

    # vocabulary tasks (both directions), excluding the retention word test
    test = json.load(open("data/private/retention/wordtest_500.json"))
    banned = {w["form"].lower() for w in test} | {w["lemma"].lower() for w in test}
    entries = [(en, ow, note) for en, v in lex.items() for ow, note in v[:1]
               if ow.lower() not in banned and len(en.split()) <= 2]
    rng.shuffle(entries)
    for en, ow, note in entries[: args.vocab_n]:
        if rng.random() < 0.6:
            q, a = rng.choice(VOCAB_ASK).format(en=en), ow + (f" ({note})" if note else "")
        else:
            q, a = rng.choice(VOCAB_BACK).format(ow=ow), en
        rows.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})

    replay, seen = [], set()
    for p in args.replay:
        if Path(p).exists():
            for l in open(p):
                m = json.loads(l)["messages"]
                if m[0]["content"][:200] not in seen:
                    seen.add(m[0]["content"][:200]); replay.append({"messages": m})
    rows += replay
    tools = [json.loads(l) for p in args.tool_replay for l in open(p) if l.strip()]
    tools = tools[: args.tool_n] if args.tool_n else tools
    rows += [{"prompt": r["prompt"], "completion": r["completion"]} for r in tools]
    rng.shuffle(rows)
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps({"human_pairs": len(human), "dict_examples": len(dict_ex), "human_rows": len(human) * args.human_repeat,
                      "bt_selected": len(chosen), "flip": len(flips), "kua_pairs": len(kua), "consensus": len(cons), "bt_nll_cutoff": round(scores[cand[min(len(cand), args.bt_n) - 1]], 3) if chosen else None,
                      "vocab": min(args.vocab_n, len(entries)), "replay": len(replay), "tool_replay": len(tools),
                      "replay_share": round((len(replay) + len(tools)) / len(rows), 3), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
