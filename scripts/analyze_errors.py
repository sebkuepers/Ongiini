#!/usr/bin/env python3
"""Where does an adapter lose chrF++ on Ndonga dev? (paper 3, CPU only)

Per domain and length bucket (vs base), added quotation marks, and whether the
reference words are missing from our Oshiwambo text (mono corpus + human pairs) or
present but not produced — i.e. vocabulary gap vs word choice. Runs in the evalsuite
image (sacrebleu); data stays in data/private.

    python3 scripts/analyze_errors.py T2
"""
import sys
import json, re, collections
from sacrebleu.metrics import CHRF
chrf = CHRF(word_order=2)
ev = {r["id"]: r for r in map(json.loads, open("data/private/export/data/eval_set.jsonl"))}
def load(lab):
    return {r["id"]: r["translation"] for r in map(json.loads, open(f"data/private/experiments/lora/gemma-4-12b-{lab}_oshindonga.jsonl"))}
dev = [i for i, r in ev.items() if not r["in_blind_split"] and r.get("oshindonga_reference")]
def corpus(h, ids): return round(chrf.corpus_score([h[i] for i in ids], [[ev[i]["oshindonga_reference"] for i in ids]]).score, 1)
T2, base = load(sys.argv[1] if len(sys.argv) > 1 else "T2"), load("base")
dev = [i for i in dev if i in T2]
print("dev n", len(dev), "T2", corpus(T2, dev), "base", corpus(base, dev))
# quotes
q = [i for i in dev if T2[i][:1] in "\"'“" and ev[i]["oshindonga_reference"][:1] not in "\"'“"]
qe = [i for i in dev if ev[i]["english"][:1] in "\"'“"]
print("T2 adds leading quote:", len(q), "| english starts with quote:", len(qe))
strip = {i: (T2[i][1:] if i in q else T2[i]) for i in dev}
strip = {i: (re.sub(r'["”]$', "", t) if i in q else t) for i, t in strip.items()}
print("T2 with added quotes stripped:", corpus(strip, dev))
# domain / length
for key in ("domain", "length_bucket"):
    g = collections.defaultdict(list)
    for i in dev: g[ev[i][key]].append(i)
    print(key, {k: (len(v), corpus(T2, v), corpus(base, v)) for k, v in sorted(g.items())})
# vocabulary coverage of reference words in training-side Oshiwambo text
seen = collections.Counter()
for l in open("data/private/corpus/osheng_v1/mono.jsonl"):
    seen.update(re.findall(r"\w+", json.loads(l)["text"].lower()))
for l in open("data/private/corpus/osheng_v1/sft_A_parallel_ndo_train.jsonl"):
    seen.update(re.findall(r"\w+", json.loads(l)["messages"][-1]["content"].lower()))
lex = json.load(open("data/private/corpora/dictionaries/viljoen-1984/lexicon_en_ow.json"))
dict_words = {w for v in lex.values() for ow, _ in v for w in re.findall(r"\w+", ow.lower())}
toks = [w for i in dev for w in re.findall(r"\w+", ev[i]["oshindonga_reference"].lower())]
print("ref tokens", len(toks), "unseen in mono+human:", round(100*sum(seen[w]==0 for w in toks)/len(toks),1), "%",
      "| seen <5x:", round(100*sum(seen[w]<5 for w in toks)/len(toks),1), "%",
      "| unseen but in dictionary:", sum(seen[w]==0 and w in dict_words for w in toks))
# does T2 produce the reference word when it is rare vs frequent?
hit = collections.defaultdict(lambda: [0, 0])
for i in dev:
    hyp = set(re.findall(r"\w+", T2[i].lower()))
    for w in set(re.findall(r"\w+", ev[i]["oshindonga_reference"].lower())):
        b = "0" if seen[w]==0 else "1-9" if seen[w]<10 else "10-99" if seen[w]<100 else "100-999" if seen[w]<1000 else "1000+"
        hit[b][0] += w in hyp; hit[b][1] += 1
print("ref word produced by T2, by corpus frequency:", {k: f"{a}/{n} ({round(100*a/n)}%)" for k, (a, n) in sorted(hit.items())})
# worst sentences
per = sorted(dev, key=lambda i: chrf.sentence_score(T2[i], [ev[i]["oshindonga_reference"]]).score)
print("length ratio hyp/ref (chars):", round(sum(len(T2[i]) for i in dev)/sum(len(ev[i]["oshindonga_reference"]) for i in dev), 3))
