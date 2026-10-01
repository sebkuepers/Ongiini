#!/usr/bin/env python3
"""Build the run manifest for Ongiini-Eval-OW baselines (docs/eval-protocol.md).

One entry per system label found in data/private/export/data/baselines/:
display name, category, region, access path, which dialects it covers,
prompt template, decoding, the run period, and attempt / redo counts from
the usage sidecars. This is the single source of the system list for
scoring and the report — add new systems to SYSTEMS below.

    python3 scripts/build_run_manifest.py   # writes baselines/manifest.json
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/private/export/data/baselines"
DIALECTS = ("oshindonga", "oshikwanyama")

# label → (display name, category, region, access, role)
#   category: frontier | open-weight | mt | specialist
#   role:     main | reference-point | diagnostic
SYSTEMS = {
    "claude-opus-5":          ("Claude Opus 5", "frontier", "USA", "OpenRouter", "main"),
    "gpt-6-astra":            ("GPT-6 Astra", "frontier", "USA", "OpenRouter", "main"),
    "gemini-3.1-pro":         ("Gemini 3.1 Pro", "frontier", "USA", "OpenRouter", "main"),
    "muse-spark-1.3":         ("Muse Spark 1.3", "frontier", "USA", "Meta Model API", "main"),
    "deepseek-v4.1-chat":     ("DeepSeek V4.1 (chat)", "frontier", "China", "OpenRouter", "main"),
    "deepseek-v4.1-reasoner": ("DeepSeek V4.1 (reasoner)", "frontier", "China", "OpenRouter", "main"),
    "glm-5.3-low":            ("GLM 5.3 (reasoning effort low)", "frontier", "China", "OpenRouter", "diagnostic"),
    "qwen3.8-27b-nothink":    ("Qwen 3.8 27B (reasoning off)", "open-weight", "China", "OpenRouter", "diagnostic"),
    "deepseek-v4.1-reasoner-redo16k": ("DeepSeek V4.1 (reasoner, empty outputs redone at 16k)", "frontier", "China",
                                       "OpenRouter", "diagnostic"),
    "kimi-k3":                ("Kimi K3", "frontier", "China", "OpenRouter", "main"),
    "glm-5.3":                ("GLM 5.3", "frontier", "China", "OpenRouter", "main"),
    "mistral-medium-3.5":     ("Mistral Medium 3.5", "frontier", "Europe", "OpenRouter", "main"),
    "gemma-4-26b-api":        ("Gemma 4 26B", "open-weight", "USA", "OpenRouter", "main"),
    "llama-4-scout":          ("Llama 4 Scout", "open-weight", "USA", "OpenRouter", "main"),
    "mistral-small-4":        ("Mistral Small 4", "open-weight", "Europe", "OpenRouter", "main"),
    "qwen3.8-27b":            ("Qwen 3.8 27B", "open-weight", "China", "OpenRouter", "main"),
    "okalm-1b_zeroshot":      ("OkaLM 1B", "specialist", "Namibia", "local (MLX)", "diagnostic"),  # base model, ignores the instruction (2026-10-01)
    "okalm-3b_zeroshot":      ("OkaLM 3B", "specialist", "Namibia", "local (MLX)", "diagnostic"),  # base model, ignores the instruction (2026-10-01)
    "okalm-8b_zeroshot":      ("OkaLM 8B", "specialist", "Namibia", "local (MLX)", "diagnostic"),  # base model, ignores the instruction (2026-10-01)
    "nllb-200-3.3b":          ("NLLB-200 3.3B → Tswana", "mt", "USA", "local", "reference-point"),
    "madlad400-3b-mt":        ("MADLAD-400 3B → Tswana", "mt", "USA", "local", "reference-point"),
    "nllb-200-3.3b-umb":      ("NLLB-200 3.3B → Umbundu", "mt", "USA", "local", "reference-point"),
    "madlad400-3b-mt-kj":     ("MADLAD-400 3B <2kj>", "mt", "USA", "local", "main"),
    "okalm-1b_fewshot":       ("OkaLM 1B (5-shot)", "specialist", "Namibia", "local (MLX)", "diagnostic"),
    "okalm-3b_fewshot":       ("OkaLM 3B (5-shot)", "specialist", "Namibia", "local (MLX)", "diagnostic"),
    "okalm-8b_fewshot":       ("OkaLM 8B (5-shot)", "specialist", "Namibia", "local (MLX)", "diagnostic"),
}
# Present on disk but excluded, with the reason (docs/eval-protocol.md §11).
EXCLUDED = {
    "mistral-large-3": "dropped: throttled, 62 of 600 items",
    "okamt-okalex-demo_sample15": "15-item manual sample via the OkaLex web UI; appendix only, unpublished",
}
FILE_RE = re.compile(r"^(?P<label>.+)_(?P<dialect>oshindonga|oshikwanyama)\.jsonl$")


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> int:
    found: dict[str, dict[str, Path]] = {}
    for p in sorted(BASE.glob("*.jsonl")):
        m = FILE_RE.match(p.name)
        if m:
            found.setdefault(m["label"], {})[m["dialect"]] = p
    unknown = sorted(set(found) - set(SYSTEMS) - set(EXCLUDED))
    if unknown:
        raise SystemExit(f"unregistered system files (add to SYSTEMS or EXCLUDED): {unknown}")

    entries = []
    for label, (name, cat, region, access, role) in SYSTEMS.items():
        files = found.get(label, {})
        if not files:
            continue
        recs = [r for p in files.values() for r in read(p)]
        templates = sorted({r.get("prompt_template_id", "?") for r in recs})
        stamps = sorted(r["timestamp"] for r in recs if r.get("timestamp"))
        usage_p = BASE / f"{label}.usage.jsonl"
        attempts, redo, max_tokens = Counter(), 0, Counter()
        if usage_p.exists():
            last = {}
            for u in read(usage_p):
                last[(u["id"], u["dialect"])] = u          # redo lines supersede
            for u in last.values():
                attempts[u.get("attempts", 1)] += 1
                redo += bool(u.get("redo_empty"))
                if u.get("max_tokens"):
                    max_tokens[u["max_tokens"]] += 1
        fewshot_ids = []
        for p in files.values():
            ids_p = p.with_suffix(".fewshot_ids.json")
            if ids_p.exists():
                fewshot_ids = json.loads(ids_p.read_text())
        entries.append({
            "label": label, "name": name, "category": cat, "region": region, "access": access,
            "role": role, "dialects": sorted(files), "files": {d: str(p.relative_to(ROOT)) for d, p in files.items()},
            "model_id": sorted({r.get("model_id", "?") for r in recs}),
            "prompt_template_id": templates,
            "seed": sorted({str(r.get("seed")) for r in recs}),
            "period": [stamps[0], stamps[-1]] if stamps else None,
            "attempts": {str(k): v for k, v in sorted(attempts.items())},
            "redo_empty_16k": redo,
            "max_tokens_recorded": {str(k): v for k, v in sorted(max_tokens.items())},
            "exclude_ids": fewshot_ids,
        })
    out = {"systems": entries, "excluded": EXCLUDED,
           "decoding_note": "API runs: temperature 0 and seed 42 where the model accepts them, "
                            "else provider default; reasoning at the provider default; max_tokens "
                            "4096 for models reporting reasoning support, else 1024; empty outputs "
                            "from reasoning models regenerated once with 16384 (docs/eval-protocol.md §3). "
                            "MT models: beam 4, no sampling. OkaLM: greedy, MLX."}
    (BASE / "manifest.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"manifest: {len(entries)} systems → {(BASE / 'manifest.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
