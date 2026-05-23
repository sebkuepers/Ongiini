#!/usr/bin/env python3
"""Side-by-side reviewer for compose-vs-revise drafts.

Reads pairs captured by ``ReviseEvalCaptureHook`` (data/revise_eval/) and
walks the operator through them one at a time, collecting human ratings
into ``data/revise_eval/ratings.jsonl``.

Goal: answer the architectural question "does revise actually improve
output, or is it just adding latency?" with real data instead of guesses.

Usage:
    python scripts/review_revises.py            # rate next unrated pair
    python scripts/review_revises.py --summary  # show aggregate verdicts
    python scripts/review_revises.py --re-rate <msg_id>  # overwrite an existing rating

Decision thresholds (rule of thumb):
    revise-better >70%   keep the loop
    revise-better <40%   kill it; save the compute
    in between           dig deeper into what kind of turns benefit
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Default location matches ReviseEvalCaptureHook's default.
DEFAULT_CAPTURE_DIR = Path("data/revise_eval")
RATINGS_FILE_NAME = "ratings.jsonl"


# Verdict keys (single character → meaning). The CLI shows the menu.
_VERDICTS = {
    "c": "compose-better",
    "r": "revise-better",
    "t": "tie",
    "b": "both-bad",
    "s": "skip",
}


def _load_ratings(ratings_path: Path) -> dict[str, dict]:
    """Return {msg_id: rating_record}. Empty dict if no file yet."""
    if not ratings_path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in ratings_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and "msg_id" in rec:
            out[rec["msg_id"]] = rec
    return out


def _append_rating(ratings_path: Path, msg_id: str, verdict: str, note: str) -> None:
    """Append one rating to the JSONL. We don't dedupe at write time;
    --summary uses the LAST rating per msg_id."""
    rec = {"msg_id": msg_id, "verdict": verdict, "note": note}
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    with ratings_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_captures(capture_dir: Path) -> list[dict]:
    """Read every <msg_id>.json file under the capture dir."""
    if not capture_dir.exists():
        return []
    out: list[dict] = []
    for p in sorted(capture_dir.glob("*.json")):
        if p.name == RATINGS_FILE_NAME:
            continue
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def _print_pair(pair: dict, idx: int, total: int) -> None:
    print(f"\n{'=' * 72}")
    print(f"[{idx}/{total}] msg_id={pair.get('msg_id')} policy={pair.get('policy')} "
          f"ts={pair.get('ts')}")
    print(f"compose_len={pair.get('compose_len')} revised_len={pair.get('revised_len')}")
    print(f"\nUSER QUESTION:\n  {pair.get('user_question', '')}")
    reasons = pair.get("critique_reasons") or []
    if reasons:
        print(f"\nCRITIQUE FLAGGED ({pair.get('critique_verdict')}):")
        for r in reasons:
            print(f"  - {r}")
    print(f"\n{'─' * 36} COMPOSE (original) {'─' * 16}")
    print(pair.get("compose_draft", "").rstrip())
    print(f"\n{'─' * 36} REVISED (shipped) {'─' * 16}")
    print(pair.get("revised_reply", "").rstrip())
    print()


def _prompt_verdict() -> tuple[str | None, str]:
    """Prompt for one rating. Returns (verdict_key, note). verdict_key
    is None if the user quits."""
    print("Rate this pair:")
    print("  c = compose was better")
    print("  r = revise was better")
    print("  t = tie / no meaningful difference")
    print("  b = both bad")
    print("  s = skip for now")
    print("  q = quit")
    while True:
        choice = input("> ").strip().lower()
        if choice == "q":
            return None, ""
        if choice in _VERDICTS:
            note = input("Optional note (or Enter to skip): ").strip()
            return _VERDICTS[choice], note
        print("Invalid. Enter c/r/t/b/s/q.")


def cmd_rate(args: argparse.Namespace) -> int:
    capture_dir = Path(args.capture_dir)
    ratings_path = capture_dir / RATINGS_FILE_NAME
    captures = _load_captures(capture_dir)
    if not captures:
        print(f"No captures found in {capture_dir}.", file=sys.stderr)
        print("Set ONGIINI_CAPTURE_REVISE_EVAL=1 on the webhook and let it "
              "collect a few REVISE turns first.", file=sys.stderr)
        return 1
    existing = _load_ratings(ratings_path)
    unrated = [c for c in captures if c.get("msg_id") not in existing]
    if not unrated:
        print(f"All {len(captures)} captures already rated. "
              f"Use --summary to see results, or --re-rate <msg_id> to redo one.")
        return 0
    print(f"{len(unrated)} unrated of {len(captures)} total. Quit any time with q.")
    for i, pair in enumerate(unrated, 1):
        _print_pair(pair, i, len(unrated))
        verdict, note = _prompt_verdict()
        if verdict is None:
            print(f"Stopped. {i - 1} new rating(s) saved.")
            break
        _append_rating(ratings_path, pair["msg_id"], verdict, note)
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    capture_dir = Path(args.capture_dir)
    ratings_path = capture_dir / RATINGS_FILE_NAME
    captures = _load_captures(capture_dir)
    ratings = _load_ratings(ratings_path)
    if not captures:
        print(f"No captures found in {capture_dir}.")
        return 1
    print(f"Captures: {len(captures)}")
    print(f"Rated:    {len(ratings)}")
    if not ratings:
        print("Nothing rated yet. Run without --summary to start rating.")
        return 0
    counts: Counter[str] = Counter(r["verdict"] for r in ratings.values())
    total_rated = sum(counts.values()) - counts.get("skip", 0)
    print()
    print("Verdicts:")
    for v in ("compose-better", "revise-better", "tie", "both-bad", "skip"):
        n = counts.get(v, 0)
        pct = (n / total_rated * 100) if total_rated and v != "skip" else 0
        suffix = f" ({pct:.1f}% of rated)" if v != "skip" and total_rated else ""
        print(f"  {v:<16} {n}{suffix}")
    print()
    if total_rated >= 1:
        revise_pct = counts.get("revise-better", 0) / total_rated * 100
        print(f"Revise-better fraction: {revise_pct:.1f}%")
        if revise_pct >= 70:
            print("→ Loop is paying off. Keep it.")
        elif revise_pct < 40:
            print("→ Loop is NOT paying off. Consider killing it.")
        else:
            print("→ Mixed signal. Need more data or finer-grained analysis "
                  "(rate per-policy, per-question-type).")
    return 0


def cmd_re_rate(args: argparse.Namespace) -> int:
    capture_dir = Path(args.capture_dir)
    ratings_path = capture_dir / RATINGS_FILE_NAME
    captures = _load_captures(capture_dir)
    pair = next((c for c in captures if c.get("msg_id") == args.msg_id), None)
    if pair is None:
        print(f"msg_id {args.msg_id!r} not found in {capture_dir}.", file=sys.stderr)
        return 1
    _print_pair(pair, 1, 1)
    verdict, note = _prompt_verdict()
    if verdict is None:
        return 0
    # We append a new line; --summary keeps only the LAST one per msg_id.
    _append_rating(ratings_path, pair["msg_id"], verdict, note)
    print("Saved.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-dir", default=str(DEFAULT_CAPTURE_DIR),
        help=f"Where to find <msg_id>.json files (default: {DEFAULT_CAPTURE_DIR})",
    )
    parser.add_argument("--summary", action="store_true", help="Print aggregate verdicts and exit.")
    parser.add_argument("--re-rate", metavar="MSG_ID", help="Re-rate a specific captured pair.")
    args = parser.parse_args()
    if args.summary:
        return cmd_summary(args)
    if args.re_rate:
        args.msg_id = args.re_rate
        return cmd_re_rate(args)
    return cmd_rate(args)


if __name__ == "__main__":
    sys.exit(main())
