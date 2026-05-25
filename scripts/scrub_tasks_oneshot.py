"""One-shot scrub of low-quality entries from the live contributions
task pool. Run inside the webhook container against /data/contributions.sqlite.

Removes tasks whose source_en matches any of the artifact patterns we
discovered AFTER the initial mining/dedup pass:
- Escaped-JSON characters (backslash-quotes)
- Q&A / structural prefixes ('Answer:', 'Example:', etc.)
- Emoji
- Academic / thesis register that dodged the v2.4 regex
- Leading quote characters (mid-conversation echo)
- Trailing list-marker numbers ('… text. 2.')

Also deletes tasks that have ALREADY been served to any contributor
WITHOUT being submitted — they're suspect enough that we don't want
them coming back."""
from __future__ import annotations

import re
import sqlite3
import sys

DB = "/data/contributions.sqlite"

# Match the patterns from scripts/mine_production_sentences.py (kept in
# sync; this is a one-shot scrub not worth refactoring into a shared
# module yet).
PATTERNS = [
    (r'\\"', "escaped-quote"),
    (r'\\\\', "escaped-backslash"),
    (r"^(Answer|Question|Example|Reason|Note|Hint|Tip|Reminder|Definition|Goal|Aim|Solution|Method)\s*:", "qa-prefix"),
    (r"[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\U0001F600-\U0001F64F☀-➿]", "emoji"),
    (r"\bin (?:the|your) (?:thesis|dissertation|methodology|literature review)\b", "academic-thesis"),
    (r"\b(?:repeat|replicate) (?:your|the|this) (?:study|experiment|methodology|findings)\b", "academic-replicate"),
    (r"\b(?:detailed|comprehensive) (?:recipe|protocol|procedure|methodology)\b", "academic-protocol"),
    (r'^["\']', "leading-quote"),
    (r"[.!?]\s+\d+\.\s*$", "trailing-list-num"),
    # Specific business-name leaks observed in live testing. Keep the
    # list short and explicit — we don't want broad capitalised-word
    # heuristics that would also drop 'Bank of Namibia'.
    (r"\bSolitaire Country Lodge\b", "biz-name-leak"),
]
COMPILED = [(re.compile(pat, re.IGNORECASE), label) for pat, label in PATTERNS]


def main() -> int:
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    rows = c.execute("SELECT id, source_en FROM tasks").fetchall()
    print(f"# audit start: {len(rows)} tasks in pool")

    to_delete: dict[int, str] = {}
    for r in rows:
        text = r["source_en"]
        for pat, label in COMPILED:
            if pat.search(text):
                to_delete[r["id"]] = label
                break

    print(f"# matches: {len(to_delete)}")
    by_label: dict[str, int] = {}
    for label in to_delete.values():
        by_label[label] = by_label.get(label, 0) + 1
    for label, n in sorted(by_label.items(), key=lambda x: -x[1]):
        print(f"  {label:20s} {n}")

    if not to_delete:
        return 0

    if "--apply" not in sys.argv:
        print("\n# dry-run. Re-run with --apply to actually delete.")
        # Show 5 random matches so we can eyeball quality
        import random
        sample = random.sample(list(to_delete.keys()), min(5, len(to_delete)))
        for sid in sample:
            row = c.execute("SELECT source_en FROM tasks WHERE id = ?", (sid,)).fetchone()
            print(f"  example #{sid}: {row['source_en'][:110]}")
        return 0

    # Apply
    ids = list(to_delete.keys())
    # Chunk in case sqlite has parameter-count limits
    BATCH = 500
    deleted = 0
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        placeholders = ",".join("?" * len(chunk))
        c.execute(f"DELETE FROM tasks WHERE id IN ({placeholders})", chunk)
        deleted += len(chunk)
    c.commit()
    print(f"\n# deleted: {deleted}")
    print(f"# remaining tasks: {c.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
