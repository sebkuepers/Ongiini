"""Mine English sentences from production Ongiini assistant replies.

Replaces the earlier mine_seeds.py one-shot. Goal: extract the
maximum number of SEMANTICALLY DISTINCT English sentences in the
WhatsApp register that the bot actually produces in production,
PII-scrubbed and categorised, ready to seed the community
contribution task pool.

Reads: ~/dev/Ongiini/data/264*.json (per-user short-term history).
Writes: /tmp/mined_sentences_v2.jsonl — one JSON object per
candidate sentence:
    {"id": int, "source_en": str, "category": str, "src_user_hash": str, "src_message_idx": int}

Runs on Spark (read-only). Designed to be cheap to re-run with
different filters during iteration. Embedding-based near-duplicate
removal happens in a separate step (dedupe_mined_sentences.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make ongiini.pii importable when run from the worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from ongiini.pii import sanitize as pii_sanitize
except Exception:
    def pii_sanitize(t: str) -> str:
        return t


DEFAULT_DATA_DIR = Path("/home/nexus/dev/Ongiini/data")
DEFAULT_OUT = Path("/tmp/mined_sentences_v2.jsonl")
DEFAULT_EVAL_PATH = Path("/tmp/eval_set_pairs.py")


# ── Welcome menu detection ────────────────────────────────────────

WELCOME_MARKERS = [
    "I'm an AI helper here on WhatsApp",
    "I can help with things like",
    "What would you like to try first",
    "Happy you found us",
    "Ek is 'n KI-helper hier op WhatsApp",
    "free WhatsApp",
    "Ondi li omukwafi",
]


def is_welcome(text: str) -> bool:
    return any(m in text for m in WELCOME_MARKERS)


# ── Oshiwambo / Afrikaans detection (cheap heuristic) ─────────────

OSHIWAMBO_MARKERS = re.compile(
    r"\b(tangi|kala po|ongiini|aame|ohandi|kuume|kandi|kwafe|kalunga|owa hala|"
    r"ombili|eewa|tu monene|owa lala|onawa|nawa)\b",
    re.IGNORECASE,
)
# Afrikaans markers — common short words / function words English doesn't have.
# "ons" (we), "die" (the), "wat" (what), "is" — but "is" is also English so
# we skip it. Pick markers that are unambiguous Afrikaans.
AFRIKAANS_MARKERS = re.compile(
    r"\b(jy|jou|ek|sien|nie|hierdie|asseblief|baie|dankie|moet|kan|jongste|"
    r"vir|wanneer|sou|wees|kry|gee|maak|ons|hulle|julle|met|hoe|"
    r"die|sodat|omdat|dieselfde|nogal|natuurlik|natuurlike|"
    r"hoeveel|hoekom|waarvan|waaroor)\b",
    re.IGNORECASE,
)
# Unambiguous Afrikaans bigrams — single hit is enough to flag.
AFRIKAANS_BIGRAMS = re.compile(
    r"\b(ons kan|ek het|jy is|jy het|jou is|sou ek|kan ons|sal jy|moet jy|"
    r"nie waar|dis 'n|is 'n |hierdie is)\b",
    re.IGNORECASE,
)


def is_mostly_non_english(text: str) -> bool:
    if OSHIWAMBO_MARKERS.search(text):
        return True
    if AFRIKAANS_BIGRAMS.search(text):
        return True
    if len(AFRIKAANS_MARKERS.findall(text)) >= 2:
        return True
    # Diacritics that aren't natural in English: à â ä ë î ï ô ö ù û ü ç
    # ñ ã õ ø æ. Any text containing two or more of these in a single
    # sentence is almost certainly Afrikaans, German, French, or
    # another European-imported orthography — drop it.
    if len(re.findall(r"[àâäëîïôöùûüçñãõøæ]", text, re.IGNORECASE)) >= 2:
        return True
    return False


# ── Markdown / structural cleanup ─────────────────────────────────

_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_MD_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_MD_CODE = re.compile(r"`([^`]+)`")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_HEADER = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_LEADING_BULLET = re.compile(r"^\s*(?:\d+[\.\)]|[\-\*\+])\s+")
_TRAILING_LIST_MARKER = re.compile(r":\s*\*+\s*$")
_URL = re.compile(r"https?://\S+")


def strip_markdown(text: str) -> str:
    """Remove WhatsApp/markdown chrome from a message before splitting."""
    text = _MD_HEADER.sub("", text)
    text = _MD_LINK.sub(r"\1", text)        # keep link text, drop URL
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_ITALIC.sub(r"\1", text)
    text = _MD_CODE.sub(r"\1", text)
    text = _URL.sub("", text)               # drop bare URLs entirely
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Sentence splitting (careful about abbreviations) ──────────────

# Common honorifics / abbreviations that look like sentence boundaries
# but aren't. We protect them by replacing the dot with a sentinel
# before splitting, then restore.
_ABBREVS = (
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Hon.", "St.", "Mt.",
    "etc.", "i.e.", "e.g.", "vs.", "no.", "No.", "Ltd.", "Inc.",
    "approx.", "p.m.", "a.m.", "P.M.", "A.M.",
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(A-Z])")


def split_sentences(text: str) -> list[str]:
    # Protect abbrevs
    for abbr in _ABBREVS:
        text = text.replace(abbr, abbr.replace(".", "\x00"))
    parts = _SENT_SPLIT.split(text)
    # Restore
    out = []
    for p in parts:
        p = p.replace("\x00", ".").strip()
        # Strip leading bullet / numbering
        p = _LEADING_BULLET.sub("", p).strip()
        # Strip trailing markdown list markers like ":*"
        p = _TRAILING_LIST_MARKER.sub("", p).strip()
        if p:
            out.append(p)
    return out


# ── Per-sentence quality gates ────────────────────────────────────

# Bot-pattern sentences we don't want as training source: meta-commentary,
# tool-use signals, role-marker leaks. These bleed into training as
# "the bot describes its own state" which isn't useful Oshiwambo content.
_BOT_PATTERNS = re.compile(
    r"^("
    r"I am an AI|I'm an AI|"
    r"As an AI|As a language model|"
    r"I am a large language model|I am Gemma|I am called Gemma|"
    r"Let me search|Let me look up|Let me check|"
    r"I cannot|I can't|"
    r"Here are some|Here is a list|"
    r"I don't have access|"
    r"Based on (?:the|my) search|"
    r"My (?:training data|knowledge) (?:cutoff|stops)|"
    r"I was (?:created|developed|trained) by"
    r")",
    re.IGNORECASE,
)
# Also catch the meta-line patterns even when they're not at the very
# start of the sentence (Gemma sometimes prefixes them after a clause:
# 'Because I am an AI, I cannot...').
_BOT_PATTERN_ANY = re.compile(
    r"\b(large language model|developed by Google|Google DeepMind|"
    r"my knowledge cutoff|my training data|cannot browse|"
    r"I don't have real-time|I am an AI|I'm an AI|as an AI|"
    r"because I am an AI|since I am an AI|"
    r"I cannot provide a medical diagnosis|"
    r"I am not a doctor|I am not a lawyer|I am not a financial)\b",
    re.IGNORECASE,
)

# Sentences ending in markdown structural junk
_ENDS_BAD = re.compile(r"[*_:`]\s*$|\bN\$\s*$")

# Sentences whose value depends on prior turn ("Yes, that's right.")
_CONTEXT_DEPENDENT = re.compile(
    r"^(Yes|No|Okay|OK|Sure|Right|Exactly|Like that|That's it|Got it|"
    r"That is|That's|Indeed)[,\s.!?]",
    re.IGNORECASE,
)


def acceptable(text: str, min_words: int = 5, max_words: int = 40) -> tuple[bool, str]:
    """Return (ok, reason) — if not ok, reason explains the rejection.
    Used in mining + in the unit tests."""
    if not text:
        return False, "empty"
    wc = len(text.split())
    if wc < min_words:
        return False, "too_short"
    if wc > max_words:
        return False, "too_long"
    if is_welcome(text):
        return False, "welcome_menu"
    if is_mostly_non_english(text):
        return False, "non_english"
    if "[REDACTED" in text:
        return False, "pii_scrubbed_changed_meaning"
    if _BOT_PATTERNS.match(text) or _BOT_PATTERN_ANY.search(text):
        return False, "bot_meta_pattern"
    # Sentences with leading "Step N:" or trailing "...: 1." are list-row
    # fragments — useful as content but they translate awkwardly because
    # they're structural, not semantic. Drop them.
    if re.match(r"^Step\s+\d+\s*[:.]", text, re.IGNORECASE):
        return False, "list_row_fragment"
    if re.search(r":\s*\d+\.?\s*$", text):
        return False, "list_row_fragment"
    if _ENDS_BAD.search(text):
        return False, "trailing_markdown_junk"
    if _CONTEXT_DEPENDENT.match(text):
        return False, "context_dependent"
    if "{" in text and "}" in text:
        return False, "json_leakage"
    if text.count('"') == 1:
        return False, "unbalanced_quote"
    if text.count("(") != text.count(")"):
        return False, "unbalanced_paren"
    # No verbatim URL fragments left after strip
    if "http" in text or "://" in text:
        return False, "url_leak"
    return True, "ok"


# ── Categorisation (same buckets as v1 plan) ──────────────────────

_RE_CV_JOBS = re.compile(
    r"\b(cv|resume|cover letter|interview|job|career|business|company|skill|"
    r"experience|profession|employer|manager|industry|salary|hire|recruit|"
    r"vacancy|apply|application|graduate)\b",
    re.IGNORECASE,
)
_RE_EDUCATION = re.compile(
    r"\b(stud(y|ies|ying)|exam|school|university|assignment|homework|learn|"
    r"lesson|grade|module|course|essay|research|nssco|nsscas|teacher|student|"
    r"class|subject|textbook|tutorial|revision|test)\b",
    re.IGNORECASE,
)
_RE_TRANSLATION = re.compile(
    r"\b(translat|oshindonga|oshikwanyama|oshiwambo|afrikaans|herero|"
    r"khoekhoegowab|nama|damara|language|pronounce|spell|meaning|dictionary|"
    r"phrase|grammar)\b",
    re.IGNORECASE,
)
_RE_CONVERSATIONAL = re.compile(
    r"\b(thank|welcome|hello|hi there|good morning|good afternoon|appreciate|"
    r"sorry|hope|glad|sounds|feel|listen|chat|here for you|happy to|"
    r"of course|no problem|tangi|nawa)\b",
    re.IGNORECASE,
)
# Business / professional / marketing / customer-service language. Catches a
# big slice of what the v1 categoriser misclassified as "niche". Note: we
# deliberately exclude "business" keyword itself from cv_jobs so this
# bucket captures pure-business-advice content (branding, sales,
# pricing, services), not just cv-with-mention-of-business.
_RE_BUSINESS = re.compile(
    r"\b(brand|branding|marketing|customer|client|sales|revenue|pricing|"
    r"product|service line|menu|profit|loss|invoice|quote|estimate|"
    r"social media|whatsapp business|instagram|facebook|website copy|"
    r"vendor|supplier|stockist|wholesale|retail margin|markup)\b",
    re.IGNORECASE,
)
# Namibian daily-life: family, money management, transport, government
# services, religion/culture, health/wellbeing, food, neighbourhood.
_RE_DAILY_LIFE = re.compile(
    r"\b(family|mother|father|grandmother|grandfather|aunt|uncle|sister|"
    r"brother|child|son|daughter|home|household|"
    r"bank|loan|account|fnb|nedbank|standard bank|debit|credit|insurance|"
    r"rent|salary day|month-end|pension|grant|"
    r"taxi|kombi|hike|road|town|village|farm|cattle|goat|sheep|"
    r"clinic|hospital|doctor|nurse|medic|prescription|pharmacy|"
    r"church|pastor|bible|sunday|prayer|"
    r"recipe|cooking|maize|pap|kapana|braai|biltong|"
    r"police|home affairs|nhc|nhi|municipality|councillor|swakopmund|windhoek|"
    r"oshakati|ondangwa|katima|rundu|walvis bay|opuwo|tsumeb|rehoboth|"
    r"caprivi|kavango|kunene|otjozondjupa|hardap|karas|omusati|oshana|oshikoto|"
    r"erongo|khomas|zambezi)\b",
    re.IGNORECASE,
)


def categorise(text: str) -> str:
    # Order matters — first match wins. We check the most specific buckets
    # first (cv_jobs / education / translation are domain-precise) and
    # only fall through to broader buckets (business / daily_life) when
    # the precise ones don't match.
    if _RE_CV_JOBS.search(text):
        return "cv_jobs"
    if _RE_EDUCATION.search(text):
        return "education"
    if _RE_TRANSLATION.search(text):
        return "translation"
    if _RE_BUSINESS.search(text):
        return "business"
    if _RE_DAILY_LIFE.search(text):
        return "daily_life"
    if _RE_CONVERSATIONAL.search(text):
        return "conversational"
    return "niche"


# ── Eval-set overlap protection ───────────────────────────────────

def normalise_for_dedup(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def load_eval_overlap(path: Path) -> set[str]:
    """Best-effort load of the eval-set English sources, normalised
    for exact-match dedup. Eval lives on Sebastian's laptop, copied
    to /tmp on Spark for the mining run. Returns empty set if missing."""
    if not path.exists():
        return set()
    out: set[str] = set()
    text = path.read_text()
    for m in re.finditer(r'"english"\s*:\s*"([^"]+)"', text):
        out.add(normalise_for_dedup(m.group(1)))
    return out


# ── Hashing user msisdn (for provenance, not for publishing) ──────

def hash_user(msisdn: str) -> str:
    return hashlib.sha256(f"mining:{msisdn}".encode()).hexdigest()[:12]


# ── Main mining loop ──────────────────────────────────────────────


def mine(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    eval_overlap = load_eval_overlap(Path(args.eval_path))
    print(f"# eval overlap protection: {len(eval_overlap)} normalised sources", file=sys.stderr)

    rejection_counts: Counter[str] = Counter()
    accepted_by_category: dict[str, list[dict]] = defaultdict(list)
    seen_normalised: set[str] = set()
    files = sorted(data_dir.glob("264*.json"))
    total_users = 0
    total_msgs_scanned = 0
    total_sentences_scanned = 0

    for p in files:
        try:
            history = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            rejection_counts["unreadable_file"] += 1
            continue
        if not isinstance(history, list):
            continue
        total_users += 1
        msisdn = p.stem
        u_hash = hash_user(msisdn)
        asst_msgs = [m for m in history if isinstance(m, dict) and m.get("role") == "assistant"]

        for msg_idx, m in enumerate(asst_msgs):
            content = m.get("content", "")
            if not isinstance(content, str):
                continue
            total_msgs_scanned += 1
            content = strip_markdown(content)
            sentences = split_sentences(content)
            for sent in sentences:
                total_sentences_scanned += 1
                ok, reason = acceptable(sent, args.min_words, args.max_words)
                if not ok:
                    rejection_counts[reason] += 1
                    continue
                cleaned = pii_sanitize(sent)
                if "[REDACTED" in cleaned:
                    rejection_counts["pii_present"] += 1
                    continue
                norm = normalise_for_dedup(cleaned)
                if not norm or norm in seen_normalised:
                    rejection_counts["duplicate_exact"] += 1
                    continue
                if norm in eval_overlap:
                    rejection_counts["overlap_eval_set"] += 1
                    continue
                seen_normalised.add(norm)
                cat = categorise(cleaned)
                accepted_by_category[cat].append({
                    "source_en": cleaned,
                    "category": cat,
                    "src_user_hash": u_hash,
                    "src_message_idx": msg_idx,
                })

    # Stats to stderr
    print("\n# === Mining summary ===", file=sys.stderr)
    print(f"# files scanned:        {len(files)}", file=sys.stderr)
    print(f"# users with history:   {total_users}", file=sys.stderr)
    print(f"# assistant msgs:       {total_msgs_scanned}", file=sys.stderr)
    print(f"# sentences seen:       {total_sentences_scanned}", file=sys.stderr)
    print(f"# accepted:             {sum(len(v) for v in accepted_by_category.values())}", file=sys.stderr)
    print(f"# rejection breakdown:", file=sys.stderr)
    for reason, count in rejection_counts.most_common():
        print(f"#   {reason:30s} {count}", file=sys.stderr)
    print(f"# acceptance by category:", file=sys.stderr)
    for cat, items in sorted(accepted_by_category.items(), key=lambda x: -len(x[1])):
        print(f"#   {cat:20s} {len(items)}", file=sys.stderr)

    # Output JSONL
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    next_id = 1
    with out_path.open("w") as f:
        for cat in ("conversational", "cv_jobs", "education", "translation",
                    "business", "daily_life", "niche"):
            for item in accepted_by_category.get(cat, []):
                item = {"id": next_id, **item}
                next_id += 1
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n# wrote {next_id - 1} sentences to {out_path}", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--eval-path", default=str(DEFAULT_EVAL_PATH))
    p.add_argument("--min-words", type=int, default=5)
    p.add_argument("--max-words", type=int, default=40)
    return p


if __name__ == "__main__":
    mine(_build_parser().parse_args())
