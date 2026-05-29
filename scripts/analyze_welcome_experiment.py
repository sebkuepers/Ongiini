"""Analysis for the welcome A/B/C experiment.

Reads /data/welcome_experiment.log (one JSONL line per FB-ad-arrival
variant assignment) and correlates against each user's per-user memory
file to compute the conversion metric: did the user send ≥2 user-side
messages within 48h of their first contact?

Run on Spark:
  docker exec ongiini-webhook python3 /data/_analyze_welcome_experiment.py

Output: per-variant assigned / engaged / deep_engaged counts + rates,
plus a chi-square-style observation if any variant looks materially
different from the others.
"""
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA = Path("/data")
LOG = DATA / "welcome_experiment.log"

# Conversion windows
ENGAGEMENT_WINDOW = timedelta(hours=48)
ENGAGED_MIN_MSGS = 2     # ≥2 user msgs in 48h = "didn't bounce"
DEEP_MIN_MSGS = 4        # ≥4 user msgs = "real conversation"


def _parse_ts(s: str):
    try:
        # JSONL stores isoformat with 'Z' or offset
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def main():
    if not LOG.exists() or LOG.stat().st_size == 0:
        print("No /data/welcome_experiment.log yet — nothing to analyze.")
        return 0

    # Load assignments (one per first-contact FB-ad arrival)
    assignments = []
    for line in LOG.read_text().splitlines():
        try:
            assignments.append(json.loads(line))
        except Exception:
            continue
    print(f"loaded {len(assignments)} variant assignments from {LOG}")

    # Build per-msisdn-hash → first user-msg count within 48h.
    # We DON'T have raw msisdns in the log (privacy), so we walk every
    # per-user memory file, hash the msisdn, and look it up.
    import hashlib
    hash_to_user_msg_times = {}
    for p in sorted(DATA.glob("*.json")):
        msisdn = p.stem
        if not msisdn.isdigit() or len(msisdn) < 9:
            continue
        try:
            turns = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(turns, list):
            continue
        h = hashlib.sha256(msisdn.encode()).hexdigest()[:12]
        # We can't reconstruct exact per-turn timestamps from the JSON
        # file alone (turns don't carry ts). For the conversion test
        # we use turn COUNT as the proxy for engagement — combined with
        # the assignment ts as t0, and walk usage.log if needed for
        # precise ts. Simple approach first: count user-role turns.
        n_user_turns = sum(
            1 for t in turns
            if isinstance(t, dict) and t.get("role") == "user"
        )
        hash_to_user_msg_times[h] = n_user_turns

    # Score each assignment. We treat an assignment as "engaged" if
    # the user has ≥ENGAGED_MIN_MSGS lifetime user turns — the FB-ad
    # arrival IS their first turn, so anything ≥2 means they sent at
    # least one follow-up. (We can tighten this with usage.log ts
    # parsing later; this is the minimum-viable analysis.)
    per_variant = defaultdict(lambda: {
        "assigned": 0, "engaged": 0, "deep": 0, "bounced": 0,
    })
    for a in assignments:
        v = a.get("variant", "?")
        h = a.get("msisdn_hash", "")
        n_turns = hash_to_user_msg_times.get(h, 0)
        b = per_variant[v]
        b["assigned"] += 1
        if n_turns >= DEEP_MIN_MSGS:
            b["deep"] += 1
            b["engaged"] += 1
        elif n_turns >= ENGAGED_MIN_MSGS:
            b["engaged"] += 1
        else:
            b["bounced"] += 1

    # Render
    print()
    print(f"=== WELCOME A/B/C — FB-ad arrival conversion ===")
    print(f"window: lifetime user-turn count as engagement proxy")
    print(f"engaged: ≥{ENGAGED_MIN_MSGS} user msgs ever  ·  deep: ≥{DEEP_MIN_MSGS}")
    print()
    print(f"  {'var':4s}  {'assigned':>8s}  {'engaged':>8s}  {'eng%':>6s}  {'deep':>5s}  {'deep%':>6s}  {'bounced':>8s}")
    for v in ("A", "B", "C"):
        b = per_variant.get(v, {"assigned": 0, "engaged": 0, "deep": 0, "bounced": 0})
        n = b["assigned"]
        if n == 0:
            print(f"  {v:4s}  {0:>8d}  {0:>8d}  {'—':>6s}  {0:>5d}  {'—':>6s}  {0:>8d}")
            continue
        eng_pct = 100 * b["engaged"] / n
        deep_pct = 100 * b["deep"] / n
        print(f"  {v:4s}  {n:>8d}  {b['engaged']:>8d}  {eng_pct:5.1f}%  {b['deep']:>5d}  {deep_pct:5.1f}%  {b['bounced']:>8d}")

    # Highlight the leader on each metric — simple, no significance
    # testing at this sample size yet.
    print()
    best_eng = max(("A", "B", "C"),
                   key=lambda v: per_variant[v]["engaged"] / max(1, per_variant[v]["assigned"]))
    best_deep = max(("A", "B", "C"),
                    key=lambda v: per_variant[v]["deep"] / max(1, per_variant[v]["assigned"]))
    print(f"  leader by engagement rate: {best_eng}")
    print(f"  leader by deep-conv rate:  {best_deep}")

    # Sample size warning
    total = sum(per_variant[v]["assigned"] for v in ("A", "B", "C"))
    if total < 60:
        print(f"\n  ⚠️  total assignments={total} — sample too small to call winners.")
        print(f"      target: ≥60 per variant (~180 total) before deciding.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
