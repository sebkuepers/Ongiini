"""Welcome A/B/C test for click-to-WhatsApp ad arrivers.

Three variant copies replace the default FIRST-MESSAGE-WELCOME for
users who arrive via a Facebook ad (Meta sends a ``referral`` block
on the first inbound message in those cases). Organic / typed-greeting
arrivers keep the default welcome unchanged.

The hypothesis we're testing: which copy best lowers the bounce rate
(=user sends ≥2 messages within 48h of first contact) for ad traffic
specifically — the cohort that drove the previous bounce-rate concern.

  A — Minimal:    "How can I help?"
  B — Suggestion: "Try me with your CV, your homework, or something on your mind."
  C — Advantage:  "Free, your chats aren't sold to big tech or used to train
                   AI. Oshiwambo support in the making. How can I help?"

Routing is deterministic via hash(msisdn) % 3 — so a given user always
sees the same variant if they bounce-then-return. We log each
assignment to /data/welcome_experiment.log as JSONL for the analysis
script to compute per-variant conversion rates.

Detection of FB-ad arrival is done from the ``referral`` block on the
inbound message (see whatsapp.extract_messages). Falling back to
prefill-text detection is a deliberate non-goal: Meta's referral block
is the authoritative signal; matching on free-text prefills would
both miss legitimate ad arrivals (custom CTA text) and capture organic
users who happen to write a similar phrase.
"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import settings

log = logging.getLogger("ongiini.welcome_experiment")

LOG_PATH = settings.data_dir / "welcome_experiment.log"

VARIANTS = ("A", "B", "C")


# Variant copy — EN + AF. Used by the runtime to inject a system
# directive that tells the LLM exactly what to send after the
# disclosure line. Keep these in lockstep with the docstring above.
_BODY_EN = {
    "A": "How can I help?",
    "B": "Try me with your CV, your homework, or something on your mind.",
    "C": (
        "Free, your chats aren't sold to big tech or used to train AI. "
        "Oshiwambo support in the making. How can I help?"
    ),
}
_BODY_AF = {
    "A": "Hoe kan ek help?",
    "B": "Probeer my met jou CV, jou huiswerk, of enigiets wat in jou kop is.",
    "C": (
        "Gratis, jou geselsies word nie aan groot tegnologiebedrywe verkoop of "
        "gebruik om KI op te lei nie. Oshiwambo-ondersteuning is op pad. "
        "Hoe kan ek help?"
    ),
}


def is_fb_ad_arrival(raw_payload: dict[str, Any] | None) -> bool:
    """True iff this inbound carries a Meta click-to-WhatsApp referral
    block. The block is only present on the FIRST message after a user
    clicks an ad — re-engagements via the same ad inside the 72h window
    don't re-emit it. That is exactly the semantics we want for "this
    user is a fresh ad arrival."""
    if not raw_payload:
        return False
    return bool(raw_payload.get("referral"))


def assign_variant(msisdn: str) -> str:
    """Deterministic, sticky variant assignment. Hash chosen rather
    than random.choice() so a returning user (e.g. who bounced then
    re-clicked the same ad days later) sees the same variant — no
    cross-variant contamination at the user level."""
    h = hashlib.sha256(msisdn.encode("utf-8")).digest()
    bucket = h[0] % 3
    return VARIANTS[bucket]


def variant_directive(variant: str, language: str = "en") -> str:
    """Return the system-message text that overrides the static
    FIRST-MESSAGE-WELCOME for this variant. Injected by the memory
    provider on first-turn + FB-ad arrival.

    The directive is shaped to tell the model exactly what to send —
    not 'follow the welcome flow' but 'use this exact body text'. This
    keeps the variant text fixed across the experiment instead of
    letting the model paraphrase it (which would make the A/B/C signal
    impossible to read).
    """
    body = (_BODY_AF if language == "af" else _BODY_EN).get(variant, _BODY_EN["A"])
    return (
        "WELCOME OVERRIDE (this user is a Facebook-ad arrival, "
        f"experiment variant {variant}):\n"
        "After the standard FIRST-MESSAGE DISCLOSURE line, send EXACTLY "
        "this as your next paragraph — do not paraphrase, do not add "
        "follow-up questions, do not append a 'What's on your mind?' "
        "if the variant body doesn't include one:\n\n"
        f"  {body}\n\n"
        "Then stop. The user's actual question (if any) gets addressed "
        "in their next message — this first reply is the welcome only."
    )


def log_assignment(
    msisdn: str,
    variant: str,
    referral: dict[str, Any] | None,
    language: str,
) -> None:
    """Append one JSONL line per (msisdn, variant) assignment. Soft-
    fails: a log write failure cannot block the welcome reply.

    msisdn is hashed before persistence — same convention as
    broadcast.log and delivery_status.log (PII contract). Referral
    fields (source_id, headline) are kept raw so we can attribute
    conversion by ad creative if Sebastian wants to drill in later;
    those are ad-side metadata, not personal data.
    """
    try:
        h = hashlib.sha256(msisdn.encode("utf-8")).hexdigest()[:12]
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "msisdn_hash": h,
            "variant": variant,
            "language": language,
            "referral": {
                "source_id": (referral or {}).get("source_id", ""),
                "source_type": (referral or {}).get("source_type", ""),
                "headline": (referral or {}).get("headline", "")[:200],
                "ctwa_clid": (referral or {}).get("ctwa_clid", ""),
            } if referral else None,
        }
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception:
        log.exception("welcome_experiment log write failed for variant=%s", variant)
