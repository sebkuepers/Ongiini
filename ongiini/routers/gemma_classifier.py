"""Owela ``Classifier`` adapter — Gemma 4 itself acts as the classifier.

This is the depth-aware evolution of ``ongiini/router.py``. The
prompt asks Gemma to emit one of five labels:

  SEARCH_SHALLOW  — one tool call is enough (single fact, single business)
  SEARCH_DEEP     — needs decomposition (comparisons, lists, multi-source)
  DOCS            — meta-question about Ongiini itself
  ADMIN           — action request (delete data, show usage)
  NONE            — general knowledge / chat

Parser accepts both new labels (SEARCH_SHALLOW / SEARCH_DEEP) and the
old bare ``SEARCH`` (degrades to SHALLOW) for backwards compatibility
during rollout.

Validation:
  - Held-out 4-way (NONE/ADMIN/DOCS/SEARCH) accuracy: still measured
    against ``ongiini/tests/router_eval_holdout.py`` after migration.
  - Depth (SHALLOW vs DEEP) accuracy: measured against an extended
    held-out set added in step 5b. Target ≥85% on depth, ≥96% on the
    4-way category.

Fail-safe: any timeout, parse failure, or network error yields
``ClassifierResult(verdict="NONE", depth="SHALLOW")``, which the policy
table maps to a sensible default — never breakage.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from owela import (
    ClassifierResult, DEPTH_DEEP, DEPTH_SHALLOW, InboundMessage,
    VERDICT_ADMIN, VERDICT_DOCS, VERDICT_NONE, VERDICT_SEARCH,
)

log = logging.getLogger("ongiini.routers.gemma")


# Application-side verdict extensions for the community-contribution
# loop. These live in ongiini/ (not owela/) per the framework anti-trap
# rule: Owela's verdict set stays small and product-agnostic. The
# policy table accepts any string as a verdict key, so wiring these
# through runtime.py needs nothing from the framework.
VERDICT_CONTRIB_INVITE   = "CONTRIBUTE_INVITE"
VERDICT_CONTRIB_DIALECT  = "CONTRIBUTE_DIALECT"
VERDICT_CONTRIB_NEXT     = "CONTRIBUTE_NEXT"
VERDICT_CONTRIB_SAVE     = "CONTRIBUTE_SAVE"
VERDICT_CONTRIB_SKIP     = "CONTRIBUTE_SKIP"
VERDICT_CONTRIB_DECLINE  = "CONTRIBUTE_DECLINE"
VERDICT_CONTRIB_STATS    = "CONTRIBUTE_STATS"


# Match common English and Afrikaans pronouns + reference words. When the
# current message contains one, we include the previous user message in
# the classifier prompt for pronoun resolution.
_PRONOUN_RE = re.compile(
    r"\b("
    r"he|his|him|she|her|hers|it|its|they|their|them|"  # EN pronouns
    r"this|that|these|those|"                            # EN references
    r"hy|sy|haar|hulle|hul|"                             # AF pronouns
    r"hierdie|daardie"                                   # AF references
    r")\b",
    re.IGNORECASE,
)


def _has_pronoun_or_reference(text: str) -> bool:
    return bool(_PRONOUN_RE.search(text))


# Five-label depth-aware classifier prompt. Extends the 4-way prompt
# from ongiini/router.py with the SEARCH_SHALLOW / SEARCH_DEEP
# split. Keep prefix-stable so vLLM's prefix cache fires on every call.
CLASSIFIER_PROMPT = """\
You classify requests for Ongiini, an AI helper for people in Namibia on WhatsApp.

Decide which of FIVE buckets the request falls in:

SEARCH_SHALLOW — needs the web AND the answer is a single fact, single
business name, single number, single price, single yes/no with brief
context. ONE search/lookup is enough.
Examples: "exchange rate today", "BIPA office hours", "is there a
Standard Bank in Walvis Bay", "what's the current malaria risk in
Oshakati", "who's the President of Namibia right now".

SEARCH_DEEP — needs the web AND the answer requires comparing options,
giving a list of 3+ items, looking up multiple data points, or
following up on initial results.
Examples: "compare home loan rates at 3 Namibian banks", "best places
to study computer science in Namibia", "what's happening with the
medicine shortage and what's being done", "give me 3 ideas for a small
business in Windhoek".

DOCS — the user is asking a SUBSTANTIVE question about Ongiini's policies,
hardware, or institutional context that genuinely needs a docs lookup:
pricing structure, privacy policy, terms of use, EU AI Act compliance, who
built it, the Common Intelligence Foundation, hardware specifics, monthly
token limits as a policy concept, how the data flows internally.

QUESTIONS ABOUT THE UNDERLYING AI MODEL also belong here — these have a
specific, factual answer in our docs (we run Gemma 4 26B locally, not a
Google/OpenAI API), and getting it wrong looks like a trust violation.
Route to DOCS:
  "which AI model are you?" / "are you Gemini?" / "are you ChatGPT?" /
  "are you GPT-4?" / "what LLM powers you?" / "are you powered by Google?" /
  "is this an API product?" / "do you call OpenAI?" / "watter KI is jy?"

DOCS is NOT for trivial capability checks like "do you speak X?" / "what
languages do you understand?" / "can you do Y?" / "what can you do?" —
those are answered by the system prompt and skill content already in the
model's context. They route to NONE. Sending them to DOCS triggers a
costly docs-lookup + critique-and-revise pipeline for a question whose
answer is a single sentence.

The distinction: "what can you help me with?" is capability → NONE.
"which AI model are you?" is identity / provenance → DOCS, because the
honest answer involves Gemma 4, open weights, local hosting, and the
absence of any Google API — facts the model doesn't reliably know about
itself without the docs lookup.

ADMIN — the user is requesting an ACTION on their own data or session:
"delete my data" / "forget everything" / "wis my data" / "vergeet alles";
"what do you remember about me?" / "wat onthou jy?" / "show me my data";
"how many tokens have I used?" / "hoeveel tokens het ek gebruik?"
These need actual tool execution (delete_my_data, whats_in_my_memory,
my_token_usage), NOT a docs lookup.

NONE — general knowledge (science, math, philosophy), generic how-to with no
local angle, emotional support, casual conversation, AND meta-questions
about THIS conversation whose answer is already in the conversation
history (asking for citations / sources / links that were cited in your
own earlier replies; asking you to recap or summarise what was just
discussed; "what did you tell me about X earlier"). These don't need a
tool call — the answer is in history.
Examples that route to NONE:
  "give me some links to your sources" (citations are in prior replies)
  "what were your sources for that?" (already cited above)
  "summarise what you just told me" (history has it)
  "remind me what the third option was" (already in history)
  "explain that more simply" (rephrase of an answer already given)

CONTRIBUTE_INVITE — the user is volunteering to help translate Oshiwambo
OR asking whether/when Ongiini supports Oshiwambo OR using Oshiwambo
for a real phrase (more than a one-word greeting like "Tangi" or "Ongiini").
Only emit this when contribute_state.recently_declined is FALSE — if
they declined recently, route normally to NONE / etc and answer their
real question.
Examples:
  "I want to help translate" → CONTRIBUTE_INVITE
  "Can I contribute Oshindonga translations?" → CONTRIBUTE_INVITE
  "Do you support Oshiwambo?" → CONTRIBUTE_INVITE (capability + offer)
  "Ondi pumbwa ekwafelo" (real Oshiwambo phrase) → CONTRIBUTE_INVITE

CONTRIBUTE_DIALECT — emit ONLY when contribute_state.pending_save is
FALSE AND contribute_state.dialect is "unknown" AND the user's
message names a dialect ("Oshindonga", "Oshikwanyama", "Ndonga",
"Kwanyama", "either", "both"). The user is answering the
dialect-question that was just asked.

CONTRIBUTE_SAVE — emit ONLY when contribute_state.pending_save is
TRUE (we just served an English sentence and are waiting for the
translation) AND the user's message looks like a translation
attempt — NOT an obvious skip / question / pivot. A short reply in
the contributor's dialect, even if you can't verify the language,
defaults to SAVE. Examples in pending_save state:
  "Ondi hala oku ku mona nawa" → CONTRIBUTE_SAVE
  "Tangi unene" → CONTRIBUTE_SAVE
  "Onawa ondi li nawa" → CONTRIBUTE_SAVE

CONTRIBUTE_SKIP — emit ONLY when contribute_state.pending_save is
TRUE AND the user's message is a clear English rejection / "I don't
know this one" / "send me a different sentence" / "skip" / "too
hard" / "easier please". Distinct from SAVE because the user is
explicitly NOT submitting a translation.

CONTRIBUTE_NEXT — emit ONLY when contribute_state.awaiting_followup
is TRUE (we just saved a translation and asked "want another?") AND
the user said yes / sure / another / one more / Tangi / Eewa / etc.

CONTRIBUTE_DECLINE — emit ONLY when contribute_state.awaiting_followup
OR contribute_state.pending_save is TRUE AND the user said no / done /
later / not now / enough / maybe later.

CONTRIBUTE_STATS — emit when the user asks how many translations have
been collected ("how many do you have?", "how much data?", "how's
the dataset doing?"). Can fire regardless of contribute_state.

The DOCS / ADMIN distinction: "what's your privacy policy" → DOCS
(asking about the document); "delete my data" → ADMIN (action on user state).
The NONE / DOCS distinction:
  - "give me your sources" → NONE (sources are in this conversation's history)
  - "do you speak Oshiwambo?" / "what languages do you support?" / "can you
    help with X?" / "what can you do?" → NONE (capability check, answer is
    in system prompt + skill content already)
  - "what's your privacy policy?" / "who built Ongiini?" / "what hardware
    do you run on?" → DOCS (substantive lookup needed)

Namibian cities (Windhoek, Walvis Bay, Oshakati, Swakopmund, Rundu, Katima
Mulilo) and institutions (BIPA, NamRA, Bank of Namibia, Ministry of Home
Affairs) imply Namibian context even when "Namibia" isn't explicitly said.

If previous conversation turns are shown for context, use them to:
  (a) resolve pronouns ("her", "his", "it", "they") and references
      like "this" or "that" in the current message;
  (b) detect whether the user is asking ABOUT something already
      established in the conversation (answer is in history → NONE)
      vs asking for new external info (needs SEARCH).
The current message is what you classify; the previous turns are
ONLY there to disambiguate what the user is talking about.

{context}{contribute_state}Current message: {user_text}

Reply with EXACTLY one of: SEARCH_SHALLOW, SEARCH_DEEP, DOCS, ADMIN, NONE,
CONTRIBUTE_INVITE, CONTRIBUTE_DIALECT, CONTRIBUTE_NEXT, CONTRIBUTE_SAVE,
CONTRIBUTE_SKIP, CONTRIBUTE_DECLINE, CONTRIBUTE_STATS.
"""


# Latency budget. Classifier should fail fast and degrade rather than
# block the user's actual reply. 2s is generous for the typical 85ms
# round-trip but tight enough that GPU contention doesn't bust p99.
_TIMEOUT_S = 2.0


# Order matters: longer labels first so we don't match SEARCH inside
# SEARCH_SHALLOW. CONTRIBUTE_* labels are all unambiguous prefixes of
# each other (CONTRIBUTE_SAVE doesn't contain CONTRIBUTE_SET-style
# collisions), so any order within that family works.
_LABEL_TOKENS = (
    "SEARCH_SHALLOW", "SEARCH_DEEP", "SEARCH",
    "CONTRIBUTE_INVITE", "CONTRIBUTE_DIALECT", "CONTRIBUTE_NEXT",
    "CONTRIBUTE_SAVE", "CONTRIBUTE_SKIP", "CONTRIBUTE_DECLINE",
    "CONTRIBUTE_STATS",
    "DOCS", "ADMIN", "NONE",
)


class GemmaClassifier:
    """Gemma-as-classifier via vLLM. See module docstring."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        *,
        client: AsyncOpenAI | None = None,
        timeout_s: float = _TIMEOUT_S,
        max_prev_chars: int = 500,
        short_msg_threshold_chars: int = 80,
    ) -> None:
        self.model_id = model_id
        self.timeout_s = timeout_s
        self.max_prev_chars = max_prev_chars
        self.short_msg_threshold_chars = short_msg_threshold_chars
        self._client = client or AsyncOpenAI(base_url=base_url, api_key="not-needed")

    async def classify(self, msg: InboundMessage) -> ClassifierResult:
        text = (msg.text or "").strip()
        if not text or len(text) < 3:
            return ClassifierResult(verdict=VERDICT_NONE, depth=DEPTH_SHALLOW)

        # Image-bearing turns skip the router. The current message's
        # informational content is in the IMAGE, not the text caption —
        # a caption like "what is this?" routinely misclassifies as DOCS.
        if msg.has_image:
            return ClassifierResult(verdict=VERDICT_NONE, depth=DEPTH_SHALLOW)

        prev_user, prev_assistant = self._extract_prev_pair(msg)
        needs_context = bool(prev_user or prev_assistant) and (
            _has_pronoun_or_reference(text)
            or len(text) < self.short_msg_threshold_chars
        )
        if needs_context:
            parts = []
            if prev_user:
                parts.append(f"Previous user message: {prev_user}")
            if prev_assistant:
                parts.append(f"Previous assistant reply: {prev_assistant}")
            context = "\n".join(parts) + "\n"
        else:
            context = ""

        # Contribution-loop state: only injected when the contributor
        # has any state worth knowing about (pending save, awaiting
        # followup, known dialect, recent decline). For users who've
        # never interacted with the contribute flow this stays empty,
        # so the prompt prefix stays cacheable for the bulk of
        # traffic.
        contribute_state = self._format_contribute_state(msg.user_id)

        try:
            resp = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self.model_id,
                    messages=[{
                        "role": "user",
                        "content": CLASSIFIER_PROMPT.format(
                            user_text=text,
                            context=context,
                            contribute_state=contribute_state,
                        ),
                    }],
                    temperature=0.0,
                    max_tokens=10,
                ),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            log.warning("classifier timed out after %ss — falling back to NONE", self.timeout_s)
            return ClassifierResult(verdict=VERDICT_NONE, depth=DEPTH_SHALLOW)
        except Exception as exc:                       # noqa: BLE001
            log.warning("classifier failed (%s) — falling back to NONE", exc)
            return ClassifierResult(verdict=VERDICT_NONE, depth=DEPTH_SHALLOW)

        billable_in, completion, cached = _billable(resp.usage)
        verdict, depth = self._parse(resp)

        return ClassifierResult(
            verdict=verdict,
            depth=depth,
            tokens_in=billable_in,
            tokens_out=completion,
            cached_tokens=cached,
        )

    # ----- internal helpers -----

    def _extract_prev_pair(self, msg: InboundMessage) -> tuple[str, str]:
        """Return the last (user, assistant) exchange from msg.history.

        v1.6: classifier needs BOTH prior turns to route "give me sources"-
        style questions correctly. The cited URLs and discussed entities
        live in the previous ASSISTANT reply, not the previous user
        question. Walking only the user side meant the classifier
        couldn't tell whether a "sources" question referred to something
        already discussed (→ NONE) vs a fresh external lookup (→ SEARCH).

        Returns ("", "") if neither role yields text. Empty strings are
        safe — the caller checks ``bool(prev_user or prev_assistant)``.
        """
        prev_user = ""
        prev_assistant = ""
        for h in reversed(msg.history):
            role = h.get("role")
            c = h.get("content", "")
            if isinstance(c, list):
                c = " ".join(
                    p.get("text", "")
                    for p in c
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            text = (c or "").strip()
            if not text:
                continue
            if role == "assistant" and not prev_assistant:
                prev_assistant = text[: self.max_prev_chars]
            elif role == "user" and not prev_user:
                prev_user = text[: self.max_prev_chars]
                break    # walk back from most-recent user; assistant came after
        return prev_user, prev_assistant

    def _format_contribute_state(self, user_id: str) -> str:
        """Render contribution-loop state for this contributor as a
        prompt block. ALWAYS returns a block (with default-false /
        unknown values for new contributors) so the classifier always
        sees consistent input — the prompt guard conditions on
        contribute_state.* read uniformly whether the user has an
        existing row or not.

        Soft-fail: any sqlite hiccup or missing hash salt yields a
        defaults-only block (everything false / unknown), so the
        prompt still has a well-formed state block but no contribute
        verdict will pass the state guards."""
        pending = False
        awaiting_followup = False
        dialect = "unknown"
        recently_declined = False
        try:
            from .. import contributions as _contrib  # lazy: import sqlite only when classifying
            h = _contrib.hash_msisdn(user_id)
            pending = bool(_contrib.get_pending_save(h))
            awaiting_followup = _contrib.is_awaiting_followup(h)
            status = _contrib.whoami(h)
            if status.startswith("known:"):
                dialect = status.split(":", 1)[1]
            recently_declined = _contrib.recently_declined(h)
        except Exception:
            pass
        return (
            "Contribute_state: "
            f"pending_save={str(pending).lower()}, "
            f"awaiting_followup={str(awaiting_followup).lower()}, "
            f"dialect={dialect}, "
            f"recently_declined={str(recently_declined).lower()}\n"
        )

    @staticmethod
    def _parse(resp: Any) -> tuple[str, str]:
        """Return (verdict, depth). Accepts the new SEARCH_SHALLOW /
        SEARCH_DEEP labels as well as the bare SEARCH (legacy → SHALLOW).
        Unparsable / unrecognised → ('NONE', 'SHALLOW')."""
        if not resp.choices:
            return VERDICT_NONE, DEPTH_SHALLOW
        raw = (resp.choices[0].message.content or "").strip().upper()
        for token in _LABEL_TOKENS:
            if token in raw:
                if token == "SEARCH_SHALLOW":
                    return VERDICT_SEARCH, DEPTH_SHALLOW
                if token == "SEARCH_DEEP":
                    return VERDICT_SEARCH, DEPTH_DEEP
                if token == "SEARCH":
                    # Legacy 4-way output — default to SHALLOW.
                    return VERDICT_SEARCH, DEPTH_SHALLOW
                if token == "DOCS":
                    return VERDICT_DOCS, DEPTH_SHALLOW
                if token == "ADMIN":
                    return VERDICT_ADMIN, DEPTH_SHALLOW
                if token == "NONE":
                    return VERDICT_NONE, DEPTH_SHALLOW
                if token == "CONTRIBUTE_INVITE":
                    return VERDICT_CONTRIB_INVITE, DEPTH_SHALLOW
                if token == "CONTRIBUTE_DIALECT":
                    return VERDICT_CONTRIB_DIALECT, DEPTH_SHALLOW
                if token == "CONTRIBUTE_NEXT":
                    return VERDICT_CONTRIB_NEXT, DEPTH_SHALLOW
                if token == "CONTRIBUTE_SAVE":
                    return VERDICT_CONTRIB_SAVE, DEPTH_SHALLOW
                if token == "CONTRIBUTE_SKIP":
                    return VERDICT_CONTRIB_SKIP, DEPTH_SHALLOW
                if token == "CONTRIBUTE_DECLINE":
                    return VERDICT_CONTRIB_DECLINE, DEPTH_SHALLOW
                if token == "CONTRIBUTE_STATS":
                    return VERDICT_CONTRIB_STATS, DEPTH_SHALLOW
        log.warning("classifier got un-parseable verdict %r — falling back to NONE", raw)
        return VERDICT_NONE, DEPTH_SHALLOW


def _billable(usage_obj: Any) -> tuple[int, int, int]:
    """Same logic as the model adapter — local copy avoids a cross-import
    just for one small helper."""
    if usage_obj is None:
        return 0, 0, 0
    prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
    cached = 0
    details = getattr(usage_obj, "prompt_tokens_details", None)
    if details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    billable_in = max(0, prompt_tokens - cached)
    return billable_in, completion_tokens, cached
