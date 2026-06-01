"""Owela ``Classifier`` adapter — Gemma 4 itself acts as the classifier.

The classifier asks Gemma to return a structured JSON object with a
verdict, a confidence band, a short reasoning trace, and a small
``extracted`` payload (named dialect, looks-like-translation, etc).

  verdict ∈ {
    SEARCH_SHALLOW, SEARCH_DEEP,       # web tool, one-shot vs decomposition
    DOCS,                              # docs lookup about Ongiini itself
    ADMIN,                             # delete/recall data, usage queries
    NONE,                              # general knowledge / chat
    CONTRIBUTE_INVITE, CONTRIBUTE_DIALECT, CONTRIBUTE_NEXT,
    CONTRIBUTE_SAVE,   CONTRIBUTE_SKIP,    CONTRIBUTE_DECLINE,
    CONTRIBUTE_STATS,                  # community-contribution loop
    OPT_OUT_BROADCAST,                 # stop receiving proactive nudges
  }

PolicyTable consumes ``verdict`` + ``depth`` (depth is derived: SEARCH_DEEP
→ DEEP, everything else → SHALLOW). Everything else returned by Gemma —
``confidence``, ``reasoning``, ``extracted``, ``state_relevance``,
``secondary_verdict`` — lands in ``ClassifierResult.attrs`` for hooks /
downstream consumers that want richer signal than a single label.

Why JSON: the previous design emitted one bare token and threw away
every signal Gemma had. It also leaned on state-as-gate rules in the
prompt (e.g. "emit SAVE only when pending_save=true") which got
brittle once state could be stale — a 23-hour-old pending_save would
still gate a SAVE on a "Yes, let's do that" button click. JSON output
+ TTL-rendered state (data, not rules) lets the model judge freshness
as part of its reasoning, and gives us per-call traces we can iterate
on.

Fail-safe: any timeout, JSON parse failure, or network error yields
``ClassifierResult(verdict="NONE", depth="SHALLOW")``, which the policy
table maps to a sensible default — never breakage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
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
VERDICT_OPT_OUT_BROADCAST = "OPT_OUT_BROADCAST"


# All verdicts we accept back from Gemma. Anything else falls through
# to NONE + SHALLOW. Kept as a set so membership is O(1).
_VALID_VERDICTS: frozenset[str] = frozenset({
    "SEARCH_SHALLOW", "SEARCH_DEEP", "SEARCH",  # legacy: bare SEARCH → SHALLOW
    "DOCS", "ADMIN", "NONE",
    VERDICT_CONTRIB_INVITE, VERDICT_CONTRIB_DIALECT, VERDICT_CONTRIB_NEXT,
    VERDICT_CONTRIB_SAVE, VERDICT_CONTRIB_SKIP, VERDICT_CONTRIB_DECLINE,
    VERDICT_CONTRIB_STATS,
    VERDICT_OPT_OUT_BROADCAST,
})


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


# Structured JSON classifier prompt. Definitions describe what each
# verdict means (i.e. what user intent it represents); the state block
# at the end carries data with timestamps (NOT gating rules). The model
# reads context + state + facts and makes the call. Kept prefix-stable
# so vLLM's prefix cache hits on every call.
CLASSIFIER_PROMPT = """\
You are the request classifier for Ongiini, a free WhatsApp AI helper for people
in Namibia. You look at one inbound message in its conversation context and
decide what kind of turn it is. The downstream policy table uses your verdict
to choose which tools the model gets, what loop shape to run, and which reply
style is appropriate.

The 13 verdicts you can choose from are:

SEARCH_SHALLOW — the question needs the web AND the answer is a single fact,
single business name, number, price, opening time, yes/no with brief context.
One search call is enough. Examples: "BoN exchange rate today", "BIPA office
hours", "is there a Standard Bank in Walvis Bay", "current malaria risk in
Oshakati".

SEARCH_DEEP — the question needs the web AND the answer requires comparing
options, listing 3+ items, looking up multiple data points, or following up on
initial results. Examples: "compare home loan rates at 3 banks", "best places
to study computer science in Namibia", "what's happening with the medicine
shortage and what's being done about it".

DOCS — the user is asking ABOUT Ongiini as a product/service. Any of:

  - The underlying model, hardware, hosting ("what AI are you?", "are you
    Gemini?", "are you ChatGPT?", "what LLM powers you?", "are you powered
    by Google?", "watter KI is jy?", "what hardware?", "where does it run?")
  - Privacy, data handling, deletion ("where is my data stored?", "do you
    keep my chats?", "is this private?", "can you forget what I told you
    yesterday?")
  - Pricing, allowances, policies ("what's my monthly token limit?", "how
    much does this cost?", "is there a paid version?", "what's the
    free-tier policy?")
  - Capabilities / coverage at the product level ("what languages do you
    support?", "do you support voice notes?", "what can Ongiini do?",
    "is this an API product?", "can I integrate this?")
  - How Ongiini works as a service ("hoe werk Ongiini?", "how does the
    Oshiwambo translation project work?", "who built this?", "what's the
    Common Intelligence Foundation?")
  - EU AI Act posture, terms of use, governance

The litmus test: "answering well requires facts about Ongiini-the-product
that aren't in the model's general knowledge or the system prompt." When in
doubt between DOCS and NONE for a question that mentions Ongiini, the model,
the data, the privacy, or the service — prefer DOCS. The docs lookup is cheap
and gives the user an accurate answer instead of a vague one.

DOCS is NOT for casual "Tangi"/"hello"/"how are you" or for generic chat
that happens to be in a session with Ongiini. The question has to be ABOUT
Ongiini, not merely WITH Ongiini.

ADMIN — the user is requesting an action on their own data or session:
"delete my data" / "forget everything" / "wis my data" / "vergeet alles" /
"what do you remember about me?" / "wat onthou jy?" / "show me my data" /
"how many tokens have I used?" / "hoeveel tokens het ek gebruik?" — these
need a tool call (delete_my_data, whats_in_my_memory, my_token_usage), not a
docs lookup.

NONE — general knowledge (science, math, philosophy), generic how-to with no
local angle, emotional support, casual conversation, AND meta-questions about
THIS conversation whose answer is already in the history (asking for the
sources you cited earlier, asking for a summary of what was discussed,
re-asking for an option you already presented). These don't need a tool call
— the answer is in the history. NONE is also the default for anything that
doesn't fit the other 12 buckets.

CONTRIBUTE_INVITE — the user is volunteering to help translate Oshiwambo OR
asking whether/when Ongiini supports Oshiwambo OR using Oshiwambo for a real
phrase (more than a one-word greeting like "Tangi" or "Ongiini"). If the
state block shows recently_declined=true, lean toward NONE — they passed
recently and a fresh invite would be nagging.

CONTRIBUTE_DIALECT — the user is naming which Oshiwambo dialect they speak,
typically in response to the bot asking. Be generous in detecting dialect
choice: any clear mention of Oshindonga / Oshikwanyama / Ndonga / Kwanyama
counts, plus "either" / "both" / "any" / "I speak both" all count. The
contribute-state block tells you whether we've already asked the dialect
question (pending_save and dialect fields).

CONTRIBUTE_NEXT — the user wants the next translation sentence. The previous
bot message should have offered another sentence ("want to try another?" /
"say yes for the next one"), and the user's message agrees (yes / sure /
another / one more / Tangi unene / Eewa). A "yes" only counts as CONTRIBUTE_
NEXT when the previous bot turn was about more translations — if it had
moved on to a different topic, "yes" is about THAT topic.

CONTRIBUTE_SAVE — the user just typed a translation we asked them for. Decide
this by reading the conversation: did the previous assistant message end
with a request like "How would you say this in Oshindonga?" or a quoted
English sentence to translate? Does the current user message look like a
plausible answer to that specific request — Oshiwambo phonology (ondi,
ohandi, okwa, shoka, iikulya, uunona, noun-class prefixes), not a generic
English yes/no/click? If the conversation has moved on since pending_save
was set, the state is stale and this is NOT a SAVE — even if pending_save
is technically still set on the contributor row.

CONTRIBUTE_SKIP — the user is clearly passing on the current English
sentence we asked them to translate ("skip", "I don't know this one", "send
me a different one", "too hard"). Context-first: the previous assistant
message must actually have been a translation request.

CONTRIBUTE_DECLINE — the user is ending the translation flow ("no thanks",
"done for today", "later", "enough for now"). Only emit when the
conversation makes it clear they're declining translations specifically —
a bare "no" outside the translation context is something else.

CONTRIBUTE_STATS — the user is asking how many translations have been
collected ("how many do you have?", "how's the dataset doing?", "how many
contributors?"). Fires regardless of contribute state.

OPT_OUT_BROADCAST — the user is asking to stop receiving proactive update or
announcement messages from us ("stop messages", "unsubscribe", "opt out",
"no more notifications", "stop boodskappe"). NOT for "delete my data" (that
is ADMIN) and NOT for "stop talking to me right now" (the user can just stop
replying — that is NONE).

Worked SEARCH examples — these are the kinds of questions that need a
web tool. Default-to-NONE on these costs the user a real answer.

  User: "are there any datacenters in Namibia?"  → SEARCH_SHALLOW.
  User: "how do I apply for a Namibian passport?"  → SEARCH_SHALLOW.
  User: "internet cafe in Tsumeb"  → SEARCH_SHALLOW. A named local
    service in a Namibian town — the model can't know what's currently
    operating there.
  User: "TransNamib train schedule Windhoek to Walvis Bay"  → SEARCH_SHALLOW.
    Current Namibian transport schedule — needs the web.
  User: "how do I create a PDF"  → SEARCH_SHALLOW. Generic how-to with
    no local angle is still SEARCH — the model could improvise, but a
    concrete answer beats general advice.
  User: "hoe registreer ek 'n besigheid by BIPA?"  → SEARCH_SHALLOW.
    Afrikaans, but BIPA is Namibian — that's enough.
  User: "best places to study computer science in Namibia"  → SEARCH_DEEP.
    Comparing 3+ institutions, multi-source.

Worked DOCS examples — questions about Ongiini-the-product itself.

  User: "what AI model are you running on"  → DOCS.
  User: "where is my data stored?"  → DOCS.
  User: "what languages do you support?"  → DOCS. Product-level capability
    question — answer it from the docs, not from the system prompt.
  User: "is this an API product"  → DOCS.
  User: "are you powered by Google"  → DOCS. Identity / provenance.
  User: "can you forget what I told you yesterday"  → DOCS. Data handling
    + memory / deletion.
  User: "hoe werk Ongiini?"  → DOCS. Same DOCS verdict in Afrikaans.
  User: "how does the Oshiwambo translation project work?"  → DOCS.

Worked CONTRIBUTE_* examples — these have caused production mistakes.

  Previous bot: 'How would you say this in Oshindonga? "The weather is
  beautiful today."'
  User: "Onkalo yombepo ombwaanawa nena"
  → CONTRIBUTE_SAVE. Looks like an answer to that specific request in
  Oshiwambo phonology.

  Previous bot: (a different topic — bot answered a question about BIPA an
  hour ago and moved on)
  User: "Yes, let's do that"
  → NONE. The state block may still show pending_save from earlier, but the
  conversation has moved on; "Yes, let's do that" is a button-style click on
  whatever the bot just said, not a translation.

  Previous bot: asked for a translation.
  User: "What does that even mean?"
  → NONE. Clarification question, not an attempt.

  Previous bot: "Want to try another one? Say yes for the next one."
  User: "Yes"
  → CONTRIBUTE_NEXT. The bot just offered another sentence; the user
  accepts.

  Previous bot: asked for a translation OR offered the next sentence.
  User: "Oshindonga"
  → CONTRIBUTE_DIALECT only if we hadn't yet recorded the user's dialect.
  Otherwise it's likely a sideways comment routing to NONE.

Tie-breakers when you're unsure:

  - For CONTRIBUTE_SAVE / CONTRIBUTE_DECLINE / OPT_OUT_BROADCAST, demand
    high confidence — a wrong save pollutes the dataset. When in doubt
    on a state-changing verdict, prefer NONE.
  - For SEARCH and DOCS, the bias goes the OTHER way. These are
    information-retrieval verdicts: a missed SEARCH leaves the user with
    a vague answer; a missed DOCS gives them generic chat when they
    asked about Ongiini. When the message could plausibly need the web
    (current/local/specific facts) or could plausibly be about
    Ongiini-the-product — prefer the tool verdict over NONE.
  - The state block in the prompt is data, not a rule. Older timestamps
    (1h+) are progressively staler; a 23-hour-old pending_save almost
    certainly doesn't apply to a fresh button click.
  - Namibian cities (Windhoek, Walvis Bay, Oshakati, Swakopmund, Rundu,
    Katima Mulilo, Tsumeb, Eros, Klein Windhoek) and institutions (BIPA,
    NamRA, Bank of Namibia, TransNamib, Ministry of Home Affairs) imply
    Namibian context even when "Namibia" isn't explicitly said. These
    are almost always SEARCH, very rarely NONE.

Output schema — return ONE JSON object, no surrounding prose:

{{
  "verdict":    one of the 13 labels above,
  "confidence": "high" | "medium" | "low",
  "reasoning":  1-2 sentences explaining what in the message drove the verdict
}}

{contribute_state}{facts}{context}Current message:
{user_text}
"""

# Order rationale: state block is ALWAYS emitted with consistent
# field set (defaults for new contributors), so it slots BEFORE the
# context block which is conditionally injected. That keeps the
# prefix-cacheable prefix as long as possible for stateful users
# (most expensive callers — mem0 hop + state lookups). Context
# breaks the cache only at the boundary where it appears, not
# 200 tokens earlier.


# Latency budget. The JSON output is up to 500 tokens vs. the old
# single-token reply, so generation time scales linearly. Live test
# on Spark (2026-05-30) saw consistent 3s-timeout under realistic
# load. 8s gives comfortable headroom while staying well under the
# 25s WhatsApp typing-window cap. The prefix cache keeps the input
# side cheap on every call regardless of timeout.
_TIMEOUT_S = 8.0

# Max output tokens for the JSON reply. Trimmed schema (verdict +
# confidence + reasoning, no extracted/state_relevance/secondary) needs
# ~30 tokens scaffolding + 60-80 tokens of reasoning text = ~120 total.
# 200 leaves headroom; truncation would corrupt the JSON and fall
# through to NONE — we surface finish_reason=="length" so it's
# monitorable from logs.
_MAX_OUTPUT_TOKENS = 200


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

        # Read the contributor's state ONCE. It feeds two decisions:
        #  - the state block we render into the prompt;
        #  - whether to force-include prior turns even when the message
        #    looks self-contained (any active contribute state means
        #    prev_assistant is exactly the context needed to judge
        #    freshness vs staleness).
        state = self._read_contribute_state(msg.user_id)
        has_active_state = (
            state["pending_save"] is not None
            or state["awaiting_followup"]
            or state["dialect"] != "unknown"
        )

        prev_user, prev_assistant = self._extract_prev_pair(msg)
        if has_active_state and (prev_user or prev_assistant):
            include_context = True
        else:
            include_context = bool(prev_user or prev_assistant) and (
                _has_pronoun_or_reference(text)
                or len(text) < self.short_msg_threshold_chars
            )

        if include_context:
            parts = []
            if prev_user:
                parts.append(f"Previous user message: {prev_user}")
            if prev_assistant:
                parts.append(f"Previous assistant reply: {prev_assistant}")
            context = "\n".join(parts) + "\n\n"
        else:
            context = ""

        contribute_state = self._format_state_with_ttls(state)

        # Mem0 facts only when state is non-empty. Stateless turns get
        # routed without the mem0 round-trip — keeps the bulk of traffic
        # fast and the prompt prefix small. format_relevant returns ""
        # when there are no facts; we add the trailing blank line only
        # when there's actually content.
        facts = ""
        if has_active_state:
            facts = self._format_recent_facts(msg.user_id)
            if facts:
                facts = facts + "\n\n"

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
                            facts=facts,
                        ),
                    }],
                    temperature=0.0,
                    max_tokens=_MAX_OUTPUT_TOKENS,
                    response_format={"type": "json_object"},
                ),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            log.warning("classifier timed out after %ss — falling back to NONE", self.timeout_s)
            return ClassifierResult(verdict=VERDICT_NONE, depth=DEPTH_SHALLOW)
        except Exception as exc:                       # noqa: BLE001
            log.warning("classifier failed (%s) — falling back to NONE", exc)
            return ClassifierResult(verdict=VERDICT_NONE, depth=DEPTH_SHALLOW)

        # Detect output truncation. When the model hits _MAX_OUTPUT_TOKENS
        # mid-JSON, json.loads silently fails and we fall through to NONE.
        # Surfacing it here as a warning makes the truncation rate
        # monitorable from logs without changing user-facing behaviour.
        try:
            finish_reason = resp.choices[0].finish_reason if resp.choices else ""
        except Exception:
            finish_reason = ""
        if finish_reason == "length":
            log.warning(
                "classifier output truncated at max_tokens=%d — JSON parse will fail "
                "and route to NONE. Consider bumping _MAX_OUTPUT_TOKENS.",
                _MAX_OUTPUT_TOKENS,
            )

        billable_in, completion, cached = _billable(resp.usage)
        verdict, depth, attrs = self._parse(resp)

        return ClassifierResult(
            verdict=verdict,
            depth=depth,
            tokens_in=billable_in,
            tokens_out=completion,
            cached_tokens=cached,
            attrs=attrs,
        )

    # ----- internal helpers -----

    def _extract_prev_pair(self, msg: InboundMessage) -> tuple[str, str]:
        """Return the last (user, assistant) exchange from msg.history.

        v1.6: classifier needs BOTH prior turns to route "give me sources"-
        style questions correctly. The cited URLs and discussed entities
        live in the previous ASSISTANT reply, not the previous user
        question.

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

    def _read_contribute_state(self, user_id: str) -> dict[str, Any]:
        """Return a structured snapshot of the contributor's state, with
        raw timestamps preserved so the renderer can decide how to age
        them. Soft-fails to defaults on any sqlite hiccup."""
        pending_save: dict[str, Any] | None = None
        awaiting_followup = False
        dialect = "unknown"
        recently_declined = False
        try:
            from .. import contributions as _contrib   # lazy import to avoid sqlite cost when unused
            h = _contrib.hash_msisdn(user_id)
            pending_save = _contrib.get_pending_save(h)
            awaiting_followup = _contrib.is_awaiting_followup(h)
            status = _contrib.whoami(h)
            if status.startswith("known:"):
                dialect = status.split(":", 1)[1]
            recently_declined = _contrib.recently_declined(h)
        except Exception:
            pass
        return {
            "pending_save":      pending_save,        # None or {task_id, dialect, set_at}
            "awaiting_followup": awaiting_followup,
            "dialect":           dialect,             # "unknown" | "Oshindonga" | "Oshikwanyama"
            "recently_declined": recently_declined,
        }

    def _format_state_with_ttls(self, state: dict[str, Any]) -> str:
        """Render the contribute_state block. Each item carries an
        explicit age ("set 23h ago" / "set 8m ago") when we have a
        timestamp, so the model can judge staleness without us hard-
        coding rules. Always emitted (with all-default values for new
        contributors) so the prompt prefix stays cacheable."""
        # pending_save
        ps = state["pending_save"]
        if ps:
            age = _format_age(ps.get("set_at"))
            age_suffix = f", {age}" if age else ""
            pending_line = (
                f"pending_save:       task_id={ps.get('task_id')}, "
                f"dialect={ps.get('dialect')}{age_suffix}"
            )
        else:
            pending_line = "pending_save:       none"

        # awaiting_followup — the contributions module already enforces
        # a 30-minute window before this returns true, so the boolean
        # carries an implicit TTL of its own. We just print true/false.
        awaiting_line = (
            f"awaiting_followup:  {str(state['awaiting_followup']).lower()}"
        )

        dialect_line = f"dialect:            {state['dialect']}"
        decline_line = (
            f"recently_declined:  {str(state['recently_declined']).lower()}"
        )

        return (
            "Contribute state (data with timestamps — judge staleness yourself):\n"
            f"  {pending_line}\n"
            f"  {awaiting_line}\n"
            f"  {dialect_line}\n"
            f"  {decline_line}\n\n"
        )

    def _format_recent_facts(self, user_id: str, limit: int = 3) -> str:
        """Up to ``limit`` short mem0 facts for the user, rendered as a
        labelled block. Used only when the contributor has active state,
        so we pay the mem0 hop only on the small fraction of traffic
        where it might disambiguate. Soft-fails to empty string.

        Mem0 failure is logged at INFO with a per-process once-only guard
        so a silent mem0 outage in prod doesn't hide the loss of the
        disambiguation signal on exactly the users that need it. After
        the first warning, individual failures drop to DEBUG to avoid
        noise."""
        try:
            from ..memory import long_term     # lazy: avoid loading mem0 at import
            facts = long_term.list_all(user_id) or []
        except Exception as exc:
            if not getattr(self, "_mem0_read_warned", False):
                log.warning(
                    "classifier mem0 read failed for %s: %s — disambiguation "
                    "facts will be skipped until mem0 recovers. (Subsequent "
                    "failures logged at DEBUG.)", user_id, exc,
                )
                self._mem0_read_warned = True
            else:
                log.debug("classifier mem0 read failed for %s: %s", user_id, exc)
            return ""
        if not facts:
            return ""
        rendered: list[str] = []
        for f in facts[:limit]:
            if not isinstance(f, dict):
                continue
            text = (f.get("memory") or f.get("text") or "").strip()
            if not text:
                continue
            # Cap each fact so a long [QUOTE] can't blow past the prompt
            # budget for what's a disambiguation hint, not a full memory
            # surface.
            if len(text) > 160:
                text = text[:160] + "…"
            rendered.append(f"- {text}")
        if not rendered:
            return ""
        return "Recent facts about this user (for disambiguation):\n" + "\n".join(rendered)

    @staticmethod
    def _parse(resp: Any) -> tuple[str, str, dict[str, Any]]:
        """Parse Gemma's JSON reply into (verdict, depth, attrs).

        On any parse failure / unrecognised verdict, returns the
        fail-safe ``(NONE, SHALLOW, {})`` — same posture as the old
        token parser. Logs a warning so it's visible in traces."""
        if not resp.choices:
            return VERDICT_NONE, DEPTH_SHALLOW, {}

        raw = (resp.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            log.warning("classifier got unparseable JSON %r — falling back to NONE", raw[:200])
            return VERDICT_NONE, DEPTH_SHALLOW, {}

        if not isinstance(parsed, dict):
            log.warning("classifier JSON was not an object: %r — falling back to NONE", raw[:200])
            return VERDICT_NONE, DEPTH_SHALLOW, {}

        # Gemma occasionally emits non-string verdict values (int, bool,
        # null, list) — strip()/upper() would AttributeError. Coerce
        # defensively: only treat as a valid verdict when it's an actual
        # string in the allowed set.
        raw_verdict_field = parsed.get("verdict")
        if not isinstance(raw_verdict_field, str):
            log.warning(
                "classifier got non-string verdict %r — falling back to NONE",
                raw_verdict_field,
            )
            return VERDICT_NONE, DEPTH_SHALLOW, {}
        raw_verdict = raw_verdict_field.strip().upper()
        if raw_verdict not in _VALID_VERDICTS:
            log.warning(
                "classifier got unrecognised verdict %r — falling back to NONE",
                raw_verdict,
            )
            return VERDICT_NONE, DEPTH_SHALLOW, {}

        # Verdict → (verdict_for_policy, depth) mapping.
        if raw_verdict == "SEARCH_SHALLOW":
            verdict, depth = VERDICT_SEARCH, DEPTH_SHALLOW
        elif raw_verdict == "SEARCH_DEEP":
            verdict, depth = VERDICT_SEARCH, DEPTH_DEEP
        elif raw_verdict == "SEARCH":
            verdict, depth = VERDICT_SEARCH, DEPTH_SHALLOW   # legacy degrade
        elif raw_verdict == "DOCS":
            verdict, depth = VERDICT_DOCS, DEPTH_SHALLOW
        elif raw_verdict == "ADMIN":
            verdict, depth = VERDICT_ADMIN, DEPTH_SHALLOW
        elif raw_verdict == "NONE":
            verdict, depth = VERDICT_NONE, DEPTH_SHALLOW
        else:
            # All CONTRIBUTE_* and OPT_OUT_BROADCAST pass through as-is.
            verdict, depth = raw_verdict, DEPTH_SHALLOW

        # Everything else flows into attrs for downstream consumers
        # (hooks, future policies that gate on confidence, eval logs).
        attrs: dict[str, Any] = {
            "verdict_raw":       raw_verdict,
            "confidence":        parsed.get("confidence"),
            "reasoning":         parsed.get("reasoning"),
            "extracted":         parsed.get("extracted") if isinstance(parsed.get("extracted"), dict) else {},
            "state_relevance":   parsed.get("state_relevance"),
            "secondary_verdict": parsed.get("secondary_verdict"),
        }
        return verdict, depth, attrs


def _format_age(iso_ts: str | None) -> str:
    """Render a stored ISO-8601 timestamp as a human age like 'set 8m
    ago' / 'set 23h ago' / 'set 5d ago'. Returns '' if the timestamp is
    missing or unparseable — caller decides whether to elide the
    suffix entirely."""
    if not iso_ts:
        return ""
    try:
        ts = datetime.fromisoformat(iso_ts)
    except (ValueError, TypeError):
        return ""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - ts
    seconds = delta.total_seconds()
    if seconds < 0:
        return "set just now"
    if seconds < 60:
        return "set just now"
    if seconds < 3600:
        return f"set {int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"set {int(seconds // 3600)}h ago"
    return f"set {int(seconds // 86400)}d ago"


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
