"""Reviewer — self-critique + revise pass before the reply ships.

After the act loop produces a draft, the executor calls the Reviewer
(if policy.enable_critique is True). The Reviewer asks Gemma — using
a separate, structured prompt — to judge the draft on six dimensions
and either pass it through or flag it for a single one-shot revise.

This is the biggest single quality unlock from the v0 → v1 plan: it
catches confabulation BEFORE the draft hits WhatsApp. The hackathon
"October 2025" reply that prompted the date/time anchor is exactly
the failure mode this pass catches.

Cost model:
  - On every turn where policy.enable_critique = True, the critique
    LLM call adds ~1.0-1.5s (small prompt, ~50-100 completion tokens).
  - On turns where the critique returns REVISE (estimated 15-25% of
    SEARCH-grounded turns), the revise call adds ~3-6s.
  - Critique is capped at 1 revise — never a loop. The revised draft
    is auto-accepted; no second critique pass.

Soft-fail contract: any timeout, parse failure, or other error in
the critique returns CritiqueStep(verdict="PASS") — the draft ships
unchanged. We'd rather miss an occasional bad reply than block a
good one because the reviewer LLM blipped.

Prompt design adapted from the Reflexion / LangGraph Generate-Reflect
pattern and open_deep_research's reflection prompt, tightened for the
WhatsApp shape — output is structured (one line per dimension + a
final VERDICT line) so the parser is robust against minor formatting
drift.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from openai import AsyncOpenAI

from owela import (
    CritiqueStep, InboundMessage, Policy, ReviseStep, Step, ToolStep,
)

log = logging.getLogger("ongiini.reviewer")


# How much of each tool result we feed to the critique.
#
# v1.3 bumped 2000 → 8000. With multi-query fan-out + advanced
# /search (include_raw_content + chunks_per_source=3) + advanced
# /extract, tool results carry substantially more content than they
# used to. The reviewer needs to see enough of each result to verify
# grounding — at 2000 chars per result, critique would routinely
# miss the actual cited content from a fetched page. Cost: ~4-6K
# extra tokens per critique call, well within Gemma 4's context.
_TOOL_RESULT_TRUNCATION = 8000


_CRITIQUE_PROMPT = """A WhatsApp helper for users in Namibia ("Ongiini") drafted the reply
below. You are the reviewer — judge whether the draft is good enough
to ship to the user, OR whether it needs a revise.

USER ASKED:
{user_question}

TOOLS THAT FIRED THIS TURN: {tool_names}

TOOL RESULTS (truncated, in the order they fired):
{tool_results}

DRAFTED REPLY:
{draft_reply}

Critique on EXACTLY these five dimensions, in this order. Be honest —
your job is to catch problems, not be polite. Format each line as
either "OK" or "FAIL: <one short reason>". Do NOT add commentary
beyond what's asked.

(Note: WhatsApp formatting is handled deterministically by the
transport layer. Do NOT critique formatting — focus only on
substance, grounding, and language.)

1. Answers the user's actual question (not a tangent or partial answer):
2. Every Namibia-specific factual claim in the reply — a business
   name, a service offering, a price, a contact method, a date — must
   be DIRECTLY VISIBLE in the TOOL RESULTS above. If the reply
   infers something the tool results don't explicitly state (even if
   the inference is reasonable), that's a FAIL. Quote a snippet from
   the tool results that supports each specific claim mentally as
   you read; if you can't quote it, it's confabulation:
3. If web_search / fetch_url / fetch_urls fired AND the reply makes
   specific factual claims drawn from those results, at least one
   DEEP URL is cited (prefixed "— source:" on its own line, with a
   path — not just a publication homepage). A SINGLE deep URL at the
   bottom is sufficient when the reply builds on the same source
   throughout. PASS replies that primarily reuse facts established
   in earlier turns of THIS conversation — those don't need
   re-citation:
4. If the tool results were thin or empty, the reply admits this
   plainly instead of inventing specifics:
5. Language matches the user's (EN or AF) — same language they asked
   in:

End with one line in the exact form:

VERDICT: PASS

OR:

VERDICT: REVISE
"""


_REVISE_PROMPT = """The reviewer flagged your draft. Here's what they said:

{reasons}

Rewrite the draft below addressing the reviewer's feedback. Keep the
same language as the original (EN or AF). Keep it warm and
conversational — same tone as before. Maintain the citation rule:
every factual claim from a search must be followed by "— source: <deep URL>"
on its own line.

Do NOT add commentary about the revision. Reply with the revised text only.

AVAILABLE DEEP URLS for citation (use these — don't invent any):
{available_urls}

Tool results from this turn (for context, in case a claim needs to
be re-grounded):
{tool_results}

Original draft:
{draft}

User's original question:
{user_question}
"""


# Anchored to the start of a line + case-insensitive multi-line so a
# verbatim echo of the prompt's "VERDICT: PASS / VERDICT: REVISE"
# instruction block doesn't trigger a false positive. We also take the
# LAST match in the output so a model that explains its reasoning
# verbatim ("first I'll say PASS, then on reflection REVISE...")
# converges on its final word.
_VERDICT_RE = re.compile(r"(?mi)^\s*VERDICT:\s*(PASS|REVISE)\s*$")

# Latency budgets. Critique should fail fast — if it doesn't come
# back in time, we ship the draft unchanged. v1.3.1 bumped revise
# 12→20s after live traces showed the 12s budget routinely timed out
# on longer drafts. v1.5 bumped critique 6→10s after a separate
# investigation: with v1.3.1's 24K-char tool block aggregate cap,
# the critique prompt itself is ~6-8K tokens. Gemma 4 26B reading
# that + generating 5-dim verdicts routinely took 5-6s, and 42% of
# recent production critiques hit the 6s ceiling. When critique
# v1.6.2: critique + revise no longer wrap the model call in
# `asyncio.wait_for`. Earlier versions had 10s/20s soft-fail budgets;
# under production load they fired ~7% of the time on revise, silently
# shipping the un-revised confabulated draft instead of waiting for a
# fix. Two failure modes were stacked:
#   - critique-timeout → PASS verdict (no quality check fires)
#   - revise-timeout   → original draft ships (the catch was wasted)
# Production data: revise_rate 67%, search_pass_rate 33% — the rescue
# loop is the main pipeline running twice. Cutting it short was making
# things worse, not safer. The AsyncOpenAI client's default 600s
# timeout is the real backstop; the soft budget below is observation-
# only and never kills the call.
_PERF_BUDGET_S = 30.0    # log a warning if critique or revise exceeds this

# Aggregate cap over the multi-step tool block fed to critique/revise.
# Per-step truncation is _TOOL_RESULT_TRUNCATION (8000); with multi-
# query fan-out + fetch_urls a turn can have 5+ ToolSteps, which
# without an aggregate cap meant ~40K chars of tool block per critique
# call. The aggregate cap caps the total at this many chars, dropping
# whole later steps (preserves at least the FIRST search + fetch
# bodies which carry the bulk of grounding).
_TOOL_BLOCK_AGGREGATE_CAP = 24000


class OngiiniReviewer:
    """Calls Gemma with critique + (conditionally) revise prompts.

    Constructed with the vLLM endpoint + model id; tests inject a
    fake AsyncOpenAI client. The critique prompt's prefix is
    byte-stable across requests so vLLM's prefix cache hits.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        *,
        client: AsyncOpenAI | None = None,
        perf_budget_s: float = _PERF_BUDGET_S,
    ) -> None:
        self.model_id = model_id
        # ``perf_budget_s`` is observation-only: critique/revise log a
        # warning if they exceed it but the call is never killed. v1.6.2
        # removed the kill-and-soft-fail timeouts that were causing
        # silent quality regressions under load.
        self.perf_budget_s = perf_budget_s
        self._client = client or AsyncOpenAI(base_url=base_url, api_key="not-needed")

    async def critique(
        self,
        msg: InboundMessage,
        draft: str,
        prior_steps: list[Step],
        policy: Policy,
    ) -> CritiqueStep:
        started = time.monotonic()
        step = CritiqueStep(started_at=started)

        user_question = (msg.text or "").strip()
        if not draft.strip() or not user_question:
            # Empty draft or question — nothing to critique. The
            # transport will supply its own fallback message anyway.
            step.verdict = "PASS"
            step.ended_at = time.monotonic()
            return step

        tool_names, tool_block = self._tool_summary(prior_steps)

        try:
            resp = await self._client.chat.completions.create(
                model=self.model_id,
                messages=[{
                    "role": "user",
                    "content": _CRITIQUE_PROMPT.format(
                        user_question=user_question,
                        tool_names=tool_names or "(none)",
                        tool_results=tool_block or "(no tool calls this turn)",
                        draft_reply=draft,
                    ),
                }],
                temperature=0.0,
                max_tokens=400,
            )
        except Exception as exc:                       # noqa: BLE001 — soft-fail
            # Real failures (network drop, model crash) still flow
            # through here. We can't critique without a model response,
            # so we PASS the draft and log loudly. This is NOT a budget
            # cap — the AsyncOpenAI client's default 600s timeout is the
            # backstop. We never kill a critique call for being slow.
            log.warning("critique failed (%s) — shipping draft unchanged", exc)
            step.verdict = "PASS"
            step.attrs["error"] = str(exc)
            step.ended_at = time.monotonic()
            return step

        elapsed = time.monotonic() - started
        if elapsed > self.perf_budget_s:
            log.warning(
                "critique exceeded perf budget: %.1fs > %.1fs (not killed; "
                "monitor revise_rate / search_pass_rate before tuning)",
                elapsed, self.perf_budget_s,
            )

        billable_in, completion, cached = _billable(resp.usage)
        step.tokens_in = billable_in
        step.tokens_out = completion
        step.cached_tokens = cached

        raw = ""
        if resp.choices:
            raw = (resp.choices[0].message.content or "").strip()
        step.attrs["raw_critique"] = raw   # for tracing / debugging

        verdict_matches = list(_VERDICT_RE.finditer(raw))
        if not verdict_matches:
            # Couldn't parse a verdict — treat as PASS rather than
            # blocking the reply on a flaky critique output.
            log.warning("critique output had no parseable VERDICT line — defaulting to PASS")
            step.verdict = "PASS"
        else:
            # Take the LAST occurrence — model's final answer wins over
            # any earlier mention while it was reasoning through.
            step.verdict = verdict_matches[-1].group(1).upper()

        # Pull the FAIL reasons out of the critique body so the revise
        # call can use them as guidance. Each line that starts with a
        # FAIL: gets captured.
        step.reasons = _extract_fail_reasons(raw)
        step.ended_at = time.monotonic()
        return step

    async def revise(
        self,
        msg: InboundMessage,
        draft: str,
        critique: CritiqueStep,
        prior_steps: list[Step],
        policy: Policy,
    ) -> ReviseStep:
        started = time.monotonic()
        step = ReviseStep(started_at=started)

        # Compose the reasons block. If the critique didn't expose any
        # reasons (parse miss), pass the raw critique text — better
        # than nothing.
        if critique.reasons:
            reasons = "\n".join(f"- {r}" for r in critique.reasons)
        else:
            reasons = critique.attrs.get("raw_critique", "").strip() or (
                "The reviewer wasn't happy with the draft. Re-ground "
                "every factual claim in the tool results below and make "
                "sure citations are present where search was used."
            )

        _, tool_block = self._tool_summary(prior_steps)
        available_urls = _extract_available_urls(prior_steps)
        user_question = (msg.text or "").strip()

        # Render the URL list for the prompt. When critique flags
        # "missing — source:" the model often invents a plausible URL;
        # surfacing the actual deep URLs from this turn's ToolSteps
        # gives revise a list to pick from.
        if available_urls:
            urls_block = "\n".join(f"- {u}" for u in available_urls)
        else:
            urls_block = "(no deep URLs gathered this turn)"

        revise_started = time.monotonic()
        try:
            resp = await self._client.chat.completions.create(
                model=self.model_id,
                messages=[{
                    "role": "user",
                    "content": _REVISE_PROMPT.format(
                        reasons=reasons,
                        available_urls=urls_block,
                        tool_results=tool_block or "(no tool calls this turn)",
                        draft=draft,
                        user_question=user_question,
                    ),
                }],
                temperature=0.4,
                max_tokens=1200,
            )
        except Exception as exc:                       # noqa: BLE001 — soft-fail
            # Real failures (network drop, model crash) still flow
            # through here. v1.6.2 removed the wait_for budget cap:
            # silently shipping the original ungrounded draft after a
            # 20s budget was masking the very confabulation critique
            # had just caught. The AsyncOpenAI client's default 600s
            # timeout is the real backstop.
            log.warning("revise failed (%s) — falling back to original draft", exc)
            step.attrs["error"] = str(exc)
            step.attrs["revised_reply"] = draft
            step.ended_at = time.monotonic()
            return step

        elapsed = time.monotonic() - revise_started
        if elapsed > self.perf_budget_s:
            log.warning(
                "revise exceeded perf budget: %.1fs > %.1fs (not killed; "
                "monitor turn latency)",
                elapsed, self.perf_budget_s,
            )

        billable_in, completion, cached = _billable(resp.usage)
        step.tokens_in = billable_in
        step.tokens_out = completion
        step.cached_tokens = cached

        revised = ""
        if resp.choices:
            revised = (resp.choices[0].message.content or "").strip()

        if not revised:
            # Empty revise output → keep the original draft. The
            # transport's empty-body fallback would otherwise fire.
            log.warning("revise returned empty content — keeping original draft")
            revised = draft

        step.attrs["revised_reply"] = revised
        step.ended_at = time.monotonic()
        return step

    @staticmethod
    def _tool_summary(prior_steps: list[Step]) -> tuple[str, str]:
        """Build (names_csv, results_block) from the act-loop steps.

        Returns ("", "") if no tools fired this turn — caller swaps in
        a friendlier placeholder for the prompt.

        Two truncation passes:
          - Per-step: each tool's body is capped at _TOOL_RESULT_TRUNCATION
            (8000 chars) so a single huge fetch can't dominate.
          - Aggregate: the assembled block is capped at
            _TOOL_BLOCK_AGGREGATE_CAP (24000 chars) so a multi-query
            fan-out (5+ ToolSteps × 8000) can't blow the critique
            context budget. The cap drops whole tail steps; the first
            steps (which carry the bulk of grounding) are preserved.
        """
        names: list[str] = []
        chunks: list[str] = []
        running_len = 0
        for s in prior_steps:
            if not isinstance(s, ToolStep) or not s.tool_name:
                continue
            names.append(s.tool_name)
            result = (s.attrs.get("result") or "")
            if len(result) > _TOOL_RESULT_TRUNCATION:
                result = result[:_TOOL_RESULT_TRUNCATION] + " […truncated]"
            chunk = f"--- {s.tool_name} ({s.result_len} chars) ---\n{result}"
            # Aggregate cap: if adding this chunk would push us past
            # the budget, drop it (and all subsequent chunks).
            if running_len + len(chunk) > _TOOL_BLOCK_AGGREGATE_CAP:
                chunks.append(
                    f"--- [{len(prior_steps) - len(chunks)} more tool result(s) "
                    f"omitted to fit reviewer context budget] ---"
                )
                break
            chunks.append(chunk)
            running_len += len(chunk) + 2   # +2 for the join separator
        return ", ".join(names), "\n\n".join(chunks)


# ----------- helpers -----------

def _extract_available_urls(prior_steps: list[Step]) -> list[str]:
    """Walk the act-loop's ToolSteps for deep URLs the revise call
    can cite from. Sources:

      - Search ToolSteps: ``attrs["urls"]`` populated by the
        web_search tool (executor-side stashing).
      - Fetch ToolSteps: the result text starts with ``Fetched: <url>``
        or contains ``## <url>`` blocks (fetch_urls format).

    Dedupes while preserving encounter order. Caps at 10 URLs — that's
    more than enough for revise to pick the right ``— source:`` line.
    """
    seen: set[str] = set()
    out: list[str] = []

    for s in prior_steps:
        if not isinstance(s, ToolStep):
            continue
        # Web_search stashes the structured URL list directly.
        stashed = s.attrs.get("urls")
        if isinstance(stashed, (list, tuple)):
            for u in stashed:
                if isinstance(u, str) and u and u not in seen:
                    seen.add(u)
                    out.append(u)
                    if len(out) >= 10:
                        return out
        # Fetch tool results carry URLs inline; extract them with a
        # cheap regex match. The result format is documented in
        # ongiini/search.py (Fetched: <url>\n\n<body>) and
        # ongiini/tools/ongiini_tools.py (## <url>\n<body>).
        result = s.attrs.get("result")
        if isinstance(result, str):
            for match in re.findall(r"^(?:Fetched:|##)\s*(https?://\S+)", result, flags=re.MULTILINE):
                # Strip trailing punctuation, but keep balanced parens
                # so Wikipedia-style URLs like .../Foo_(bar) survive.
                url = _strip_trailing_punct_balanced(match)
                if url and url not in seen:
                    seen.add(url)
                    out.append(url)
                    if len(out) >= 10:
                        return out
    return out


def _strip_trailing_punct_balanced(url: str) -> str:
    """Strip trailing ``.,;:!?)`` from a URL while preserving a final
    ``)`` that matches an opening ``(`` earlier in the URL. Handles
    Wikipedia-style paths like ``.../Foo_(bar)`` cleanly."""
    # Always strip these from the tail (never legitimate in a URL).
    while url and url[-1] in ".,;:!?":
        url = url[:-1]
    # Only strip trailing ``)`` if it's UNMATCHED (no opening ``(``
    # before it). Balanced paren stays as-is.
    if url.endswith(")") and url.count("(") < url.count(")"):
        url = url[:-1]
    return url


def _billable(usage_obj: Any) -> tuple[int, int, int]:
    if usage_obj is None:
        return 0, 0, 0
    prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
    cached = 0
    details = getattr(usage_obj, "prompt_tokens_details", None)
    if details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    return max(0, prompt_tokens - cached), completion_tokens, cached


_REASON_PREFIXES = ("FAIL:", "ISSUE:", "PROBLEM:", "FAILS:", "NO:")

# Strip only ACTUAL list decoration: "1." / "2)" / "-" / "*" / "•",
# followed by whitespace. Previously a greedy character class also ate
# leading digits of content (e.g. "2025." would become "."). The
# anchored alternation is conservative — it only matches recognised
# list-marker shapes.
_DECORATION_RE = re.compile(r"^(?:\d+[\.\)]|[\-\*•])\s+")

# A line is an "OK / PASS" verdict on a dimension if it begins with
# that word followed by a word boundary (whitespace, punctuation,
# end-of-string). Earlier code only checked for trailing space, which
# missed "OK." / "OK," / "OK -".
_OK_OR_PASS_RE = re.compile(r"^(?:OK|PASS)\b", re.IGNORECASE)

# A line counts as a verdict line. The VERDICT marker is the cue we
# use to enable narrative-only fallback parsing.
_VERDICT_LINE_RE = re.compile(r"^\s*VERDICT:\s*(PASS|REVISE)\s*$", re.IGNORECASE | re.MULTILINE)


def _extract_fail_reasons(critique_text: str) -> list[str]:
    """Pull reason lines out of the critique body.

    Gemma 4 doesn't always use the exact "FAIL:" prefix we asked for —
    in live testing it uses "Issue:", "Problem:", "Fails:" interchangeably.
    Catch all the common variants. Each matching line yields one reason.

    Also handles the numbered-list-with-prose case where Gemma writes
    "2. The claim about X isn't grounded in the search results" rather
    than "2. FAIL: claim not grounded". We capture those when:

      (a) the critique already has at least one structured reason
          (Gemma is mixing both styles in the same response), OR
      (b) the critique's verdict is REVISE but pass 1 found zero
          structured reasons (Gemma went pure-narrative for this one).

    Both branches skip lines that explicitly say "OK" / "PASS" so
    positive dimensions don't get captured as failures.
    """
    lines = critique_text.splitlines()
    reasons: list[str] = []
    structured_reasons_seen = False

    # Pass 1: structured "PREFIX: <reason>" lines (any of the variants).
    for line in lines:
        stripped = _DECORATION_RE.sub("", line.strip())
        upper = stripped.upper()
        for prefix in _REASON_PREFIXES:
            if upper.startswith(prefix):
                body = stripped[len(prefix):].strip()
                if body:
                    reasons.append(body)
                    structured_reasons_seen = True
                break

    # Pass 2 trigger:
    #   - mixed-mode: at least one structured reason already → grab the
    #     narrative siblings too;
    #   - narrative-only mode: verdict was REVISE but pass 1 found
    #     nothing — Gemma went pure-prose this time.
    verdict_match = _VERDICT_LINE_RE.search(critique_text)
    revise_verdict = verdict_match and verdict_match.group(1).upper() == "REVISE"
    run_pass_2 = structured_reasons_seen or (revise_verdict and not structured_reasons_seen)

    if run_pass_2:
        for line in lines:
            stripped = line.strip()
            num_match = re.match(r"^[\d]+[\.\)]\s+(.+)$", stripped)
            if not num_match:
                continue
            body = num_match.group(1).strip()
            upper = body.upper()
            # Skip if it's already captured by pass 1 (starts with a
            # known prefix) or if it's an explicit OK / PASS verdict.
            if any(upper.startswith(p) for p in _REASON_PREFIXES):
                continue
            if _OK_OR_PASS_RE.match(body):
                continue
            reasons.append(body)

    return reasons
