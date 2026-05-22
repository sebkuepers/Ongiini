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

import asyncio
import logging
import re
import time
from typing import Any

from openai import AsyncOpenAI

from owela import (
    CritiqueStep, InboundMessage, Policy, ReviseStep, Step, ToolStep,
)

log = logging.getLogger("ongiini.reviewer")


# How much of each tool result we feed to the critique. 2k chars is
# enough context to judge "is this claim backed?" while keeping the
# critique prompt prefix-cacheable across requests.
_TOOL_RESULT_TRUNCATION = 2000


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

Critique on EXACTLY these six dimensions, in this order. Be honest —
your job is to catch problems, not be polite. Format each line as
either "OK" or "FAIL: <one short reason>". Do NOT add commentary
beyond what's asked.

1. Answers the user's actual question (not a tangent or partial answer):
2. Every factual claim about Namibia (places, businesses, prices,
   dates, schedules) is grounded in the tool results above — no
   training-data confabulation:
3. If web_search / fetch_url / fetch_urls fired, at least one DEEP
   URL is cited in the reply (prefixed "— source:" on its own line),
   not just a publication homepage:
4. If the tool results were thin or empty, the reply admits this
   plainly instead of inventing specifics:
5. Language matches the user's (EN or AF) — same language they asked
   in:
6. Plain WhatsApp text — no Markdown, no asterisks, no #headers, no
   "1." numbered lists, no bullet hyphens used as bullets, no tables,
   no backticks:

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
# back in time, we ship the draft unchanged.
_CRITIQUE_TIMEOUT_S = 6.0
_REVISE_TIMEOUT_S = 12.0


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
        critique_timeout_s: float = _CRITIQUE_TIMEOUT_S,
        revise_timeout_s: float = _REVISE_TIMEOUT_S,
    ) -> None:
        self.model_id = model_id
        self.critique_timeout_s = critique_timeout_s
        self.revise_timeout_s = revise_timeout_s
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
            resp = await asyncio.wait_for(
                self._client.chat.completions.create(
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
                ),
                timeout=self.critique_timeout_s,
            )
        except asyncio.TimeoutError:
            log.warning(
                "critique timed out after %ss — shipping draft unchanged",
                self.critique_timeout_s,
            )
            step.verdict = "PASS"
            step.attrs["error"] = "timeout"
            step.ended_at = time.monotonic()
            return step
        except Exception as exc:                       # noqa: BLE001 — soft-fail
            log.warning("critique failed (%s) — shipping draft unchanged", exc)
            step.verdict = "PASS"
            step.attrs["error"] = str(exc)
            step.ended_at = time.monotonic()
            return step

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
        user_question = (msg.text or "").strip()

        try:
            resp = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self.model_id,
                    messages=[{
                        "role": "user",
                        "content": _REVISE_PROMPT.format(
                            reasons=reasons,
                            tool_results=tool_block or "(no tool calls this turn)",
                            draft=draft,
                            user_question=user_question,
                        ),
                    }],
                    temperature=0.4,
                    max_tokens=1200,
                ),
                timeout=self.revise_timeout_s,
            )
        except asyncio.TimeoutError:
            log.warning(
                "revise timed out after %ss — falling back to original draft",
                self.revise_timeout_s,
            )
            step.attrs["error"] = "timeout"
            step.attrs["revised_reply"] = draft   # fall back to original
            step.ended_at = time.monotonic()
            return step
        except Exception as exc:                       # noqa: BLE001 — soft-fail
            log.warning("revise failed (%s) — falling back to original draft", exc)
            step.attrs["error"] = str(exc)
            step.attrs["revised_reply"] = draft
            step.ended_at = time.monotonic()
            return step

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
        """
        names: list[str] = []
        chunks: list[str] = []
        for s in prior_steps:
            if not isinstance(s, ToolStep) or not s.tool_name:
                continue
            names.append(s.tool_name)
            result = (s.attrs.get("result") or "")
            if len(result) > _TOOL_RESULT_TRUNCATION:
                result = result[:_TOOL_RESULT_TRUNCATION] + " […truncated]"
            chunks.append(f"--- {s.tool_name} ({s.result_len} chars) ---\n{result}")
        return ", ".join(names), "\n\n".join(chunks)


# ----------- helpers -----------

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


def _extract_fail_reasons(critique_text: str) -> list[str]:
    """Pull "FAIL: <reason>" lines out of the critique body. Each line
    that begins (after optional digit + dot prefix and whitespace) with
    FAIL: contributes one reason."""
    reasons: list[str] = []
    for line in critique_text.splitlines():
        stripped = line.strip()
        # Strip any leading "1." / "2)" / "-" decoration
        stripped = re.sub(r"^[\d\.\)\-\s]+", "", stripped)
        if stripped.upper().startswith("FAIL:"):
            reasons.append(stripped[len("FAIL:"):].strip())
    return reasons
