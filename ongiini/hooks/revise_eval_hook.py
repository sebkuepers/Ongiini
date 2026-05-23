"""Capture both compose and revise drafts for offline human evaluation.

Purpose: the compose → critique → revise loop is the most expensive
piece of Ongiini's pipeline (one extra Gemma call per REVISE-flagged
turn). We've measured how often it FIRES (revise_rate ~67% over 24h)
but never whether it actually IMPROVES output. This hook captures the
data needed to answer that question.

For every turn that produced a ReviseStep, dumps a single JSON file to
``data/revise_eval/<msg_id>.json`` containing the user question, the
critique verdict + reasons, the original compose draft, the revised
draft, and tool-result metadata. ``scripts/review_revises.py`` shows
pairs side-by-side and collects ratings into a JSONL.

Privacy: this hook DELIBERATELY breaks the "no message content on
disk" contract because the data IS the question + the reply. It's
opt-in via ``ONGIINI_CAPTURE_REVISE_EVAL=1`` with a loud startup
warning, gitignored, never exported. The capture dir can be removed at
any time. Re-disable when the eval window closes.

Soft-fail: any IO error logs a warning but never crashes the turn.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from owela import CritiqueStep, ReviseStep, Step, ToolStep, TurnContext

from ..config import settings

log = logging.getLogger("ongiini.hooks.revise_eval")


class ReviseEvalCaptureHook:
    """on_turn_complete → capture both drafts for offline review.

    Only writes when ``settings.capture_revise_eval`` is True AND the
    turn actually produced a ReviseStep (i.e. critique flagged REVISE
    and revise ran). Compose-only turns are not captured — there's
    nothing to compare them against.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        # Per-instance base_dir so tests can substitute a tmp path
        # without monkeypatching settings.
        self._base_dir = base_dir or (settings.data_dir / "revise_eval")

    async def on_turn_complete(self, steps: list[Step], ctx: TurnContext) -> None:
        if not settings.capture_revise_eval:
            return
        try:
            payload = self._build_payload(steps, ctx)
        except Exception as exc:                       # noqa: BLE001 — soft-fail
            log.warning("revise_eval payload build failed: %s", exc)
            return
        if payload is None:
            return
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            # msg_id is unique per inbound WhatsApp message. Safe-ish
            # filename — strip anything weird just in case.
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in payload["msg_id"])[:128]
            path = self._base_dir / f"{safe}.json"
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            tmp.replace(path)
        except OSError as exc:
            log.warning("revise_eval write failed: %s", exc)

    @staticmethod
    def _build_payload(steps: list[Step], ctx: TurnContext) -> dict | None:
        """Walk steps, extract the compose / revise / critique data.
        Returns None if there's no ReviseStep (nothing to capture).
        """
        revise = next((s for s in reversed(steps) if isinstance(s, ReviseStep)), None)
        if revise is None:
            return None
        compose_draft = revise.attrs.get("compose_draft")
        revised_reply = revise.attrs.get("revised_reply")
        if not compose_draft or not revised_reply:
            # Defensive: shouldn't happen post v1.7-eval but skip if so.
            return None

        critique = next((s for s in reversed(steps) if isinstance(s, CritiqueStep)), None)

        # Tool-result metadata only — no result bodies (kilobytes of
        # search content) since they aren't what a reviewer needs to
        # judge "is the revise better than compose". If they're needed
        # later they can be pulled from trace.jsonl + this file's
        # msg_id.
        tool_results = []
        for s in steps:
            if isinstance(s, ToolStep):
                tool_results.append({
                    "name": s.tool_name,
                    "result_len": s.result_len,
                    "error": s.error,
                })

        return {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "msg_id": ctx.msg.msg_id,
            "policy": ctx.policy.name,
            "user_question": (ctx.msg.text or "").strip(),
            "tool_results": tool_results,
            "critique_verdict": critique.verdict if critique else None,
            "critique_reasons": list(critique.reasons) if critique else [],
            "raw_critique": critique.attrs.get("raw_critique", "") if critique else "",
            "compose_draft": compose_draft,
            "revised_reply": revised_reply,
            "compose_len": len(compose_draft),
            "revised_len": len(revised_reply),
            "revise_error": revise.attrs.get("error"),
        }
