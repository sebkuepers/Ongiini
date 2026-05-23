"""Owela ``Hook`` that persists cited URLs across turns.

After each turn completes, walks the step list for web_search and
fetch ToolSteps, extracts their URLs from ``attrs["urls"]`` (web_search)
or the result body (fetch_url, fetch_urls), and appends to the user's
per-msisdn source index file.

Pairs with ``OngiiniMemoryProvider.assemble_messages`` which injects a
compact system message listing the user's recent URLs when the index
is non-empty — so "give me sources" 60 turns later still has data.

Soft-fail: any exception is caught and logged. Broken persistence
must NEVER crash a successful reply.
"""

from __future__ import annotations

import logging
import re

from owela import Step, ToolStep, TurnContext

from ..memory import source_index

log = logging.getLogger("ongiini.hooks.source_index")


# Same URL-line patterns the reviewer recognises for fetch results.
# Kept local to avoid a cross-import for one regex.
_FETCH_URL_RE = re.compile(r"^(?:Fetched:|##)\s*(https?://\S+)", flags=re.MULTILINE)


class SourceIndexHook:
    """Subscribes to ``on_turn_complete``. Stateless across calls —
    safe to share one instance across all users (Runtime is a
    singleton)."""

    async def on_turn_complete(self, steps: list[Step], ctx: TurnContext) -> None:
        try:
            urls = self._collect_urls(steps)
            if urls:
                source_index.append(ctx.msg.user_id, urls)
        except Exception as exc:                       # noqa: BLE001 — soft-fail
            log.warning("source_index hook failed: %s", exc)

    @staticmethod
    def _collect_urls(steps: list[Step]) -> list[str]:
        """Return de-duplicated URLs in encounter order across this
        turn's web_search / fetch_url / fetch_urls ToolSteps."""
        seen: set[str] = set()
        out: list[str] = []
        for s in steps:
            if not isinstance(s, ToolStep):
                continue
            if s.tool_name not in ("web_search", "fetch_url", "fetch_urls"):
                continue
            # web_search stashes the structured URL list (executor uses
            # it for auto-followup synthesis; we re-use it here).
            stashed = s.attrs.get("urls") if isinstance(s.attrs, dict) else None
            if isinstance(stashed, (list, tuple)):
                for u in stashed:
                    if isinstance(u, str) and u and u not in seen:
                        seen.add(u)
                        out.append(u)
            # fetch_url(s) result text starts with "Fetched: <url>" or
            # contains "## <url>" headers — extract those.
            result = s.attrs.get("result") if isinstance(s.attrs, dict) else None
            if isinstance(result, str):
                for url in _FETCH_URL_RE.findall(result):
                    if url not in seen:
                        seen.add(url)
                        out.append(url)
        return out
