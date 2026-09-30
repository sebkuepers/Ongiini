"""Owela ``Transport`` implementation for WhatsApp Cloud API.

This is the only place in the codebase that knows about Meta's
``graph.facebook.com`` endpoint and the 25s typing-window constraint.
The Owela executor only sees ``transport.acknowledge(msg)``,
``transport.send_interstitial(user_id, policy)``, and
``transport.send(user_id, body, policy, ctx)``.

Transport-internal reply hygiene (see ``reply_hygiene``):
  1. **Truncated drafts** are trimmed to the last sentence/paragraph and
     get a one-line offer to continue.
  2. **Markdown → WhatsApp** formatting, deterministic.
  3. **URL allowlist** on citation turns — URLs the tools did not return
     this turn are removed (plus HTML-fragment URLs).
  4. **Policy notice** — ``policy.reply_notice`` (e.g. "search is down")
     is appended in italics.
  5. **Char cap** at a boundary inside WhatsApp's 4096 limit.
"""

from __future__ import annotations

import asyncio
import logging
import re

from owela import InboundMessage, Policy, ReplyContext, SendResult

from . import reply_hygiene

# We delegate the actual Meta HTTP calls to the existing low-level helpers
# in ``webhook.app.whatsapp`` so the retry policy, signature checks, and
# media handling all keep working unchanged. The Transport is purely a
# wrapper that imposes the Owela protocol shape.
from ..whatsapp import mark_as_read as _mark_as_read
from ..whatsapp import send_text as _send_text

log = logging.getLogger("ongiini.transports.whatsapp")


# Markdown-table helpers used by ``WhatsAppTransport._tables_to_bullets``.
# Kept module-private to avoid surfacing them outside this file.

_TABLE_ALIGNMENT_CELL_RE = re.compile(r"^\s*:?-+:?\s*$")


def _is_table_row(line: str) -> bool:
    """A table row is a line that starts with ``|`` and has ≥2 cells
    (i.e. at least one extra ``|``). The trailing ``|`` is optional."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    return stripped.count("|") >= 2


def _is_table_alignment_row(line: str) -> bool:
    """Alignment rows look like ``|:---|---:|:---:|`` — cells contain
    only dashes and optional leading/trailing colons."""
    if not _is_table_row(line):
        return False
    cells = _parse_table_row(line)
    if not cells:
        return False
    return all(_TABLE_ALIGNMENT_CELL_RE.match(c) for c in cells)


def _parse_table_row(line: str) -> list[str]:
    """Split a ``|a|b|c|`` row into ``["a", "b", "c"]``. Strips empty
    leading/trailing cells (table rows usually have outer pipes)."""
    parts = [c.strip() for c in line.strip().split("|")]
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def _render_table_as_bullets(headers: list[str], rows: list[list[str]]) -> str:
    """Render parsed table rows as ``*{cell[0]}*\\n- {h1}: {c1}\\n...``
    blocks separated by blank lines. Empty cells are skipped silently.
    """
    blocks: list[str] = []
    for row in rows:
        if not row:
            continue
        # First cell becomes the bolded "row header"; remaining cells
        # become labelled bullets paired with the header row's labels.
        primary = row[0].strip()
        body_lines: list[str] = []
        for col_idx in range(1, len(row)):
            cell = row[col_idx].strip()
            if not cell:
                continue
            header = headers[col_idx].strip() if col_idx < len(headers) else ""
            if header:
                body_lines.append(f"- {header}: {cell}")
            else:
                body_lines.append(f"- {cell}")
        if primary or body_lines:
            block = f"*{primary}*" if primary else ""
            if body_lines:
                block = (block + "\n" if block else "") + "\n".join(body_lines)
            blocks.append(block)
    return "\n\n".join(blocks)


class WhatsAppTransport:
    """The Ongiini transport. Constructed once at runtime build."""

    # Owela protocol metadata.
    name = "whatsapp"
    typing_window_s = 25.0          # Meta-side hard limit, not configurable.
    max_message_chars = 4096        # WhatsApp's per-text limit.
    format = "plain_text"           # No markdown — WhatsApp shows raw chars.

    def __init__(
        self,
        *,
        interstitial_text: str = (
            "Looking into this for you — give me a moment, this kind of "
            "question usually needs a few sources."
        ),
        followup_interstitial_text: str = (
            "Still working on this — pulling extra sources to make sure "
            "the answer is grounded."
        ),
        followup_delay_s: float = 15.0,
        empty_reply_text: str = "Sorry, I couldn't come up with a reply.",
        continue_offer_text: str = "Want the rest? Reply *more*.",
    ) -> None:
        self.interstitial_text = interstitial_text
        self.followup_interstitial_text = followup_interstitial_text
        self.followup_delay_s = followup_delay_s
        self.empty_reply_text = empty_reply_text
        self.continue_offer_text = continue_offer_text
        # In-flight followup tasks, keyed by user_id. The send() method
        # cancels these when the real reply lands. If two consecutive
        # turns from the same user fire send_interstitial in close
        # succession, the second overwrites the first task (the first's
        # send_interstitial run has already finished anyway).
        self._followup_tasks: dict[str, asyncio.Task] = {}

    async def acknowledge(self, msg: InboundMessage) -> None:
        """Read receipt + typing indicator. Soft-fail (logged inside
        the underlying helper) — UX is not allowed to break a reply."""
        if not msg.msg_id:
            return
        await _mark_as_read(msg.msg_id, with_typing=True)

    async def send_interstitial(self, user_id: str, policy: Policy) -> None:
        """v1 — sends a "still working" message during long turns. The
        text is configurable at construction time so other transports
        with different latency budgets (Signal, Telegram) can override.

        v1.3 — also schedules a follow-up interstitial at T+
        ``followup_delay_s`` seconds. The follow-up fires if and only
        if the real reply hasn't landed by then. ``send()`` cancels
        the pending follow-up cleanly. Multi-query fan-out + advanced
        Tavily extract can push tail latency past WhatsApp's 25s
        typing-indicator window; the follow-up keeps the user informed.
        """
        await _send_text(user_id, self.interstitial_text)
        # Replace any previous still-pending followup for this user.
        prev = self._followup_tasks.pop(user_id, None)
        if prev is not None and not prev.done():
            prev.cancel()
        self._followup_tasks[user_id] = asyncio.create_task(
            self._followup_after_delay(user_id),
        )

    async def _followup_after_delay(self, user_id: str) -> None:
        """Sleep for ``followup_delay_s`` then send the followup
        interstitial. ``send()`` cancels this task via
        ``asyncio.CancelledError`` when the real reply lands; we treat
        cancellation as the success path and stay silent. Any other
        error is logged but not raised — interstitial UX is soft-fail
        per the transport contract.
        """
        try:
            await asyncio.sleep(self.followup_delay_s)
            await _send_text(user_id, self.followup_interstitial_text)
        except asyncio.CancelledError:
            # Normal: the reply landed before T+15s.
            raise
        except Exception as exc:                       # noqa: BLE001
            log.warning("followup interstitial failed for %s: %s", user_id, exc)
        finally:
            # Self-cleanup so the dict doesn't grow unbounded.
            self._followup_tasks.pop(user_id, None)

    async def send(
        self,
        user_id: str,
        body: str,
        policy: Policy,
        ctx: ReplyContext,
    ) -> SendResult:
        """Post-process and deliver the final reply.

        Steps, in order:
          1. trim; a truncated draft is cut to a boundary + continue offer
          2. normalise Markdown → WhatsApp formatting (deterministic)
          3. on citation turns: drop URLs the tools didn't return
          4. append ``policy.reply_notice``
          5. cap at ``max_message_chars`` on a boundary
          6. send via the underlying WhatsApp helper

        Markdown normalisation runs BEFORE the URL check so that links
        like ``[The Namibian](https://...)`` are bare URLs by then.

        Transport-side failures bubble up; the executor records
        ReplyStep.sent=False.
        """
        # Cancel any pending follow-up interstitial — the real reply is
        # arriving, no need for "still working" any more.
        pending = self._followup_tasks.pop(user_id, None)
        if pending is not None and not pending.done():
            pending.cancel()

        attrs: dict[str, object] = {}
        cleaned = (body or "").strip()
        if cleaned and ctx.truncated:
            cleaned = reply_hygiene.trim_to_boundary(cleaned)
            cleaned = f"{cleaned}\n\n{self.continue_offer_text}"
            attrs["trimmed_at_boundary"] = True
        if not cleaned:
            cleaned = self.empty_reply_text
            attrs["empty_fallback"] = True

        # Gemma 4 emits **double-asterisk** bold and `# heading` Markdown
        # that WhatsApp shows literally; convert deterministically.
        cleaned = self._normalise_markdown_for_whatsapp(cleaned)

        if reply_hygiene.cites_tools(ctx.used_tools):
            cleaned, dropped = reply_hygiene.drop_unlisted_urls(cleaned, ctx.allowed_urls)
            attrs["urls_dropped"] = dropped
            if dropped:
                log.info("dropped %d URL(s) not returned by this turn's tools", dropped)
            if not cleaned:
                cleaned = self.empty_reply_text
                attrs["empty_fallback"] = True

        notice = f"\n\n_{policy.reply_notice}_" if policy.reply_notice else ""
        limit = self.max_message_chars - len(notice)
        if len(cleaned) > limit:
            cleaned = reply_hygiene.cap_at_boundary(cleaned, limit)
            attrs["chars_capped"] = True
        cleaned += notice

        await _send_text(user_id, cleaned)
        return SendResult(sent=True, attrs=attrs)

    # ----------- internal hygiene -----------

    @staticmethod
    def _normalise_markdown_for_whatsapp(text: str) -> str:
        """Convert common Markdown idioms into the SUBSET WhatsApp
        actually renders. Deterministic, no LLM call.

        WhatsApp supports:
          ``*bold*`` (single asterisks)
          ``_italic_`` (single underscores)
          ``~strikethrough~``
          ```code``` / triple-backtick blocks
          ``- bullet`` / ``* bullet`` at line start (renders as •)
          ``1. item`` at line start
          ``> quote`` at line start

        WhatsApp does NOT render (shows literal characters):
          ``**double-asterisk bold**`` — Markdown
          ``__double-underscore italic__`` — some MD dialects
          ``# heading`` / ``## subheading``
          ``[link text](https://url)`` — Markdown links

        Transforms applied here:
          ``**X**``       → ``*X*``      (the most common failure)
          ``__X__``       → ``_X_``      (rare but cheap to handle)
          ``# X`` at SOL  → ``*X*``      (treat headers as bold)
          ``[T](URL)``    → ``T (URL)``  (preserves both text and URL,
                                          and exposes the URL so the
                                          dead-link HEAD-check can run)

        Other Markdown features (tables, footnotes, etc.) are left as-is.
        We prefer leaving raw text over guessing badly.
        """
        if not text:
            return text

        # **bold** → *bold* — non-greedy, must have content, can't span
        # a line break. The `[^*]` inner class prevents collapsing
        # adjacent runs of asterisks into a single match.
        text = re.sub(
            r"\*\*([^\s*][^*\n]*?[^\s*]|[^\s*])\*\*",
            r"*\1*",
            text,
        )

        # __italic__ → _italic_ — same pattern, with underscores. Use a
        # word-boundary on the outside so we don't munch in-the-middle
        # underscores in things like file names (``a__b``).
        text = re.sub(
            r"(?<![A-Za-z0-9_])__([^\s_][^_\n]*?[^\s_]|[^\s_])__(?![A-Za-z0-9_])",
            r"_\1_",
            text,
        )

        # Markdown headers `#`/`##`/`###` at start of line → *bold*.
        # We strip the # and surrounding spaces, then bold the heading
        # text. Limit to up to 6 leading `#` characters (standard MD).
        text = re.sub(
            r"^[ \t]*#{1,6}[ \t]+(.+?)[ \t]*$",
            r"*\1*",
            text,
            flags=re.MULTILINE,
        )

        # [text](url) → text (url). Preserves both halves; the URL is
        # exposed so the dead-link HEAD-check can probe it. Skip the
        # match if the bracketed text is empty or the URL doesn't look
        # like an http/https URL (might be an image alt or footnote ref).
        text = re.sub(
            r"\[([^\]\n]+?)\]\((https?://[^\s)]+)\)",
            r"\1 (\2)",
            text,
        )

        # Markdown tables → labelled bullet groups. WhatsApp doesn't
        # render tables at all (no pipe syntax, no HTML tables), so we
        # convert each row into a *header* + per-cell bullet block.
        # See the helper for the parse rules + fallback behaviour.
        text = WhatsAppTransport._tables_to_bullets(text)

        return text

    @staticmethod
    def _tables_to_bullets(text: str) -> str:
        """Detect Markdown tables and convert to WhatsApp-friendly
        labelled bullet groups.

        A Markdown table is 3+ consecutive lines where:
          - Line 1 is the header row (starts with ``|``, has ≥2 cells)
          - Line 2 is the alignment row (cells are dashes / colons only)
          - Lines 3+ are data rows (same shape as line 1)

        Each data row becomes a block::

            *{cell[0]}*
            - {header[1]}: {row[1]}
            - {header[2]}: {row[2]}

        Empty cells are skipped. Falls back gracefully: if any expected
        line doesn't match the shape, the block is left as-is (no
        partial conversion that could break formatting).
        """
        lines = text.splitlines(keepends=False)
        out: list[str] = []
        i = 0
        in_fence = False
        while i < len(lines):
            # Skip table conversion inside fenced code blocks. A line
            # starting with ``` (any number of additional backticks /
            # info string) toggles the fence flag. WhatsApp renders
            # triple-backtick blocks in monospace — leave them alone.
            stripped_line = lines[i].lstrip()
            if stripped_line.startswith("```"):
                in_fence = not in_fence
                out.append(lines[i])
                i += 1
                continue
            if in_fence:
                out.append(lines[i])
                i += 1
                continue

            # Look-ahead: is the current line a plausible header row,
            # and is the NEXT line an alignment row?
            if (
                _is_table_row(lines[i])
                and i + 1 < len(lines)
                and _is_table_alignment_row(lines[i + 1])
            ):
                header_cells = _parse_table_row(lines[i])
                # Collect data rows until we hit a non-table-row line.
                j = i + 2
                data_rows: list[list[str]] = []
                while j < len(lines) and _is_table_row(lines[j]):
                    data_rows.append(_parse_table_row(lines[j]))
                    j += 1

                # If we got at least one data row, convert. Otherwise
                # bail (header + alignment with no data is malformed).
                if data_rows and header_cells:
                    out.append(_render_table_as_bullets(header_cells, data_rows))
                    i = j
                    continue
                # Malformed: pass through unchanged.
            out.append(lines[i])
            i += 1
        return "\n".join(out)
