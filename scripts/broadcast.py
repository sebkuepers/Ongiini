"""scripts/broadcast.py — proactive WhatsApp broadcast CLI.

Sends a pre-approved MARKETING template (`ongiini_announcement`) to
every user who has a per-user memory file in ``/data/`` MINUS anyone
in the broadcast opt-out store.

USAGE
-----

    # Always do this first
    python -m scripts.broadcast --dry-run --message "test announcement"

    # Smoke test to one msisdn (your own)
    python -m scripts.broadcast \\
        --message "Voice notes are live — try sending one." \\
        --only-msisdn +264811000000

    # Full broadcast
    python -m scripts.broadcast \\
        --message "Voice notes are live — try sending one." \\
        --url-suffix ""

Each successful send writes the rendered template body to the
user's short-term memory as an assistant turn, so the AI has context
when the user replies. STOP keyword opt-outs are honoured before any
send via the opt-out store.

This script must be run from inside the webhook container (so
``/data`` is bind-mounted + settings env vars are loaded):

    docker exec -it ongiini-webhook python -m scripts.broadcast ...
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add repo root to sys.path so the script can be run via
# `python scripts/broadcast.py` as well as `python -m scripts.broadcast`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ongiini.broadcast import opt_outs  # noqa: E402
from ongiini.broadcast.sender import broadcast_to, BroadcastResult  # noqa: E402
from ongiini.config import settings  # noqa: E402
from ongiini.contributions import hash_msisdn  # noqa: E402
from ongiini.filters import is_allowed, normalize  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
)
log = logging.getLogger("broadcast")


# ── Recipient enumeration ──────────────────────────────────────────


def enumerate_recipients(only_msisdn: list[str] | None = None) -> list[str]:
    """Return the list of msisdns to broadcast to, with opt-outs
    already filtered. Order is deterministic (sorted) so partial
    runs are resumable mentally.

    If ``only_msisdn`` is non-empty, returns just those (still
    filtered through is_allowed + opt-outs) — used for smoke tests.
    """
    if only_msisdn:
        candidates = [normalize(m) for m in only_msisdn]
    else:
        candidates = sorted(p.stem for p in settings.data_dir.glob("*.json"))

    # Apply the same allowlist the webhook uses (Namibian +264 etc).
    candidates = [m for m in candidates if is_allowed(m)]

    excluded = opt_outs.all_opted_out_hashes()
    if not excluded:
        return candidates

    # Hash each candidate once, filter against the set.
    filtered = [m for m in candidates if hash_msisdn(m) not in excluded]
    return filtered


# ── Logging ────────────────────────────────────────────────────────


def _append_log(line: str) -> None:
    """Append-only structured log of every broadcast attempt. Lives
    next to /data so it survives container restarts."""
    log_path = settings.data_dir / "broadcast.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(line + "\n")


def _format_log_line(result: BroadcastResult, run_id: str) -> str:
    """One JSON-shaped line per recipient. Uses the msisdn HASH —
    never the raw number — so the log is safe to ship with debug
    bundles. Mirrors usage.log's privacy posture."""
    import json
    h = hash_msisdn(result.msisdn)
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_id,
        "msisdn_hash": h[:12],   # truncate — short enough to grep, no value to leak
        "ok": result.ok,
        "memory_written": result.memory_written,
    }
    if result.skipped_reason:
        payload["skipped"] = result.skipped_reason
    if result.error:
        # Truncate error to keep the log line bounded
        payload["error"] = result.error[:200]
    if result.meta_message_id:
        payload["meta_id"] = result.meta_message_id
    return json.dumps(payload, separators=(",", ":"))


# ── Main loop ──────────────────────────────────────────────────────


@dataclass
class RunStats:
    total: int = 0
    ok: int = 0
    failed: int = 0
    skipped: int = 0


async def _run(args: argparse.Namespace) -> int:
    # Lazy warmup so the script can run outside the FastAPI lifespan
    opt_outs.warmup()

    recipients = enumerate_recipients(only_msisdn=args.only_msisdn)
    stats = RunStats(total=len(recipients))
    log.info(
        "broadcast plan: %d recipients%s, message %d chars, suffix=%r, dry_run=%s",
        len(recipients),
        " (smoke list)" if args.only_msisdn else "",
        len(args.message),
        args.url_suffix,
        args.dry_run,
    )

    if not recipients:
        log.warning("no recipients to send to; exiting")
        return 0

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    # Throttle by spacing requests evenly. rate_per_sec=5 → one every 200ms.
    interval = 1.0 / max(0.1, args.rate_per_sec)

    for i, msisdn in enumerate(recipients, start=1):
        loop_start = time.monotonic()
        try:
            result = await broadcast_to(
                msisdn=msisdn,
                body_text=args.message,
                url_suffix=args.url_suffix,
                dry_run=args.dry_run,
            )
        except Exception as exc:                  # noqa: BLE001
            log.exception("broadcast_to raised for %s — recording as failure", msisdn)
            result = BroadcastResult(
                msisdn=msisdn, ok=False, error=f"{type(exc).__name__}: {exc}"
            )

        if result.ok:
            stats.ok += 1
        elif result.skipped_reason:
            stats.skipped += 1
        else:
            stats.failed += 1

        _append_log(_format_log_line(result, run_id))

        # Progress every 50 recipients
        if i % 50 == 0 or i == len(recipients):
            log.info(
                "progress: %d/%d (ok=%d, failed=%d, skipped=%d)",
                i, stats.total, stats.ok, stats.failed, stats.skipped,
            )

        # Sleep the remainder of the throttle window
        elapsed = time.monotonic() - loop_start
        sleep_for = interval - elapsed
        if sleep_for > 0 and i < len(recipients):
            await asyncio.sleep(sleep_for)

    log.info(
        "broadcast complete: %d total, %d ok, %d failed, %d skipped",
        stats.total, stats.ok, stats.failed, stats.skipped,
    )
    return 0 if stats.failed == 0 else 1


# ── argparse + entrypoint ─────────────────────────────────────────


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Send a proactive WhatsApp template to opted-in users",
    )
    p.add_argument(
        "--message", required=True,
        help="The {{1}} body text. Wrapped by 'Update from Ongiini AI: ...'",
    )
    p.add_argument(
        "--url-suffix", default="",
        help="The {{2}} URL suffix appended to https://ongiini.ai/. "
             "Empty → homepage. e.g. 'contribute/' or 'statistics/'",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Log what WOULD send. Writes no memory, sends nothing.",
    )
    p.add_argument(
        "--rate-per-sec", type=float, default=5.0,
        help="Throttle. Default 5 req/sec (≈3 min for ~860 users).",
    )
    p.add_argument(
        "--only-msisdn", action="append", default=None,
        help="Restrict to one or more msisdns (repeatable). For smoke tests.",
    )
    args = p.parse_args(argv)
    if not args.message.strip():
        p.error("--message cannot be empty or whitespace-only")
    if len(args.message) > 800:
        # Body wrapper adds ~25 chars; total stays comfortably under
        # WhatsApp's 1024 limit. Bail rather than truncate silently.
        p.error(
            f"--message is {len(args.message)} chars; keep under 800 "
            "(template wrapper adds more, WhatsApp limit is 1024)"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
