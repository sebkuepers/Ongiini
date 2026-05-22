import re
from datetime import datetime, timezone

from .config import settings

LOG_PATH = settings.data_dir / "usage.log"


def billable_from_usage(usage_obj) -> tuple[int, int, int]:
    """Extract (billable_in, completion, cached) from a vLLM/OpenAI usage object.

    vLLM with ``--enable-prompt-tokens-details`` populates
    ``usage.prompt_tokens_details.cached_tokens`` with the number of prompt
    tokens that hit the prefix cache (i.e. cost no real GPU work). We bill the
    user only the MARGINAL tokens — uncached prompt + completion — so the
    static SYSTEM_PROMPT and product.md overhead don't eat the monthly
    allowance once they're cached on the second request and onward.

    Falls back gracefully to full ``prompt_tokens`` if the cached field is
    absent (e.g. vLLM running without the flag, or a non-vLLM OpenAI-compat
    backend that doesn't surface cache details).

    Returns (0, 0, 0) when usage_obj is None.
    """
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

# Parses the lines written by record(); kept in lockstep with the format below.
# The trailing `| kind=...` field is optional so old log lines (pre-v3) still
# match — they implicitly count as kind=chat for the summary_for() rollup.
_LINE_RE = re.compile(
    r"^(?P<ts>\S+)\s\|\s(?P<msisdn>\S+)\s\|\stokens_in=(?P<in>\d+)\s"
    r"tokens_out=(?P<out>\d+)\s\|\ssearch=(?P<search>yes|no)"
)


def record(
    msisdn: str,
    tokens_in: int,
    tokens_out: int,
    used_search: bool,
    kind: str = "chat",
) -> None:
    """Append one accounting line to usage.log.

    `kind` distinguishes WHERE the tokens were spent so we can answer the
    "what counts toward my monthly allowance" question precisely:

        chat     — the user-facing reply call (everything in respond())
                   AND any image-content tokens (Gemma 4 vision is part
                   of the same prompt, billed in prompt_tokens).
        memory   — mem0's internal extraction + reconciliation calls
                   that turn a chat turn into typed long-term facts.
        summary  — the rolling-summary LLM call that compresses old
                   turns into a leading system message when history
                   crosses the soft threshold.

    summary_for() sums everything, so the user sees one combined number
    against their 1M/month allowance — but the kind tag is useful for
    auditing and for the FAQ explanation.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    search = "yes" if used_search else "no"
    line = (
        f"{ts} | {msisdn} | tokens_in={tokens_in} tokens_out={tokens_out} | "
        f"search={search} | kind={kind}\n"
    )
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


_KIND_RE = re.compile(r"\skind=(?P<kind>[a-zA-Z_]+)")


def summary_for(msisdn: str) -> dict:
    """Aggregate this user's token usage for the current calendar month (UTC).

    Splits by `kind` so the user can see how their monthly allowance is
    spent: the bulk goes to `chat` (their replies, with image content
    folded in via prompt_tokens), a smaller share goes to `memory`
    (mem0's typed-fact extraction across all turns), and `summary` is
    the occasional rolling-summary call that compresses old turns.

    "messages" counts only `chat` lines so it reflects what the USER
    sent, not how many internal calls the assistant made handling it.
    `tokens_total` sums across all kinds so it accurately reports the
    full month's spend against the 1M allowance.

    Missing log file / empty match → zero usage, never raises.
    """
    now = datetime.now(timezone.utc)
    month_prefix = now.strftime("%Y-%m")

    breakdown: dict[str, dict[str, int]] = {}

    if LOG_PATH.exists():
        with LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                m = _LINE_RE.match(line)
                if not m:
                    continue
                if m["msisdn"] != msisdn:
                    continue
                if not m["ts"].startswith(month_prefix):
                    continue
                kind_m = _KIND_RE.search(line)
                kind = kind_m.group("kind") if kind_m else "chat"
                slot = breakdown.setdefault(
                    kind, {"count": 0, "tokens_in": 0, "tokens_out": 0}
                )
                slot["count"] += 1
                slot["tokens_in"] += int(m["in"])
                slot["tokens_out"] += int(m["out"])

    # Kinds excluded from the user's monthly cap. The 3-way router
    # classifier ("kind=router") is system overhead — a fixed cost
    # per turn that doesn't correspond to anything the user explicitly
    # asked for. We log it under its own kind for auditability and
    # cost monitoring, but it shouldn't count against the 1M allowance
    # the user is told about in product.md / FAQ. Likewise we keep
    # `chat`, `memory`, and `summary` billable — those are all
    # explicitly documented in the FAQ as counting toward the monthly
    # allowance.
    NON_BILLABLE_KINDS = {"router"}

    tokens_in = sum(
        s["tokens_in"] for k, s in breakdown.items() if k not in NON_BILLABLE_KINDS
    )
    tokens_out = sum(
        s["tokens_out"] for k, s in breakdown.items() if k not in NON_BILLABLE_KINDS
    )
    messages = breakdown.get("chat", {}).get("count", 0)
    total = tokens_in + tokens_out
    limit = settings.monthly_token_limit
    pct = (total / limit * 100) if limit else 0.0
    return {
        "month": month_prefix,
        "messages": messages,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_total": total,
        "limit": limit,
        "percent_used": round(pct, 2),
        "breakdown": breakdown,
    }
