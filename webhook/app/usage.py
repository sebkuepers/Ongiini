import re
from datetime import datetime, timezone

from .config import settings

LOG_PATH = settings.data_dir / "usage.log"

# Parses the lines written by record(); kept in lockstep with the format below.
_LINE_RE = re.compile(
    r"^(?P<ts>\S+)\s\|\s(?P<msisdn>\S+)\s\|\stokens_in=(?P<in>\d+)\s"
    r"tokens_out=(?P<out>\d+)\s\|\ssearch=(?P<search>yes|no)"
)


def record(msisdn: str, tokens_in: int, tokens_out: int, used_search: bool) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    search = "yes" if used_search else "no"
    line = (
        f"{ts} | {msisdn} | tokens_in={tokens_in} tokens_out={tokens_out} | "
        f"search={search}\n"
    )
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def summary_for(msisdn: str) -> dict:
    """Aggregate this user's token usage for the current calendar month (UTC).

    Returns a flat dict the LLM can turn into prose. Missing log file or
    empty match is treated as zero usage — never raises.
    """
    now = datetime.now(timezone.utc)
    month_prefix = now.strftime("%Y-%m")

    messages = 0
    tokens_in = 0
    tokens_out = 0

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
                messages += 1
                tokens_in += int(m["in"])
                tokens_out += int(m["out"])

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
    }
