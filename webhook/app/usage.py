from datetime import datetime, timezone

from .config import settings

LOG_PATH = settings.data_dir / "usage.log"


def record(msisdn: str, tokens_in: int, tokens_out: int, used_search: bool) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    search = "yes" if used_search else "no"
    line = (
        f"{ts} | {msisdn} | tokens_in={tokens_in} tokens_out={tokens_out} | "
        f"search={search}\n"
    )
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)
