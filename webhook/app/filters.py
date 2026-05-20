import re

from .config import settings

# E.164-shaped: 6-18 digits, first digit non-zero. Rejects anything with
# slashes, dots, whitespace, or other punctuation that could be used for
# filesystem path traversal in memory.py, OR shell injection further down
# the line.
_MSISDN_RE = re.compile(r"^[1-9]\d{5,17}$")


class InvalidMsisdn(ValueError):
    """Raised when a sender field doesn't look like a phone number."""


def normalize(msisdn: str) -> str:
    """Strip leading + and whitespace, then validate the digits-only shape.

    Raises InvalidMsisdn for anything that wouldn't survive as a safe
    filename component. Callers MUST treat the failure path explicitly —
    typically: log + drop the message.
    """
    if not isinstance(msisdn, str):
        raise InvalidMsisdn(f"msisdn must be a string, got {type(msisdn).__name__}")
    cleaned = msisdn.lstrip("+")
    cleaned = "".join(cleaned.split())  # strip ALL whitespace, internal or trailing
    if not _MSISDN_RE.match(cleaned):
        raise InvalidMsisdn(f"msisdn must be 6-18 digits, got {msisdn!r}")
    return cleaned


def is_allowed(msisdn: str) -> bool:
    n = normalize(msisdn)
    if n in settings.whitelist:
        return True
    return n.startswith(settings.namibia_country_code)
