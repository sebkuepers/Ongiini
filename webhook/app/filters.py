from .config import settings


def normalize(msisdn: str) -> str:
    return msisdn.lstrip("+").strip()


def is_allowed(msisdn: str) -> bool:
    n = normalize(msisdn)
    if n in settings.whitelist:
        return True
    return n.startswith(settings.namibia_country_code)
