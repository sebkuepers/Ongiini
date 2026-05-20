import os
from pathlib import Path


def _whitelist() -> set[str]:
    raw = os.getenv("WHITELIST", "")
    return {n.strip().lstrip("+") for n in raw.split(",") if n.strip()}


class Settings:
    whatsapp_token: str = os.getenv("WHATSAPP_TOKEN", "")
    whatsapp_phone_id: str = os.getenv("WHATSAPP_PHONE_ID", "")
    whatsapp_verify_token: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
    # Meta App Secret — used to verify the X-Hub-Signature-256 header on
    # every incoming webhook POST. When empty we log a warning and accept
    # unsigned requests (dev mode).
    whatsapp_app_secret: str = os.getenv("WHATSAPP_APP_SECRET", "")

    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://host.docker.internal:8000/v1")
    vllm_model: str = os.getenv("VLLM_MODEL", "google/gemma-3-27b-it")

    data_dir: Path = Path(os.getenv("DATA_DIR", "/data"))
    whitelist: set[str] = _whitelist()

    memory_window: int = 10
    # Rolling-summary trigger: when len(history) exceeds the threshold, the
    # oldest entries are LLM-compressed into a leading system message and the
    # last `memory_keep_recent` entries are kept verbatim. Keeps long-running
    # conversations bounded without losing earlier context entirely.
    memory_summary_threshold: int = 14
    memory_keep_recent: int = 8
    namibia_country_code: str = "264"

    # Abuse / cost protection
    message_max_chars: int = 4096          # WhatsApp's own per-text limit
    rate_limit_per_5min: int = 20
    rate_limit_per_day: int = 200


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
