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

    # Short-term verbatim history. memory.save caps at memory_window*2
    # entries (each turn = 1 user + 1 assistant message = 2 entries).
    # 50 means we keep about 50 turns of back-and-forth verbatim before
    # the rolling summary starts compressing the oldest.
    memory_window: int = 50
    # Rolling-summary trigger: when len(history) exceeds the threshold, the
    # oldest entries are LLM-compressed into a leading system message and the
    # last `memory_keep_recent` entries are kept verbatim. Threshold sits
    # above the working window so most conversations never trigger it;
    # only marathon chats fold. keep_recent stays large so even after a
    # fold the user has plenty of immediate context.
    memory_summary_threshold: int = 70
    memory_keep_recent: int = 40
    # Free-tier monthly token allowance per user. Surfaced via the
    # my_token_usage tool and quoted on the website's "Free, with a fair
    # limit" section.
    monthly_token_limit: int = 1_000_000
    namibia_country_code: str = "264"

    # Abuse / cost protection
    message_max_chars: int = 4096          # WhatsApp's own per-text limit
    rate_limit_per_5min: int = 20
    rate_limit_per_day: int = 200


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
