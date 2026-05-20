import os
from pathlib import Path


def _whitelist() -> set[str]:
    raw = os.getenv("WHITELIST", "")
    return {n.strip().lstrip("+") for n in raw.split(",") if n.strip()}


class Settings:
    whatsapp_token: str = os.getenv("WHATSAPP_TOKEN", "")
    whatsapp_phone_id: str = os.getenv("WHATSAPP_PHONE_ID", "")
    whatsapp_verify_token: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "")

    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://host.docker.internal:8000/v1")
    vllm_model: str = os.getenv("VLLM_MODEL", "google/gemma-3-27b-it")

    data_dir: Path = Path(os.getenv("DATA_DIR", "/data"))
    whitelist: set[str] = _whitelist()

    memory_window: int = 10
    namibia_country_code: str = "264"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
