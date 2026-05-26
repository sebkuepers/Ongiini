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

    # Pre-approved MARKETING template used by scripts/broadcast.py
    # for proactive announcements. Default values reflect the name +
    # language Sebastian submits to Meta Business Manager — change
    # only if the template is renamed or a new language variant ships.
    whatsapp_template_announcement_name: str = os.getenv(
        "WHATSAPP_TEMPLATE_ANNOUNCEMENT_NAME", "ongiini_announcement"
    )
    whatsapp_template_announcement_language: str = os.getenv(
        "WHATSAPP_TEMPLATE_ANNOUNCEMENT_LANGUAGE", "en"
    )

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

    # Transparency-reporting / /stats.json endpoint.
    # stats_cache_ttl_seconds: how long the assembled aggregate JSON stays
    # cached in-process before recomputation. Aggregation is cheap (a few
    # ten-thousand log lines) but the Pages Function CDN also honours
    # Cache-Control: max-age of the same number, so this is the de-facto
    # refresh interval seen by visitors. 5 minutes is responsive enough
    # for an internal dashboard, idle enough not to thrash the disk.
    stats_cache_ttl_seconds: int = 300
    # How often the background topic-classification task wakes up to
    # process newly-arrived user messages through the local model. The
    # work is bounded — only messages not yet in topic_cache.sqlite get
    # classified. Default 10 minutes.
    topic_classify_interval_seconds: int = 600
    # Privacy floor: any distribution-category whose count is strictly
    # below this number is collapsed into "Other" before publication.
    # Protects against accidentally publishing a category with only one
    # user in it, which would be reverse-identifiable.
    stats_minimum_bucket: int = 5

    # v1 quality-phase kill switches. Each defaults to ON; set the
    # env var to "1" / "true" / "yes" to disable WITHOUT redeploying
    # any code (env-var change + container restart only). Useful if
    # post-deploy we discover the REVISE rate is too high or the
    # planner is wasting budget on shallow questions the classifier
    # mis-tagged as DEEP. Read by ongiini.runtime.build_policy_table.
    disable_planner: bool = (
        os.getenv("ONGIINI_DISABLE_PLANNER", "").lower() in ("1", "true", "yes")
    )
    disable_critique: bool = (
        os.getenv("ONGIINI_DISABLE_CRITIQUE", "").lower() in ("1", "true", "yes")
    )
    disable_interstitial: bool = (
        os.getenv("ONGIINI_DISABLE_INTERSTITIAL", "").lower() in ("1", "true", "yes")
    )

    # Debug-mode trace detail. When set, the TracingHook includes the
    # raw critique body + planner plan_text snippet inside each turn's
    # trace.jsonl entry. Default OFF — production traces stay
    # structural-only (no content). The operator flips this on during
    # experiments to understand WHY critique flipped REVISE or what
    # the planner actually said, then flips it off again.
    #
    # Privacy note: these fields contain LLM output that references
    # user content (the draft reply, parts of the user's question,
    # snippets from search results). Don't ship traces written under
    # this flag to external systems / public dashboards.
    trace_critique_detail: bool = (
        os.getenv("ONGIINI_TRACE_CRITIQUE_DETAIL", "").lower() in ("1", "true", "yes")
    )

    # Revise-loop validation capture (v1.7 eval work). When set, every
    # turn that triggers REVISE writes BOTH drafts (compose + revise) to
    # data/revise_eval/<msg_id>.json so the operator can compare them
    # side by side and decide whether the critique-revise loop is
    # actually improving output quality. See scripts/review_revises.py.
    #
    # Privacy note: this DELIBERATELY breaks the "no message content on
    # disk" PII contract — both drafts contain reply text, and the user
    # question is preserved so a human reviewer has context. Capture is
    # local-only, gitignored, never exported. Flip OFF when the eval
    # window closes; the dir can be rm -rf'd at any time.
    capture_revise_eval: bool = (
        os.getenv("ONGIINI_CAPTURE_REVISE_EVAL", "").lower() in ("1", "true", "yes")
    )

    # Salt for hashing contributor msisdns in the community-contribution
    # database (ongiini.contributions). Lives in .env on the host as
    # CONTRIBUTIONS_HASH_SALT. Without it, hash_msisdn() raises — we
    # never want to hash with an empty salt (would defeat the purpose
    # of pseudonymisation). Generate once with `openssl rand -hex 32`.
    contributions_hash_salt: str = os.getenv("CONTRIBUTIONS_HASH_SALT", "")


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
