"""Long-term semantic memory via mem0.

This is the "tier 2" memory layer that sits ALONGSIDE the existing
short-term JSON memory in ongiini.memory.short_term. The split:

- ongiini.memory.short_term  → recent raw turns for THIS conversation (10 turns,
                rolling summary kicks in beyond that)
- ongiini.memory.long_term     → durable semantic facts about THIS user, extracted by
                an LLM call when new info appears, retrieved by vector
                similarity to the current query

Stack:
- LLM (for fact extraction): the same vLLM-served Gemma 4 we use for
  replies, via mem0's "vllm" provider against settings.vllm_base_url.
- Vector store: qdrant in embedded-file mode under /data/qdrant/.
  No separate qdrant container needed at pilot scale.
- Embedder: sentence-transformers/all-MiniLM-L6-v2, CPU-only (image
  pre-downloads it during build).

The mem0 client is initialised lazily on first use because constructing
it loads ~80MB of model weights into RAM — we don't want to pay that
cost just to import the module.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

# long_term_llm carries the mem0 LLM bridge (formerly app/mem_llm.py).
# Aliased to ``mem_llm`` to preserve the in-file call sites verbatim.
from . import long_term_llm as mem_llm
from ..config import settings
from ..filters import normalize

log = logging.getLogger("ongiini.mem")

# Patch mem0's LlmFactory so its "vllm" provider resolves to our
# TrackedVllmLLM, which records token usage against the calling user.
# Must run BEFORE Memory.from_config builds the LLM instance.
mem_llm.install()

_memory_singleton: Any = None
_init_lock = Lock()


# Replaces mem0's default FACT_RETRIEVAL_PROMPT. Drives WHAT we extract
# and HOW we tag it. Returns one JSON list of typed third-person facts.
#
# The output contract mem0 cares about is exactly {"facts": [str, ...]}
# (see mem0/memory/main.py::_add_to_vector_store — it json.loads the LLM
# response and reads response["facts"]). The TAG prefix is our own
# convention; mem0 treats the whole string as one memory.
#
# Why English-only output regardless of input language:
#   - cross-language retrieval works (embedding similarity holds)
#   - one canonical phrasing per fact prevents duplicates across languages
#
# Why no absolute-date resolution: the prompt is set at startup, so
# "today" inside it is frozen at process-start. Keep relative dates as
# the user said them; the retrieval / next-turn layer can resolve.
_FACT_EXTRACTION_PROMPT = """You are the memory keeper for Ongiini, a free WhatsApp AI assistant for Namibians.

Extract user-specific information from this conversation that will help in FUTURE conversations.
Be generous, not minimalist — capture preferences, life context, ongoing situations, emotional
state, and verbatim phrasing when memorable. Not every turn warrants a fact: small talk and
pure how-to questions usually don't.

For each fact, output ONE short, third-person ENGLISH statement prefixed with a type tag:

  [PROFILE]     Stable identity, location, family, role
                Example: "Lives in Oshakati", "Has a daughter named Anna",
                         "Works as a teacher", "Native Afrikaans speaker"
  [PREFERENCE]  What they like / dislike / how they want to be addressed
                Example: "Prefers Afrikaans replies", "Likes step-by-step explanations",
                         "Dislikes mielie pap"
  [SITUATION]   Ongoing context: current job, project, problem they're working through
                Example: "Registering farm with BIPA as a (Pty) Ltd",
                         "Maize has yellowing on lower leaves",
                         "Looking for a new job after redundancy"
  [COMMITMENT]  Reminders, follow-ups, things to track over time
                Example: "Wants reminder on Friday to call the clinic",
                         "Promised to share CV draft tomorrow"
  [QUOTE]       Striking verbatim phrasing in the user's own words (use sparingly —
                only when paraphrase would lose something real)
                Example: 'User said: "I just want to know my child won\\'t go to bed hungry tonight."'
  [EMOTION]     Notable emotional state at the time
                Example: "Frustrated with the school principal",
                         "Recently relieved after diagnosis came back clear"

Rules:
- Output facts in ENGLISH regardless of input language. The user may write in Afrikaans
  ("Ek hou nie van mielie pap nie"); store as "[PREFERENCE] Dislikes mielie pap (maize porridge)".
- When the user uses a relative date ("Friday", "tomorrow", "next week"), keep it relative —
  do NOT compute an absolute date. Storing "Friday" is fine; the retrieval layer resolves.
- DO NOT store: small talk, the assistant's own replies, generic factual queries
  ("what is photosynthesis", "capital of France"), schoolwork problem solutions,
  hypotheticals, weather lookups, current-events queries.
- DO NOT store anything that looks like personal credentials (passwords, full card numbers,
  pure ID numbers) — those should already be redacted upstream as [REDACTED:kind].
- If nothing in this turn is worth remembering, return an empty list.

Output ONLY valid JSON, no markdown, no commentary:
  {"facts": ["[TAG] fact 1", "[TAG] fact 2", ...]}

Empty case:
  {"facts": []}

Examples:

Input:
user: Hi! Im Taraneh, just moved to Oshakati. I'm a farmer.
assistant: Welcome! What crops are you growing?
user: maize and a bit of mahangu on 3 hectares
Output:
{"facts": [
  "[PROFILE] Name is Taraneh",
  "[PROFILE] Recently moved to Oshakati",
  "[PROFILE] Works as a small-scale farmer",
  "[SITUATION] Currently growing maize and mahangu on 3 hectares"
]}

Input:
user: please remind me on Friday to call BIPA about my form
assistant: I'll mention BIPA when you message me on Friday.
Output:
{"facts": [
  "[COMMITMENT] Wants reminder on Friday to call BIPA about a form"
]}

Input:
user: ek hou nie van mielie pap nie
assistant: Reg so, sal dit onthou.
Output:
{"facts": [
  "[PREFERENCE] Dislikes mielie pap (maize porridge)",
  "[PROFILE] Speaks Afrikaans"
]}

Input:
user: lol thanks
assistant: anytime!
Output:
{"facts": []}

Input:
user: what is photosynthesis?
assistant: [long explanation]
Output:
{"facts": []}

IMAGES specifically:
- When the user's message includes a shared photo (a "[image attached]" placeholder
  in their text and/or image content), extract a [SITUATION] fact capturing WHAT they
  showed you AND their apparent CONCERN. Use the assistant's described observation
  in its reply as your ground truth — that's the model's own description of the image.
- Do NOT paraphrase the assistant's whole reply into a fact. Capture the SUBJECT and
  the user's apparent intent in one tight sentence.
- Do NOT extract [QUOTE] facts from the assistant's image-description text — those
  came from the bot, not the user.

Input:
user: [image attached] I think my maize leaves look off. Anything you notice?
assistant: I see four green leaves with some yellowing at the tips. In maize,
that can mean a nitrogen or potassium deficiency, water stress, or early pest damage.
Is the yellowing on the older leaves or the new growth?
Output:
{"facts": [
  "[SITUATION] Shared photo of maize leaves with yellowing at the tips; worried about crop health"
]}

Input:
user: [image attached]
assistant: I see a printed letter that looks like a rental contract clause. It says
the tenant agrees to pay an annual escalation of 8%.
Output:
{"facts": [
  "[SITUATION] Shared photo of a rental contract clause about an 8% annual escalation"
]}

Input:
user: [image attached] just wanted to show you
assistant: That's a nice photo of a sunset over what looks like the coast. Anything
specific you wanted to chat about?
Output:
{"facts": []}
"""


def _qdrant_storage_path() -> str:
    p = settings.data_dir / "qdrant"
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _build_config() -> dict:
    return {
        "llm": {
            "provider": "vllm",
            "config": {
                "model": settings.vllm_model,
                "vllm_base_url": settings.vllm_base_url,
                "api_key": "not-needed",
                # Keep fact-extraction calls cheap and deterministic.
                "temperature": 0.1,
                "max_tokens": 600,
                # Gemma 4 is multimodal. Turning vision on here means
                # mem0's extraction call will pass through image_url
                # content parts unchanged when we send them. Text-only
                # turns are unaffected — mem0 only adds vision payload
                # when the message actually contains image content.
                "enable_vision": True,
                "vision_details": "auto",
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "ongiini_memories",
                "path": _qdrant_storage_path(),
                "embedding_model_dims": 384,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dims": 384,
            },
        },
        # Custom extraction prompt: replaces mem0's default minimal
        # fact-stripping prompt with a Namibia-tuned, type-tagged version
        # that captures preferences, situations, commitments, quotes,
        # and emotional state — not just bald identity facts.
        "custom_fact_extraction_prompt": _FACT_EXTRACTION_PROMPT,
        # Persistent SQLite history of all memory mutations. Lets
        # whats_in_my_memory potentially surface "when did I learn this".
        "history_db_path": str(settings.data_dir / "mem0_history.db"),
    }


def _client():
    """Lazy singleton — the first call pays the model-load cost, subsequent calls are free."""
    global _memory_singleton
    if _memory_singleton is not None:
        return _memory_singleton
    with _init_lock:
        if _memory_singleton is not None:
            return _memory_singleton
        # Defer the heavy imports until the first call so that simply
        # importing ongiini.memory.long_term doesn't load torch.
        from mem0 import Memory  # noqa: WPS433
        log.info("initialising mem0 (this loads the embedding model — ~80MB RAM)")
        _memory_singleton = Memory.from_config(_build_config())
    return _memory_singleton


def warmup() -> None:
    """Eagerly initialise mem0 so the first real request doesn't pay
    the ~10s embedding-model load cost. Called from FastAPI's lifespan
    handler at startup."""
    try:
        _client()
    except Exception as exc:
        log.warning("mem0 warmup failed: %s", exc)


def format_relevant(memories: list[dict]) -> str:
    """Render mem0 search hits as a system-prompt snippet the LLM can use.

    Returns "" when there's nothing worth injecting — caller should skip
    adding the system message in that case rather than passing an empty
    block that costs tokens for no signal. Type tags ([PROFILE], etc.)
    are kept inline so the model can use them as semantic hints
    (a [COMMITMENT] warrants proactive follow-up; a [QUOTE] is verbatim).
    """
    facts = [
        (m.get("memory") or "").strip()
        for m in memories
        if isinstance(m, dict) and (m.get("memory") or "").strip()
    ]
    if not facts:
        return ""
    lines = ["What you know about this user from prior conversations:"]
    for fact in facts:
        lines.append(f"- {fact}")
    return "\n".join(lines)


# Order matches the priority for surfacing in whats_in_my_memory:
# identity first, then long-running situations, then how they want to be
# talked to, then trackables, then voice / mood.
_TAG_ORDER = ("PROFILE", "SITUATION", "PREFERENCE", "COMMITMENT", "QUOTE", "EMOTION")
_TAG_HEADINGS = {
    "PROFILE":    "About you",
    "PREFERENCE": "Your preferences",
    "SITUATION":  "Things you're currently working on",
    "COMMITMENT": "Things to follow up on",
    "QUOTE":      "Things you've said in your own words",
    "EMOTION":    "Recent emotional context",
}


def format_grouped_by_tag(memories: list[dict]) -> str:
    """Group typed facts under per-tag headings for whats_in_my_memory.

    Untagged facts (e.g. those written by earlier mem0 prompt versions)
    land in a trailing "Other" group rather than being dropped.
    """
    buckets: dict[str, list[str]] = {}
    other: list[str] = []
    for m in memories:
        text = (m.get("memory") or "").strip() if isinstance(m, dict) else ""
        if not text:
            continue
        # Trim long entries so the tool result stays readable.
        if len(text) > 240:
            text = text[:240] + "…"
        if text.startswith("[") and "]" in text:
            tag = text[1:text.index("]")].strip().upper()
            body = text[text.index("]") + 1:].strip()
            buckets.setdefault(tag, []).append(body)
        else:
            other.append(text)

    parts: list[str] = []
    for tag in _TAG_ORDER:
        items = buckets.pop(tag, None)
        if not items:
            continue
        parts.append(f"{_TAG_HEADINGS[tag]}:")
        for item in items:
            parts.append(f"  - {item}")
    # Any unexpected tags we didn't enumerate (defensive — prompt could
    # invent a new tag) get their own sections too.
    for tag in sorted(buckets):
        parts.append(f"{tag.title()}:")
        for item in buckets[tag]:
            parts.append(f"  - {item}")
    if other:
        parts.append("Other:")
        for item in other:
            parts.append(f"  - {item}")
    return "\n".join(parts)


def add_turn(msisdn: str, user_content, assistant_text: str) -> None:
    """Feed one completed turn to mem0 so it can extract / update facts.

    `user_content` is a plain string for text-only turns. mem0 runs its
    extraction + reconciliation pipeline (two LLM calls) and may store
    zero or more typed facts depending on whether the turn warrants it.

    For IMAGE turns, use add_image_turn() instead — feeding a multipart
    list containing a multi-KB base64 data URL into mem0's extraction
    LLM wastes tokens on garbage and reliably produces zero facts in
    practice. add_image_turn synthesises a text-only version that the
    extractor handles cleanly.

    The internal LLM calls mem0 makes are tracked against the caller's
    msisdn via the _current_msisdn context var, so every token mem0
    spends counts toward the user's monthly allowance.

    Never raises — long-term memory is a soft enhancement; if mem0
    hiccups we don't want to break the live reply path.
    """
    norm = normalize(msisdn)
    cv_token = mem_llm._current_msisdn.set(norm)
    try:
        m = _client()
        msgs = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_text},
        ]
        m.add(msgs, user_id=norm)
    except Exception as exc:
        log.warning("mem0 add_turn failed for %s: %s", msisdn, exc)
    finally:
        mem_llm._current_msisdn.reset(cv_token)


def add_image_turn(msisdn: str, caption: str, assistant_text: str) -> None:
    """Feed an image-bearing turn to mem0 as text only.

    The user's actual image bytes are NOT useful to mem0's extractor —
    they show up as kilobytes of base64 noise that distracts the model
    and pushes the real conversation out of its attention window. What
    IS useful is:
      - the caption (or its absence), so we know what the user wanted
      - the assistant's reply, which contains its own description of the
        image (e.g. "I see four green leaves with yellowing tips…")

    From those two strings the extraction prompt's IMAGES examples can
    derive a clean [SITUATION] fact like "Shared photo of maize leaves
    with yellowing tips; worried about crop health". The image example
    block in _FACT_EXTRACTION_PROMPT is calibrated for exactly this shape.

    Never raises.
    """
    placeholder = "[image attached]"
    if caption:
        placeholder += f" {caption}"
    add_turn(msisdn, placeholder, assistant_text)


def search(msisdn: str, query: str, limit: int = 5) -> list[dict]:
    """Return up to `limit` memories relevant to the query, ranked by similarity.

    Each item is a dict from mem0; the only field we rely on is
    "memory" (the stored fact string). Returns [] on any error.
    """
    try:
        m = _client()
        result = m.search(query=query, user_id=normalize(msisdn), limit=limit)
        # mem0 0.1.x returns {"results": [...]}; older versions returned a bare list.
        if isinstance(result, dict):
            return list(result.get("results", []))
        return list(result or [])
    except Exception as exc:
        log.warning("mem0 search failed for %s: %s", msisdn, exc)
        return []


def list_all(msisdn: str) -> list[dict]:
    """Return every stored memory for a user. Used by the whats_in_my_memory tool."""
    try:
        m = _client()
        result = m.get_all(user_id=normalize(msisdn))
        if isinstance(result, dict):
            return list(result.get("results", []))
        return list(result or [])
    except Exception as exc:
        log.warning("mem0 list_all failed for %s: %s", msisdn, exc)
        return []


def delete_all(msisdn: str) -> bool:
    """Wipe every stored memory for a user. Returns True on success."""
    try:
        m = _client()
        m.delete_all(user_id=normalize(msisdn))
        return True
    except Exception as exc:
        log.warning("mem0 delete_all failed for %s: %s", msisdn, exc)
        return False
