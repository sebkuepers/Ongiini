"""WhatsApp voice-note transcription via faster-whisper.

faster-whisper wraps CTranslate2 — INT8-quantised Whisper that runs
comfortably on CPU. For Ongiini's typical voice notes (< 30s) the
large-v3-turbo model transcribes in 2-5s on the Spark's CPU, keeping
user-visible end-to-end latency around 7-10s (transcribe + vLLM reply).

Why CPU: vLLM is already at --gpu-memory-utilization 0.70 serving
Gemma 4. Putting Whisper on the GPU would compete for KV cache. CPU
is fast enough for pilot scale. If voice traffic ever justifies it,
flip to GPU by passing device="cuda" and dropping vLLM's allocation.

Lazy singleton: model load is ~1GB of weights + a few seconds of init.
We defer until first use and trigger via warmup() from main.py's
FastAPI lifespan so the first real voice note doesn't pay the cold
cost.

Returns (transcript_text, detected_language, duration_seconds). Never
raises — caller decides how to respond to ("", "", 0.0).
"""

from __future__ import annotations

import io
import logging
from threading import Lock
from typing import Any

log = logging.getLogger("ongiini.audio")

_model_singleton: Any = None
_init_lock = Lock()

# deepdml's CTranslate2 conversion of openai/whisper-large-v3-turbo.
# Standard community model; lives under HF_HOME (=/data/hf_cache) so
# the bytes persist across container restarts after the first download.
_MODEL_NAME = "deepdml/faster-whisper-large-v3-turbo-ct2"

# beam_size=5 is the quality baseline. Drop to 1 if we ever need to
# squeeze latency further; quality cost on conversational speech is
# small at typical SNR.
_BEAM_SIZE = 5

# Hard cap on transcribable audio length. WhatsApp's own voice-note
# limit is 60s under normal recording but file uploads can be longer.
# Anything past 90s is either a long-form recording (not the use case)
# or abuse — refuse politely. Whisper itself happily processes hours
# of audio, but per-request latency would balloon.
_MAX_DURATION_S = 90.0


def _client():
    """Lazy singleton — first call pays the load cost (~5s)."""
    global _model_singleton
    if _model_singleton is not None:
        return _model_singleton
    with _init_lock:
        if _model_singleton is not None:
            return _model_singleton
        # Defer the import too so just importing app.audio doesn't
        # pull CTranslate2 into RAM.
        from faster_whisper import WhisperModel  # noqa: WPS433
        log.info(
            "loading faster-whisper %s on CPU (int8) — first run downloads ~1GB",
            _MODEL_NAME,
        )
        _model_singleton = WhisperModel(
            _MODEL_NAME,
            device="cpu",
            compute_type="int8",
        )
        log.info("faster-whisper ready")
    return _model_singleton


def warmup() -> None:
    """Eagerly initialise the model from FastAPI's lifespan handler.

    Mirrors mem.warmup() — we'd rather pay the cold-load cost at
    startup than on the first user voice note. Failure is logged but
    not raised; lazy init will retry on the first real call.
    """
    try:
        _client()
    except Exception as exc:
        log.warning("audio warmup failed: %s", exc)


def transcribe(audio_bytes: bytes) -> tuple[str, str, float]:
    """Decode + transcribe one audio clip.

    Returns (transcript_text, detected_lang_code, duration_seconds).
    On any failure or refusal returns ("", lang_or_empty, duration_or_0).
    Never raises — caller decides how to respond.

    Refusal cases that yield "":
      - empty / unreadable bytes
      - duration > _MAX_DURATION_S
      - no speech detected (all VAD-filtered)
      - faster-whisper / ffmpeg error

    Note: language auto-detect runs on every clip; the returned
    `lang_code` is what Whisper thinks the audio is. The caller does
    NOT need to gate on it — the existing EN/AF-only redirect in the
    system prompt fires on the transcript itself once it lands in
    handle_message, so foreign-language voice notes get the same
    polite redirect as foreign-language text.
    """
    if not audio_bytes:
        return ("", "", 0.0)

    try:
        model = _client()
    except Exception as exc:
        log.warning("transcribe: model unavailable: %s", exc)
        return ("", "", 0.0)

    try:
        # faster-whisper accepts a BinaryIO. It shells out to ffmpeg
        # under the hood, which handles WhatsApp's OGG/Opus, MP3, M4A,
        # WAV, and most other container/codec combos. Passing BytesIO
        # avoids needing a temp file on the read-only rootfs.
        buf = io.BytesIO(audio_bytes)
        segments, info = model.transcribe(
            buf,
            beam_size=_BEAM_SIZE,
            vad_filter=True,
            language=None,  # auto-detect
        )

        duration = float(info.duration or 0.0)
        lang = info.language or ""

        if duration > _MAX_DURATION_S:
            log.warning(
                "audio %.1fs exceeds max %.1fs — refusing transcription",
                duration, _MAX_DURATION_S,
            )
            return ("", lang, duration)

        # segments is a generator; force it now and join the texts.
        parts: list[str] = []
        for segment in segments:
            text = (segment.text or "").strip()
            if text:
                parts.append(text)
        transcript = " ".join(parts).strip()
        return (transcript, lang, duration)
    except Exception as exc:
        log.warning("transcribe failed: %s", exc)
        return ("", "", 0.0)
