"""Phase D smoke: voice-note transcription end-to-end.

Three scenarios — all run faster-whisper for real (not mocked) so we
also confirm the model loaded and ffmpeg/CTranslate2 are wired up.

  case A — 1.5s of silence (OGG/Opus mono, WhatsApp container shape).
           Whisper should return a non-zero duration and a graceful
           empty/short transcript. Confirms the pipeline survives
           audio with no speech.

  case B — English speech synthesised via espeak-ng:
           "Hello, I am a maize farmer in Oshakati." Asserts that
           Whisper picks up at least 2/3 of the salient nouns. Robotic
           but reliable for keyword-level transcription.

  case C — Real Afrikaans speech: a 1.5s recording of "Suid-Afrika"
           by Jan Schutte (SABC, 1960), sourced from Wikimedia Commons
           and shipped in webhook/tests/fixtures/. Vendored because
           espeak-ng's AF voice transcribes too poorly to be useful,
           and we need to prove the pipeline handles a natural human
           voice — which is what WhatsApp will actually deliver.

Run from inside the rebuilt webhook container:

    docker cp webhook/tests/audio_smoke.py ongiini-webhook:/data/audio_smoke.py
    docker cp webhook/tests/fixtures/. ongiini-webhook:/data/fixtures/
    docker exec ongiini-webhook python3 /data/audio_smoke.py
"""

from __future__ import annotations

import asyncio
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, "/app")

from app import audio  # noqa: E402

# Real-speech fixtures live next to this file. The AF case reads a
# public-domain Wikimedia Commons clip rather than synthesising AF
# speech with espeak-ng — espeak's robotic AF voice tokenises poorly
# and gives Whisper nothing to grip on. See fixtures/NOTICE.md for
# provenance.
_FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"
_AF_FIXTURE = _FIXTURES_DIR / "suid_afrika.ogg"


def _make_silence_ogg(duration_s: float = 1.5) -> bytes:
    """Generate <duration_s> of silence as OGG/Opus — matches WhatsApp's
    voice-note container/codec. Uses the system ffmpeg installed via
    the Dockerfile."""
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
        out_path = f.name
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=mono:sample_rate=16000",
        "-t", str(duration_s),
        "-c:a", "libopus", "-b:a", "32k",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    with open(out_path, "rb") as f:
        return f.read()


def _make_speechlike_ogg(text: str, voice: str = "en") -> bytes:
    """Render a short phrase to OGG/Opus via espeak-ng + ffmpeg.

    espeak-ng's robotic voice is far from natural speech, but
    large-v3-turbo is trained on enough varied data that it still
    transcribes the WORDS reliably — which is all the smoke needs.
    Raises FileNotFoundError if espeak-ng isn't installed.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
        ogg_path = f.name
    subprocess.run(
        ["espeak-ng", "-v", voice, "-s", "150", "-w", wav_path, text],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "32k", ogg_path],
        check=True, capture_output=True,
    )
    with open(ogg_path, "rb") as f:
        return f.read()


def _summarize(label: str, transcript: str, lang: str, duration: float) -> None:
    short_t = (transcript[:160] + "…") if len(transcript) > 160 else transcript
    print(f"  {label}: lang={lang!r:>10}  duration={duration:5.2f}s  "
          f"transcript_len={len(transcript)}")
    if transcript:
        print(f"    └─ {short_t!r}")


async def main() -> None:
    print("=== audio smoke ===")

    print("\n--- case A: 1.5s of silence (OGG/Opus, mimics WhatsApp container) ---")
    silence = _make_silence_ogg(1.5)
    print(f"  generated {len(silence)} bytes")
    transcript, lang, duration = await asyncio.to_thread(audio.transcribe, silence)
    _summarize("silence", transcript, lang, duration)
    assert duration > 0, f"expected non-zero duration, got {duration}"
    # transcript may be empty (no speech) or some Whisper hallucination —
    # both are acceptable. We only insist that transcribe() returned.

    # Case B - EN: real synthesised speech. Whisper should transcribe it
    # and the transcript should contain the salient nouns. We don't
    # require an exact match (espeak's robotic voice yields some word
    # variance) — just that the key content survived.
    print("\n--- case B (EN): 'I am a maize farmer in Oshakati' via espeak-ng ---")
    try:
        speech = _make_speechlike_ogg(
            "Hello, I am a maize farmer in Oshakati.", voice="en"
        )
        print(f"  generated {len(speech)} bytes")
        transcript, lang, duration = await asyncio.to_thread(audio.transcribe, speech)
        _summarize("EN speech", transcript, lang, duration)
        assert duration > 0, f"expected non-zero duration, got {duration}"
        low = transcript.lower()
        en_hits = sum(1 for word in ("maize", "farmer", "oshakati") if word in low)
        assert en_hits >= 2, (
            f"expected ≥2 of [maize, farmer, oshakati] in EN transcript, "
            f"got {en_hits}: {transcript!r}"
        )
        print(f"  ✓ transcript captured {en_hits}/3 expected keywords")
    except FileNotFoundError:
        print("  espeak-ng not installed — skipping EN content check")

    # Case C - AF: real-voice Afrikaans recording from Wikimedia
    # Commons (1.5s, "Suid-Afrika" pronounced by Jan Schutte, SABC
    # 1960, public domain). This proves the pipeline transcribes
    # ACTUAL natural speech, not just espeak's robotic synthesis —
    # which is the shape WhatsApp voice notes will arrive in.
    print("\n--- case C (AF): real recording 'Suid-Afrika' (Wikimedia Commons) ---")
    if not _AF_FIXTURE.exists():
        raise AssertionError(
            f"AF fixture missing at {_AF_FIXTURE} — run from a checkout that "
            f"includes webhook/tests/fixtures/"
        )
    speech_af = _AF_FIXTURE.read_bytes()
    print(f"  loaded {len(speech_af)} bytes from {_AF_FIXTURE.name}")
    transcript_af, lang_af, duration_af = await asyncio.to_thread(
        audio.transcribe, speech_af
    )
    _summarize("AF speech", transcript_af, lang_af, duration_af)
    assert duration_af > 0, f"expected non-zero duration, got {duration_af}"
    # The recording is the single phrase "Suid-Afrika" — Whisper may
    # render it as "Suid-Afrika", "Suid Afrika", "Sudafrika" or
    # similar. Asserting the case-folded substring "afrika" survives
    # is the robust check across those variants.
    assert "afrika" in transcript_af.lower(), (
        f"expected 'afrika' in AF transcript, got {transcript_af!r}"
    )
    print(f"  ✓ 'Suid-Afrika' captured (lang detected: {lang_af!r})")

    print("\n=== smoke passed ===")


if __name__ == "__main__":
    asyncio.run(main())
