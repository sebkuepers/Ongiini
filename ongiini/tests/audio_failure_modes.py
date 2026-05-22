"""Failure-mode coverage for the voice-note pipeline.

Pokes the edges that real WhatsApp traffic + Whisper can throw at us
and asserts each one degrades to ("", lang_or_empty, duration_or_0)
rather than raising out of handle_audio_message and crashing the
worker.

Run from inside the rebuilt webhook container:

    docker cp ongiini/tests/audio_failure_modes.py ongiini-webhook:/data/aft.py
    docker exec ongiini-webhook python3 /data/aft.py
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "/app")

from ongiini import audio  # noqa: E402


def _check(name: str, ok: bool, note: str = "") -> bool:
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name}{(' — ' + note) if note else ''}")
    return ok


def test_empty_bytes() -> bool:
    print("\n--- empty bytes ---")
    t, lang, d = audio.transcribe(b"")
    return all([
        _check("returned without raising", isinstance(t, str)),
        _check("transcript is empty", t == "", f"got {t!r}"),
        _check("duration is 0.0", d == 0.0, f"got {d}"),
    ])


def test_garbage_bytes() -> bool:
    print("\n--- garbage bytes (4 KB of random data) ---")
    junk = (b"\xa9\xff\x00\x12\x34" * 800)[:4096]
    t, lang, d = audio.transcribe(junk)
    return all([
        _check("returned without raising", isinstance(t, str)),
        _check("transcript is empty on undecodable input", t == ""),
        _check("duration is 0.0 when ffmpeg refuses", d == 0.0),
    ])


def test_truncated_ogg() -> bool:
    """First few bytes of a valid OGG header + truncation."""
    print("\n--- truncated OGG (header only, no payload) ---")
    # 'OggS' magic + version 0 + flags + granule position + serial + page seq
    head = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x00\x00\x00\x00\x00"
    t, lang, d = audio.transcribe(head)
    return all([
        _check("returned without raising", isinstance(t, str)),
        _check("transcript is empty", t == ""),
        _check("duration is 0.0", d == 0.0),
    ])


async def main() -> None:
    results = [
        test_empty_bytes(),
        test_garbage_bytes(),
        test_truncated_ogg(),
    ]
    passed = sum(1 for r in results if r)
    print(f"\n{'=' * 40}")
    print(f"audio failure-mode coverage: {passed}/{len(results)} passed")
    print(f"{'=' * 40}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
