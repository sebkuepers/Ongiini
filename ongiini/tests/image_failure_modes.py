"""Failure-mode coverage for image messages.

Image handling has more failure paths than text — Meta media downloads
can fail, the image bytes can be corrupt, PIL might choke on a weird
format, vLLM can hiccup on the vision pathway. This test pokes each
failure mode and asserts we degrade gracefully rather than crashing
the webhook or leaving the user stranded.

Each case calls a slice of the live pipeline directly (no real
WhatsApp inbound traffic) and asserts the observed behaviour.

Run from inside the rebuilt webhook container:

    docker cp webhook/tests/image_failure_modes.py ongiini-webhook:/data/imft.py
    docker exec ongiini-webhook python3 /data/imft.py
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "/app")

from ongiini.memory import long_term as mem, memory  # noqa: E402
from ongiini.api.main import _resize_for_gemma4  # noqa: E402


_PASS = "✓"   # ✓
_FAIL = "✗"   # ✗


def _check(name: str, ok: bool, note: str = "") -> bool:
    print(f"  {_PASS if ok else _FAIL} {name}{(' — ' + note) if note else ''}")
    return ok


def test_resizer_on_empty_bytes() -> bool:
    """Zero-byte payload must NOT raise; resizer's contract is
    'never raises, returns either resized bytes or the original'."""
    print("\n--- empty bytes through resizer ---")
    out = _resize_for_gemma4(b"")
    return _check("returned without raising", isinstance(out, bytes), f"len={len(out)}")


def test_resizer_on_garbage_bytes() -> bool:
    """Random bytes that aren't a valid image format. PIL will fail
    to open; the resizer should fall through and return the input
    unchanged rather than blowing up."""
    print("\n--- garbage bytes through resizer ---")
    garbage = b"\xff" * 1024 + b"NOT AN IMAGE" + b"\x00" * 256
    out = _resize_for_gemma4(garbage)
    return _check(
        "returned without raising",
        isinstance(out, bytes),
        f"input={len(garbage)} output={len(out)}",
    )


def test_resizer_on_truncated_jpeg() -> bool:
    """A JPEG header that lies — claims valid format but truncates
    mid-payload. PIL's image.load() is what catches these, sometimes
    only when you actually access pixel data. Our resizer calls
    convert('RGB') which forces a full decode."""
    print("\n--- truncated JPEG ---")
    # Minimal JPEG SOI marker followed by junk.
    truncated = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 8 + b"GARBAGE"
    out = _resize_for_gemma4(truncated)
    return _check(
        "returned without raising",
        isinstance(out, bytes),
        f"input={len(truncated)} output={len(out)}",
    )


def test_resizer_on_tiny_image() -> bool:
    """Below the minimum useful resolution. Resizer must upscale to
    at least 336x192 (its declared minimum)."""
    print("\n--- 50x50 tiny PNG ---")
    from PIL import Image
    import io
    img = Image.new("RGB", (50, 50), (128, 200, 128))
    buf = io.BytesIO(); img.save(buf, format="PNG")

    out = _resize_for_gemma4(buf.getvalue())
    ok1 = isinstance(out, bytes) and len(out) > 0

    # Verify the actual dimensions came out 48-aligned and ≥ min.
    decoded = Image.open(io.BytesIO(out))
    w, h = decoded.size
    aligned = (w % 48 == 0) and (h % 48 == 0)
    above_min = w >= 336 and h >= 192

    return all([
        _check("returned without raising", ok1),
        _check("dimensions multiple of 48", aligned, f"got {w}x{h}"),
        _check("dimensions ≥ min (336x192)", above_min, f"got {w}x{h}"),
    ])


def test_resizer_on_huge_aspect() -> bool:
    """Pathological aspect ratio (16:1) — 1600x100. Resizer should
    clamp the long side to ≤896 and snap both dims to multiples of 48."""
    print("\n--- 1600x100 huge-aspect ---")
    from PIL import Image
    import io
    img = Image.new("RGB", (1600, 100), (200, 200, 200))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=85)

    out = _resize_for_gemma4(buf.getvalue())
    decoded = Image.open(io.BytesIO(out))
    w, h = decoded.size

    return all([
        _check("returned without raising", isinstance(out, bytes)),
        _check("long side clamped to ≤896", max(w, h) <= 896, f"got {w}x{h}"),
        _check("dimensions multiple of 48", (w % 48 == 0) and (h % 48 == 0), f"got {w}x{h}"),
        _check("dimensions ≥ min (336x192)", w >= 336 and h >= 192, f"got {w}x{h}"),
    ])


def test_mem_failures_are_silent() -> bool:
    """mem0 calls in ongiini.memory.long_term are documented as 'no-op-on-failure'.
    Force a failure path (bad msisdn shape) and verify nothing raises
    out of the wrapper."""
    print("\n--- mem.list_all with empty msisdn ---")
    out = mem.list_all("")
    return _check("returned empty list", out == [], f"got {out!r}")


def test_memory_load_missing() -> bool:
    """memory.load on a never-seen msisdn must return [] cleanly."""
    print("\n--- memory.load on missing user ---")
    out = memory.load("99000000000003")
    return _check("returned empty list", out == [], f"got {out!r}")


async def main() -> None:
    results = [
        test_resizer_on_empty_bytes(),
        test_resizer_on_garbage_bytes(),
        test_resizer_on_truncated_jpeg(),
        test_resizer_on_tiny_image(),
        test_resizer_on_huge_aspect(),
        test_mem_failures_are_silent(),
        test_memory_load_missing(),
    ]
    passed = sum(1 for r in results if r)
    print(f"\n{'=' * 40}")
    print(f"failure-mode coverage: {passed}/{len(results)} passed")
    print(f"{'=' * 40}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
