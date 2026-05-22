"""Phase C smoke + iteration tests: image messages through the live pipeline.

Drives multiple image scenarios through `respond()` + the full save path
(short-term JSON + mem0 long-term fact extraction) so we can watch the
behaviour on:

  case A — clean 384x384 caption + image (the original smoke)
  case B — off-grid 320x200 that the main.py resizer must snap to clean dims
  case C — palette / 1-bit PNG that must convert to RGB before vLLM
  case D — image with NO caption (model must steer the conversation)
  case E — large image (2000x2000) that must clamp to ≤896 per side

After each case we dump mem0's stored facts so we can see whether the
custom extraction prompt actually captures image-derived [SITUATION]
facts in each scenario.

Run from inside the rebuilt webhook container:

    docker cp ongiini/tests/image_smoke.py ongiini-webhook:/data/image_smoke.py
    docker exec ongiini-webhook python3 /data/image_smoke.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import sys

sys.path.insert(0, "/app")

from PIL import Image, ImageDraw, ImageFont   # noqa: E402 (bundled via sentence-transformers)

from ongiini.memory import long_term as mem, memory, pii   # noqa: E402
from ongiini.tests._legacy_respond import respond  # noqa: E402
from ongiini.api.main import _resize_for_gemma4  # noqa: E402  (re-uses the live preprocessor)

# One synthetic msisdn per case keeps mem0 state isolated and the runs idempotent.
_MSISDN_BASE = "9900066"


def _msisdn(case: str) -> str:
    return _MSISDN_BASE + str(abs(hash(case)) % 10**6).zfill(6)[:6]


def _maize_image(width: int, height: int) -> Image.Image:
    """Pale-green background with 4 darker-green ellipse 'leaves' that
    have yellow 'tips' — clear enough that Gemma 4's vision tower
    consistently describes them as plant leaves with discolouration."""
    img = Image.new("RGB", (width, height), (210, 230, 160))
    draw = ImageDraw.Draw(img)
    leaf = (60, 130, 60)
    yellow = (210, 200, 80)
    # Lay out 4 leaves vertically centred on a band roughly 2/3 down.
    band_y = int(height * 0.55)
    leaf_h = max(40, height // 5)
    leaf_w = max(40, width // 8)
    gap = (width - 4 * leaf_w) // 5
    for i in range(4):
        x = gap + i * (leaf_w + gap)
        draw.ellipse((x, band_y, x + leaf_w, band_y + leaf_h), fill=leaf)
        # tip at top of each leaf
        tip_w = leaf_w // 2
        tip_h = leaf_h // 3
        draw.ellipse(
            (x + (leaf_w - tip_w) // 2, band_y - tip_h // 2,
             x + (leaf_w + tip_w) // 2, band_y + tip_h // 2),
            fill=yellow,
        )
    try:
        font = ImageFont.load_default()
        draw.text((10, 10), f"MAIZE LEAVES (test {width}x{height})",
                  fill=(40, 40, 40), font=font)
    except Exception:
        pass
    return img


def _encode_jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _encode_palette_png(img: Image.Image) -> bytes:
    """Force palette mode + alpha layer — the kind of awkward asset
    that crashes vLLM if we don't normalize to RGB."""
    p = img.convert("P", palette=Image.ADAPTIVE, colors=64)
    buf = io.BytesIO()
    p.save(buf, format="PNG")
    return buf.getvalue()


def _data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.standard_b64encode(image_bytes).decode('ascii')}"


def _dump_mem(msisdn: str, label: str) -> None:
    facts = mem.list_all(msisdn)
    print(f"  [{label}] long-term: {len(facts)} fact(s)")
    for f in facts:
        text = (f.get("memory") or "").strip()
        if text:
            print(f"    - {text}")


async def _run_image_turn(
    case_id: str,
    label: str,
    raw_bytes: bytes,
    caption: str,
    follow_up: str | None = None,
) -> None:
    msisdn = _msisdn(case_id)
    print(f"\n=== case {case_id}: {label} ===")
    print(f"  msisdn={msisdn}  raw_input_bytes={len(raw_bytes)}")

    # Clean slate so each case is reproducible.
    memory.delete(msisdn)
    mem.delete_all(msisdn)

    # Mirror main.py: ALL inbound images go through the 48-grid resizer
    # before they ever reach vLLM. This is the bug-fix path; the test
    # exercises it on each off-grid input.
    processed = _resize_for_gemma4(raw_bytes)
    if len(processed) != len(raw_bytes):
        print(f"  resizer changed bytes: {len(raw_bytes)} → {len(processed)}")

    # The resized output is always JPEG per the implementation.
    data_url = _data_url(processed, mime="image/jpeg")

    user_text_part = caption or (
        "I just sent you a photo. Have a look and tell me what you see — "
        "if there's something specific worth pointing out, mention it."
    )
    user_content = [
        {"type": "text", "text": user_text_part},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    history = memory.load(msisdn)
    result = await respond(history, user_content, msisdn)
    print(f"  reply ({result.tokens_in} in / {result.tokens_out} out):")
    for line in result.reply.splitlines():
        print(f"    {line}")

    # Save short-term + feed mem0 long-term (same path main.py uses).
    placeholder = "[image attached]"
    if caption:
        placeholder += f" {caption}"
    history.append({"role": "user", "content": placeholder})
    history.append({"role": "assistant", "content": result.reply})
    memory.save(msisdn, history)
    # Mirror main.py: image turns feed mem0 a text-only synthesis, not
    # the raw multipart with base64. The extraction prompt's IMAGES
    # examples are calibrated for this shape.
    await asyncio.to_thread(
        mem.add_image_turn, msisdn, caption, pii.sanitize(result.reply)
    )

    _dump_mem(msisdn, "post-image")

    # Optional text follow-up — proves the image context survived to T2
    # via short-term memory + extracted facts.
    if follow_up:
        history = memory.load(msisdn)
        result2 = await respond(history, follow_up, msisdn)
        print(f"\n  follow-up reply ({result2.tokens_in} in / {result2.tokens_out} out):")
        for line in result2.reply.splitlines():
            print(f"    {line}")

    # Per-case cleanup so reruns are stable.
    memory.delete(msisdn)
    mem.delete_all(msisdn)


async def main() -> None:
    print("=== image smoke (multi-case) ===")

    # Case A — clean 384x384 + caption. The baseline that worked before.
    await _run_image_turn(
        "A_clean_384",
        "clean 384x384 image + descriptive caption",
        _encode_jpeg(_maize_image(384, 384)),
        "I think my maize leaves look off. Anything you notice?",
        follow_up="should I be worried?",
    )

    # Case B — off-grid 320x200. Without _resize_for_gemma4 this used to
    # crash vLLM's vision pooler with cudaErrorNotPermitted. The resizer
    # should snap it to 336x192 (or larger) before it hits the model.
    await _run_image_turn(
        "B_offgrid_320x200",
        "off-grid 320x200 — resizer must snap to 48-multiple",
        _encode_jpeg(_maize_image(320, 200)),
        "what do you see here?",
    )

    # Case C — palette/PNG. Some Android image pickers emit indexed-color
    # PNGs. _resize_for_gemma4 converts to RGB before re-encoding as JPEG.
    await _run_image_turn(
        "C_palette_png",
        "palette-mode PNG — must convert to RGB",
        _encode_palette_png(_maize_image(400, 400)),
        "anything to flag on these leaves?",
    )

    # Case D — no caption. Model must give a one-line description and
    # ask what the user wants to know (per WHEN YOU GET AN IMAGE rule).
    await _run_image_turn(
        "D_no_caption",
        "no caption — model must steer with a question",
        _encode_jpeg(_maize_image(384, 384)),
        caption="",
    )

    # Case E — large 2000x2000 image, must clamp to ≤896 per side then
    # snap to a 48-multiple. Tests the clamp+snap branch of the resizer.
    await _run_image_turn(
        "E_oversized_2000",
        "oversized 2000x2000 — must clamp to ≤896 and 48-align",
        _encode_jpeg(_maize_image(2000, 2000)),
        "what's wrong with these?",
    )


if __name__ == "__main__":
    asyncio.run(main())
