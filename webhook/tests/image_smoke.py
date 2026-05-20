"""Phase C smoke test: image messages through the live respond() pipeline.

Generates a small synthetic image in-process (no external file needed),
passes it through `respond()` with OpenAI-style multipart content, and
verifies:

  1. vLLM Gemma 4 actually processes the image (reply describes it)
  2. mem0 with enable_vision stores a typed long-term fact about the image
  3. Short-term memory.save persists the compact "[image attached]" placeholder

Run from inside the rebuilt webhook container:

    docker cp webhook/tests/image_smoke.py ongiini-webhook:/data/image_smoke.py
    docker exec ongiini-webhook python3 /data/image_smoke.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import sys

sys.path.insert(0, "/app")

from PIL import Image, ImageDraw, ImageFont  # bundled via sentence-transformers

from app import mem, memory, pii  # noqa: E402
from app.llm import respond  # noqa: E402

MSISDN = "99000000666666"  # synthetic test number


def _make_test_image() -> tuple[bytes, str]:
    """Create a small image with a recognisable subject.

    384x384 — multiple of 48 on both dimensions, satisfies Gemma 4's
    vision pooler grid requirement. Off-grid sizes (like 320x200) crash
    `_avg_pool_by_positions` with cudaErrorNotPermitted on the Spark.
    """
    img = Image.new("RGB", (384, 384), (210, 230, 160))   # pale-green background
    draw = ImageDraw.Draw(img)
    # Draw a few darker-green "leaf" shapes with yellow tips so the
    # model has something specific to describe.
    leaf_color = (60, 130, 60)
    yellow = (210, 200, 80)
    for x in (40, 130, 220, 300):
        # leaf base
        draw.ellipse((x, 220, x + 60, 320), fill=leaf_color)
        # yellow tip
        draw.ellipse((x + 5, 210, x + 30, 245), fill=yellow)
    # caption text on the image
    try:
        font = ImageFont.load_default()
        draw.text((10, 10), "MAIZE LEAVES (test)", fill=(40, 40, 40), font=font)
    except Exception:
        pass

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), "image/jpeg"


def _dump_mem() -> None:
    facts = mem.list_all(MSISDN)
    print(f"\n[long-term] {len(facts)} stored facts:")
    for f in facts:
        text = (f.get("memory") or "").strip()
        if text:
            print(f"  - {text}")

    stored = memory.load(MSISDN)
    print(f"\n[short-term] {len(stored)} entries:")
    for s in stored:
        role = s.get("role", "?")
        content = (s.get("content") or "")
        if len(content) > 140:
            content = content[:140] + "…"
        print(f"  - [{role}] {content}")


async def main() -> None:
    print(f"=== image smoke for {MSISDN} ===")

    # Clean slate so reruns are deterministic.
    memory.delete(MSISDN)
    mem.delete_all(MSISDN)

    image_bytes, mime = _make_test_image()
    data_url = (
        f"data:{mime};base64,{base64.standard_b64encode(image_bytes).decode('ascii')}"
    )
    print(f"generated test image: {len(image_bytes)} bytes  mime={mime}")

    # Turn 1: image WITH caption — clearest signal that the model is doing
    # vision and not just hallucinating.
    user_content = [
        {"type": "text", "text": "I think my maize leaves look off. Anything you notice?"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    print("\n--- T1: image + caption ---")
    history = memory.load(MSISDN)
    result = await respond(history, user_content, MSISDN)
    print(f"REPLY ({result.tokens_in} in / {result.tokens_out} out):")
    print(result.reply)

    # Mirror the live save path so mem0 gets fed.
    history.append({"role": "user", "content": "[image attached] " + user_content[0]["text"]})
    history.append({"role": "assistant", "content": result.reply})
    memory.save(MSISDN, history)
    await asyncio.to_thread(
        mem.add_turn, MSISDN, user_content, pii.sanitize(result.reply)
    )

    _dump_mem()

    # Turn 2: text-only follow-up. Tests that the model carries the image
    # context forward via memory (it can't see the image bytes any more —
    # mem0 + the [image attached] placeholder are what it has now).
    print("\n--- T2: text follow-up (no new image) ---")
    history = memory.load(MSISDN)
    result2 = await respond(history, "should I be worried?", MSISDN)
    print(f"REPLY ({result2.tokens_in} in / {result2.tokens_out} out):")
    print(result2.reply)

    print("\ncleanup…")
    memory.delete(MSISDN)
    mem.delete_all(MSISDN)


if __name__ == "__main__":
    asyncio.run(main())
