"""Concurrency stress test for the image pipeline.

Fires N parallel image requests through `respond()` and reports the
latency distribution. Each request uses a different synthetic msisdn so
they don't block on the per-user memory.lock_for() — this measures
vLLM's batching behaviour for vision input, not our serialisation.

Also runs one mixed batch (text + image requests interleaved) which
matches the most realistic production shape — most messages are text,
a fraction are images.

Run from inside the rebuilt webhook container:

    docker cp ongiini/tests/image_stress.py ongiini-webhook:/data/imstress.py
    docker exec ongiini-webhook python3 /data/imstress.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import statistics
import sys
import time

sys.path.insert(0, "/app")

from PIL import Image, ImageDraw   # noqa: E402

from ongiini.tests._legacy_respond import respond  # noqa: E402

_BASE_MSISDN = "990001"   # one-shot stress msisdns; we don't seed mem0 for them


def _make_jpeg(w: int = 384, h: int = 384) -> bytes:
    img = Image.new("RGB", (w, h), (210, 230, 160))
    draw = ImageDraw.Draw(img)
    for i in range(4):
        x = 40 + i * 80
        draw.ellipse((x, 200, x + 60, 300), fill=(60, 130, 60))
        draw.ellipse((x + 5, 190, x + 30, 225), fill=(210, 200, 80))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


_TEST_JPEG = _make_jpeg()
_DATA_URL = f"data:image/jpeg;base64,{base64.standard_b64encode(_TEST_JPEG).decode('ascii')}"


async def one_image_call(rid: int) -> dict:
    msisdn = _BASE_MSISDN + f"{rid:08d}"
    user_content = [
        {"type": "text", "text": "What do you see in this image?"},
        {"type": "image_url", "image_url": {"url": _DATA_URL}},
    ]
    t0 = time.perf_counter()
    try:
        result = await respond([], user_content, msisdn)
        t1 = time.perf_counter()
        return {
            "rid": rid, "ok": True, "wall_s": t1 - t0,
            "in": result.tokens_in, "out": result.tokens_out,
            "reply_len": len(result.reply),
        }
    except Exception as exc:
        t1 = time.perf_counter()
        return {"rid": rid, "ok": False, "wall_s": t1 - t0, "error": repr(exc)}


async def one_text_call(rid: int) -> dict:
    msisdn = _BASE_MSISDN + f"{(rid + 1000):08d}"
    t0 = time.perf_counter()
    try:
        result = await respond([], "Briefly explain photosynthesis in 100 words.", msisdn)
        t1 = time.perf_counter()
        return {
            "rid": rid, "ok": True, "kind": "text", "wall_s": t1 - t0,
            "in": result.tokens_in, "out": result.tokens_out,
        }
    except Exception as exc:
        t1 = time.perf_counter()
        return {"rid": rid, "ok": False, "kind": "text", "wall_s": t1 - t0, "error": repr(exc)}


def _summary(label: str, results: list[dict]) -> None:
    walls = sorted(r["wall_s"] for r in results)
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\n=== {label} ===")
    print(f"  total: {len(results)}  ok: {n_ok}  failed: {len(results) - n_ok}")
    if walls:
        print(
            f"  wall  min={walls[0]:.2f}s "
            f"p50={statistics.median(walls):.2f}s "
            f"p95={walls[int(0.95 * len(walls))] if len(walls) > 1 else walls[0]:.2f}s "
            f"max={walls[-1]:.2f}s "
            f"mean={statistics.mean(walls):.2f}s"
        )
    for r in results:
        if not r.get("ok"):
            print(f"  FAILED rid={r['rid']}: {r.get('error')}")


async def run_image_concurrency(n: int) -> list[dict]:
    t0 = time.perf_counter()
    results = await asyncio.gather(*(one_image_call(i) for i in range(n)))
    wall = time.perf_counter() - t0
    print(f"\n>>> {n} concurrent IMAGE requests — wall {wall:.2f}s")
    _summary(f"image x{n}", results)
    return results


async def run_mixed(n_text: int, n_image: int) -> list[dict]:
    t0 = time.perf_counter()
    coros = (
        [one_image_call(i) for i in range(n_image)]
        + [one_text_call(i) for i in range(n_text)]
    )
    results = await asyncio.gather(*coros)
    wall = time.perf_counter() - t0
    print(f"\n>>> mixed batch ({n_image} image + {n_text} text) — wall {wall:.2f}s")
    images = [r for r in results if "kind" not in r]
    texts = [r for r in results if r.get("kind") == "text"]
    _summary(f"image x{n_image}", images)
    _summary(f"text x{n_text}",  texts)
    return results


async def main() -> None:
    print(f"test JPEG: {len(_TEST_JPEG)} bytes  data-URL: {len(_DATA_URL)} chars")
    print("WARM-UP: 1 image request to bring vision tower into the GPU cache")
    await one_image_call(rid=-1)

    for n in (1, 2, 4):
        await run_image_concurrency(n)

    await run_mixed(n_text=4, n_image=2)


if __name__ == "__main__":
    asyncio.run(main())
