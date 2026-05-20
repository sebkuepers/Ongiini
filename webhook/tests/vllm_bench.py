"""vLLM decode-throughput micro-benchmark.

Hits the same OpenAI-compatible endpoint the webhook uses, streaming, at
concurrency N = 1, 3, 5, 8. For each N reports per-request TTFT, decode
seconds, and decode tok/s, plus the aggregate cross-request tok/s.

Run from inside the webhook container so we hit vLLM through the same
client and base URL the real traffic does:

    docker cp webhook/tests/vllm_bench.py ongiini-webhook:/data/vllm_bench.py
    docker exec ongiini-webhook python3 /data/vllm_bench.py
"""

from __future__ import annotations

import asyncio
import os
import time

from openai import AsyncOpenAI

BASE_URL = os.getenv("VLLM_BASE_URL", "http://host.docker.internal:8000/v1")
MODEL = os.getenv("VLLM_MODEL", "google/gemma-3-27b-it")
PROMPT = (
    "Write a roughly 300-word, friendly explanation of photosynthesis aimed at "
    "a 12-year-old. Use one concrete analogy and avoid bullet points."
)

client = AsyncOpenAI(base_url=BASE_URL, api_key="not-needed")


async def one_request(rid: int) -> dict:
    t0 = time.perf_counter()
    first_tok_at: float | None = None
    final_usage = None

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=400,
        temperature=0.6,
        stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            if first_tok_at is None:
                first_tok_at = time.perf_counter()
        if getattr(chunk, "usage", None):
            final_usage = chunk.usage

    t1 = time.perf_counter()
    out_tokens = final_usage.completion_tokens if final_usage else 0
    in_tokens = final_usage.prompt_tokens if final_usage else 0
    ttft = (first_tok_at - t0) if first_tok_at else 0.0
    decode_s = (t1 - first_tok_at) if first_tok_at else 0.0
    decode_tps = (out_tokens / decode_s) if decode_s > 0 else 0.0
    return {
        "rid": rid,
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "ttft_s": ttft,
        "decode_s": decode_s,
        "decode_tps": decode_tps,
        "total_s": t1 - t0,
    }


async def run_at_concurrency(n: int) -> None:
    print(f"\n=== concurrency={n} ===")
    t0 = time.perf_counter()
    results = await asyncio.gather(*[one_request(i) for i in range(n)])
    wall = time.perf_counter() - t0

    total_out = sum(r["out_tokens"] for r in results)
    aggregate_tps = total_out / wall if wall > 0 else 0.0
    median_decode = sorted(r["decode_tps"] for r in results)[len(results) // 2]
    median_ttft = sorted(r["ttft_s"] for r in results)[len(results) // 2]

    print(f"  wall={wall:.2f}s  total_out={total_out}  aggregate={aggregate_tps:.1f} tok/s")
    print(f"  median TTFT={median_ttft:.2f}s  median decode={median_decode:.1f} tok/s/req")
    for r in results:
        print(
            f"  rid={r['rid']:>2}  in={r['in_tokens']:>4} out={r['out_tokens']:>3}  "
            f"ttft={r['ttft_s']:.2f}s  decode={r['decode_s']:.2f}s  "
            f"per-req={r['decode_tps']:.1f} tok/s  total={r['total_s']:.2f}s"
        )


async def main() -> None:
    print(f"endpoint: {BASE_URL}")
    print(f"model:    {MODEL}")
    print("warming up (single fresh request to load prefix cache)…")
    await one_request(-1)
    for n in (1, 3, 5, 8):
        await run_at_concurrency(n)


if __name__ == "__main__":
    asyncio.run(main())
