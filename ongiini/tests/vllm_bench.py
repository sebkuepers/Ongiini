"""vLLM throughput benchmark — REALISTIC Ongiini-shaped traffic.

This is v2 of the benchmark. The first version used a 47-token prompt
and produced misleading numbers — real Ongiini calls carry ~5000 tokens
of input (system prompt ~2800 + tool definitions ~600 + ~7 turns of
history + user message). Prefill cost and KV-cache pressure both scale
with input length, so the v1 numbers were not representative.

This version imports SYSTEM_PROMPT and TOOLS from the actual webhook
code, seeds a plausible 7-turn farming-chat history, and benchmarks
two regimes:

  - "fresh" runs: every concurrent caller hits with a different prior
    history (no prefix cache hit beyond the system prompt itself).
    Mirrors the worst-case "different users, simultaneously".

  - "warm" runs: same prior history across all concurrent callers, so
    the prefix cache hits hard. Mirrors a single user typing fast or
    a small set of returning users.

For each regime + concurrency level we report:
  - aggregate output tok/s (cluster throughput)
  - per-request decode tok/s (sustained generation speed)
  - median TTFT (latency to first byte — dominated by prefill)
  - p50 / p95 wall-clock latency (what the user experiences)

Run from inside the webhook container so we hit vLLM through the same
client + base URL the real traffic does:

    docker cp webhook/tests/vllm_bench.py ongiini-webhook:/data/vllm_bench.py
    docker exec ongiini-webhook python3 /data/vllm_bench.py
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time

sys.path.insert(0, "/app")

from openai import AsyncOpenAI

# Import the REAL system prompt + tool definitions, so the benchmark
# pays the same prefill cost the live webhook pays per call.
from ongiini.system_prompt import SYSTEM_PROMPT
from ongiini.tools import ALL_TOOLS
from owela import ToolRegistry
TOOLS = ToolRegistry(list(ALL_TOOLS)).schemas()  # noqa: E402

BASE_URL = os.getenv("VLLM_BASE_URL", "http://host.docker.internal:8000/v1")
MODEL = os.getenv("VLLM_MODEL", "google/gemma-3-27b-it")

# A 7-turn farming chat — same shape as M3_rolling_summary_en's seed,
# without the rolling summary applied yet (so input length is roughly
# what a moderately deep conversation looks like in practice).
SHARED_HISTORY: list[dict] = [
    {"role": "user", "content": "Hi, I'm a small-scale farmer near Oshakati."},
    {"role": "assistant", "content": "Hello! Are you focused on crops or livestock?"},
    {"role": "user", "content": "I grow maize and mahangu on 3 hectares."},
    {"role": "assistant", "content": "Good size. Mahangu does well in sandy soil. Any pest issues?"},
    {"role": "user", "content": "Sandy soil with rainy-season floods."},
    {"role": "assistant", "content": "Sandy drains fast — watch nitrogen. Consider compost to retain moisture."},
    {"role": "user", "content": "How often should I fertilise?"},
    {"role": "assistant", "content": "Split into two or three smaller applications — once before planting, again around six weeks in."},
    {"role": "user", "content": "What about irrigation if rain is unreliable?"},
    {"role": "assistant", "content": "Drip irrigation is most efficient for your soil. Furrow is cheaper but wastes water."},
    {"role": "user", "content": "I want to register the farm with BIPA next."},
    {"role": "assistant", "content": "Good plan — gives you a proper business identity. Start by reserving a name."},
    {"role": "user", "content": "Got my passport ready for the application."},
    {"role": "assistant", "content": "Passport works as ID. You'll also need a proof of address."},
]

# Picked a question the SYSTEM_PROMPT's WHEN-TO-SEARCH rule explicitly
# steers AWAY from web_search ("basic science / definitions / schoolwork
# explanations") so the model produces a long free-text reply rather
# than a 25-token tool-call payload. That gives us a clean decode signal.
USER_QUESTION = (
    "Could you briefly explain how photosynthesis works, in about 200 words?"
)

client = AsyncOpenAI(base_url=BASE_URL, api_key="not-needed")


def _build_messages(per_request_history: list[dict]) -> list[dict]:
    """Same message-construction logic as app.llm.respond()."""
    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + per_request_history
        + [{"role": "user", "content": USER_QUESTION}]
    )


async def one_request(rid: int, history: list[dict]) -> dict:
    messages = _build_messages(history)
    t0 = time.perf_counter()
    first_tok_at: float | None = None
    final_usage = None

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=400,
        temperature=0.6,
        stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        # ANY non-empty delta counts as the first generated token — be it
        # free-text content or a tool-call argument fragment. The previous
        # version only watched `delta.content` and missed tool-call cases
        # entirely, making TTFT look like 0s and decode look like 0 tok/s.
        delta = chunk.choices[0].delta if chunk.choices else None
        if first_tok_at is None and delta is not None and (
            getattr(delta, "content", None) or getattr(delta, "tool_calls", None)
        ):
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


def _make_unique_history(rid: int) -> list[dict]:
    """Slight per-rid mutation so the prefix cache can't reuse the suffix."""
    h = [dict(m) for m in SHARED_HISTORY]
    # Replace the LAST assistant turn's content with rid-specific text;
    # the assistant role keeps cache-friendly structure but the content
    # diff invalidates the cache for everything after that point.
    h[-1] = {
        "role": "assistant",
        "content": (
            f"Passport works as ID. (Note for request {rid}: also bring "
            "a proof of address — utility bill or municipal letter is fine.)"
        ),
    }
    return h


async def run_block(label: str, n: int, fresh: bool) -> None:
    print(f"\n=== {label}  concurrency={n}  ({'fresh per-rid' if fresh else 'warm shared'}) ===")
    histories = [(_make_unique_history(i) if fresh else SHARED_HISTORY) for i in range(n)]
    t0 = time.perf_counter()
    results = await asyncio.gather(*[one_request(i, h) for i, h in enumerate(histories)])
    wall = time.perf_counter() - t0

    in_avg = statistics.mean(r["in_tokens"] for r in results)
    out_total = sum(r["out_tokens"] for r in results)
    aggregate_tps = out_total / wall if wall > 0 else 0.0
    decode_med = statistics.median(r["decode_tps"] for r in results)
    ttft_med = statistics.median(r["ttft_s"] for r in results)
    totals = sorted(r["total_s"] for r in results)
    p50 = statistics.median(totals)
    p95 = totals[int(0.95 * len(totals))] if len(totals) > 1 else totals[0]

    print(
        f"  in_avg={in_avg:.0f} tok  out_total={out_total}  wall={wall:.2f}s\n"
        f"  aggregate={aggregate_tps:.1f} tok/s  per-req decode median={decode_med:.1f} tok/s\n"
        f"  TTFT median={ttft_med:.2f}s  total p50={p50:.2f}s  p95={p95:.2f}s"
    )


async def main() -> None:
    print(f"endpoint: {BASE_URL}")
    print(f"model:    {MODEL}")
    print(f"prompt:   system + tools + {len(SHARED_HISTORY)}-msg history + user")
    print("warming up (single fresh request to load model + page in weights)…")
    await one_request(-1, SHARED_HISTORY)

    print("\n--- FRESH histories (different users) — worst case ---")
    for n in (1, 3, 5, 8):
        await run_block("fresh", n, fresh=True)

    print("\n--- WARM shared history (same user / prefix cache hit) ---")
    for n in (1, 3, 5, 8):
        await run_block("warm", n, fresh=False)


if __name__ == "__main__":
    asyncio.run(main())
