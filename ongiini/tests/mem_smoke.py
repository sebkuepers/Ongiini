"""Phase 1 smoke test for the mem0 integration.

Drives ongiini.memory.long_term directly (no respond() loop yet) to verify:

  1. mem0 initialises against our vLLM + qdrant + sentence-transformers stack
  2. add_turn() actually extracts facts from a conversation
  3. search() returns relevant memories ranked by similarity
  4. list_all() and delete_all() behave correctly

Run from inside the rebuilt webhook container:

    docker cp webhook/tests/mem_smoke.py ongiini-webhook:/data/mem_smoke.py
    docker exec ongiini-webhook python3 /data/mem_smoke.py
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "/app")

from ongiini.memory import long_term as mem  # noqa: E402

MSISDN = "99000000888888"  # synthetic test number, distinct from live_smoke's


def _dump(label: str, items: list[dict]) -> None:
    print(f"\n--- {label} ({len(items)}) ---")
    for i, it in enumerate(items):
        # mem0 returns dicts with at least {"id", "memory", "score"?}.
        text = it.get("memory", str(it))
        score = it.get("score")
        suffix = f"  (score={score:.3f})" if isinstance(score, float) else ""
        print(f"  {i}. {text}{suffix}")


async def main() -> None:
    print(f"=== mem0 smoke for {MSISDN} ===")

    # Start clean so reruns are deterministic.
    print("delete_all (initial cleanup)…")
    mem.delete_all(MSISDN)

    print("\nadding conversational turns…")
    mem.add_turn(
        MSISDN,
        user_content="Hi! I'm Taraneh, I live in Oshakati and farm maize.",
        assistant_text="Nice to meet you, Taraneh. Oshakati gets seasonal rain — useful for maize.",
    )
    mem.add_turn(
        MSISDN,
        user_content="I have 3 hectares and the soil is sandy.",
        assistant_text="Sandy soil drains fast — watch nitrogen levels.",
    )
    mem.add_turn(
        MSISDN,
        user_content="I'm trying to register the farm with BIPA as a (Pty) Ltd.",
        assistant_text="(Pty) Ltd is the more formal structure — good for growth and outside investment.",
    )
    mem.add_turn(
        MSISDN,
        user_content="By the way I prefer answers in Afrikaans where possible.",
        assistant_text="Geen probleem — ek antwoord in Afrikaans waar dit pas.",
    )

    all_mem = mem.list_all(MSISDN)
    _dump("all stored memories (post-ingest)", all_mem)

    # Now query for relevant facts on different topics.
    for q in [
        "Where does the user live?",
        "What language does the user prefer?",
        "How big is their farm?",
        "Is the user married?",   # nothing relevant — should return weak / nothing
    ]:
        results = mem.search(MSISDN, q, limit=3)
        _dump(f"search: {q!r}", results)

    print("\ndelete_all (cleanup)…")
    mem.delete_all(MSISDN)
    after = mem.list_all(MSISDN)
    print(f"  after delete_all: {len(after)} entries (expect 0)")


if __name__ == "__main__":
    asyncio.run(main())
