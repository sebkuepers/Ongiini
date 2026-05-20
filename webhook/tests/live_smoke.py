"""Live smoke-test of Memory v2 end-to-end.

Drives a fresh user's history through the same load → respond →
sanitize → maybe_summarize → save pipeline that main.py uses, and
prints the on-disk memory state after each turn. Lets us watch:

  - PII patterns getting scrubbed
  - the rolling-summary trigger firing once history crosses threshold
  - the `whats_in_my_memory` tool surfacing the stored state
  - `delete_my_data` wiping the file

Run from inside the container:

    docker cp webhook/tests/live_smoke.py ongiini-webhook:/data/live_smoke.py
    docker exec ongiini-webhook python3 /data/live_smoke.py
"""

from __future__ import annotations

import asyncio
import json
import sys

sys.path.insert(0, "/app")

from app import memory, pii, usage  # noqa: E402
from app.llm import maybe_summarize, respond  # noqa: E402

MSISDN = "99000000777777"  # synthetic test number, 14 digits, '99' prefix


def _dump_disk() -> None:
    path = memory._path_for(MSISDN)
    if not path.exists():
        print("  [disk] (no file)")
        return
    data = json.loads(path.read_text())
    print(f"  [disk] {len(data)} entries:")
    for i, m in enumerate(data):
        role = m.get("role", "?")
        content = (m.get("content") or "").replace("\n", " ")
        if len(content) > 110:
            content = content[:110] + "…"
        print(f"    {i:>2}. [{role}] {content}")


async def turn(label: str, user_text: str) -> None:
    print(f"\n--- {label} ---")
    print(f"USER: {user_text}")
    history = memory.load(MSISDN)
    result = await respond(history, user_text, MSISDN)
    print(f"BOT : {result.reply}")
    tools = []
    if result.used_web_search:           tools.append("web_search")
    if result.used_fetch_url:            tools.append("fetch_url")
    if result.deleted_data:              tools.append("delete_my_data")
    if result.used_whats_in_my_memory:   tools.append("whats_in_my_memory")
    print(f"      tools={tools or '-'}  in={result.tokens_in} out={result.tokens_out}")

    if not result.deleted_data:
        history.append(pii.sanitize_message({"role": "user", "content": user_text}))
        history.append(pii.sanitize_message({"role": "assistant", "content": result.reply}))
        history = await maybe_summarize(history)
        memory.save(MSISDN, history)
    # Mirror main.py's usage tracking so the my_token_usage tool sees real data.
    usage.record(MSISDN, result.tokens_in, result.tokens_out, result.used_search)
    _dump_disk()


async def main() -> None:
    memory.delete(MSISDN)
    print(f"=== Memory v2 live smoke for {MSISDN} ===")

    await turn("T1 (intro)",        "Hi! I'm a small-scale farmer near Oshakati.")
    await turn("T2 (PII)",          "Quick note: my email is taraneh.example@gmail.com and my ID is 80123456789.")
    await turn("T3 (filler)",       "I grow maize and mahangu on 3 hectares.")
    await turn("T4 (filler)",       "Soil is sandy and we get rainy-season floods.")
    await turn("T5 (filler)",       "How often should I fertilise?")
    await turn("T6 (filler)",       "What about irrigation if rain is unreliable?")
    await turn("T7 (filler)",       "I want to register the farm with BIPA next.")
    await turn("T8 (summary trip)", "Should I use a CC or a (Pty) Ltd for that?")
    await turn("T9 (recall)",       "What do you actually remember about me?")
    await turn("T10 (tokens)",      "By the way, how many tokens have I used this month?")
    await turn("T11 (delete)",      "Please forget everything and delete my data.")


if __name__ == "__main__":
    asyncio.run(main())
