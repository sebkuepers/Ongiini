# Ongiini — contributor guide

You are working inside `ongiini/`, the application package that
implements the WhatsApp helper for Namibia on top of the
[Owela](../owela/CLAUDE.md) framework.

If you're here to change Owela framework code, you're in the wrong
directory — open `owela/` instead. This file is for changes to the
application: a new tool, a new policy, a system-prompt edit, a new
WhatsApp behaviour.

---

## Orientation

`ongiini/` is purely application code. Every Python file here is
either:

1. **A pre-flight step `api/main.py` does** before `agent.handle()` —
   transport-receive concerns (signature verify, dedup, rate limit),
   media handling (image resize, audio transcribe), msisdn
   normalisation.
2. **An implementation of an Owela Protocol** — `Model`, `Transport`,
   `MemoryProvider`, `Classifier`, `Hook`.
3. **A Hook fired by the executor** during `on_step` /
   `on_turn_complete`.
4. **A tool** decorated with `@owela.tool`.

If a piece of code doesn't fit one of those four buckets, ask whether
it should.

---

## The runtime composition pattern

The one mental model that makes this codebase make sense:

```
WhatsApp webhook POST
  → api/main.py extracts message, normalises into InboundMessage
  → agent.handle(msg)                          [owela/agent.py]
    → execute_turn(runtime, msg)               [owela/executor.py]
      → classifier.classify(msg)               [ongiini/routers/]
      → policies.lookup(verdict, depth)        [ongiini/runtime.py builds the table]
      → memory.assemble_messages(...)          [ongiini/memory/provider.py]
      → for turn in 1..max_steps:
          → model.complete(req)                [ongiini/models/vllm_gemma.py]
          → tools.execute_parallel(...)        [ongiini/tools/]
      → transport.send(...)                    [ongiini/transports/whatsapp_transport.py]
      → hooks.on_turn_complete(...)            [ongiini/hooks/]
        → BillingHook  → ongiini/usage.py
        → TracingHook  → /data/trace.jsonl
        → MemoryRecordingHook → memory/provider.record_turn
```

Every file in `ongiini/` is somewhere on that diagram. If you can't
place it, stop and figure out where it goes before adding more.

---

## The eight Owela anti-trap principles, applied to this app

See [`../owela/CLAUDE.md`](../owela/CLAUDE.md) for the canonical
list. How each shows up here:

1. **No special cases in the executor** — There are none. Every
   conditional in the executor is gated on a Policy flag. The
   image-skip-router behaviour lives in `routers/gemma_classifier.py`
   (the impl decides), not in `owela/executor.py`.
2. **Tools are decorated functions** — see `tools/ongiini_tools.py`.
3. **Steps are typed** — we don't add new step types from Ongiini;
   we consume the Owela ones via hooks.
4. **Hooks for cross-cutting concerns** — `hooks/billing_hook.py`,
   `hooks/tracing_hook.py`, `hooks/memory_recording_hook.py`.
   Everything observe-only.
5. **Transport is an adapter** — `transports/whatsapp_transport.py`
   owns the WhatsApp 25s typing window, the 4096-char cap, dead-URL
   hygiene. The executor knows none of those.
6. **One Runtime object** — `runtime.py::build_runtime()`. Built
   once in the FastAPI lifespan. Frozen dataclass.
7. **No god-functions** — `runtime.py` is composition (no logic),
   `api/main.py` does normalisation + dispatch (no agent logic).
8. **Owela is pure library** — no Ongiini specifics in `owela/`.

---

## Where Ongiini-specific quirks live

Every product-specific quirk is intentionally localised. Don't add
quirks to `owela/`; add them here, in the right module.

| Quirk | Lives in |
|---|---|
| Gemma 4 vision pooler needs image dims as multiples of 48 | `api/main.py::_resize_for_gemma4` |
| WhatsApp typing-indicator timeout is 25s (Meta-side hard limit) | `transports/whatsapp_transport.py` |
| WhatsApp max message length is 4096 chars | `transports/whatsapp_transport.py` |
| Selective reasoning with `reasoning_budget=500` | `models/vllm_gemma.py` + per-policy in `runtime.py` |
| Date/time anchor in Namibia CAT timezone | `memory/provider.py::_today_in_namibia_prompt` |
| PII scrub at write time (LLM sees raw, disk sees redacted) | `pii.py` + `hooks/memory_recording_hook.py` |
| Image bytes never persisted to either memory tier | `memory/provider.py::record_image_turn` + `tools/ongiini_tools.py` |
| Dead-URL HEAD-check on search-grounded replies | `transports/whatsapp_transport.py::_strip_dead_urls` |
| Caption-router for admin-intent on image messages | `api/main.py::_caption_is_admin_intent` |
| 1M-tokens-per-user-per-month allowance | `usage.py` + `tools/ongiini_tools.py::my_token_usage` |
| Namibia-only filter (+264 country code) | `filters.py::is_allowed` |
| Common Intelligence Foundation / Spark / German number context | `system_prompt.py` |
| Oshiwambo greeting / code-switching reference | `skills/oshiwambo/SKILL.md` |

If you add a new quirk: pick the right module, NOT `owela/`. If
you can't find a right module, the quirk probably needs its own
abstraction — that's a design decision worth flagging.

---

## Skills (Claude-compatible reference content)

Drop named reference blocks into `ongiini/skills/<name>/SKILL.md` with
YAML frontmatter. The Owela framework (`owela.Skill` /
`owela.SkillRegistry`) renders the manifest into the system prompt and
exposes a `load_skill` tool for on-demand content.

**Required frontmatter (Claude spec):**

```yaml
---
name: <kebab-or-snake-case>
description: >
  One paragraph that BOTH explains what the skill is AND when the model
  should use it. The model decides relevance from the description, so
  bake the "when to use" guidance directly into it.
---
```

**Optional Owela extension (ignored by Claude):**

```yaml
load: always  # or 'on_demand' (default)
```

- `load: always` — full content embedded in the system prompt every
  turn. Use for small skills (~1k tokens) needed unpredictably (e.g.
  greetings).
- `load: on_demand` — only the manifest entry (name + description) goes
  in the system prompt; full content fetched via `load_skill(name)`.
  Use for large skills (~5k+ tokens) needed only occasionally.

**Where the wiring lives:**

- `ongiini/skills_loader.py` — scans `skills/`, parses frontmatter
- `ongiini/tools/skill_tools.py::load_skill` — the on-demand tool
- `ongiini/memory/provider.py::assemble_messages` — injects the
  manifest as a system message after SYSTEM_PROMPT, before the date
  anchor
- `ongiini/runtime.py::build_runtime` — loads skills and passes the
  registry to both the MemoryProvider and the Runtime

**Anti-trap discipline:**

- Skills don't bundle tools, hooks, or policies — those abstractions
  already exist and shouldn't be wrapped
- Skills are static reference content registered at startup, not
  per-turn state
- The Owela `Skill` and `SkillRegistry` are pure-library types — every
  Ongiini specific (the `oshiwambo` content, the loader path, the
  on-demand tool) lives here in `ongiini/`

---

## The PII contract

**LLM sees raw text. Disk + mem0 see redacted text.**

This is non-negotiable. The model can answer "what's my email?" by
seeing the raw value in the current turn's content; what gets
persisted for future turns is the `[REDACTED:email]` placeholder.

Implementation:
- `pii.sanitize(text)` — pure-regex scrub, returns redacted text
- `hooks/memory_recording_hook.py::OngiiniMemoryRecordingHook` calls
  `pii.sanitize` on user text + reply BEFORE handing them to
  `memory.record_turn`
- The provider in `memory/provider.py` does NOT sanitise itself —
  it trusts inputs are already clean

If you add a new persistence path (a new tool that writes user-
visible content somewhere), it MUST go through `pii.sanitize` first.
There's no `@enforce_pii` decorator — it's enforced by code review +
the tests in `tests/test_ongiini_memory_recording_hook.py`.

---

## The delete contract

When `delete_my_data` fires successfully (`ToolStep.error is None`),
NO persistence happens that turn. The deletion request itself must
not be re-recorded.

Implementation:
- `hooks/memory_recording_hook.py::OngiiniMemoryRecordingHook._delete_fired`
  walks the step list for a successful `delete_my_data` ToolStep
- On hit, the hook returns early — `record_turn` is NOT called

Tested in `tests/test_ongiini_memory_recording_hook.py`.
Don't break it.

---

## Adding a new tool

```python
# 1. Write the function in ongiini/tools/ongiini_tools.py

from owela import ToolContext, tool

@tool(
    name="weather_namibia",
    description=(
        "Get the current weather for a Namibian town. Use ONLY for "
        "Namibian locations; for elsewhere, fall back to web_search."
    ),
    params={"town": "Town name. E.g. 'Windhoek', 'Oshakati'."},
)
async def weather_namibia(town: str) -> str:
    # ... implementation
    return f"Weather for {town}: …"


# 2. Add to ALL_TOOLS in ongiini/tools/__init__.py
# (the existing pattern — just add weather_namibia to the tuple)


# 3. (Optional) If you want this tool forced on turn 1 for a specific
# classifier verdict, add a PolicyTable entry in ongiini/runtime.py:

table.set(
    VERDICT_WEATHER, DEPTH_SHALLOW,  # if you also added a new verdict
    Policy(name="weather", first_tool=force_tool("weather_namibia")),
)


# 4. Add a unit test in ongiini/tests/test_ongiini_tools.py
```

If the tool needs runtime access (msisdn, memory provider, etc.),
declare its first param as `ctx: ToolContext`. The decorator
excludes it from the schema.

---

## Adding a new policy / verdict

The depth-aware classifier is one axis. If you want a NEW axis
(e.g. "VOICE_NOTE" turns get a different loop shape):

1. **Extend the classifier prompt** in
   `routers/gemma_classifier.py::CLASSIFIER_PROMPT` to teach the
   model the new bucket.
2. **Add the new verdict constant** to `owela.policy` (if it's
   genuinely a framework concern) OR to `ongiini/routers/gemma_classifier.py`
   as a local constant + map it to one of the existing
   `VERDICT_*` values via your prompt's output rule. Prefer the
   latter — the framework's verdict set is small on purpose.
3. **Extend `ClassifierResult` parsing** to recognise the new label.
4. **Add a Policy entry** in `runtime.py::build_policy_table` for
   the new (verdict, depth) pair.
5. **Re-run the held-out eval** at `tests/router_eval_holdout.py`
   to confirm you haven't regressed the existing 4-way accuracy.

Target: ≥96% existing accuracy preserved, ≥85% on the new axis.

---

## Touching the system prompt

The whole prompt is `system_prompt.py::SYSTEM_PROMPT`. Sections:
LANGUAGES, FIRST-MESSAGE DISCLOSURE, TONE & FORMAT, CAUTIONS,
WHEN TO SEARCH, HONESTY WHEN SEARCH DOESN'T HELP, CITATIONS, MEMORY,
TOOL DISPATCH FOR DATA/USAGE/SELF, NAMIBIA CONTEXT, BOUNDARIES.

Small edits land directly. Anything bigger (new section,
restructure) should be eval-tested via the `eval.py` /
`router_eval_holdout.py` scripts before deploying — prompt changes
have user-facing impact within seconds of deploy.

---

## The stats subsystem

`stats/` is its own world. It runs a background qualitative-analysis
loop that pulls anonymous topic/role/region/language signals out of
recent messages, clusters them, and exposes the aggregate at
`/stats.json` for the website's `/statistics` page.

Largely independent of the agent loop. Read
[`../docs/statistics.md`](../docs/statistics.md) for the design.
Bug fixes there don't typically need agent-loop tests.

---

## Testing patterns

- **Fakes over MagicMock** for adapter unit tests. `FakeTransport`,
  `FakeClassifier`, etc. in `tests/test_executor.py` (Owela side) and
  the per-adapter test files (Ongiini side).
- **Patch the import path that the code uses**, not the source
  module — see `tests/test_whatsapp_transport.py` patching
  `ongiini.transports.whatsapp_transport._send_text`.
- **Smoke scripts** (`*_smoke.py`) run inside the live container via
  `docker exec` — they need vLLM + mem0 + Tavily reachable.
- **Eval scripts** (`eval.py`, `router_eval_holdout.py`) use the
  `_legacy_respond.py` shim — DO NOT depend on that shim in new
  tests; call `Agent.handle` directly via `runtime.build_agent()`.

Run:

```sh
pytest owela/tests ongiini/tests   # 149 unit tests, no live stack
```

---

## What stays in `ongiini/` vs `owela/`

Repeat of the table from the README, because it's the most common
"where do I put this" question:

- If the change would benefit ANY chat agent (different model,
  different transport, different memory backend), it belongs in
  `owela/`.
- If the change is specific to Gemma 4, vLLM, WhatsApp, mem0,
  Namibia, or the Common Intelligence Foundation, it belongs here.

When in doubt: keep it here. The framework's job is to NOT know
about your product; it's a design goal, not a constraint to push
against.

---

## See also

- [`README.md`](./README.md) — what Ongiini is, module guide,
  how to run it.
- [`../owela/CLAUDE.md`](../owela/CLAUDE.md) — the eight anti-trap
  principles in their canonical form + framework extension guide.
- [`../README.md`](../README.md) — root README with the operator
  manual + AI Act compliance + foundation context.
- [`../SECURITY.md`](../SECURITY.md) — privacy + container
  hardening posture.
- [`../docs/statistics.md`](../docs/statistics.md) — transparency
  reporting design.
