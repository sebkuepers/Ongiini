# Owela

*An opinionated chat-agent framework for small open-source models on
inference engines, delivered through messenger transports.*

---

## The gap

Most agent frameworks were designed for a specific shape: a frontier
model (Claude, GPT-4) accessed through an API, with a wide latency
budget, in a UI surface the developer controls. LangGraph, smolagents,
PydanticAI, Agno — all good frameworks for that shape.

Owela is for the opposite shape:

- **Small open-source models** — Gemma 4, Qwen 3, Llama 3.x — that
  need more deliberate orchestration to get near-frontier results.
  Tool-choice="auto" and "let the model decide" don't work the same
  way on a 26B model as they do on Claude.
- **Inference engines** — vLLM, llama.cpp, ollama — accessed directly,
  with engine-specific knobs (reasoning_budget, prefix-cached tokens)
  surfaced as first-class concerns rather than abstracted away.
- **Self-hosted hardware** — a DGX Spark, an M-series Mac, a
  single-GPU box. You're the one paying for the GPU minutes; the
  framework should help you budget them.
- **Messenger transports** — WhatsApp, Signal, Telegram. A 25-second
  typing-indicator window. No Markdown rendering. No streaming UI.
  Replies are atomic, asynchronous, plain-text-only.
- **No UI control** — unlike ChatGPT or Claude.ai, the user's chat
  surface isn't yours. You don't get to show a thinking spinner, a
  citation card, or a streaming token stream. The reply lands as a
  single WhatsApp message and that has to be enough.

If your stack is "GPT-5 via OpenAI's API, rendered in a React UI you
control" — Owela is not for you. Use LangGraph or the Anthropic SDK.

If your stack is "Gemma 4 on a DGX Spark in your living room,
answering WhatsApp messages from users in Namibia" — Owela is the
framework you've been hand-rolling.

---

## The core insight

> **The loop's shape should be decided by a classifier, not by the
> model.**

Existing frameworks delegate the turn's shape to the model itself via
`tool_choice="auto"` and prompt instructions. That works when the model
is smart enough to consistently pick the right tool and reasoning
strategy. It does not work as reliably on smaller models.

Owela makes the loop shape an explicit **Policy table**, indexed by a
classifier's verdict + depth. The classifier (a separate, prefix-cached
LLM call ~85ms p50) decides the turn's category (SEARCH / DOCS / ADMIN /
NONE) and its complexity (SHALLOW / DEEP). The PolicyTable maps that
pair to a frozen Policy object that drives the rest of the turn:

```
Verdict           First tool        Planner  Critique  Max steps  Reasoning
─────────────────────────────────────────────────────────────────────────────
NONE              auto              no       no        6          on-demand
ADMIN             auto              no       no        4          on-demand
DOCS              lookup_docs       no       yes (v1)  4          on-demand
SEARCH_SHALLOW    web_search        no       yes (v1)  6          on-demand
SEARCH_DEEP       auto              yes (v1) yes (v1)  8          on-demand
```

The model still chooses *what to say* and *which tools to chain* — but
the *shape* of the loop is set before the first model call.

---

## Architectural shape

A turn flows through a thread of typed steps. Hence the name (Owela is
Oshiwambo for "thread"):

```
InboundMessage
     │
     ▼
  router  ──────────▶ RouterStep  (verdict, depth)
     │                     │
     │                     ▼
     │              PolicyTable.lookup
     │                     │
     │                     ▼
     │                  Policy
     │                     │
     ├──► (interstitial — optional, v1)
     │
     ├──► (planner — optional, v1) ──────────▶ PlanStep
     │
     ▼
  act loop (1..max_steps)
     │
     ├──► model.complete() ─────────────────▶ ModelCallStep
     │                │
     │                └─ tool_calls? ──┐
     │                                 ▼
     │            asyncio.gather(execute(tc)…) ─▶ ToolStep ×N
     │                                 │
     │                                 └─▶ next iteration
     │
     │  (no tool_calls → exit loop with the draft)
     │
     ├──► (critique — optional, v1) ──────────▶ CritiqueStep
     │             │
     │             └─ verdict=REVISE? ─▶ revise ─▶ ReviseStep
     │
     ▼
  transport.send() ───────────────────────────▶ ReplyStep
     │
     ▼
  hooks.on_turn_complete(steps)
     │
     └─ BillingHook, TracingHook, MemoryRecordingHook, …
```

The `Step` list is the canonical record of the turn. Hooks subscribe
to step events; they observe, they don't transform. Anything that
needs to mutate the reply (dead-URL stripping, char capping,
format normalisation) lives inside the Transport adapter — because
those rules are medium-specific.

---

## Quick start

A minimal complete app, ~30 lines:

```python
import asyncio
from openai import AsyncOpenAI
from owela import (
    Agent, AUTO, ClassifierResult, DEPTH_SHALLOW, HookRegistry,
    InboundMessage, Policy, PolicyTable, Runtime, ToolRegistry,
    VERDICT_NONE, tool,
)

# Implement Owela's protocols (Model, Transport, Memory, Classifier).
# Each is a small adapter — typically <100 lines for a real backend.
# See ongiini/ in this repo for a fully wired production example.

class StdoutTransport:
    name = "stdout"
    typing_window_s = 999
    max_message_chars = 100_000
    format = "plain_text"
    async def acknowledge(self, msg): pass
    async def send_interstitial(self, user_id, policy): pass
    async def send(self, user_id, body, policy, *, used_search=False):
        print(f"[{user_id}] {body}")
        return True

class NullClassifier:
    async def classify(self, msg):
        return ClassifierResult(verdict=VERDICT_NONE, depth=DEPTH_SHALLOW)

class SimpleMemory:
    async def assemble_messages(self, msg, policy, steps):
        return [{"role": "system", "content": "You are helpful."},
                {"role": "user", "content": msg.text}]
    async def record_turn(self, *args, **kw): pass
    async def delete_all(self, user_id): return True
    async def list_all(self, user_id): return []
    def format_facts(self, facts): return ""

# Use your inference engine of choice. Anything with an OpenAI-compatible
# /v1/chat/completions endpoint works — see ongiini/models/vllm_gemma.py
# for a real adapter that surfaces reasoning_budget + cached_tokens.

@tool(name="add", params={"a": "First number.", "b": "Second."})
async def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)

# Wire the runtime
runtime = Runtime(
    model=YourModelAdapter(),
    transport=StdoutTransport(),
    memory=SimpleMemory(),
    classifier=NullClassifier(),
    tools=ToolRegistry([add]),
    policies=PolicyTable().set(VERDICT_NONE, DEPTH_SHALLOW,
                               Policy(name="default", first_tool=AUTO)),
    hooks=HookRegistry(),
)
agent = Agent(runtime)

async def main():
    msg = InboundMessage(
        user_id="cli", msg_id="1", text="What's 7 + 5?",
        content_parts=[{"type": "text", "text": "What's 7 + 5?"}],
    )
    await agent.handle(msg)

asyncio.run(main())
```

For a fully wired production application — depth-aware Gemma
classifier, dual-tier memory, WhatsApp transport with dead-URL
hygiene, transparency-reporting hooks — see the sibling `ongiini/`
package in this repository.

---

## What's in the box

| Module | Job |
|---|---|
| `owela.Agent` | Top-level orchestrator. `agent.handle(msg)` runs one turn. |
| `owela.Runtime` | Composition root. Frozen dataclass holding model, transport, memory, classifier, tools, hooks, policies. Built once at app startup. |
| `owela.executor.execute_turn` | The only orchestrator function (~180 LOC). No special cases. |
| `owela.Policy` + `owela.PolicyTable` | Loop-shape configuration indexed by (verdict, depth). |
| `owela.Step` (and subclasses) | Typed dataclasses for each phase of a turn. |
| `owela.tool` (decorator) | Declares a Python async function as a tool. Auto-generates the OpenAI schema from type hints + docstring. |
| `owela.ToolRegistry` | Holds tools and parallel-dispatches them via `asyncio.gather`. |
| `owela.Hook` + `owela.HookRegistry` | Observe step events. Soft-fail by contract. |
| `owela.Classifier` (protocol) | Decides the turn's verdict + depth. |
| `owela.Model` (protocol) | One LLM round-trip. Adapter surfaces engine knobs. |
| `owela.Transport` (protocol) | The medium. Owns reply hygiene + transport metadata. |
| `owela.MemoryProvider` (protocol) | Assembles messages + records turns. |
| `owela.hooks_builtin.MemoryRecordingHook` | Opt-in hook for persistence via the provider. |

About 1700 lines of framework code, ~900 lines of tests. Designed to be
read end-to-end in an afternoon.

---

## Status

v0, in production. Powers the **Ongiini** WhatsApp helper (a free
chatbot for users in Namibia, built on Gemma 4 26B running on an NVIDIA
DGX Spark). The whole stack — framework, application, and infrastructure
— lives in this repository.

API stability:
- `Step`, `Policy`, `PolicyTable`, `tool`, `Runtime`, `Agent` —
  considered stable for v0.
- The Planner / Reviewer / interstitial-UX hooks are designed in but
  not yet wired (v1 work).
- Streaming responses, multi-tenant Runtime, async-iteration over
  steps — not yet supported.

If you adopt Owela today, you should be comfortable reading the source.

---

## Why "Owela"

[Oshiwambo](https://en.wikipedia.org/wiki/Ovambo_language) for "thread."
Honors the project's roots (the application that birthed the framework
is for users in Namibia) and the literal architecture (a turn is a
thread of typed steps).

---

## License

MIT. See [`LICENSE`](../LICENSE).

---

## See also

- [`CLAUDE.md`](./CLAUDE.md) — extension guide + anti-trap rules for
  contributors (humans and AI agents alike).
- [`ongiini/README.md`](../ongiini/README.md) — the production
  application built on Owela. Read this for a concrete end-to-end
  example of every protocol in the framework.
- [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
  — the "augmented LLM" pattern Owela formalises for the
  inference-engine-on-messenger shape.
