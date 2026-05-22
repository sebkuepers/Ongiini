"""mem0 LLM provider that records token usage against the calling user.

mem0's default vllm provider creates an OpenAI client and returns only
the text content from chat completions — `response.usage` is discarded.
That makes mem0's extraction + reconciliation calls invisible to our
`usage` counter even though they bill real vLLM tokens against the
Spark's capacity. For the user-facing 1M-tokens-per-month allowance
that's a real gap: a user sending 10 messages a day was effectively
getting double the LLM compute the counter showed.

This module subclasses mem0's VllmLLM to:
  1. Capture `response.usage` from every chat completion
  2. Post it to `ongiini.usage.record()` against the msisdn in
     `_current_msisdn`
  3. Tag the log line with `kind=memory` so it's distinguishable

`install()` does an import-time monkey-patch of
`mem0.utils.factory.LlmFactory.provider_to_class["vllm"]` so mem0's
existing `provider: "vllm"` config picks up our wrapper transparently —
no change to the mem0 config needed.

`_current_msisdn` is set by `mem.add_turn` etc. before calling mem0.
Background calls (e.g. warmup) leave it None and the record is skipped
rather than billing an unknown user.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import logging

from mem0.llms.vllm import VllmLLM as _VllmLLM
from mem0.utils.factory import LlmFactory

from .. import usage

log = logging.getLogger("ongiini.mem.llm")

_current_msisdn: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ongiini_current_msisdn", default=None
)


class TrackedVllmLLM(_VllmLLM):
    """VllmLLM subclass that emits a usage.record() per chat completion."""

    def generate_response(
        self, messages, response_format=None, tools=None, tool_choice="auto", **kwargs
    ):
        params = self._get_supported_params(messages=messages, **kwargs)
        params.update({"model": self.config.model, "messages": messages})
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)

        try:
            msisdn = _current_msisdn.get()
            if msisdn:
                billable_in, completion, _cached = usage.billable_from_usage(
                    response.usage
                )
                if billable_in or completion:
                    usage.record(
                        msisdn,
                        billable_in,
                        completion,
                        used_search=False,
                        kind="memory",
                    )
        except Exception as exc:
            log.warning("mem0 usage record failed: %s", exc)

        return self._parse_response(response, tools)


_installed = False
_orig_executor_submit = concurrent.futures.ThreadPoolExecutor.submit


def _ctx_aware_submit(self, fn, /, *args, **kwargs):
    """Wrap submitted callables so contextvars propagate into the worker.

    Stock ThreadPoolExecutor.submit starts the worker thread with a fresh
    context — so the `_current_msisdn` we set in mem.add_turn is invisible
    inside mem0's `_add_to_vector_store` thread, where the LLM call
    actually happens. Wrapping the callable in `copy_context().run` makes
    it inherit the caller's contextvars. This is the idiomatic Python
    pattern for the same problem (see PEP 567).
    """
    ctx = contextvars.copy_context()
    return _orig_executor_submit(self, ctx.run, fn, *args, **kwargs)


def install() -> None:
    """Patch mem0's LlmFactory so the `vllm` provider resolves to us,
    AND patch ThreadPoolExecutor.submit so contextvars survive into the
    worker threads mem0 spawns for `_add_to_vector_store`.

    Safe to call multiple times — second and subsequent calls are no-ops.
    Called from ongiini.memory.long_term at module load, before any mem0 Memory is built.
    """
    global _installed
    if _installed:
        return
    entry = LlmFactory.provider_to_class.get("vllm")
    if entry is None:
        log.warning("mem0 LlmFactory missing 'vllm' entry; tracking disabled")
        return
    # entry is (module_path, ConfigClass). Swap the module path for ours
    # while keeping the same config class so mem0's create() path works.
    LlmFactory.provider_to_class["vllm"] = (
        "ongiini.memory.long_term_llm.TrackedVllmLLM",
        entry[1],
    )
    # Global patch — affects all ThreadPoolExecutor.submit calls in the
    # process, but the change is transparent for code that doesn't read
    # contextvars (which is the vast majority of stdlib + library code).
    concurrent.futures.ThreadPoolExecutor.submit = _ctx_aware_submit
    _installed = True
    log.info(
        "installed TrackedVllmLLM for mem0 'vllm' provider "
        "and patched ThreadPoolExecutor.submit for contextvar propagation"
    )
