"""Ongiini package init.

The ONLY runtime work this file does is kill PostHog at import time —
see the comment block below. Everything else is normal sub-module
resolution. If you need package-wide config, put it in ``ongiini.config``
not here; this file fires very early in the import chain and pulling
heavy modules from it would slow cold-start measurably.
"""

# ── Kill PostHog before mem0 (or anything else) can spawn a Consumer
# ── thread. Root cause of the 2026-05-24 leak: mem0 unconditionally
# ── creates a ``posthog.Posthog(...)`` client during its telemetry
# ── module import; the Posthog client's constructor spawns a Consumer
# ── thread that blocks forever on ``queue.get()``. Setting
# ── ``MEM0_TELEMETRY=False`` only flips ``client.disabled = True``
# ── AFTER instantiation — the thread is already alive and never gets
# ── joined. Every ``AnonymousTelemetry()`` call spawns another one,
# ── and ``get_oss_telemetry()`` is invoked per ``Memory()`` init, so
# ── thread count grows linearly with conversations until the kernel
# ── starts starving the healthcheck (~8000 threads) and autoheal
# ── restarts the container.
#
# Fix: replace ``posthog.Posthog`` with a no-op stub BEFORE any other
# import that might pull in mem0. Because this file is
# ``ongiini/__init__.py``, it runs as the very first thing whenever
# ``ongiini.*`` is imported — including ``ongiini.api.main`` which
# uvicorn loads at startup.
#
# Why monkey-patch rather than uninstall posthog: posthog is a hard
# dependency of mem0 (declared in mem0's pyproject.toml). Uninstalling
# it would break mem0's import. We need the symbol to exist so
# ``from posthog import Posthog`` succeeds — we just don't want any
# actual telemetry threads spawned.
try:
    import posthog as _posthog

    class _NoOpPosthog:
        """Drop-in replacement for ``posthog.Posthog`` that doesn't
        spawn any threads. Implements just enough surface for the
        ``mem0.memory.telemetry`` consumer to think it has a client."""

        disabled = True

        def __init__(self, *_args, **_kwargs):  # noqa: D401
            # Set the public attrs mem0 reads/writes so attribute access
            # doesn't blow up. mem0 specifically does
            # ``self.posthog.disabled = True`` after construction.
            self.disabled = True

        def capture(self, *_args, **_kwargs):
            return None

        def shutdown(self, *_args, **_kwargs):
            return None

        # Other methods mem0 / posthog clients sometimes call. All
        # no-op — we never want to actually send telemetry.
        def identify(self, *_args, **_kwargs):
            return None

        def set(self, *_args, **_kwargs):
            return None

        def group(self, *_args, **_kwargs):
            return None

        def alias(self, *_args, **_kwargs):
            return None

        def flush(self, *_args, **_kwargs):
            return None

    _posthog.Posthog = _NoOpPosthog  # type: ignore[assignment]
except ImportError:
    # posthog isn't installed (e.g. a stripped test env) — nothing
    # to patch, nothing leaks. Carry on.
    pass
