"""Regression tests for the PostHog kill-switch in ongiini/__init__.py.

Root cause of the 2026-05-24 thread-leak incident: mem0 spawns a
``posthog.Posthog(...)`` client during its telemetry module init; the
Posthog constructor immediately spawns a ``Consumer`` background thread
that blocks forever on ``queue.get()``. Setting ``MEM0_TELEMETRY=False``
only flips ``client.disabled = True`` AFTER instantiation, so the
already-running consumer thread persists. Every ``AnonymousTelemetry()``
call spawned another thread, growing linearly with conversations until
the kernel starved the healthcheck and autoheal restarted the container.

Fix: ``ongiini/__init__.py`` monkey-patches ``posthog.Posthog`` to a
no-op stub BEFORE mem0 (or anyone else) can import the real class.
These tests pin that behaviour.
"""
from __future__ import annotations

import threading

import pytest


# posthog is a prod-only dep (pulled transitively via mem0). The unit-test
# env may not have it. Each test that touches posthog uses importorskip
# so it runs in CI / container builds but silently skips on a stripped
# dev env — without weakening the regression guarantee where it matters.


def test_posthog_is_patched_to_no_op_when_ongiini_is_imported():
    """Importing the ongiini package must replace posthog.Posthog with
    our no-op stub. Without this, mem0's telemetry import spawns a
    Consumer thread that never exits."""
    posthog = pytest.importorskip("posthog")
    # The act of importing ongiini.* should already have patched posthog.
    import ongiini  # noqa: F401 (the import is the test)

    # The patched class must be ours, not the real posthog Client.
    assert posthog.Posthog.__name__ == "_NoOpPosthog", (
        "ongiini/__init__.py is supposed to monkey-patch posthog.Posthog "
        f"to a no-op stub, but it's still {posthog.Posthog!r}. This means "
        "the kill-switch is not running early enough, or someone removed "
        "it. Restore it — without it the webhook leaks threads."
    )


def test_no_op_posthog_does_not_spawn_threads():
    """Instantiating the no-op stub must not spawn any background thread.
    The whole point of the patch is to avoid posthog.Consumer threads."""
    posthog = pytest.importorskip("posthog")
    import ongiini  # noqa: F401

    before = threading.active_count()
    # Spawn 100 stubs — if any spawn a thread, count goes up
    stubs = [posthog.Posthog(project_api_key="dummy", host="https://x") for _ in range(100)]
    after = threading.active_count()

    assert after == before, (
        f"_NoOpPosthog should not spawn threads, but creating 100 stubs "
        f"raised thread count from {before} to {after}. Likely someone "
        f"reverted the patch, or the stub class is starting a real "
        f"posthog.Consumer somewhere."
    )
    # Use stubs so a linter doesn't flag the loop as a no-op
    assert all(s.disabled for s in stubs)


def test_no_op_posthog_has_required_attributes():
    """mem0's telemetry code reads/writes specific attributes on the
    Posthog client (``capture()``, ``shutdown()``, ``disabled``). The
    stub must implement all of them so mem0 doesn't crash."""
    posthog = pytest.importorskip("posthog")
    import ongiini  # noqa: F401

    stub = posthog.Posthog(project_api_key="dummy")
    # mem0 sets disabled=True after instantiation
    assert hasattr(stub, "disabled")
    stub.disabled = True
    # mem0 calls capture() to fire events
    assert callable(stub.capture)
    stub.capture(event="test", distinct_id="x")
    # mem0 calls shutdown() on cleanup
    assert callable(stub.shutdown)
    stub.shutdown()


def test_mem0_telemetry_import_does_not_spawn_threads():
    """End-to-end: importing mem0's telemetry module (which is what
    triggers the leak in production) should not spawn any background
    thread. This test fails if our kill-switch ever stops working.

    Skipped if mem0 isn't installed (unit-test env doesn't ship it)."""
    pytest.importorskip("mem0.memory.telemetry")
    # Confirm posthog is already patched by the ongiini import
    import ongiini  # noqa: F401
    import posthog
    assert posthog.Posthog.__name__ == "_NoOpPosthog"

    before = threading.active_count()
    # Re-importing telemetry: if the singleton already ran without
    # the patch, this won't help — but the patch was applied at
    # ongiini import time which is before any mem0 import in our
    # actual entrypoint chain.
    import importlib

    import mem0.memory.telemetry as t
    importlib.reload(t)
    # Plus actively instantiate an AnonymousTelemetry — this is what
    # mem0 does inside get_oss_telemetry()
    if hasattr(t, "AnonymousTelemetry"):
        t.AnonymousTelemetry()
    after = threading.active_count()

    assert after == before, (
        f"Importing mem0.memory.telemetry spawned {after - before} "
        f"new thread(s) — the PostHog kill-switch is not effective."
    )
