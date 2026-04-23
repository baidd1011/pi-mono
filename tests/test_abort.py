# tests/test_abort.py
"""Tests for AbortController and AbortSignal."""

import pytest
import asyncio


@pytest.mark.asyncio
async def test_abort_controller_basic():
    """AbortController can abort signal."""
    from pi_agent_core._abort import AbortController

    controller = AbortController()
    assert controller.signal.aborted is False

    controller.abort()
    assert controller.signal.aborted is True


@pytest.mark.asyncio
async def test_abort_signal_wait():
    """AbortSignal can be awaited for abort."""
    from pi_agent_core._abort import AbortController

    controller = AbortController()

    # Start waiting in background
    task = asyncio.create_task(controller.signal.wait_for_abort())

    # Abort after short delay
    await asyncio.sleep(0.1)
    controller.abort()

    # Wait should complete
    await asyncio.wait_for(task, timeout=0.5)
    assert controller.signal.aborted is True


@pytest.mark.asyncio
async def test_abort_signal_already_aborted():
    """AbortSignal.wait_for_abort returns immediately if already aborted."""
    from pi_agent_core._abort import AbortController

    controller = AbortController()
    controller.abort()

    # Should return immediately, not hang
    await asyncio.wait_for(controller.signal.wait_for_abort(), timeout=0.1)
    assert controller.signal.aborted is True


@pytest.mark.asyncio
async def test_abort_signal_multiple_waiters():
    """Multiple waiters are all notified on abort."""
    from pi_agent_core._abort import AbortController

    controller = AbortController()

    # Start multiple waiters
    tasks = [asyncio.create_task(controller.signal.wait_for_abort()) for _ in range(3)]

    # Give tasks time to start and register
    await asyncio.sleep(0.01)
    assert len(controller.signal._waiters) == 3

    # Abort after short delay
    await asyncio.sleep(0.1)
    controller.abort()

    # All waiters should complete
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=0.5)
    assert controller.signal.aborted is True

    # Waiters list should be cleared after abort
    assert len(controller.signal._waiters) == 0


@pytest.mark.asyncio
async def test_abort_signal_waiter_cancelled():
    """Cancelled waiter is removed from waiters list."""
    from pi_agent_core._abort import AbortController

    controller = AbortController()

    # Start waiter
    task = asyncio.create_task(controller.signal.wait_for_abort())

    # Give it a moment to start and register
    await asyncio.sleep(0.01)
    assert len(controller.signal._waiters) == 1

    # Cancel the waiter
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Waiter should be removed from list
    assert len(controller.signal._waiters) == 0

    # Signal should still work for other waiters
    task2 = asyncio.create_task(controller.signal.wait_for_abort())
    await asyncio.sleep(0.01)
    assert len(controller.signal._waiters) == 1

    controller.abort()
    await asyncio.wait_for(task2, timeout=0.5)
    assert len(controller.signal._waiters) == 0