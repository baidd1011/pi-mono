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