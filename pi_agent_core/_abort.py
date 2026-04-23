"""Abort handling for Python."""

import asyncio
from typing import List


class AbortSignal:
    """Python equivalent of JavaScript AbortSignal."""

    def __init__(self):
        self._aborted: bool = False
        self._waiters: List[asyncio.Future] = []

    @property
    def aborted(self) -> bool:
        return self._aborted

    async def wait_for_abort(self) -> None:
        """Wait until this signal is aborted."""
        if self._aborted:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        future: asyncio.Future = loop.create_future()
        self._waiters.append(future)

        try:
            await future
        except asyncio.CancelledError:
            pass
        finally:
            # Remove from waiters if still present (already aborted clears list)
            if future in self._waiters:
                self._waiters.remove(future)

    def _notify_abort(self) -> None:
        """Notify all waiters of abort."""
        for waiter in self._waiters:
            if not waiter.done():
                waiter.set_result(None)
        self._waiters.clear()


class AbortController:
    """Python equivalent of JavaScript AbortController."""

    def __init__(self):
        self._signal = AbortSignal()

    @property
    def signal(self) -> AbortSignal:
        return self._signal

    def abort(self) -> None:
        """Abort the signal."""
        self._signal._aborted = True
        self._signal._notify_abort()