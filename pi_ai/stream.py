"""
pi-ai stream module.

Generic async streaming utilities for LLM responses.
"""

from typing import Generic, TypeVar, Callable, Awaitable
from collections.abc import AsyncIterator
import asyncio
import time

from pi_ai.exceptions import StreamError
from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
    StartEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThinkingStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    DoneEvent,
    ErrorEvent,
)


# Generic type variables
T = TypeVar("T")  # Event type
R = TypeVar("R")  # Result type


class EventStream(Generic[T, R], AsyncIterator[T]):
    """
    Generic async iterator with result collection.

    A stream of events of type T that can be collected into a result of type R.
    Supports pushing events, ending with a result, and async iteration.

    Example:
        stream = EventStream[str, str]()
        stream.push("Hello")
        stream.push(" World")
        stream.end("Hello World")
        async for event in stream:
            print(event)
        result = await stream.result()
    """

    def __init__(self) -> None:
        """Initialize the event stream."""
        self._events: list[T] = []
        self._result: R | None = None
        self._ended = False
        self._event_available = asyncio.Event()
        self._result_available = asyncio.Event()
        self._index = 0

    def push(self, event: T) -> None:
        """
        Push an event to the stream.

        Args:
            event: The event to push.
        """
        if self._ended:
            raise StreamError("Cannot push to an ended stream")
        self._events.append(event)
        self._event_available.set()

    def end(self, result: R) -> None:
        """
        End the stream with a final result.

        Args:
            result: The final result of the stream.
        """
        if self._ended:
            raise StreamError("Stream already ended")
        self._result = result
        self._ended = True
        self._result_available.set()
        self._event_available.set()  # Signal to waiting iterators

    async def result(self) -> R:
        """
        Wait for and return the final result.

        Returns:
            The final result of the stream.

        Raises:
            StreamError: If the stream has not ended yet.
        """
        if self._result is not None:
            return self._result

        await self._result_available.wait()

        if self._result is None:
            raise StreamError("Stream ended without a result")

        return self._result

    def __aiter__(self) -> "EventStream[T, R]":
        """Return self as the async iterator."""
        return self

    async def __anext__(self) -> T:
        """
        Get the next event from the stream.

        Returns:
            The next event.

        Raises:
            StopAsyncIteration: When all events have been consumed.
        """
        while self._index >= len(self._events):
            if self._ended:
                raise StopAsyncIteration
            await self._event_available.wait()
            self._event_available.clear()

        event = self._events[self._index]
        self._index += 1
        return event


# -----------------------------------------------------------------------------
# AssistantMessageEventStream (Task 8)
# -----------------------------------------------------------------------------

class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    """
    EventStream specialization for AssistantMessage events.

    Provides typed streaming for assistant message content, including
    text, thinking, and tool call events.

    Example:
        stream = AssistantMessageEventStream()
        stream.push({"type": "start", "api": "anthropic", ...})
        stream.push({"type": "textDelta", "index": 0, "delta": "Hello"})
        stream.end(assistant_message)
        async for event in stream:
            print(event)
    """

    def __init__(self) -> None:
        """Initialize the assistant message event stream."""
        super().__init__()
        self._text_buffers: dict[int, str] = {}
        self._thinking_buffers: dict[int, str] = {}
        self._tool_call_buffers: dict[int, dict[str, str]] = {}
        self._start_info: dict[str, str] = {}
        self._done_info: dict[str, object] = {}

    def _build_message(self) -> AssistantMessage:
        """Build the final AssistantMessage from collected events."""
        content: list[TextContent | ThinkingContent | ToolCall] = []

        # Add text content blocks
        for index, text in sorted(self._text_buffers.items()):
            content.append(TextContent(text=text))

        # Add thinking content blocks
        for index, thinking in sorted(self._thinking_buffers.items()):
            content.append(ThinkingContent(thinking=thinking))

        # Add tool call content blocks
        for index, tool_info in sorted(self._tool_call_buffers.items()):
            import json
            arguments_str = tool_info.get("arguments", "{}")
            try:
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError:
                arguments = {}
            content.append(ToolCall(
                id=tool_info["id"],
                name=tool_info["name"],
                arguments=arguments
            ))

        # Build usage from done event
        usage_data = self._done_info.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cache_read_tokens=usage_data.get("cache_read_tokens", 0),
            cache_write_tokens=usage_data.get("cache_write_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            cost=Usage.Cost(
                input=usage_data.get("cost", {}).get("input", 0.0),
                output=usage_data.get("cost", {}).get("output", 0.0),
                cache_read=usage_data.get("cost", {}).get("cache_read", 0.0),
                cache_write=usage_data.get("cost", {}).get("cache_write", 0.0),
                total=usage_data.get("cost", {}).get("total", 0.0),
            )
        )

        return AssistantMessage(
            content=content,
            api=self._start_info.get("api", ""),
            provider=self._start_info.get("provider", ""),
            model=self._start_info.get("model", ""),
            response_id=self._done_info.get("response_id"),
            usage=usage,
            stop_reason=self._done_info.get("stop_reason", "stop"),
            timestamp=int(time.time() * 1000),
        )

    async def result(self) -> AssistantMessage:
        """
        Wait for and return the final AssistantMessage.

        Returns:
            The final AssistantMessage built from all events.
        """
        # If we have an explicit result, return it
        if self._result is not None:
            return self._result

        # Otherwise build from collected events
        await self._result_available.wait()

        if self._result is not None:
            return self._result

        # Build from events if no explicit result
        return self._build_message()


# Stream function types
StreamFunction = Callable[[Context, Model], Awaitable[AssistantMessageEventStream]]
CompleteFunction = Callable[[Context, Model], Awaitable[AssistantMessage]]


async def stream_simple(
    stream_func: StreamFunction,
    context: Context,
    model: Model,
) -> AssistantMessageEventStream:
    """
    Simple streaming function wrapper.

    Args:
        stream_func: The provider's stream function.
        context: The conversation context.
        model: The model to use.

    Returns:
        An AssistantMessageEventStream for iterating over events.
    """
    return await stream_func(context, model)


async def complete_simple(
    stream_func: StreamFunction,
    context: Context,
    model: Model,
) -> AssistantMessage:
    """
    Simple complete function - collects stream into final message.

    Args:
        stream_func: The provider's stream function.
        context: The conversation context.
        model: The model to use.

    Returns:
        The final AssistantMessage.
    """
    stream = await stream_func(context, model)
    return await stream.result()