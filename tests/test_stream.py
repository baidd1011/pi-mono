# tests/test_stream.py
"""Tests for pi_ai stream module."""

import pytest
import asyncio
import time

from pi_ai.stream import EventStream, AssistantMessageEventStream
from pi_ai.exceptions import StreamError
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    Usage,
    UserMessage,
    StartEvent,
    TextDeltaEvent,
    DoneEvent,
)


class TestEventStream:
    """Tests for EventStream class."""

    @pytest.mark.asyncio
    async def test_event_stream_push_and_iterate(self):
        """EventStream can push events and iterate over them."""
        stream: EventStream[str, str] = EventStream()
        stream.push("event1")
        stream.push("event2")
        stream.end("result")

        events = []
        async for event in stream:
            events.append(event)

        assert events == ["event1", "event2"]

    @pytest.mark.asyncio
    async def test_event_stream_result(self):
        """EventStream can return the final result."""
        stream: EventStream[str, int] = EventStream()
        stream.push("event")
        stream.end(42)

        result = await stream.result()
        assert result == 42

    @pytest.mark.asyncio
    async def test_event_stream_result_waits_for_end(self):
        """EventStream.result() waits for the stream to end."""
        stream: EventStream[str, str] = EventStream()

        async def end_later():
            await asyncio.sleep(0.01)
            stream.end("delayed result")

        task = asyncio.create_task(end_later())
        result = await stream.result()
        await task

        assert result == "delayed result"

    @pytest.mark.asyncio
    async def test_event_stream_push_after_end_raises(self):
        """EventStream.push() raises error after stream ended."""
        stream: EventStream[str, str] = EventStream()
        stream.end("result")

        with pytest.raises(StreamError, match="Cannot push to an ended stream"):
            stream.push("late event")

    @pytest.mark.asyncio
    async def test_event_stream_end_twice_raises(self):
        """EventStream.end() raises error if called twice."""
        stream: EventStream[str, str] = EventStream()
        stream.end("result1")

        with pytest.raises(StreamError, match="Stream already ended"):
            stream.end("result2")

    @pytest.mark.asyncio
    async def test_event_stream_empty_stream(self):
        """EventStream can be empty and still return result."""
        stream: EventStream[str, str] = EventStream()
        stream.end("empty result")

        events = []
        async for event in stream:
            events.append(event)

        assert events == []
        result = await stream.result()
        assert result == "empty result"

    @pytest.mark.asyncio
    async def test_event_stream_different_types(self):
        """EventStream can handle different event and result types."""
        stream: EventStream[int, str] = EventStream()
        stream.push(1)
        stream.push(2)
        stream.push(3)
        stream.end("sum is 6")

        events = []
        async for event in stream:
            events.append(event)

        assert events == [1, 2, 3]
        result = await stream.result()
        assert result == "sum is 6"

    @pytest.mark.asyncio
    async def test_event_stream_concurrent_iteration(self):
        """EventStream can handle events added during iteration."""
        stream: EventStream[str, str] = EventStream()

        async def add_events():
            for i in range(3):
                stream.push(f"event{i}")
                await asyncio.sleep(0.005)
            stream.end("done")

        task = asyncio.create_task(add_events())

        events = []
        async for event in stream:
            events.append(event)

        await task

        assert events == ["event0", "event1", "event2"]
        result = await stream.result()
        assert result == "done"


class TestAssistantMessageEventStream:
    """Tests for AssistantMessageEventStream class."""

    @pytest.mark.asyncio
    async def test_assistant_message_event_stream_iteration(self):
        """AssistantMessageEventStream can iterate over events."""
        stream = AssistantMessageEventStream()

        start_event: StartEvent = {
            "type": "start",
            "api": "anthropic",
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514"
        }
        stream.push(start_event)

        delta_event: TextDeltaEvent = {
            "type": "textDelta",
            "index": 0,
            "delta": "Hello"
        }
        stream.push(delta_event)

        done_event: DoneEvent = {
            "type": "done",
            "response_id": "msg_123",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "total_tokens": 110,
                "cost": {"input": 0.001, "output": 0.002, "cache_read": 0.0, "cache_write": 0.0, "total": 0.003}
            },
            "stop_reason": "stop"
        }
        stream.push(done_event)

        # End with explicit AssistantMessage
        message = AssistantMessage(
            content=[{"type": "text", "text": "Hello"}],
            api="anthropic",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            response_id="msg_123",
            usage=Usage(
                input_tokens=100,
                output_tokens=10,
                total_tokens=110,
                cost=Usage.Cost(input=0.001, output=0.002, total=0.003)
            ),
            stop_reason="stop",
            timestamp=int(time.time() * 1000)
        )
        stream.end(message)

        events = []
        async for event in stream:
            events.append(event)

        assert len(events) == 3
        assert events[0]["type"] == "start"
        assert events[1]["type"] == "textDelta"
        assert events[2]["type"] == "done"

    @pytest.mark.asyncio
    async def test_assistant_message_event_stream_result(self):
        """AssistantMessageEventStream can return the final AssistantMessage."""
        stream = AssistantMessageEventStream()

        message = AssistantMessage(
            content=[{"type": "text", "text": "Hello"}],
            api="anthropic",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            usage=Usage(
                input_tokens=100,
                output_tokens=10,
                total_tokens=110,
                cost=Usage.Cost(input=0.001, output=0.002, total=0.003)
            ),
            stop_reason="stop",
            timestamp=int(time.time() * 1000)
        )
        stream.end(message)

        result = await stream.result()
        assert isinstance(result, AssistantMessage)
        assert result.api == "anthropic"
        assert result.model == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_assistant_message_event_stream_inherits_from_event_stream(self):
        """AssistantMessageEventStream inherits from EventStream."""
        from pi_ai.stream import EventStream

        stream = AssistantMessageEventStream()
        assert isinstance(stream, EventStream)

    @pytest.mark.asyncio
    async def test_assistant_message_event_stream_push_after_end_raises(self):
        """AssistantMessageEventStream.push raises error after stream ended."""
        stream = AssistantMessageEventStream()

        message = AssistantMessage(
            content=[{"type": "text", "text": "Hello"}],
            api="anthropic",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            usage=Usage(
                input_tokens=100,
                output_tokens=10,
                total_tokens=110,
                cost=Usage.Cost(input=0.001, output=0.002, total=0.003)
            ),
            stop_reason="stop",
            timestamp=int(time.time() * 1000)
        )
        stream.end(message)

        with pytest.raises(StreamError, match="Cannot push to an ended stream"):
            stream.push({"type": "textDelta", "index": 0, "delta": "late"})

    @pytest.mark.asyncio
    async def test_assistant_message_event_stream_empty(self):
        """AssistantMessageEventStream can be empty."""
        stream = AssistantMessageEventStream()

        message = AssistantMessage(
            content=[],
            api="anthropic",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            usage=Usage(
                input_tokens=100,
                output_tokens=0,
                total_tokens=100,
                cost=Usage.Cost(input=0.001, output=0.0, total=0.001)
            ),
            stop_reason="stop",
            timestamp=int(time.time() * 1000)
        )
        stream.end(message)

        events = []
        async for event in stream:
            events.append(event)

        assert events == []
        result = await stream.result()
        assert result.content == []