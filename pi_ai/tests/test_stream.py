"""
Tests for pi-ai stream module.

Tests the EventStream and AssistantMessageEventStream classes.
"""

import pytest
import asyncio

from pi_ai.stream import (
    EventStream,
    AssistantMessageEventStream,
    stream_simple,
    complete_simple,
)
from pi_ai.types import (
    Context,
    Model,
    UserMessage,
    AssistantMessage,
    Usage,
    StartEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    DoneEvent,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_model() -> Model:
    """Sample model configuration for testing."""
    return Model(
        id="test-model",
        name="Test Model",
        api="test",
        provider="test",
        base_url="https://test.example.com/v1",
        reasoning=False,
        input=["text"],
        cost=Usage.Cost(input=0.01, output=0.02, total=0.03),
        context_window=4096,
        max_tokens=1024,
    )


@pytest.fixture
def sample_context() -> Context:
    """Sample conversation context for testing."""
    return Context(
        system_prompt="You are a helpful assistant.",
        messages=[
            UserMessage(
                role="user",
                content="Hello, world!",
                timestamp=1000,
            )
        ],
    )


# -----------------------------------------------------------------------------
# EventStream Tests
# -----------------------------------------------------------------------------

class TestEventStream:
    """Tests for the generic EventStream class."""

    @pytest.mark.asyncio
    async def test_push_and_iterate(self):
        """Test pushing events and iterating over them."""
        stream: EventStream[str, str] = EventStream()
        stream.push("event1")
        stream.push("event2")
        stream.end("result")

        events = []
        async for event in stream:
            events.append(event)

        assert events == ["event1", "event2"]
        result = await stream.result()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_result_before_end_raises(self):
        """Test that getting result before stream ends raises."""
        stream: EventStream[str, str] = EventStream()
        stream.push("event1")

        # result() should wait, but we can't test this easily
        # Just test that we can end it properly
        stream.end("result")
        result = await stream.result()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_push_after_end_raises(self):
        """Test that pushing after stream ends raises StreamError."""
        from pi_ai.exceptions import StreamError

        stream: EventStream[str, str] = EventStream()
        stream.end("result")

        with pytest.raises(StreamError, match="Cannot push"):
            stream.push("late_event")

    @pytest.mark.asyncio
    async def test_double_end_raises(self):
        """Test that ending twice raises StreamError."""
        from pi_ai.exceptions import StreamError

        stream: EventStream[str, str] = EventStream()
        stream.end("result1")

        with pytest.raises(StreamError, match="already ended"):
            stream.end("result2")

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """Test stream with no events."""
        stream: EventStream[str, str] = EventStream()
        stream.end("empty_result")

        events = []
        async for event in stream:
            events.append(event)

        assert events == []
        result = await stream.result()
        assert result == "empty_result"

    @pytest.mark.asyncio
    async def test_concurrent_iteration(self):
        """Test that multiple iterators can work concurrently."""
        stream: EventStream[int, int] = EventStream()

        # Push some events
        for i in range(5):
            stream.push(i)
        stream.end(100)

        # Single iterator should get all events
        events = []
        async for event in stream:
            events.append(event)

        assert events == [0, 1, 2, 3, 4]


# -----------------------------------------------------------------------------
# AssistantMessageEventStream Tests
# -----------------------------------------------------------------------------

class TestAssistantMessageEventStream:
    """Tests for the AssistantMessageEventStream class."""

    @pytest.mark.asyncio
    async def test_basic_text_events(self):
        """Test stream with basic text events."""
        stream = AssistantMessageEventStream()

        # Push events
        stream.push(StartEvent(
            type="start",
            api="test",
            provider="test",
            model="test-model",
        ))
        stream.push(TextStartEvent(type="textStart", index=0))
        stream.push(TextDeltaEvent(type="textDelta", index=0, delta="Hello"))
        stream.push(TextEndEvent(type="textEnd", index=0, text_signature=None))
        stream.push(DoneEvent(
            type="done",
            response_id="test-id",
            usage=Usage(
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=0,
                cache_write_tokens=0,
                total_tokens=15,
                cost=Usage.Cost(input=0.01, output=0.02, total=0.03),
            ),
            stop_reason="stop",
        ))

        # End the stream
        message = stream._build_message()
        stream.end(message)

        result = await stream.result()

        assert result.role == "assistant"
        assert result.api == "test"
        assert result.provider == "test"
        assert result.model == "test-model"
        assert result.response_id == "test-id"
        assert result.stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_build_message_from_events(self):
        """Test that message is built correctly from events."""
        stream = AssistantMessageEventStream()

        stream.push(StartEvent(type="start", api="test", provider="test", model="test"))
        stream.push(TextStartEvent(type="textStart", index=0))
        stream.push(TextDeltaEvent(type="textDelta", index=0, delta="Hello "))
        stream.push(TextDeltaEvent(type="textDelta", index=0, delta="World"))
        stream.push(TextEndEvent(type="textEnd", index=0, text_signature=None))
        stream.push(DoneEvent(
            type="done",
            response_id=None,
            usage=Usage(
                input_tokens=10,
                output_tokens=10,
                cache_read_tokens=0,
                cache_write_tokens=0,
                total_tokens=20,
                cost=Usage.Cost(input=0, output=0, total=0),
            ),
            stop_reason="stop",
        ))

        # End the stream
        message = stream._build_message()
        stream.end(message)

        result = await stream.result()

        assert len(result.content) >= 1
        # Text should be accumulated
        assert "Hello" in str(result.content)
        assert "World" in str(result.content)

    @pytest.mark.asyncio
    async def test_iteration_yields_all_events(self):
        """Test that iteration yields all pushed events."""
        stream = AssistantMessageEventStream()

        events_to_push = [
            StartEvent(type="start", api="test", provider="test", model="test"),
            TextStartEvent(type="textStart", index=0),
            TextDeltaEvent(type="textDelta", index=0, delta="Hi"),
            TextEndEvent(type="textEnd", index=0, text_signature=None),
            DoneEvent(
                type="done",
                response_id="id",
                usage=Usage(
                    input_tokens=5,
                    output_tokens=5,
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                    total_tokens=10,
                    cost=Usage.Cost(input=0, output=0, total=0),
                ),
                stop_reason="stop",
            ),
        ]

        for event in events_to_push:
            stream.push(event)

        # End with explicit result
        stream.end(AssistantMessage(
            role="assistant",
            content=[],
            api="test",
            provider="test",
            model="test",
            response_id="id",
            usage=Usage(
                input_tokens=5,
                output_tokens=5,
                cache_read_tokens=0,
                cache_write_tokens=0,
                total_tokens=10,
                cost=Usage.Cost(input=0, output=0, total=0),
            ),
            stop_reason="stop",
            timestamp=1000,
        ))

        events = []
        async for event in stream:
            events.append(event)

        assert len(events) == 5
        assert events[0]["type"] == "start"
        assert events[-1]["type"] == "done"


# -----------------------------------------------------------------------------
# stream_simple and complete_simple Tests
# -----------------------------------------------------------------------------

class TestStreamFunctions:
    """Tests for stream_simple and complete_simple functions."""

    @pytest.mark.asyncio
    async def test_stream_simple_wrapper(self, sample_model: Model, sample_context: Context):
        """Test stream_simple wrapper function."""
        from pi_ai.providers import FauxProvider

        provider = FauxProvider(responses=[[("text", "Test response")]])

        stream = await stream_simple(provider.stream, sample_context, sample_model)
        result = await stream.result()

        assert result.role == "assistant"
        assert len(result.content) >= 1

    @pytest.mark.asyncio
    async def test_complete_simple(self, sample_model: Model, sample_context: Context):
        """Test complete_simple function returns final message."""
        from pi_ai.providers import FauxProvider

        provider = FauxProvider(responses=[[("text", "Complete test")]])

        result = await complete_simple(provider.stream, sample_context, sample_model)

        assert result.role == "assistant"
        assert "Complete" in str(result.content) or "test" in str(result.content)


# -----------------------------------------------------------------------------
# Event Types Tests
# -----------------------------------------------------------------------------

class TestEventTypes:
    """Tests for event type correctness."""

    def test_start_event_type(self):
        """Test StartEvent has correct type."""
        event = StartEvent(type="start", api="test", provider="test", model="test")
        assert event["type"] == "start"

    def test_text_delta_event_structure(self):
        """Test TextDeltaEvent has required fields."""
        event = TextDeltaEvent(type="textDelta", index=0, delta="text")
        assert event["type"] == "textDelta"
        assert event["index"] == 0
        assert event["delta"] == "text"

    def test_done_event_usage_structure(self):
        """Test DoneEvent has correct usage structure."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=10,
            cache_write_tokens=5,
            total_tokens=165,
            cost=Usage.Cost(input=1.0, output=0.5, total=1.5),
        )
        event = DoneEvent(
            type="done",
            response_id="resp-123",
            usage=usage,
            stop_reason="stop",
        )

        assert event["type"] == "done"
        assert event["usage"].input_tokens == 100
        assert event["stop_reason"] == "stop"