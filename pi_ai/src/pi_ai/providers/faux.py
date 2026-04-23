"""
Faux provider implementation.

A mock provider for testing that returns predetermined responses
and emits configurable event sequences without making real API calls.
"""

import asyncio
from typing import Any, Sequence

from pi_ai.registry import ProviderInterface
from pi_ai.stream import AssistantMessageEventStream
from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    Model,
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


class FauxProvider(ProviderInterface):
    """
    Mock provider for testing purposes.

    Returns predetermined responses and emits configurable event sequences.
    Used in tests to simulate streaming without real API calls.

    Example:
        provider = FauxProvider(
            responses=[
                [("text", "Hello, world!")],
                [("thinking", "Let me think..."), ("text", "The answer is 42.")],
            ]
        )
        stream = await provider.stream(context, model)
        async for event in stream:
            print(event)
    """

    def __init__(
        self,
        responses: list[list[tuple[str, str]]] | None = None,
        events: list[list[AssistantMessageEvent]] | None = None,
        delay: float = 0.0,
        error: str | None = None,
    ):
        """
        Initialize the faux provider.

        Args:
            responses: List of response sequences. Each response is a list of
                       (type, content) tuples. Types: "text", "thinking".
            events: List of event sequences. Each sequence is a list of events
                    to emit. Takes precedence over responses if both provided.
            delay: Optional delay in seconds between events.
            error: Optional error message to emit instead of successful response.
        """
        self.responses = responses or [[("text", "Default faux response.")]]
        self.events = events
        self.delay = delay
        self.error = error
        self._call_count = 0

    def _build_events_from_response(
        self,
        response: list[tuple[str, str]],
        model: Model,
    ) -> list[AssistantMessageEvent]:
        """
        Build event sequence from a response tuple list.

        Args:
            response: List of (type, content) tuples.
            model: The model configuration.

        Returns:
            List of events to emit.
        """
        events: list[AssistantMessageEvent] = []

        # Start event
        events.append(StartEvent(
            type="start",
            api=model.api,
            provider=model.provider,
            model=model.id,
        ))

        text_index = 0
        thinking_index = 0

        for content_type, content in response:
            if content_type == "thinking":
                events.append(ThinkingStartEvent(
                    type="thinkingStart",
                    index=thinking_index,
                ))
                # Split into deltas for realistic simulation
                words = content.split()
                for i, word in enumerate(words):
                    delta = word if i == 0 else f" {word}"
                    events.append(ThinkingDeltaEvent(
                        type="thinkingDelta",
                        index=thinking_index,
                        delta=delta,
                    ))
                events.append(ThinkingEndEvent(
                    type="thinkingEnd",
                    index=thinking_index,
                    thinking_signature=None,
                ))
                thinking_index += 1

            elif content_type == "text":
                events.append(TextStartEvent(
                    type="textStart",
                    index=text_index,
                ))
                # Split into deltas for realistic simulation
                words = content.split()
                for i, word in enumerate(words):
                    delta = word if i == 0 else f" {word}"
                    events.append(TextDeltaEvent(
                        type="textDelta",
                        index=text_index,
                        delta=delta,
                    ))
                events.append(TextEndEvent(
                    type="textEnd",
                    index=text_index,
                    text_signature=None,
                ))
                text_index += 1

            elif content_type == "toolCall":
                # Parse tool call format: "name|id|arguments_json"
                parts = content.split("|", 2)
                name = parts[0] if len(parts) > 0 else "unknown"
                tc_id = parts[1] if len(parts) > 1 else "call_123"
                args_json = parts[2] if len(parts) > 2 else "{}"

                import json
                try:
                    args = json.loads(args_json)
                except json.JSONDecodeError:
                    args = {}

                events.append(ToolCallStartEvent(
                    type="toolCallStart",
                    index=0,
                    id=tc_id,
                    name=name,
                ))
                events.append(ToolCallDeltaEvent(
                    type="toolCallDelta",
                    index=0,
                    id=tc_id,
                    delta=args_json,
                ))
                events.append(ToolCallEndEvent(
                    type="toolCallEnd",
                    index=0,
                    id=tc_id,
                    arguments=args,
                ))

        # Done event
        events.append(DoneEvent(
            type="done",
            response_id=f"faux_response_{self._call_count}",
            usage=Usage(
                input_tokens=10,
                output_tokens=20,
                cache_read_tokens=0,
                cache_write_tokens=0,
                total_tokens=30,
                cost=Usage.Cost(
                    input=model.cost.input,
                    output=model.cost.output,
                    total=model.cost.total,
                )
            ),
            stop_reason="stop",
        ))

        return events

    async def stream(
        self,
        context: Context,
        model: Model,
    ) -> AssistantMessageEventStream:
        """
        Stream predetermined events without making real API calls.

        Args:
            context: The conversation context (ignored).
            model: The model configuration (used for metadata).

        Returns:
            An AssistantMessageEventStream with predetermined events.
        """
        stream_result = AssistantMessageEventStream()

        # Handle error case
        if self.error:
            stream_result.push(ErrorEvent(
                type="error",
                error_message=self.error,
            ))
            message = AssistantMessage(
                content=[],
                api=model.api,
                provider=model.provider,
                model=model.id,
                response_id=None,
                usage=Usage(
                    input_tokens=0,
                    output_tokens=0,
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                    total_tokens=0,
                    cost=Usage.Cost(input=0, output=0, total=0),
                ),
                stop_reason="error",
                error_message=self.error,
                timestamp=0,
            )
            stream_result.end(message)
            return stream_result

        # Get the response for this call
        response_index = self._call_count % len(self.responses)
        self._call_count += 1

        # Build events
        if self.events:
            event_sequence = self.events[response_index % len(self.events)]
        else:
            event_sequence = self._build_events_from_response(
                self.responses[response_index], model
            )

        # Emit events with optional delay
        for event in event_sequence:
            if self.delay > 0:
                await asyncio.sleep(self.delay)
            stream_result.push(event)

        # Build and end the stream
        message = stream_result._build_message()
        stream_result.end(message)
        return stream_result

    async def stream_simple(
        self,
        context: Context,
        model: Model,
    ) -> AssistantMessageEventStream:
        """
        Simple streaming wrapper - same as stream for faux provider.

        Args:
            context: The conversation context.
            model: The model configuration.

        Returns:
            An AssistantMessageEventStream with predetermined events.
        """
        return await self.stream(context, model)

    def reset(self) -> None:
        """Reset the call counter."""
        self._call_count = 0


# Predefined faux providers for common test scenarios

def create_text_faux_provider(text: str = "Hello, world!") -> FauxProvider:
    """Create a faux provider that returns a simple text response."""
    return FauxProvider(responses=[[("text", text)]])


def create_thinking_faux_provider(
    thinking: str = "Let me think about this...",
    text: str = "The answer is 42.",
) -> FauxProvider:
    """Create a faux provider that returns thinking followed by text."""
    return FauxProvider(responses=[[("thinking", thinking), ("text", text)]])


def create_tool_call_faux_provider(
    name: str = "get_weather",
    tc_id: str = "call_123",
    arguments: dict[str, Any] = None,
) -> FauxProvider:
    """Create a faux provider that returns a tool call."""
    import json
    args = arguments or {"location": "San Francisco"}
    args_json = json.dumps(args)
    return FauxProvider(
        responses=[[("toolCall", f"{name}|{tc_id}|{args_json}")]]
    )


def create_multi_response_faux_provider(
    responses: list[str],
) -> FauxProvider:
    """Create a faux provider that returns different responses on each call."""
    return FauxProvider(
        responses=[[("text", r)] for r in responses]
    )


def create_error_faux_provider(error: str = "Simulated error") -> FauxProvider:
    """Create a faux provider that always returns an error."""
    return FauxProvider(error=error)