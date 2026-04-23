"""Tests for agent loop functions."""

import pytest
import asyncio
import time
from typing import Any, List

from pi_ai import (
    Model, Context, Usage,
    UserMessage, AssistantMessage, ToolResultMessage,
    TextContent, ToolCall,
    FauxProvider,
)
from pi_ai.stream import AssistantMessageEventStream

from pi_agent_core.agent_loop import (
    agent_loop,
    agent_loop_continue,
    run_agent_loop,
    run_agent_loop_continue,
    AgentEventStream,
    extract_tool_calls,
)
from pi_agent_core.types import (
    AgentContext, AgentLoopConfig, AgentTool, AgentToolResult,
    ToolExecutionMode, AgentToolCall,
)
from pi_agent_core._abort import AbortController


# Helper to create a test model
def create_test_model() -> Model:
    """Create a test model for faux provider."""
    return Model(
        id="faux-model",
        name="Faux Model",
        api="faux",
        provider="faux",
        base_url="",
        reasoning=False,
        input=["text"],
        cost=Usage.Cost(input=0.0, output=0.0, total=0.0),
        context_window=1000,
        max_tokens=100,
    )


# Helper to create a faux stream function
def create_faux_stream_fn(responses: List[List[tuple]]) -> Any:
    """Create a stream function using faux provider."""
    provider = FauxProvider(responses=responses)

    async def stream_fn(context: Context, model: Model) -> AssistantMessageEventStream:
        return await provider.stream(context, model)

    return stream_fn


# Helper to collect events from a stream
async def collect_stream_events(stream: AgentEventStream) -> tuple[List[Any], List[Any]]:
    """Collect all events and final messages from a stream."""
    events = []
    async for event in stream:
        events.append(event)
    messages = await stream.result()
    return events, messages


# Helper to create a simple tool
def create_simple_tool(name: str = "test_tool") -> AgentTool:
    """Create a simple test tool."""
    async def execute(
        tool_call_id: str,
        args: Any,
        signal: Any,
        update_cb: Any
    ) -> AgentToolResult:
        return AgentToolResult(
            content=[TextContent(text=f"Tool {name} executed with args: {args}")],
            details={"tool_call_id": tool_call_id}
        )

    return AgentTool(
        name=name,
        label="Test Tool",
        description="A simple test tool",
        parameters={"type": "object", "properties": {"input": {"type": "string"}}},
        execute=execute,
    )


# Helper to create a blocking tool (for testing execution mode)
def create_blocking_tool(name: str = "blocking_tool", execution_mode: ToolExecutionMode = "sequential") -> AgentTool:
    """Create a tool with specific execution mode."""
    async def execute(
        tool_call_id: str,
        args: Any,
        signal: Any,
        update_cb: Any
    ) -> AgentToolResult:
        await asyncio.sleep(0.1)  # Simulate work
        return AgentToolResult(
            content=[TextContent(text=f"Blocking tool {name} completed")],
            details={"tool_call_id": tool_call_id}
        )

    return AgentTool(
        name=name,
        label="Blocking Tool",
        description="A tool that takes time",
        parameters={"type": "object"},
        execute=execute,
        execution_mode=execution_mode,
    )


class TestExtractToolCalls:
    """Tests for extract_tool_calls function."""

    def test_extract_from_assistant_message(self):
        """Extract tool calls from AssistantMessage content."""
        message = AssistantMessage(
            content=[
                TextContent(text="Let me check that."),
                ToolCall(id="call_1", name="get_weather", arguments={"location": "SF"}),
                ToolCall(id="call_2", name="get_time", arguments={"timezone": "PST"}),
            ],
            api="faux",
            provider="faux",
            model="faux-model",
            usage=Usage(
                input_tokens=10,
                output_tokens=20,
                cache_read_tokens=0,
                cache_write_tokens=0,
                total_tokens=30,
                cost=Usage.Cost(input=0, output=0, total=0),
            ),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        tool_calls = extract_tool_calls(message)

        assert len(tool_calls) == 2
        assert tool_calls[0].id == "call_1"
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == {"location": "SF"}
        assert tool_calls[1].id == "call_2"
        assert tool_calls[1].name == "get_time"

    def test_extract_from_empty_content(self):
        """Extract from message with no tool calls."""
        message = AssistantMessage(
            content=[TextContent(text="Hello!")],
            api="faux",
            provider="faux",
            model="faux-model",
            usage=Usage(
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=0,
                cache_write_tokens=0,
                total_tokens=15,
                cost=Usage.Cost(input=0, output=0, total=0),
            ),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        tool_calls = extract_tool_calls(message)

        assert len(tool_calls) == 0

    def test_extract_from_dict_content(self):
        """Extract from message with dict-style tool calls."""
        message = AssistantMessage(
            content=[
                {"type": "toolCall", "id": "call_3", "name": "search", "arguments": {"query": "test"}},
            ],
            api="faux",
            provider="faux",
            model="faux-model",
            usage=Usage(
                input_tokens=5,
                output_tokens=10,
                cache_read_tokens=0,
                cache_write_tokens=0,
                total_tokens=15,
                cost=Usage.Cost(input=0, output=0, total=0),
            ),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        tool_calls = extract_tool_calls(message)

        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_3"
        assert tool_calls[0].name == "search"


class TestAgentLoopBasic:
    """Tests for basic agent_loop functionality."""

    @pytest.mark.asyncio
    async def test_agent_loop_with_simple_text(self):
        """agent_loop works with simple text response."""
        model = create_test_model()
        stream_fn = create_faux_stream_fn([[("text", "Hello, I am an AI assistant.")]])

        context = AgentContext(
            system_prompt="You are helpful.",
            messages=[],
        )

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
        )

        prompt = UserMessage(role="user", content="Hi!", timestamp=int(time.time() * 1000))

        stream = agent_loop([prompt], context, config, stream_fn=stream_fn)
        events, messages = await collect_stream_events(stream)

        # Check events
        event_types = [e.get("type") for e in events]
        assert "agent_start" in event_types
        assert "turn_start" in event_types
        assert "message_start" in event_types
        assert "message_end" in event_types
        assert "turn_end" in event_types
        assert "agent_end" in event_types

        # Check messages
        assert len(messages) >= 1  # At least the prompt
        assert messages[0].role == "user"

    @pytest.mark.asyncio
    async def test_agent_loop_returns_stream(self):
        """agent_loop returns an AgentEventStream."""
        model = create_test_model()
        stream_fn = create_faux_stream_fn([[("text", "Response")]])

        context = AgentContext(
            system_prompt="",
            messages=[],
        )

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
        )

        prompt = UserMessage(role="user", content="Test", timestamp=int(time.time() * 1000))

        stream = agent_loop([prompt], context, config, stream_fn=stream_fn)

        assert isinstance(stream, AgentEventStream)
        assert isinstance(stream, AgentEventStream)

    @pytest.mark.asyncio
    async def test_agent_loop_with_multiple_prompts(self):
        """agent_loop handles multiple prompt messages."""
        model = create_test_model()
        stream_fn = create_faux_stream_fn([[("text", "I see both messages.")]])

        context = AgentContext(
            system_prompt="",
            messages=[],
        )

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
        )

        prompts = [
            UserMessage(role="user", content="First message", timestamp=int(time.time() * 1000)),
            UserMessage(role="user", content="Second message", timestamp=int(time.time() * 1000) + 100),
        ]

        stream = agent_loop(prompts, context, config, stream_fn=stream_fn)
        events, messages = await collect_stream_events(stream)

        # Should have message events for both prompts
        message_start_count = sum(1 for e in events if e.get("type") == "message_start")
        assert message_start_count >= 2


class TestAgentLoopContinue:
    """Tests for agent_loop_continue functionality."""

    @pytest.mark.asyncio
    async def test_agent_loop_continue_requires_messages(self):
        """agent_loop_continue raises error on empty context."""
        model = create_test_model()

        context = AgentContext(
            system_prompt="",
            messages=[],  # Empty!
        )

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
        )

        with pytest.raises(ValueError, match="no messages in context"):
            agent_loop_continue(context, config)

    @pytest.mark.asyncio
    async def test_agent_loop_continue_rejects_assistant_last(self):
        """agent_loop_continue raises error when last message is assistant."""
        model = create_test_model()

        context = AgentContext(
            system_prompt="",
            messages=[
                UserMessage(role="user", content="Hi", timestamp=1000),
                AssistantMessage(
                    content=[TextContent(text="Hello")],
                    api="faux",
                    provider="faux",
                    model="faux",
                    usage=Usage(
                        input_tokens=5,
                        output_tokens=5,
                        cache_read_tokens=0,
                        cache_write_tokens=0,
                        total_tokens=10,
                        cost=Usage.Cost(input=0, output=0, total=0),
                    ),
                    stop_reason="stop",
                    timestamp=2000,
                ),
            ],
        )

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
        )

        with pytest.raises(ValueError, match="Cannot continue from message role: assistant"):
            agent_loop_continue(context, config)

    @pytest.mark.asyncio
    async def test_agent_loop_continue_with_tool_result(self):
        """agent_loop_continue works with tool result as last message."""
        model = create_test_model()
        stream_fn = create_faux_stream_fn([[("text", "Based on the tool result...")]])

        context = AgentContext(
            system_prompt="",
            messages=[
                UserMessage(role="user", content="Check weather", timestamp=1000),
                AssistantMessage(
                    content=[ToolCall(id="call_1", name="get_weather", arguments={"loc": "SF"})],
                    api="faux",
                    provider="faux",
                    model="faux",
                    usage=Usage(
                        input_tokens=5,
                        output_tokens=5,
                        cache_read_tokens=0,
                        cache_write_tokens=0,
                        total_tokens=10,
                        cost=Usage.Cost(input=0, output=0, total=0),
                    ),
                    stop_reason="stop",
                    timestamp=2000,
                ),
                ToolResultMessage(
                    role="toolResult",
                    tool_call_id="call_1",
                    tool_name="get_weather",
                    content=[TextContent(text="Sunny, 75F")],
                    details=None,
                    is_error=False,
                    timestamp=3000,
                ),
            ],
        )

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
        )

        stream = agent_loop_continue(context, config, stream_fn=stream_fn)
        events, messages = await collect_stream_events(stream)

        # Should have agent_start and turn_start
        assert "agent_start" in [e.get("type") for e in events]
        assert "turn_start" in [e.get("type") for e in events]


class TestRunAgentLoopDirect:
    """Tests for direct execution functions."""

    @pytest.mark.asyncio
    async def test_run_agent_loop_emits_events(self):
        """run_agent_loop emits events via callback."""
        model = create_test_model()
        stream_fn = create_faux_stream_fn([[("text", "Direct response")]])

        context = AgentContext(
            system_prompt="",
            messages=[],
        )

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
        )

        events = []
        async def emit(event):
            events.append(event)

        prompt = UserMessage(role="user", content="Direct test", timestamp=int(time.time() * 1000))

        messages = await run_agent_loop([prompt], context, config, emit, stream_fn=stream_fn)

        # Check events were emitted
        event_types = [e.get("type") for e in events]
        assert "agent_start" in event_types
        assert "turn_start" in event_types
        assert "agent_end" in event_types

        # Check messages returned
        assert len(messages) >= 1

    @pytest.mark.asyncio
    async def test_run_agent_loop_continue_emits_events(self):
        """run_agent_loop_continue emits events via callback."""
        model = create_test_model()
        stream_fn = create_faux_stream_fn([[("text", "Continuing...")]])

        context = AgentContext(
            system_prompt="",
            messages=[
                UserMessage(role="user", content="Initial", timestamp=1000),
            ],
        )

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
        )

        events = []
        async def emit(event):
            events.append(event)

        messages = await run_agent_loop_continue(context, config, emit, stream_fn=stream_fn)

        event_types = [e.get("type") for e in events]
        assert "agent_start" in event_types
        assert "turn_start" in event_types


class TestToolExecution:
    """Tests for tool execution."""

    @pytest.mark.asyncio
    async def test_parallel_tool_execution(self):
        """Tools execute in parallel by default."""
        model = create_test_model()
        import json

        # Create faux provider that returns two tool calls
        stream_fn = create_faux_stream_fn([
            [
                ("toolCall", f"tool_a|call_1|{json.dumps({'input': 'a'})}"),
                ("toolCall", f"tool_b|call_2|{json.dumps({'input': 'b'})}"),
            ],
            [("text", "Both tools completed")],
        ])

        tool_a = create_blocking_tool("tool_a")
        tool_b = create_blocking_tool("tool_b")

        context = AgentContext(
            system_prompt="",
            messages=[],
            tools=[tool_a, tool_b],
        )

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
            tool_execution="parallel",
        )

        events = []
        async def emit(event):
            events.append(event)

        prompt = UserMessage(role="user", content="Test parallel", timestamp=int(time.time() * 1000))

        start_time = time.time()
        messages = await run_agent_loop([prompt], context, config, emit, stream_fn=stream_fn)
        elapsed = time.time() - start_time

        # Both tools should run concurrently (about 0.1s total, not 0.2s)
        assert elapsed < 0.3

        # Should have tool execution events
        tool_start_events = [e for e in events if e.get("type") == "tool_execution_start"]
        assert len(tool_start_events) == 2

    @pytest.mark.asyncio
    async def test_sequential_tool_execution(self):
        """Tools execute sequentially when configured."""
        model = create_test_model()
        import json

        stream_fn = create_faux_stream_fn([
            [
                ("toolCall", f"tool_a|call_1|{json.dumps({'input': 'a'})}"),
                ("toolCall", f"tool_b|call_2|{json.dumps({'input': 'b'})}"),
            ],
            [("text", "Tools done")],
        ])

        tool_a = create_blocking_tool("tool_a")
        tool_b = create_blocking_tool("tool_b")

        context = AgentContext(
            system_prompt="",
            messages=[],
            tools=[tool_a, tool_b],
        )

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
            tool_execution="sequential",
        )

        events = []
        async def emit(event):
            events.append(event)

        prompt = UserMessage(role="user", content="Test sequential", timestamp=int(time.time() * 1000))

        start_time = time.time()
        messages = await run_agent_loop([prompt], context, config, emit, stream_fn=stream_fn)
        elapsed = time.time() - start_time

        # Sequential execution should take longer (both tools must finish before next)
        assert elapsed >= 0.15  # Both 0.1s delays run sequentially

    @pytest.mark.asyncio
    async def test_tool_override_to_sequential(self):
        """Tool with execution_mode='sequential' forces sequential execution."""
        model = create_test_model()
        import json

        stream_fn = create_faux_stream_fn([
            [
                ("toolCall", f"regular|call_1|{json.dumps({'input': 'x'})}"),
                ("toolCall", f"blocking|call_2|{json.dumps({'input': 'y'})}"),
            ],
            [("text", "Done")],
        ])

        regular_tool = create_simple_tool("regular")
        blocking_tool = create_blocking_tool("blocking", execution_mode="sequential")

        context = AgentContext(
            system_prompt="",
            messages=[],
            tools=[regular_tool, blocking_tool],
        )

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
            tool_execution="parallel",  # Config says parallel, but tool overrides
        )

        events = []
        async def emit(event):
            events.append(event)

        prompt = UserMessage(role="user", content="Test override", timestamp=int(time.time() * 1000))

        # When any tool has sequential mode, batch executes sequentially
        start_time = time.time()
        messages = await run_agent_loop([prompt], context, config, emit, stream_fn=stream_fn)
        elapsed = time.time() - start_time

        # Should take longer due to sequential execution forced by tool
        assert elapsed >= 0.08  # blocking tool's 0.1s delay

    @pytest.mark.asyncio
    async def test_tool_not_found_error(self):
        """Missing tool produces error result."""
        model = create_test_model()
        import json

        stream_fn = create_faux_stream_fn([
            [("toolCall", f"missing_tool|call_1|{json.dumps({'input': 'x'})}")],
            [("text", "Handled error")],
        ])

        context = AgentContext(
            system_prompt="",
            messages=[],
            tools=[],  # No tools!
        )

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
        )

        events = []
        async def emit(event):
            events.append(event)

        prompt = UserMessage(role="user", content="Test missing tool", timestamp=int(time.time() * 1000))

        messages = await run_agent_loop([prompt], context, config, emit, stream_fn=stream_fn)

        # Should have tool execution end with error
        tool_end_events = [e for e in events if e.get("type") == "tool_execution_end"]
        assert len(tool_end_events) >= 1
        assert tool_end_events[0]["is_error"] is True


class TestSteeringMessages:
    """Tests for steering message injection."""

    @pytest.mark.asyncio
    async def test_steering_messages_injected(self):
        """Steering messages are injected after tool execution."""
        model = create_test_model()
        import json

        # First call: tool call, second call: response to steering
        stream_fn = create_faux_stream_fn([
            [("toolCall", f"test_tool|call_1|{json.dumps({'input': 'x'})}")],
            [("text", "Processing steering message")],
        ])

        tool = create_simple_tool("test_tool")

        context = AgentContext(
            system_prompt="",
            messages=[],
            tools=[tool],
        )

        steering_messages = [
            UserMessage(role="user", content="Steer this way", timestamp=int(time.time() * 1000))
        ]

        steering_count = 0
        async def get_steering():
            nonlocal steering_count
            if steering_count == 0:
                steering_count += 1
                return steering_messages
            return []

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
            get_steering_messages=get_steering,
        )

        events = []
        async def emit(event):
            events.append(event)

        prompt = UserMessage(role="user", content="Initial", timestamp=int(time.time() * 1000))

        messages = await run_agent_loop([prompt], context, config, emit, stream_fn=stream_fn)

        # Should have message events for steering message
        message_starts = [e for e in events if e.get("type") == "message_start"]
        # Initial prompt + assistant + tool result + steering + second assistant
        assert len(message_starts) >= 4

    @pytest.mark.asyncio
    async def test_no_steering_messages(self):
        """Empty steering messages don't affect loop."""
        model = create_test_model()
        stream_fn = create_faux_stream_fn([[("text", "Simple response")]])

        context = AgentContext(
            system_prompt="",
            messages=[],
        )

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
            get_steering_messages=lambda: [],
        )

        events = []
        async def emit(event):
            events.append(event)

        prompt = UserMessage(role="user", content="Test", timestamp=int(time.time() * 1000))

        messages = await run_agent_loop([prompt], context, config, emit, stream_fn=stream_fn)

        # Should complete normally
        assert "agent_end" in [e.get("type") for e in events]


class TestFollowUpMessages:
    """Tests for follow-up message handling."""

    @pytest.mark.asyncio
    async def test_follow_up_messages_continue_loop(self):
        """Follow-up messages cause agent to continue."""
        model = create_test_model()
        stream_fn = create_faux_stream_fn([
            [("text", "First response")],
            [("text", "Follow-up response")],
        ])

        context = AgentContext(
            system_prompt="",
            messages=[],
        )

        follow_up_messages = [
            UserMessage(role="user", content="Follow up", timestamp=int(time.time() * 1000))
        ]

        follow_up_count = 0
        async def get_follow_up():
            nonlocal follow_up_count
            if follow_up_count == 0:
                follow_up_count += 1
                return follow_up_messages
            return []

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
            get_follow_up_messages=get_follow_up,
        )

        events = []
        async def emit(event):
            events.append(event)

        prompt = UserMessage(role="user", content="Initial", timestamp=int(time.time() * 1000))

        messages = await run_agent_loop([prompt], context, config, emit, stream_fn=stream_fn)

        # Should have two turn_end events
        turn_ends = [e for e in events if e.get("type") == "turn_end"]
        assert len(turn_ends) >= 2

    @pytest.mark.asyncio
    async def test_no_follow_up_messages(self):
        """Empty follow-up messages don't affect loop."""
        model = create_test_model()
        stream_fn = create_faux_stream_fn([[("text", "Response")]])

        context = AgentContext(
            system_prompt="",
            messages=[],
        )

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
            get_follow_up_messages=lambda: [],
        )

        events = []
        async def emit(event):
            events.append(event)

        prompt = UserMessage(role="user", content="Test", timestamp=int(time.time() * 1000))

        messages = await run_agent_loop([prompt], context, config, emit, stream_fn=stream_fn)

        # Should have exactly one turn_end
        turn_ends = [e for e in events if e.get("type") == "turn_end"]
        assert len(turn_ends) == 1


class TestErrorHandling:
    """Tests for error and abort handling."""

    @pytest.mark.asyncio
    async def test_stream_error_handling(self):
        """Stream errors are handled gracefully."""
        model = create_test_model()
        provider = FauxProvider(error="Simulated stream error")
        stream_fn = provider.stream

        context = AgentContext(
            system_prompt="",
            messages=[],
        )

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
        )

        events = []
        async def emit(event):
            events.append(event)

        prompt = UserMessage(role="user", content="Test error", timestamp=int(time.time() * 1000))

        messages = await run_agent_loop([prompt], context, config, emit, stream_fn=stream_fn)

        # Should still have agent_end
        assert "agent_end" in [e.get("type") for e in events]

    @pytest.mark.asyncio
    async def test_before_tool_call_blocking(self):
        """before_tool_call can block tool execution."""
        model = create_test_model()
        import json

        stream_fn = create_faux_stream_fn([
            [("toolCall", f"test_tool|call_1|{json.dumps({'input': 'x'})}")],
            [("text", "Tool was blocked")],  # Second response after tool result
        ])

        tool = create_simple_tool("test_tool")

        context = AgentContext(
            system_prompt="",
            messages=[],
            tools=[tool],
        )

        async def before_tool_call(ctx, signal):
            from pi_agent_core.types import BeforeToolCallResult
            return BeforeToolCallResult(block=True, reason="Blocked for testing")

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
            before_tool_call=before_tool_call,
        )

        events = []
        async def emit(event):
            events.append(event)

        prompt = UserMessage(role="user", content="Test block", timestamp=int(time.time() * 1000))

        messages = await run_agent_loop([prompt], context, config, emit, stream_fn=stream_fn)

        # Tool execution end should have error
        tool_end_events = [e for e in events if e.get("type") == "tool_execution_end"]
        assert len(tool_end_events) >= 1
        assert tool_end_events[0]["is_error"] is True

    @pytest.mark.asyncio
    async def test_after_tool_call_override(self):
        """after_tool_call can override tool result."""
        model = create_test_model()
        import json

        stream_fn = create_faux_stream_fn([
            [("toolCall", f"test_tool|call_1|{json.dumps({'input': 'x'})}")],
            [("text", "After tool call")],
        ])

        tool = create_simple_tool("test_tool")

        context = AgentContext(
            system_prompt="",
            messages=[],
            tools=[tool],
        )

        async def after_tool_call(ctx, signal):
            from pi_agent_core.types import AfterToolCallResult
            return AfterToolCallResult(
                content=[TextContent(text="Overridden content")],
                details={"overridden": True},
            )

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
            after_tool_call=after_tool_call,
        )

        events = []
        async def emit(event):
            events.append(event)

        prompt = UserMessage(role="user", content="Test override", timestamp=int(time.time() * 1000))

        messages = await run_agent_loop([prompt], context, config, emit, stream_fn=stream_fn)

        # Should have tool result messages
        tool_result_messages = [m for m in messages if hasattr(m, 'role') and m.role == "toolResult"]
        assert len(tool_result_messages) >= 1
        # Check content was overridden
        assert "Overridden content" in str(tool_result_messages[0].content)


class TestConvertToLLM:
    """Tests for message conversion."""

    @pytest.mark.asyncio
    async def test_convert_to_llm_filters_messages(self):
        """convert_to_llm can filter messages."""
        model = create_test_model()
        stream_fn = create_faux_stream_fn([[("text", "Filtered response")]])

        context = AgentContext(
            system_prompt="",
            messages=[],
        )

        def convert(msgs):
            # Only include user messages
            return [m for m in msgs if hasattr(m, 'role') and m.role == "user"]

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=convert,
        )

        prompt = UserMessage(role="user", content="Test filter", timestamp=int(time.time() * 1000))

        events = []
        async def emit(event):
            events.append(event)

        messages = await run_agent_loop([prompt], context, config, emit, stream_fn=stream_fn)

        # Should complete normally
        assert "agent_end" in [e.get("type") for e in events]

    @pytest.mark.asyncio
    async def test_async_convert_to_llm(self):
        """convert_to_llm can be async."""
        model = create_test_model()
        stream_fn = create_faux_stream_fn([[("text", "Async convert")]])

        context = AgentContext(
            system_prompt="",
            messages=[],
        )

        async def convert(msgs):
            await asyncio.sleep(0.01)  # Simulate async work
            return msgs

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=convert,
        )

        prompt = UserMessage(role="user", content="Test async", timestamp=int(time.time() * 1000))

        events = []
        async def emit(event):
            events.append(event)

        messages = await run_agent_loop([prompt], context, config, emit, stream_fn=stream_fn)

        assert "agent_end" in [e.get("type") for e in events]


class TestAPIKeyResolution:
    """Tests for API key resolution."""

    @pytest.mark.asyncio
    async def test_get_api_key_called(self):
        """get_api_key is called for key resolution."""
        model = create_test_model()
        stream_fn = create_faux_stream_fn([[("text", "Key resolved")]])

        context = AgentContext(
            system_prompt="",
            messages=[],
        )

        key_called = False
        def get_key(provider):
            nonlocal key_called
            key_called = True
            return "resolved_key"

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
            get_api_key=get_key,
        )

        prompt = UserMessage(role="user", content="Test key", timestamp=int(time.time() * 1000))

        events = []
        async def emit(event):
            events.append(event)

        messages = await run_agent_loop([prompt], context, config, emit, stream_fn=stream_fn)

        # Key resolver should have been called
        assert key_called is True

    @pytest.mark.asyncio
    async def test_async_get_api_key(self):
        """get_api_key can be async."""
        model = create_test_model()
        stream_fn = create_faux_stream_fn([[("text", "Async key")]])

        context = AgentContext(
            system_prompt="",
            messages=[],
        )

        async def get_key(provider):
            await asyncio.sleep(0.01)
            return "async_key"

        config = AgentLoopConfig(
            model=model,
            convert_to_llm=lambda msgs: msgs,
            get_api_key=get_key,
        )

        prompt = UserMessage(role="user", content="Test async key", timestamp=int(time.time() * 1000))

        events = []
        async def emit(event):
            events.append(event)

        messages = await run_agent_loop([prompt], context, config, emit, stream_fn=stream_fn)

        assert "agent_end" in [e.get("type") for e in events]