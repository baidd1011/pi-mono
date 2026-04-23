# tests/test_agent_types.py
"""Tests for pi_agent_core tool types."""

import pytest


class TestAgentToolResult:
    """Tests for AgentToolResult model."""

    def test_agent_tool_result_creation(self):
        """AgentToolResult can be created with content and details."""
        from pi_agent_core.types import AgentToolResult
        from pi_ai import TextContent

        result = AgentToolResult(
            content=[TextContent(text="Result")],
            details={"processed": True}
        )
        assert len(result.content) == 1
        assert result.content[0].text == "Result"
        assert result.details == {"processed": True}

    def test_agent_tool_result_with_image(self):
        """AgentToolResult can contain ImageContent."""
        from pi_agent_core.types import AgentToolResult
        from pi_ai import TextContent, ImageContent

        result = AgentToolResult(
            content=[
                TextContent(text="Here's the image:"),
                ImageContent(data="base64data", mime_type="image/png")
            ],
            details=None
        )
        assert len(result.content) == 2
        assert result.content[1].type == "image"


class TestAgentToolCall:
    """Tests for AgentToolCall model."""

    def test_agent_tool_call_creation(self):
        """AgentToolCall can be created with required fields."""
        from pi_agent_core.types import AgentToolCall

        call = AgentToolCall(
            id="call_123",
            name="get_weather",
            arguments={"city": "Beijing"}
        )
        assert call.id == "call_123"
        assert call.name == "get_weather"
        assert call.arguments == {"city": "Beijing"}

    def test_agent_tool_call_empty_arguments(self):
        """AgentToolCall can have empty arguments."""
        from pi_agent_core.types import AgentToolCall

        call = AgentToolCall(id="call_456", name="ping", arguments={})
        assert call.arguments == {}


class TestToolExecutionMode:
    """Tests for ToolExecutionMode literal type."""

    def test_tool_execution_mode_values(self):
        """ToolExecutionMode accepts valid values."""
        from pi_agent_core.types import ToolExecutionMode

        # Just verify the type exists
        assert ToolExecutionMode is not None
        # Valid values are: "sequential", "parallel"


class TestAgentTool:
    """Tests for AgentTool model."""

    def test_agent_tool_creation(self):
        """AgentTool defines a tool for agent execution."""
        from pi_agent_core.types import AgentTool, AgentToolResult
        from pi_ai import TextContent

        async def execute(tool_call_id, args, signal, update_cb):
            return AgentToolResult(
                content=[TextContent(text="Result")],
                details={"processed": True}
            )

        tool = AgentTool(
            name="test_tool",
            label="Test Tool",
            description="A test tool",
            parameters={"type": "object"},
            execute=execute
        )

        assert tool.name == "test_tool"
        assert tool.label == "Test Tool"
        assert tool.description == "A test tool"
        assert tool.parameters == {"type": "object"}
        assert tool.execution_mode is None
        assert tool.prepare_arguments is None

    def test_agent_tool_with_execution_mode(self):
        """AgentTool can have execution_mode set."""
        from pi_agent_core.types import AgentTool, AgentToolResult, ToolExecutionMode
        from pydantic import BaseModel
        from pi_ai import TextContent

        class SearchParams(BaseModel):
            query: str
            limit: int = 10

        async def execute(tool_call_id, args, signal, update_cb):
            return AgentToolResult(
                content=[TextContent(text="Search results")],
                details={"count": 5}
            )

        tool = AgentTool(
            name="search",
            label="Search",
            description="Search for items",
            parameters=SearchParams,
            execute=execute,
            execution_mode="sequential"
        )

        assert tool.execution_mode == "sequential"
        assert tool.parameters == SearchParams

    def test_agent_tool_with_prepare_arguments(self):
        """AgentTool can have prepare_arguments callback."""
        from pi_agent_core.types import AgentTool, AgentToolResult
        from pi_ai import TextContent

        def prepare_args(raw_args):
            # Transform raw args to validated params
            return {"query": raw_args.get("q", ""), "limit": 10}

        async def execute(tool_call_id, args, signal, update_cb):
            return AgentToolResult(
                content=[TextContent(text=f"Results for: {args['query']}")],
                details=None
            )

        tool = AgentTool(
            name="search",
            label="Search",
            description="Search for items",
            parameters={"type": "object"},
            prepare_arguments=prepare_args,
            execute=execute
        )

        assert tool.prepare_arguments is not None
        # Test the prepare_arguments function
        prepared = tool.prepare_arguments({"q": "test"})
        assert prepared == {"query": "test", "limit": 10}

    def test_agent_tool_with_pydantic_parameters(self):
        """AgentTool can use Pydantic model for parameters."""
        from pi_agent_core.types import AgentTool, AgentToolResult
        from pydantic import BaseModel
        from pi_ai import TextContent

        class WeatherParams(BaseModel):
            city: str
            unit: str = "celsius"

        async def execute(tool_call_id, args, signal, update_cb):
            return AgentToolResult(
                content=[TextContent(text=f"Weather for {args.city}")],
                details={"temp": 25}
            )

        tool = AgentTool(
            name="get_weather",
            label="Get Weather",
            description="Get weather for a city",
            parameters=WeatherParams,
            execute=execute
        )

        assert tool.name == "get_weather"
        assert tool.parameters == WeatherParams

    def test_agent_tool_execute_callable(self):
        """AgentTool execute function is callable."""
        from pi_agent_core.types import AgentTool, AgentToolResult
        from pi_ai import TextContent
        import asyncio

        async def execute(tool_call_id, args, signal, update_cb):
            return AgentToolResult(
                content=[TextContent(text="Done")],
                details=None
            )

        tool = AgentTool(
            name="test",
            label="Test",
            description="Test tool",
            parameters={"type": "object"},
            execute=execute
        )

        # Verify execute is callable
        result = asyncio.run(tool.execute("call_1", {}, None, None))
        assert result.content[0].text == "Done"

    def test_agent_tool_update_callback(self):
        """AgentTool can stream partial results via update callback."""
        from pi_agent_core.types import AgentTool, AgentToolResult, AgentToolUpdateCallback
        from pi_ai import TextContent
        import asyncio

        updates = []

        async def execute(tool_call_id, args, signal, update_cb):
            # Stream partial updates
            if update_cb:
                update_cb(AgentToolResult(
                    content=[TextContent(text="Partial 1")],
                    details={"progress": 50}
                ))
                update_cb(AgentToolResult(
                    content=[TextContent(text="Partial 2")],
                    details={"progress": 100}
                ))
            return AgentToolResult(
                content=[TextContent(text="Final")],
                details={"progress": 100}
            )

        def on_update(partial_result):
            updates.append(partial_result)

        tool = AgentTool(
            name="streaming_tool",
            label="Streaming Tool",
            description="A tool that streams updates",
            parameters={"type": "object"},
            execute=execute
        )

        result = asyncio.run(tool.execute("call_1", {}, None, on_update))
        assert result.content[0].text == "Final"
        assert len(updates) == 2
        assert updates[0].content[0].text == "Partial 1"
        assert updates[1].content[0].text == "Partial 2"


class TestAgentState:
    """Tests for AgentState model."""

    def test_agent_state_creation(self):
        """AgentState holds current agent state."""
        from pi_agent_core.types import AgentState
        from pi_ai import Model, Usage

        model = Model(
            id="gpt-4o", name="GPT-4o", api="openai", provider="openai",
            base_url="https://api.openai.com/v1", reasoning=False,
            input=["text"], cost=Usage.Cost(input=5, output=15, cache_read=0, cache_write=0, total=20),
            context_window=128000, max_tokens=4096
        )

        state = AgentState(
            system_prompt="Hello",
            model=model,
            thinking_level="off",
            tools=[],
            messages=[],
            is_streaming=False,
            streaming_message=None,
            pending_tool_calls=frozenset(),
            error_message=None
        )

        assert state.system_prompt == "Hello"
        assert state.is_streaming is False

    def test_agent_state_with_tools_and_messages(self):
        """AgentState can hold tools and messages."""
        from pi_agent_core.types import AgentState, AgentTool, AgentToolResult
        from pi_ai import Model, Usage, TextContent

        model = Model(
            id="gpt-4o", name="GPT-4o", api="openai", provider="openai",
            base_url="https://api.openai.com/v1", reasoning=False,
            input=["text"], cost=Usage.Cost(input=5, output=15, cache_read=0, cache_write=0, total=20),
            context_window=128000, max_tokens=4096
        )

        async def execute(tool_call_id, args, signal, update_cb):
            return AgentToolResult(content=[TextContent(text="Done")], details=None)

        tool = AgentTool(
            name="test_tool",
            label="Test Tool",
            description="A test tool",
            parameters={"type": "object"},
            execute=execute
        )

        state = AgentState(
            system_prompt="System prompt",
            model=model,
            thinking_level="medium",
            tools=[tool],
            messages=[],
            is_streaming=True,
            streaming_message=None,
            pending_tool_calls=frozenset(["call_123"]),
            error_message="Something went wrong"
        )

        assert len(state.tools) == 1
        assert state.tools[0].name == "test_tool"
        assert state.thinking_level == "medium"
        assert state.is_streaming is True
        assert "call_123" in state.pending_tool_calls
        assert state.error_message == "Something went wrong"


class TestAgentContext:
    """Tests for AgentContext model."""

    def test_agent_context_creation(self):
        """AgentContext holds context for LLM calls."""
        from pi_agent_core.types import AgentContext

        context = AgentContext(
            system_prompt="You are a helpful assistant",
            messages=[]
        )

        assert context.system_prompt == "You are a helpful assistant"
        assert context.messages == []
        assert context.tools is None

    def test_agent_context_with_tools(self):
        """AgentContext can include tools."""
        from pi_agent_core.types import AgentContext, AgentTool, AgentToolResult
        from pi_ai import TextContent

        async def execute(tool_call_id, args, signal, update_cb):
            return AgentToolResult(content=[TextContent(text="Done")], details=None)

        tool = AgentTool(
            name="get_weather",
            label="Get Weather",
            description="Get weather info",
            parameters={"type": "object"},
            execute=execute
        )

        context = AgentContext(
            system_prompt="Helpful assistant",
            messages=[],
            tools=[tool]
        )

        assert context.tools is not None
        assert len(context.tools) == 1
        assert context.tools[0].name == "get_weather"


class TestAgentEventTypes:
    """Tests for AgentEvent types."""

    def test_agent_start_event_creation(self):
        """AgentStartEvent can be created with type field."""
        from pi_agent_core.types import AgentStartEvent

        event: AgentStartEvent = {"type": "agent_start"}
        assert event["type"] == "agent_start"

    def test_agent_end_event_creation(self):
        """AgentEndEvent can be created with messages field."""
        from pi_agent_core.types import AgentEndEvent

        event: AgentEndEvent = {
            "type": "agent_end",
            "messages": []
        }
        assert event["type"] == "agent_end"
        assert event["messages"] == []

    def test_turn_start_event_creation(self):
        """TurnStartEvent can be created with type field."""
        from pi_agent_core.types import TurnStartEvent

        event: TurnStartEvent = {"type": "turn_start"}
        assert event["type"] == "turn_start"

    def test_turn_end_event_creation(self):
        """TurnEndEvent can be created with message and tool_results."""
        from pi_agent_core.types import TurnEndEvent

        event: TurnEndEvent = {
            "type": "turn_end",
            "message": {"role": "assistant", "content": []},
            "tool_results": []
        }
        assert event["type"] == "turn_end"
        assert event["message"]["role"] == "assistant"
        assert event["tool_results"] == []

    def test_message_start_event_creation(self):
        """MessageStartEvent can be created with message field."""
        from pi_agent_core.types import MessageStartEvent

        event: MessageStartEvent = {
            "type": "message_start",
            "message": {"role": "assistant", "content": []}
        }
        assert event["type"] == "message_start"
        assert event["message"]["role"] == "assistant"

    def test_message_update_event_creation(self):
        """MessageUpdateEvent can be created with message and assistant_message_event."""
        from pi_agent_core.types import MessageUpdateEvent

        event: MessageUpdateEvent = {
            "type": "message_update",
            "message": {"role": "assistant", "content": []},
            "assistant_message_event": {"type": "text", "text": "Hello"}
        }
        assert event["type"] == "message_update"
        assert event["assistant_message_event"]["type"] == "text"

    def test_message_end_event_creation(self):
        """MessageEndEvent can be created with message field."""
        from pi_agent_core.types import MessageEndEvent

        event: MessageEndEvent = {
            "type": "message_end",
            "message": {"role": "assistant", "content": []}
        }
        assert event["type"] == "message_end"

    def test_tool_execution_start_event_creation(self):
        """ToolExecutionStartEvent can be created with all fields."""
        from pi_agent_core.types import ToolExecutionStartEvent

        event: ToolExecutionStartEvent = {
            "type": "tool_execution_start",
            "tool_call_id": "call_123",
            "tool_name": "get_weather",
            "args": {"city": "Beijing"}
        }
        assert event["type"] == "tool_execution_start"
        assert event["tool_call_id"] == "call_123"
        assert event["tool_name"] == "get_weather"
        assert event["args"] == {"city": "Beijing"}

    def test_tool_execution_update_event_creation(self):
        """ToolExecutionUpdateEvent can be created with all fields."""
        from pi_agent_core.types import ToolExecutionUpdateEvent

        event: ToolExecutionUpdateEvent = {
            "type": "tool_execution_update",
            "tool_call_id": "call_123",
            "tool_name": "get_weather",
            "args": {"city": "Beijing"},
            "partial_result": {"temperature": 25}
        }
        assert event["type"] == "tool_execution_update"
        assert event["tool_call_id"] == "call_123"
        assert event["tool_name"] == "get_weather"
        assert event["args"] == {"city": "Beijing"}
        assert event["partial_result"] == {"temperature": 25}

    def test_tool_execution_end_event_creation(self):
        """ToolExecutionEndEvent can be created with is_error field."""
        from pi_agent_core.types import ToolExecutionEndEvent

        event: ToolExecutionEndEvent = {
            "type": "tool_execution_end",
            "tool_call_id": "call_123",
            "tool_name": "get_weather",
            "result": {"temperature": 25, "humidity": 60},
            "is_error": False
        }
        assert event["type"] == "tool_execution_end"
        assert event["tool_call_id"] == "call_123"
        assert event["tool_name"] == "get_weather"
        assert event["result"] == {"temperature": 25, "humidity": 60}
        assert event["is_error"] is False

    def test_tool_execution_end_event_with_error(self):
        """ToolExecutionEndEvent can represent an error result."""
        from pi_agent_core.types import ToolExecutionEndEvent

        event: ToolExecutionEndEvent = {
            "type": "tool_execution_end",
            "tool_call_id": "call_456",
            "tool_name": "divide",
            "result": "Division by zero",
            "is_error": True
        }
        assert event["is_error"] is True
        assert event["result"] == "Division by zero"

    def test_agent_event_union_type(self):
        """AgentEvent union type can hold any event type."""
        from pi_agent_core.types import AgentEvent, AgentStartEvent, ToolExecutionEndEvent

        start_event: AgentEvent = {"type": "agent_start"}
        assert start_event["type"] == "agent_start"

        end_event: AgentEvent = {
            "type": "tool_execution_end",
            "tool_call_id": "call_789",
            "tool_name": "test",
            "result": None,
            "is_error": False
        }
        assert end_event["type"] == "tool_execution_end"

    def test_agent_event_sink_type(self):
        """AgentEventSink is a callable type for emitting events."""
        from pi_agent_core.types import AgentEventSink, AgentStartEvent

        received_events = []

        def sink(event: AgentEventSink) -> None:
            received_events.append(event)

        # Verify the type is callable
        assert callable(sink)

        # Can be called with an event
        event: AgentStartEvent = {"type": "agent_start"}
        sink(event)
        assert len(received_events) == 1
        assert received_events[0]["type"] == "agent_start"