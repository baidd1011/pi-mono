# tests/test_agent.py
"""Tests for Agent class state management."""

import pytest
from typing import Any


class TestAgentCreation:
    """Tests for Agent creation and initialization."""

    def test_agent_creation_with_defaults(self):
        """Agent can be created with default options."""
        from pi_agent_core.agent import Agent

        agent = Agent()
        state = agent.state

        assert state.system_prompt == ""
        assert state.model.id == "unknown"
        assert state.thinking_level == "off"
        assert state.tools == []
        assert state.messages == []
        assert state.is_streaming is False
        assert state.streaming_message is None
        assert state.pending_tool_calls == frozenset()
        assert state.error_message is None

    def test_agent_creation_with_options(self):
        """Agent can be created with AgentOptions."""
        from pi_agent_core.agent import Agent, AgentOptions
        from pi_ai import Model, Usage

        model = Model(
            id="gpt-4o", name="GPT-4o", api="openai", provider="openai",
            base_url="https://api.openai.com/v1", reasoning=False,
            input=["text"], cost=Usage.Cost(input=5, output=15, cache_read=0, cache_write=0, total=20),
            context_window=128000, max_tokens=4096
        )

        options = AgentOptions(
            initial_state={
                "system_prompt": "You are helpful",
                "model": model,
            }
        )

        agent = Agent(options)
        state = agent.state

        assert state.system_prompt == "You are helpful"
        assert state.model.id == "gpt-4o"

    def test_agent_options_all_fields(self):
        """AgentOptions can have all configurable fields."""
        from pi_agent_core.agent import AgentOptions

        def convert_fn(msgs):
            return msgs

        def stream_fn(ctx, model):
            pass

        options = AgentOptions(
            convert_to_llm=convert_fn,
            stream_fn=stream_fn,
            steering_mode="all",
            follow_up_mode="all",
            session_id="test-session",
            tool_execution="sequential",
        )

        assert options.convert_to_llm is not None
        assert options.steering_mode == "all"
        assert options.follow_up_mode == "all"
        assert options.session_id == "test-session"
        assert options.tool_execution == "sequential"


class TestAgentStateMutation:
    """Tests for Agent state mutation methods."""

    def test_set_system_prompt(self):
        """Agent can set system prompt."""
        from pi_agent_core.agent import Agent

        agent = Agent()
        agent.set_system_prompt("New system prompt")

        assert agent.state.system_prompt == "New system prompt"

    def test_set_model(self):
        """Agent can set model."""
        from pi_agent_core.agent import Agent
        from pi_ai import Model, Usage

        model = Model(
            id="claude-3", name="Claude 3", api="anthropic", provider="anthropic",
            base_url="https://api.anthropic.com/v1", reasoning=True,
            input=["text"], cost=Usage.Cost(input=3, output=15, cache_read=0, cache_write=0, total=18),
            context_window=200000, max_tokens=4096
        )

        agent = Agent()
        agent.set_model(model)

        assert agent.state.model.id == "claude-3"
        assert agent.state.model.reasoning is True

    def test_set_thinking_level(self):
        """Agent can set thinking level."""
        from pi_agent_core.agent import Agent

        agent = Agent()
        agent.set_thinking_level("high")

        assert agent.state.thinking_level == "high"

    def test_set_tools(self):
        """Agent can set tools list."""
        from pi_agent_core.agent import Agent
        from pi_agent_core.types import AgentTool, AgentToolResult
        from pi_ai import TextContent

        async def execute(tool_call_id, args, signal, update_cb):
            return AgentToolResult(content=[TextContent(text="Done")], details=None)

        tool = AgentTool(
            name="test_tool",
            label="Test Tool",
            description="A test tool",
            parameters={"type": "object"},
            execute=execute
        )

        agent = Agent()
        agent.set_tools([tool])

        assert len(agent.state.tools) == 1
        assert agent.state.tools[0].name == "test_tool"

    def test_set_messages(self):
        """Agent can set messages list."""
        from pi_agent_core.agent import Agent
        from pi_ai import UserMessage

        message = UserMessage(
            role="user",
            content="Hello",
            timestamp=1234567890
        )

        agent = Agent()
        agent.set_messages([message])

        assert len(agent.state.messages) == 1
        assert agent.state.messages[0].content == "Hello"


class TestAgentStateImmutability:
    """Tests for Agent state immutability and copying."""

    def test_state_snapshot_is_read_only(self):
        """Agent.state returns a read-only snapshot."""
        from pi_agent_core.agent import Agent
        from pi_agent_core.types import AgentState

        agent = Agent()
        state = agent.state

        # Verify it's an AgentState dataclass instance
        assert isinstance(state, AgentState)

        # Verify it's frozen (cannot be modified)
        # Note: frozen dataclasses raise FrozenInstanceError on assignment
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            state.system_prompt = "modified"

    def test_tools_are_copied_on_assignment(self):
        """Tools array is copied when set, not stored directly."""
        from pi_agent_core.agent import Agent
        from pi_agent_core.types import AgentTool, AgentToolResult
        from pi_ai import TextContent

        async def execute(tool_call_id, args, signal, update_cb):
            return AgentToolResult(content=[TextContent(text="Done")], details=None)

        tool1 = AgentTool(
            name="tool1",
            label="Tool 1",
            description="First tool",
            parameters={"type": "object"},
            execute=execute
        )

        tool2 = AgentTool(
            name="tool2",
            label="Tool 2",
            description="Second tool",
            parameters={"type": "object"},
            execute=execute
        )

        tools = [tool1, tool2]
        agent = Agent()
        agent.set_tools(tools)

        # Modify original list
        tools.append(AgentTool(
            name="tool3",
            label="Tool 3",
            description="Third tool",
            parameters={"type": "object"},
            execute=execute
        ))

        # Agent state should not be affected
        assert len(agent.state.tools) == 2

    def test_messages_are_copied_on_assignment(self):
        """Messages array is copied when set, not stored directly."""
        from pi_agent_core.agent import Agent
        from pi_ai import UserMessage

        msg1 = UserMessage(role="user", content="Hello", timestamp=1000)
        msg2 = UserMessage(role="user", content="World", timestamp=2000)

        messages = [msg1, msg2]
        agent = Agent()
        agent.set_messages(messages)

        # Modify original list
        messages.append(UserMessage(role="user", content="Added", timestamp=3000))

        # Agent state should not be affected
        assert len(agent.state.messages) == 2

    def test_state_tools_returns_copy(self):
        """AgentState.tools returns a copy each time."""
        from pi_agent_core.agent import Agent
        from pi_agent_core.types import AgentTool, AgentToolResult
        from pi_ai import TextContent

        async def execute(tool_call_id, args, signal, update_cb):
            return AgentToolResult(content=[TextContent(text="Done")], details=None)

        tool = AgentTool(
            name="tool",
            label="Tool",
            description="A tool",
            parameters={"type": "object"},
            execute=execute
        )

        agent = Agent()
        agent.set_tools([tool])

        # Get state twice
        state1 = agent.state
        state2 = agent.state

        # Each call should return a new copy
        assert state1.tools is not state2.tools

    def test_state_messages_returns_copy(self):
        """AgentState.messages returns a copy each time."""
        from pi_agent_core.agent import Agent
        from pi_ai import UserMessage

        msg = UserMessage(role="user", content="Test", timestamp=1000)
        agent = Agent()
        agent.set_messages([msg])

        # Get state twice
        state1 = agent.state
        state2 = agent.state

        # Each call should return a new copy
        assert state1.messages is not state2.messages

    def test_pending_tool_calls_is_frozen(self):
        """AgentState.pending_tool_calls is a frozenset."""
        from pi_agent_core.agent import Agent

        agent = Agent()
        state = agent.state

        assert isinstance(state.pending_tool_calls, frozenset)
        assert state.pending_tool_calls == frozenset()


class TestAgentCallbacks:
    """Tests for Agent configurable callbacks."""

    def test_default_convert_to_llm(self):
        """Agent has default convert_to_llm function."""
        from pi_agent_core.agent import Agent
        from pi_ai import UserMessage

        agent = Agent()

        # Default converter filters valid roles
        user_msg = UserMessage(role="user", content="Hello", timestamp=1000)
        converted = agent.convert_to_llm([user_msg])

        assert len(converted) == 1
        assert converted[0].role == "user"

    def test_custom_convert_to_llm(self):
        """Agent can use custom convert_to_llm function."""
        from pi_agent_core.agent import Agent, AgentOptions
        from pi_ai import UserMessage

        def custom_convert(messages):
            # Custom logic: only take first message
            return messages[:1] if messages else []

        options = AgentOptions(convert_to_llm=custom_convert)
        agent = Agent(options)

        msgs = [
            UserMessage(role="user", content="First", timestamp=1000),
            UserMessage(role="user", content="Second", timestamp=2000),
        ]

        converted = agent.convert_to_llm(msgs)
        assert len(converted) == 1
        assert converted[0].content == "First"

    def test_agent_stores_callbacks(self):
        """Agent stores all callback references."""
        from pi_agent_core.agent import Agent, AgentOptions

        def on_payload(data):
            pass

        def on_response(resp):
            pass

        options = AgentOptions(
            on_payload=on_payload,
            on_response=on_response,
        )

        agent = Agent(options)

        assert agent.on_payload is on_payload
        assert agent.on_response is on_response


class TestDefaultModel:
    """Tests for default model placeholder."""

    def test_default_model_is_placeholder(self):
        """Agent starts with placeholder model."""
        from pi_agent_core.agent import Agent, DEFAULT_MODEL

        agent = Agent()

        assert agent.state.model.id == "unknown"
        assert agent.state.model.name == "unknown"
        assert agent.state.model.api == "unknown"
        assert agent.state.model.provider == "unknown"

        # Verify DEFAULT_MODEL constant exists
        assert DEFAULT_MODEL.id == "unknown"

    def test_default_model_has_zero_cost(self):
        """Default model has zero cost."""
        from pi_agent_core.agent import DEFAULT_MODEL

        assert DEFAULT_MODEL.cost.input == 0
        assert DEFAULT_MODEL.cost.output == 0
        assert DEFAULT_MODEL.cost.total == 0

    def test_default_model_has_zero_limits(self):
        """Default model has zero context/token limits."""
        from pi_agent_core.agent import DEFAULT_MODEL

        assert DEFAULT_MODEL.context_window == 0
        assert DEFAULT_MODEL.max_tokens == 0


class TestAgentOptionsModes:
    """Tests for AgentOptions mode settings."""

    def test_default_steering_mode(self):
        """Default steering mode is one-at-a-time."""
        from pi_agent_core.agent import AgentOptions

        options = AgentOptions()
        assert options.steering_mode == "one-at-a-time"

    def test_default_follow_up_mode(self):
        """Default follow-up mode is one-at-a-time."""
        from pi_agent_core.agent import AgentOptions

        options = AgentOptions()
        assert options.follow_up_mode == "one-at-a-time"

    def test_default_tool_execution(self):
        """Default tool execution is parallel."""
        from pi_agent_core.agent import AgentOptions

        options = AgentOptions()
        assert options.tool_execution == "parallel"

    def test_default_transport(self):
        """Default transport is SSE."""
        from pi_agent_core.agent import AgentOptions

        options = AgentOptions()
        assert options.transport == "sse"