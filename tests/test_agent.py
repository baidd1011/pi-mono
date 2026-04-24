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


class TestAgentEventSubscription:
    """Tests for Agent event subscription system."""

    def test_subscribe_returns_unsubscribe_function(self):
        """subscribe returns a callable that removes the listener."""
        from pi_agent_core.agent import Agent

        agent = Agent()

        call_count = 0

        def listener(event, signal):
            nonlocal call_count
            call_count += 1

        unsubscribe = agent.subscribe(listener)

        # Listener is registered
        assert callable(unsubscribe)

        # Unsubscribe removes the listener
        unsubscribe()
        # Should not raise - listener is removed

    def test_unsubscribe_is_idempotent(self):
        """Calling unsubscribe multiple times is safe."""
        from pi_agent_core.agent import Agent

        agent = Agent()

        def listener(event, signal):
            pass

        unsubscribe = agent.subscribe(listener)
        unsubscribe()
        unsubscribe()  # Should not raise

    def test_emit_event_calls_listeners(self):
        """_emit_event calls all registered listeners."""
        from pi_agent_core.agent import Agent
        from pi_agent_core.types import TurnStartEvent

        agent = Agent()
        events_received = []

        def listener1(event, signal):
            events_received.append(("listener1", event))

        def listener2(event, signal):
            events_received.append(("listener2", event))

        agent.subscribe(listener1)
        agent.subscribe(listener2)

        import asyncio
        event = TurnStartEvent(type="turn_start")
        asyncio.run(agent._emit_event(event))

        assert len(events_received) == 2
        assert events_received[0][0] == "listener1"
        assert events_received[1][0] == "listener2"
        assert events_received[0][1]["type"] == "turn_start"

    def test_emit_event_with_async_listener(self):
        """_emit_event handles async listeners."""
        from pi_agent_core.agent import Agent
        from pi_agent_core.types import TurnStartEvent

        agent = Agent()
        events_received = []

        async def async_listener(event, signal):
            events_received.append(("async", event))

        agent.subscribe(async_listener)

        import asyncio
        event = TurnStartEvent(type="turn_start")
        asyncio.run(agent._emit_event(event))

        assert len(events_received) == 1
        assert events_received[0][0] == "async"


class TestAgentMessageQueues:
    """Tests for Agent message queue methods."""

    def test_steer_queues_message(self):
        """steer queues a message to steering queue."""
        from pi_agent_core.agent import Agent
        from pi_ai import UserMessage

        agent = Agent()
        msg = UserMessage(role="user", content="Steer this", timestamp=1000)

        agent.steer(msg)

        assert agent.has_queued_messages() is True

    def test_follow_up_queues_message(self):
        """follow_up queues a message to follow-up queue."""
        from pi_agent_core.agent import Agent
        from pi_ai import UserMessage

        agent = Agent()
        msg = UserMessage(role="user", content="Follow up", timestamp=1000)

        agent.follow_up(msg)

        assert agent.has_queued_messages() is True

    def test_clear_steering_queue(self):
        """clear_steering_queue empties the steering queue."""
        from pi_agent_core.agent import Agent
        from pi_ai import UserMessage

        agent = Agent()
        agent.steer(UserMessage(role="user", content="msg1", timestamp=1000))
        agent.steer(UserMessage(role="user", content="msg2", timestamp=2000))

        agent.clear_steering_queue()

        # After clear, draining should return empty
        result = agent._drain_steering()
        assert result == []

    def test_clear_follow_up_queue(self):
        """clear_follow_up_queue empties the follow-up queue."""
        from pi_agent_core.agent import Agent
        from pi_ai import UserMessage

        agent = Agent()
        agent.follow_up(UserMessage(role="user", content="msg1", timestamp=1000))
        agent.follow_up(UserMessage(role="user", content="msg2", timestamp=2000))

        agent.clear_follow_up_queue()

        result = agent._drain_follow_up()
        assert result == []

    def test_clear_all_queues(self):
        """clear_all_queues empties both queues."""
        from pi_agent_core.agent import Agent
        from pi_ai import UserMessage

        agent = Agent()
        agent.steer(UserMessage(role="user", content="steer", timestamp=1000))
        agent.follow_up(UserMessage(role="user", content="follow", timestamp=2000))

        agent.clear_all_queues()

        assert agent.has_queued_messages() is False

    def test_has_queued_messages(self):
        """has_queued_messages returns correct status."""
        from pi_agent_core.agent import Agent
        from pi_ai import UserMessage

        agent = Agent()

        assert agent.has_queued_messages() is False

        agent.steer(UserMessage(role="user", content="steer", timestamp=1000))
        assert agent.has_queued_messages() is True

        agent.clear_all_queues()
        assert agent.has_queued_messages() is False

        agent.follow_up(UserMessage(role="user", content="follow", timestamp=2000))
        assert agent.has_queued_messages() is True


class TestAgentQueueModes:
    """Tests for steering and follow-up queue modes."""

    def test_steering_mode_default(self):
        """Default steering mode is one-at-a-time."""
        from pi_agent_core.agent import Agent

        agent = Agent()
        assert agent.steering_mode == "one-at-a-time"

    def test_follow_up_mode_default(self):
        """Default follow-up mode is one-at-a-time."""
        from pi_agent_core.agent import Agent

        agent = Agent()
        assert agent.follow_up_mode == "one-at-a-time"

    def test_steering_mode_can_be_set(self):
        """steering_mode can be set to all or one-at-a-time."""
        from pi_agent_core.agent import Agent

        agent = Agent()

        agent.steering_mode = "all"
        assert agent.steering_mode == "all"

        agent.steering_mode = "one-at-a-time"
        assert agent.steering_mode == "one-at-a-time"

    def test_follow_up_mode_can_be_set(self):
        """follow_up_mode can be set to all or one-at-a-time."""
        from pi_agent_core.agent import Agent

        agent = Agent()

        agent.follow_up_mode = "all"
        assert agent.follow_up_mode == "all"

        agent.follow_up_mode = "one-at-a-time"
        assert agent.follow_up_mode == "one-at-a-time"

    def test_drain_steering_all_mode(self):
        """_drain_steering returns all messages in 'all' mode."""
        from pi_agent_core.agent import Agent, AgentOptions
        from pi_ai import UserMessage

        agent = Agent(AgentOptions(steering_mode="all"))
        agent.steer(UserMessage(role="user", content="msg1", timestamp=1000))
        agent.steer(UserMessage(role="user", content="msg2", timestamp=2000))
        agent.steer(UserMessage(role="user", content="msg3", timestamp=3000))

        result = agent._drain_steering()

        assert len(result) == 3
        assert result[0].content == "msg1"
        assert result[1].content == "msg2"
        assert result[2].content == "msg3"

        # Queue should be empty after drain
        assert agent._drain_steering() == []

    def test_drain_steering_one_at_a_time_mode(self):
        """_drain_steering returns one message in 'one-at-a-time' mode."""
        from pi_agent_core.agent import Agent, AgentOptions
        from pi_ai import UserMessage

        agent = Agent(AgentOptions(steering_mode="one-at-a-time"))
        agent.steer(UserMessage(role="user", content="msg1", timestamp=1000))
        agent.steer(UserMessage(role="user", content="msg2", timestamp=2000))
        agent.steer(UserMessage(role="user", content="msg3", timestamp=3000))

        # First drain should return only first message
        result1 = agent._drain_steering()
        assert len(result1) == 1
        assert result1[0].content == "msg1"

        # Second drain should return second message
        result2 = agent._drain_steering()
        assert len(result2) == 1
        assert result2[0].content == "msg2"

        # Third drain should return third message
        result3 = agent._drain_steering()
        assert len(result3) == 1
        assert result3[0].content == "msg3"

        # Fourth drain should return empty
        result4 = agent._drain_steering()
        assert result4 == []

    def test_drain_follow_up_all_mode(self):
        """_drain_follow_up returns all messages in 'all' mode."""
        from pi_agent_core.agent import Agent, AgentOptions
        from pi_ai import UserMessage

        agent = Agent(AgentOptions(follow_up_mode="all"))
        agent.follow_up(UserMessage(role="user", content="msg1", timestamp=1000))
        agent.follow_up(UserMessage(role="user", content="msg2", timestamp=2000))

        result = agent._drain_follow_up()

        assert len(result) == 2
        assert result[0].content == "msg1"
        assert result[1].content == "msg2"

        # Queue should be empty after drain
        assert agent._drain_follow_up() == []

    def test_drain_follow_up_one_at_a_time_mode(self):
        """_drain_follow_up returns one message in 'one-at-a-time' mode."""
        from pi_agent_core.agent import Agent, AgentOptions
        from pi_ai import UserMessage

        agent = Agent(AgentOptions(follow_up_mode="one-at-a-time"))
        agent.follow_up(UserMessage(role="user", content="msg1", timestamp=1000))
        agent.follow_up(UserMessage(role="user", content="msg2", timestamp=2000))

        # First drain returns first message
        result1 = agent._drain_follow_up()
        assert len(result1) == 1
        assert result1[0].content == "msg1"

        # Second drain returns second message
        result2 = agent._drain_follow_up()
        assert len(result2) == 1
        assert result2[0].content == "msg2"

        # Third drain returns empty
        result3 = agent._drain_follow_up()
        assert result3 == []

    def test_drain_steering_empty_queue(self):
        """_drain_steering returns empty list when queue is empty."""
        from pi_agent_core.agent import Agent

        agent = Agent()
        result = agent._drain_steering()
        assert result == []

    def test_drain_follow_up_empty_queue(self):
        """_drain_follow_up returns empty list when queue is empty."""
        from pi_agent_core.agent import Agent

        agent = Agent()
        result = agent._drain_follow_up()
        assert result == []


class TestAgentAbortHandling:
    """Tests for Agent abort handling."""

    def test_signal_is_none_when_not_running(self):
        """signal property is None when agent is not running."""
        from pi_agent_core.agent import Agent

        agent = Agent()
        assert agent.signal is None

    def test_abort_does_nothing_when_not_running(self):
        """abort() is safe to call when agent is not running."""
        from pi_agent_core.agent import Agent

        agent = Agent()
        agent.abort()  # Should not raise

    def test_wait_for_idle_when_not_running(self):
        """wait_for_idle completes immediately when not running."""
        from pi_agent_core.agent import Agent
        import asyncio

        agent = Agent()
        # Should complete immediately without waiting
        asyncio.run(agent.wait_for_idle())

    def test_reset_clears_messages(self):
        """reset() clears agent state and queues."""
        from pi_agent_core.agent import Agent
        from pi_ai import UserMessage

        agent = Agent()
        agent.set_messages([UserMessage(role="user", content="Hello", timestamp=1000)])
        agent.steer(UserMessage(role="user", content="Steer", timestamp=2000))
        agent.follow_up(UserMessage(role="user", content="Follow", timestamp=3000))

        agent.reset()

        assert agent.state.messages == []
        assert agent.has_queued_messages() is False

    def test_reset_clears_streaming_state(self):
        """reset() clears streaming state."""
        from pi_agent_core.agent import Agent

        agent = Agent()
        # Manually set streaming state (normally set during run)
        agent._state.is_streaming = True
        agent._state.streaming_message = "some message"
        agent._state.pending_tool_calls = {"call_1"}
        agent._state.error_message = "some error"

        agent.reset()

        assert agent.state.is_streaming is False
        assert agent.state.streaming_message is None
        assert agent.state.pending_tool_calls == frozenset()
        assert agent.state.error_message is None

    def test_reset_clears_system_prompt(self):
        """reset() does NOT clear system prompt (only messages and queues)."""
        from pi_agent_core.agent import Agent

        agent = Agent()
        agent.set_system_prompt("You are helpful")

        agent.reset()

        # System prompt should remain
        assert agent.state.system_prompt == "You are helpful"


class TestAgentOptionsModesFromOptions:
    """Tests that Agent initializes modes from AgentOptions."""

    def test_agent_initializes_steering_mode_from_options(self):
        """Agent reads steering_mode from AgentOptions."""
        from pi_agent_core.agent import Agent, AgentOptions

        agent = Agent(AgentOptions(steering_mode="all"))
        assert agent.steering_mode == "all"

    def test_agent_initializes_follow_up_mode_from_options(self):
        """Agent reads follow_up_mode from AgentOptions."""
        from pi_agent_core.agent import Agent, AgentOptions

        agent = Agent(AgentOptions(follow_up_mode="all"))
        assert agent.follow_up_mode == "all"


class TestAgentPromptMethod:
    """Tests for Agent prompt() method."""

    def test_prompt_throws_when_already_running(self):
        """prompt() raises RuntimeError when already has active run."""
        from pi_agent_core.agent import Agent
        import asyncio

        agent = Agent()
        # Simulate an active run
        agent._active_run = {
            "abort_controller": None,
            "promise": None,
        }

        with pytest.raises(RuntimeError, match="already has an active run"):
            asyncio.run(agent.prompt("Hello"))

    def test_prompt_accepts_string_input(self):
        """prompt() accepts a string as input."""
        from pi_agent_core.agent import Agent
        from pi_ai import Model, Usage
        import asyncio

        # Create agent with a valid model
        model = Model(
            id="faux-model", name="Faux Model", api="faux", provider="faux",
            base_url="", reasoning=False, input=["text"],
            cost=Usage.Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
            context_window=1000, max_tokens=100,
        )

        agent = Agent()
        agent.set_model(model)

        # Run prompt
        asyncio.run(agent.prompt("Hello, agent!"))

        # Messages should be updated
        messages = agent.state.messages
        assert len(messages) >= 1
        # First message should be the user prompt
        first_msg = messages[0]
        assert hasattr(first_msg, 'role')
        assert first_msg.role == "user"

    def test_prompt_accepts_single_message(self):
        """prompt() accepts a single message as input."""
        from pi_agent_core.agent import Agent
        from pi_ai import Model, Usage, UserMessage, TextContent
        import asyncio

        model = Model(
            id="faux-model", name="Faux Model", api="faux", provider="faux",
            base_url="", reasoning=False, input=["text"],
            cost=Usage.Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
            context_window=1000, max_tokens=100,
        )

        agent = Agent()
        agent.set_model(model)

        user_msg = UserMessage(
            role="user",
            content=[TextContent(text="Single message")],
            timestamp=1000
        )

        asyncio.run(agent.prompt(user_msg))

        messages = agent.state.messages
        assert len(messages) >= 1
        assert messages[0].role == "user"

    def test_prompt_accepts_message_list(self):
        """prompt() accepts a list of messages as input."""
        from pi_agent_core.agent import Agent
        from pi_ai import Model, Usage, UserMessage, TextContent
        import asyncio

        model = Model(
            id="faux-model", name="Faux Model", api="faux", provider="faux",
            base_url="", reasoning=False, input=["text"],
            cost=Usage.Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
            context_window=1000, max_tokens=100,
        )

        agent = Agent()
        agent.set_model(model)

        messages_input = [
            UserMessage(role="user", content=[TextContent(text="First")], timestamp=1000),
            UserMessage(role="user", content=[TextContent(text="Second")], timestamp=2000),
        ]

        asyncio.run(agent.prompt(messages_input))

        messages = agent.state.messages
        assert len(messages) >= 2
        assert messages[0].role == "user"
        assert messages[1].role == "user"

    def test_prompt_emits_events(self):
        """prompt() emits agent events to listeners."""
        from pi_agent_core.agent import Agent
        from pi_ai import Model, Usage
        import asyncio

        model = Model(
            id="faux-model", name="Faux Model", api="faux", provider="faux",
            base_url="", reasoning=False, input=["text"],
            cost=Usage.Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
            context_window=1000, max_tokens=100,
        )

        agent = Agent()
        agent.set_model(model)

        events_received = []

        def listener(event, signal):
            events_received.append(event)

        agent.subscribe(listener)

        asyncio.run(agent.prompt("Hello"))

        # Should have received events
        assert len(events_received) > 0
        event_types = [e.get("type") for e in events_received]
        assert "agent_start" in event_types
        assert "agent_end" in event_types

    def test_prompt_clears_active_run_after_completion(self):
        """prompt() clears active_run after completion."""
        from pi_agent_core.agent import Agent
        from pi_ai import Model, Usage
        import asyncio

        model = Model(
            id="faux-model", name="Faux Model", api="faux", provider="faux",
            base_url="", reasoning=False, input=["text"],
            cost=Usage.Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
            context_window=1000, max_tokens=100,
        )

        agent = Agent()
        agent.set_model(model)

        asyncio.run(agent.prompt("Hello"))

        # Active run should be cleared
        assert agent._active_run is None

    def test_prompt_clears_active_run_on_error(self):
        """prompt() clears active_run even on error."""
        from pi_agent_core.agent import Agent
        import asyncio

        agent = Agent()
        # Model is invalid, will likely cause error

        try:
            asyncio.run(agent.prompt("Hello"))
        except Exception:
            pass

        # Active run should be cleared even on error
        assert agent._active_run is None


class TestAgentContinueMethod:
    """Tests for Agent continue_() method."""

    def test_continue_throws_when_already_running(self):
        """continue_() raises RuntimeError when already has active run."""
        from pi_agent_core.agent import Agent
        import asyncio

        agent = Agent()
        # Simulate an active run
        agent._active_run = {
            "abort_controller": None,
            "promise": None,
        }

        with pytest.raises(RuntimeError, match="already has an active run"):
            asyncio.run(agent.continue_())

    def test_continue_throws_when_no_messages(self):
        """continue_() raises ValueError when there are no messages."""
        from pi_agent_core.agent import Agent
        import asyncio

        agent = Agent()

        with pytest.raises(ValueError, match="no messages"):
            asyncio.run(agent.continue_())

    def test_continue_throws_when_last_message_is_assistant(self):
        """continue_() raises ValueError when last message is assistant."""
        from pi_agent_core.agent import Agent
        from pi_ai import AssistantMessage, Usage, UserMessage, TextContent
        import asyncio

        agent = Agent()

        # Set up messages ending with assistant
        user_msg = UserMessage(role="user", content=[TextContent(text="Hello")], timestamp=1000)
        assistant_msg = AssistantMessage(
            content=[TextContent(text="Response")],
            api="faux", provider="faux", model="faux-model",
            response_id=None,
            usage=Usage(input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0, total_tokens=0,
                        cost=Usage.Cost(input=0.0, output=0.0, total=0.0)),
            stop_reason="stop",
            timestamp=2000
        )

        agent.set_messages([user_msg, assistant_msg])

        with pytest.raises(ValueError, match="Cannot continue from assistant message"):
            asyncio.run(agent.continue_())

    def test_continue_from_user_message(self):
        """continue_() works when last message is user."""
        from pi_agent_core.agent import Agent
        from pi_ai import Model, Usage, UserMessage, TextContent
        import asyncio

        model = Model(
            id="faux-model", name="Faux Model", api="faux", provider="faux",
            base_url="", reasoning=False, input=["text"],
            cost=Usage.Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
            context_window=1000, max_tokens=100,
        )

        agent = Agent()
        agent.set_model(model)

        # Set up messages ending with user message
        user_msg = UserMessage(role="user", content=[TextContent(text="Hello")], timestamp=1000)
        agent.set_messages([user_msg])

        asyncio.run(agent.continue_())

        # Should have added assistant response
        messages = agent.state.messages
        assert len(messages) >= 2
        # Last message should be assistant
        assert messages[-1].role == "assistant"

    def test_continue_emits_events(self):
        """continue_() emits agent events to listeners."""
        from pi_agent_core.agent import Agent
        from pi_ai import Model, Usage, UserMessage, TextContent
        import asyncio

        model = Model(
            id="faux-model", name="Faux Model", api="faux", provider="faux",
            base_url="", reasoning=False, input=["text"],
            cost=Usage.Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
            context_window=1000, max_tokens=100,
        )

        agent = Agent()
        agent.set_model(model)

        user_msg = UserMessage(role="user", content=[TextContent(text="Hello")], timestamp=1000)
        agent.set_messages([user_msg])

        events_received = []

        def listener(event, signal):
            events_received.append(event)

        agent.subscribe(listener)

        asyncio.run(agent.continue_())

        # Should have received events
        assert len(events_received) > 0
        event_types = [e.get("type") for e in events_received]
        assert "agent_start" in event_types
        assert "agent_end" in event_types

    def test_continue_clears_active_run_after_completion(self):
        """continue_() clears active_run after completion."""
        from pi_agent_core.agent import Agent
        from pi_ai import Model, Usage, UserMessage, TextContent
        import asyncio

        model = Model(
            id="faux-model", name="Faux Model", api="faux", provider="faux",
            base_url="", reasoning=False, input=["text"],
            cost=Usage.Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
            context_window=1000, max_tokens=100,
        )

        agent = Agent()
        agent.set_model(model)

        user_msg = UserMessage(role="user", content=[TextContent(text="Hello")], timestamp=1000)
        agent.set_messages([user_msg])

        asyncio.run(agent.continue_())

        # Active run should be cleared
        assert agent._active_run is None