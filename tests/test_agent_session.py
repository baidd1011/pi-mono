# tests/test_agent_session.py
"""Tests for AgentSession class."""

import pytest
from pathlib import Path
import tempfile


class TestAgentSessionCreation:
    """Tests for AgentSession creation and initialization."""

    def test_agent_session_creation_with_defaults(self):
        """AgentSession can be created with default configuration."""
        from pi_coding_agent import AgentSession, DEFAULT_SYSTEM_PROMPT

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            state = session.state

            assert session.workspace_dir == Path(workspace)
            assert state.system_prompt == DEFAULT_SYSTEM_PROMPT
            assert state.thinking_level == "off"
            assert state.is_streaming is False
            assert state.streaming_message is None
            assert state.pending_tool_calls == frozenset()
            assert state.error_message is None

    def test_agent_session_with_custom_model(self):
        """AgentSession can be created with a custom model."""
        from pi_coding_agent import AgentSession
        from pi_ai import Model, Usage

        custom_model = Model(
            id="gpt-4o",
            name="GPT-4o",
            api="openai",
            provider="openai",
            base_url="https://api.openai.com/v1",
            reasoning=False,
            input=["text", "image"],
            cost=Usage.Cost(input=5, output=15, cache_read=0, cache_write=0, total=20),
            context_window=128000,
            max_tokens=4096,
        )

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace, model=custom_model)

            assert session.model.id == "gpt-4o"
            assert session.model.name == "GPT-4o"
            assert session.state.model.id == "gpt-4o"

    def test_agent_session_with_custom_session_dir(self):
        """AgentSession can use a custom session directory."""
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            with tempfile.TemporaryDirectory() as session_dir:
                session = AgentSession(workspace, session_dir=session_dir)

                assert session._session_dir == Path(session_dir)

    def test_agent_session_default_session_dir(self):
        """AgentSession uses default .pi/sessions directory."""
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)

            expected_path = Path(workspace) / ".pi" / "sessions"
            assert session._session_dir == expected_path


class TestAgentSessionTools:
    """Tests for AgentSession tool initialization."""

    def test_tools_list_has_7_tools(self):
        """AgentSession initializes with 7 coding tools."""
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)

            tools = session.tools
            assert len(tools) == 7

            # Check tool names
            tool_names = [t.name for t in tools]
            assert "read" in tool_names
            assert "write" in tool_names
            assert "edit" in tool_names
            assert "bash" in tool_names
            assert "grep" in tool_names
            assert "find" in tool_names
            assert "ls" in tool_names

    def test_tools_have_labels(self):
        """All tools have human-readable labels."""
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)

            for tool in session.tools:
                assert tool.label is not None
                assert len(tool.label) > 0

    def test_tools_have_descriptions(self):
        """All tools have descriptions."""
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)

            for tool in session.tools:
                assert tool.description is not None
                assert len(tool.description) > 0


class TestAgentSessionProperties:
    """Tests for AgentSession properties."""

    def test_model_property_returns_model(self):
        """model property returns the Model configuration."""
        from pi_coding_agent import AgentSession, DEFAULT_MODEL

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)

            assert session.model.id == DEFAULT_MODEL.id
            assert session.model.name == DEFAULT_MODEL.name

    def test_session_id_is_none_initially(self):
        """session_id is None on creation."""
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)

            assert session.session_id is None

    def test_state_returns_agent_state(self):
        """state property returns AgentState from underlying agent."""
        from pi_coding_agent import AgentSession
        from pi_agent_core.types import AgentState

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)

            assert isinstance(session.state, AgentState)


class TestAgentSessionSetters:
    """Tests for AgentSession setter methods."""

    def test_set_model_updates_agent(self):
        """set_model updates both session and agent model."""
        from pi_coding_agent import AgentSession
        from pi_ai import Model, Usage

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)

            new_model = Model(
                id="qwen-max",
                name="Qwen Max",
                api="bailian",
                provider="bailian",
                base_url="https://dashscope.aliyuncs.com/api/v1",
                reasoning=True,
                input=["text", "image"],
                cost=Usage.Cost(input=20, output=60, cache_read=0, cache_write=0, total=80),
                context_window=32768,
                max_tokens=8192,
            )

            session.set_model(new_model)

            assert session.model.id == "qwen-max"
            assert session.state.model.id == "qwen-max"

    def test_set_thinking_level(self):
        """set_thinking_level updates agent thinking level."""
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)

            session.set_thinking_level("high")

            assert session.state.thinking_level == "high"

    def test_set_system_prompt(self):
        """set_system_prompt updates agent system prompt."""
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)

            session.set_system_prompt("Custom system prompt")

            assert session.state.system_prompt == "Custom system prompt"


class TestAgentSessionSubscription:
    """Tests for AgentSession event subscription."""

    def test_subscribe_returns_unsubscribe_function(self):
        """subscribe returns a callable that removes the listener."""
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)

            call_count = 0

            def listener(event, signal):
                nonlocal call_count
                call_count += 1

            unsubscribe = session.subscribe(listener)

            assert callable(unsubscribe)

            # Unsubscribe removes the listener
            unsubscribe()

    def test_unsubscribe_is_idempotent(self):
        """Calling unsubscribe multiple times is safe."""
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)

            def listener(event, signal):
                pass

            unsubscribe = session.subscribe(listener)
            unsubscribe()
            unsubscribe()  # Should not raise


class TestAgentSessionAbortHandling:
    """Tests for AgentSession abort handling."""

    def test_abort_does_nothing_when_not_running(self):
        """abort() is safe to call when agent is not running."""
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            session.abort()  # Should not raise

    def test_wait_for_idle_when_not_running(self):
        """wait_for_idle completes immediately when not running."""
        from pi_coding_agent import AgentSession
        import asyncio

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            asyncio.run(session.wait_for_idle())  # Should complete immediately


class TestAgentSessionMessageQueues:
    """Tests for AgentSession message queue methods."""

    def test_steer_queues_message(self):
        """steer queues a message to steering queue."""
        from pi_coding_agent import AgentSession
        from pi_ai import UserMessage

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            msg = UserMessage(role="user", content="Steer this", timestamp=1000)

            session.steer(msg)

            assert session.has_queued_messages() is True

    def test_follow_up_queues_message(self):
        """follow_up queues a message to follow-up queue."""
        from pi_coding_agent import AgentSession
        from pi_ai import UserMessage

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            msg = UserMessage(role="user", content="Follow up", timestamp=1000)

            session.follow_up(msg)

            assert session.has_queued_messages() is True

    def test_clear_all_queues(self):
        """clear_all_queues empties both queues."""
        from pi_coding_agent import AgentSession
        from pi_ai import UserMessage

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            session.steer(UserMessage(role="user", content="steer", timestamp=1000))
            session.follow_up(UserMessage(role="user", content="follow", timestamp=2000))

            session.clear_all_queues()

            assert session.has_queued_messages() is False


class TestAgentSessionReset:
    """Tests for AgentSession reset method."""

    def test_reset_clears_messages(self):
        """reset() clears agent messages."""
        from pi_coding_agent import AgentSession
        from pi_ai import UserMessage

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            session._agent.set_messages([UserMessage(role="user", content="Hello", timestamp=1000)])

            session.reset()

            assert session.state.messages == []

    def test_reset_clears_queues(self):
        """reset() clears message queues."""
        from pi_coding_agent import AgentSession
        from pi_ai import UserMessage

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            session.steer(UserMessage(role="user", content="Steer", timestamp=1000))
            session.follow_up(UserMessage(role="user", content="Follow", timestamp=2000))

            session.reset()

            assert session.has_queued_messages() is False


class TestDefaults:
    """Tests for default configuration values."""

    def test_default_model_is_qwen_plus(self):
        """DEFAULT_MODEL is configured for Qwen Plus."""
        from pi_coding_agent import DEFAULT_MODEL

        assert DEFAULT_MODEL.id == "qwen-plus-latest"
        assert DEFAULT_MODEL.name == "Qwen Plus"
        assert DEFAULT_MODEL.api == "bailian-chat"
        assert DEFAULT_MODEL.provider == "bailian"

    def test_default_model_has_context_window(self):
        """DEFAULT_MODEL has context window configured."""
        from pi_coding_agent import DEFAULT_MODEL

        assert DEFAULT_MODEL.context_window == 128000
        assert DEFAULT_MODEL.max_tokens == 8192

    def test_default_thinking_level_is_off(self):
        """DEFAULT_THINKING_LEVEL is off."""
        from pi_coding_agent import DEFAULT_THINKING_LEVEL

        assert DEFAULT_THINKING_LEVEL == "off"

    def test_default_system_prompt_not_empty(self):
        """DEFAULT_SYSTEM_PROMPT is not empty."""
        from pi_coding_agent import DEFAULT_SYSTEM_PROMPT

        assert len(DEFAULT_SYSTEM_PROMPT) > 0

    def test_default_limits(self):
        """DEFAULT_MAX_LINES and DEFAULT_MAX_BYTES are configured."""
        from pi_coding_agent import DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES

        assert DEFAULT_MAX_LINES == 2000
        assert DEFAULT_MAX_BYTES == 100000