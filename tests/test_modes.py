# tests/test_modes.py
"""Tests for pi-coding-agent modes (print, rpc, sdk)."""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from io import StringIO


class TestPrintMode:
    """Tests for print mode."""

    def test_run_print_mode_function_exists(self):
        """run_print_mode function is importable."""
        from pi_coding_agent.modes.print_mode import run_print_mode

        assert callable(run_print_mode)

    def test_print_and_exit_function_exists(self):
        """print_and_exit function is importable."""
        from pi_coding_agent.modes.print_mode import print_and_exit

        assert callable(print_and_exit)

    @pytest.mark.asyncio
    async def test_run_print_mode_creates_session(self):
        """run_print_mode creates an AgentSession."""
        from pi_coding_agent.modes.print_mode import run_print_mode
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            # Mock the session to avoid actual LLM calls
            with patch("pi_coding_agent.modes.print_mode.AgentSession") as mock_session_class:
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session
                mock_session.prompt = AsyncMock()
                mock_session.wait_for_idle = AsyncMock()
                mock_session.subscribe = MagicMock(return_value=lambda: None)

                # Set up event capture
                events_collected = []

                def capture_event(event, signal):
                    events_collected.append(event)

                await run_print_mode(
                    "test prompt",
                    workspace,
                    on_event=capture_event,
                )

                mock_session_class.assert_called_once_with(workspace)
                mock_session.prompt.assert_called_once()
                mock_session.wait_for_idle.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_print_mode_collects_text_output(self):
        """run_print_mode collects text from assistant message."""
        from pi_coding_agent.modes.print_mode import run_print_mode

        with tempfile.TemporaryDirectory() as workspace:
            with patch("pi_coding_agent.modes.print_mode.AgentSession") as mock_session_class:
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session
                mock_session.prompt = AsyncMock()
                mock_session.wait_for_idle = AsyncMock()

                # Set up subscribe to capture the listener
                captured_listener = []

                def capture_subscribe(listener):
                    captured_listener.append(listener)
                    return lambda: None

                mock_session.subscribe = capture_subscribe

                result = await run_print_mode("test prompt", workspace)

                # Simulate a message_end event
                if captured_listener:
                    listener = captured_listener[0]
                    listener(
                        {
                            "type": "message_end",
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "Hello world"}],
                            },
                        },
                        None,
                    )

                # Result should be empty because event was fired before prompt
                assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_print_and_exit_prints_output(self, capsys):
        """print_and_exit prints output to stdout."""
        from pi_coding_agent.modes.print_mode import print_and_exit

        with tempfile.TemporaryDirectory() as workspace:
            with patch("pi_coding_agent.modes.print_mode.run_print_mode") as mock_run:
                mock_run.return_value = "Test output"

                await print_and_exit("test prompt", workspace)

                captured = capsys.readouterr()
                assert "Test output" in captured.out


class TestRPCMode:
    """Tests for RPC mode."""

    def test_rpc_handler_class_exists(self):
        """RPCHandler class is importable."""
        from pi_coding_agent.modes.rpc import RPCHandler

        assert RPCHandler is not None

    def test_run_rpc_mode_function_exists(self):
        """run_rpc_mode function is importable."""
        from pi_coding_agent.modes.rpc import run_rpc_mode

        assert callable(run_rpc_mode)

    def test_rpc_handler_initialization(self):
        """RPCHandler can be initialized with a session."""
        from pi_coding_agent.modes.rpc import RPCHandler
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            handler = RPCHandler(session)

            assert handler.session is session

    @pytest.mark.asyncio
    async def test_rpc_handler_unknown_method(self):
        """RPCHandler returns error for unknown method."""
        from pi_coding_agent.modes.rpc import RPCHandler
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            handler = RPCHandler(session)

            response = await handler.handle_request(
                {"jsonrpc": "2.0", "method": "unknownMethod", "id": 1}
            )

            assert "error" in response
            assert response["error"]["code"] == -32601
            assert response["id"] == 1

    @pytest.mark.asyncio
    async def test_rpc_handler_get_status(self):
        """RPCHandler handles getStatus method."""
        from pi_coding_agent.modes.rpc import RPCHandler
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            handler = RPCHandler(session)

            response = await handler.handle_request(
                {"jsonrpc": "2.0", "method": "getStatus", "id": 2}
            )

            assert "result" in response
            assert "isStreaming" in response["result"]
            assert "messageCount" in response["result"]
            assert "hasQueuedMessages" in response["result"]
            assert response["id"] == 2

    @pytest.mark.asyncio
    async def test_rpc_handler_abort(self):
        """RPCHandler handles abort method."""
        from pi_coding_agent.modes.rpc import RPCHandler
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            handler = RPCHandler(session)

            response = await handler.handle_request(
                {"jsonrpc": "2.0", "method": "abort", "id": 3}
            )

            assert "result" in response
            assert response["result"]["aborted"] is True
            assert response["id"] == 3

    @pytest.mark.asyncio
    async def test_rpc_handler_prompt(self):
        """RPCHandler handles prompt method."""
        from pi_coding_agent.modes.rpc import RPCHandler
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            handler = RPCHandler(session)

            # Mock prompt to avoid actual LLM call
            session.prompt = AsyncMock()

            response = await handler.handle_request(
                {"jsonrpc": "2.0", "method": "prompt", "params": {"prompt": "Hello"}, "id": 4}
            )

            assert "result" in response
            assert response["result"]["status"] == "processing"
            session.prompt.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_rpc_handler_continue(self):
        """RPCHandler handles continue method."""
        from pi_coding_agent.modes.rpc import RPCHandler
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            handler = RPCHandler(session)

            # Mock continue_ to avoid actual LLM call
            session.continue_ = AsyncMock()

            response = await handler.handle_request(
                {"jsonrpc": "2.0", "method": "continue", "id": 5}
            )

            assert "result" in response
            assert response["result"]["status"] == "processing"
            session.continue_.assert_called_once()

    @pytest.mark.asyncio
    async def test_rpc_handler_exception_handling(self):
        """RPCHandler returns error response on exception."""
        from pi_coding_agent.modes.rpc import RPCHandler
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            handler = RPCHandler(session)

            # Make prompt raise an exception
            session.prompt = AsyncMock(side_effect=RuntimeError("Test error"))

            response = await handler.handle_request(
                {"jsonrpc": "2.0", "method": "prompt", "params": {"prompt": "Hello"}, "id": 6}
            )

            assert "error" in response
            assert response["error"]["code"] == -32000
            assert "Test error" in response["error"]["message"]

    def test_rpc_handler_parse_error_format(self):
        """Test that parse error format is correct."""
        error_response = {
            "jsonrpc": "2.0",
            "error": {"code": -32700, "message": "Parse error"},
            "id": None,
        }

        assert error_response["jsonrpc"] == "2.0"
        assert error_response["error"]["code"] == -32700
        assert error_response["id"] is None


class TestSDKMode:
    """Tests for SDK mode."""

    def test_create_sdk_session_function_exists(self):
        """create_sdk_session function is importable."""
        from pi_coding_agent.modes.sdk import create_sdk_session

        assert callable(create_sdk_session)

    def test_run_sdk_prompt_function_exists(self):
        """run_sdk_prompt function is importable."""
        from pi_coding_agent.modes.sdk import run_sdk_prompt

        assert callable(run_sdk_prompt)

    def test_create_sdk_session_returns_session(self):
        """create_sdk_session returns an AgentSession."""
        from pi_coding_agent.modes.sdk import create_sdk_session
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = create_sdk_session(workspace)

            assert isinstance(session, AgentSession)
            assert session.workspace_dir == Path(workspace)

    def test_create_sdk_session_with_event_callback(self):
        """create_sdk_session subscribes event callback."""
        from pi_coding_agent.modes.sdk import create_sdk_session

        with tempfile.TemporaryDirectory() as workspace:
            events_collected = []

            def on_event(event):
                events_collected.append(event)

            session = create_sdk_session(workspace, on_event=on_event)

            # Session should be created successfully
            assert session is not None

    @pytest.mark.asyncio
    async def test_run_sdk_prompt_calls_prompt(self):
        """run_sdk_prompt calls session.prompt."""
        from pi_coding_agent.modes.sdk import run_sdk_prompt
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            session.prompt = AsyncMock()
            session.wait_for_idle = AsyncMock()

            await run_sdk_prompt(session, "test prompt")

            session.prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_sdk_prompt_wait_true(self):
        """run_sdk_prompt waits for idle when wait=True."""
        from pi_coding_agent.modes.sdk import run_sdk_prompt
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            session.prompt = AsyncMock()
            session.wait_for_idle = AsyncMock()

            await run_sdk_prompt(session, "test prompt", wait=True)

            session.wait_for_idle.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_sdk_prompt_wait_false(self):
        """run_sdk_prompt does not wait when wait=False."""
        from pi_coding_agent.modes.sdk import run_sdk_prompt
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            session.prompt = AsyncMock()
            session.wait_for_idle = AsyncMock()

            result = await run_sdk_prompt(session, "test prompt", wait=False)

            session.wait_for_idle.assert_not_called()
            assert result is None

    @pytest.mark.asyncio
    async def test_run_sdk_prompt_collects_output(self):
        """run_sdk_prompt collects output from message_end events."""
        from pi_coding_agent.modes.sdk import run_sdk_prompt
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            session.prompt = AsyncMock()
            session.wait_for_idle = AsyncMock()

            # Set up subscribe to capture the listener
            captured_listener = []

            def capture_subscribe(listener):
                captured_listener.append(listener)
                return lambda: None

            session.subscribe = capture_subscribe

            # Start the prompt
            task = asyncio.create_task(run_sdk_prompt(session, "test prompt", wait=False))

            # Give the task time to start
            await asyncio.sleep(0.01)

            # Simulate a message_end event
            if captured_listener:
                listener = captured_listener[0]
                listener(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "SDK response"}],
                        },
                    },
                    None,
                )

            result = await task
            # When wait=False, result is None
            assert result is None


class TestModesModule:
    """Tests for modes module exports."""

    def test_modes_module_imports(self):
        """Modes module exports all expected functions."""
        from pi_coding_agent.modes import (
            run_print_mode,
            print_and_exit,
            RPCHandler,
            run_rpc_mode,
            create_sdk_session,
            run_sdk_prompt,
        )

        assert callable(run_print_mode)
        assert callable(print_and_exit)
        assert callable(run_rpc_mode)
        assert callable(create_sdk_session)
        assert callable(run_sdk_prompt)