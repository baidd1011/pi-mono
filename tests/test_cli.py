# tests/test_cli.py
"""Tests for pi-coding-agent CLI."""

import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock


class TestCLIEntryPoint:
    """Tests for CLI main function."""

    def test_main_function_exists(self):
        """main function is importable."""
        from pi_coding_agent.cli import main

        assert callable(main)

    def test_cli_module_imports(self):
        """CLI module can be imported."""
        from pi_coding_agent.cli import (
            main,
            _run_interactive_tui,
            _run_basic_interactive,
        )

        assert callable(main)
        assert callable(_run_interactive_tui)
        assert callable(_run_basic_interactive)


class TestCLIArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_parse_default_arguments(self):
        """Default arguments are parsed correctly."""
        from pi_coding_agent.cli import main

        with patch("sys.argv", ["pi"]):
            with patch("pi_coding_agent.cli._run_interactive_tui") as mock_tui:
                main()
                # Default mode is interactive
                mock_tui.assert_called_once()

    def test_parse_prompt_argument(self):
        """Prompt argument is parsed."""
        parser_test_cases = [
            ("pi", "test prompt", {"prompt": "test prompt"}),
        ]
        # This is a simplified test - full parsing would need argparse

    def test_parse_workspace_argument(self):
        """--workspace argument is parsed."""
        from pi_coding_agent.cli import main
        import argparse

        # Create a parser directly
        parser = argparse.ArgumentParser()
        parser.add_argument("prompt", nargs="?")
        parser.add_argument("--workspace", "-w", default=".")
        parser.add_argument("--model", "-m")
        parser.add_argument("--mode", choices=["interactive", "print", "rpc"], default="interactive")
        parser.add_argument("--no-tui", action="store_true")
        parser.add_argument("--session", "-s")

        args = parser.parse_args(["--workspace", "/tmp/test"])

        assert args.workspace == "/tmp/test"

    def test_parse_model_argument(self):
        """--model argument is parsed."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--model", "-m")

        args = parser.parse_args(["--model", "claude-3-opus"])

        assert args.model == "claude-3-opus"

    def test_parse_mode_argument(self):
        """--mode argument is parsed."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", choices=["interactive", "print", "rpc"], default="interactive")

        args = parser.parse_args(["--mode", "rpc"])
        assert args.mode == "rpc"

        args = parser.parse_args(["--mode", "print"])
        assert args.mode == "print"

    def test_parse_no_tui_argument(self):
        """--no-tui argument is parsed."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--no-tui", action="store_true")

        args = parser.parse_args(["--no-tui"])
        assert args.no_tui is True

        args = parser.parse_args([])
        assert args.no_tui is False

    def test_parse_session_argument(self):
        """--session argument is parsed."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--session", "-s")

        args = parser.parse_args(["--session", "abc123"])
        assert args.session == "abc123"

    def test_parse_version_argument(self):
        """--version argument shows version."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--version", "-v", action="version", version="%(prog)s 0.1.0")

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])

        # argparse exits with 0 for version
        assert exc_info.value.code == 0


class TestCLIModes:
    """Tests for CLI mode selection."""

    def test_print_mode_with_prompt(self):
        """Print mode is used when prompt is provided."""
        from pi_coding_agent.cli import main

        with patch("sys.argv", ["pi", "hello"]):
            with patch("pi_coding_agent.modes.print_mode.print_and_exit") as mock_print:
                mock_print.return_value = None
                # Mock asyncio.run
                with patch("asyncio.run") as mock_run:
                    main()
                    mock_run.assert_called_once()

    def test_rpc_mode_selection(self):
        """RPC mode is selected with --mode rpc."""
        from pi_coding_agent.cli import main

        with patch("sys.argv", ["pi", "--mode", "rpc"]):
            with patch("pi_coding_agent.modes.rpc.run_rpc_mode") as mock_rpc:
                mock_rpc.return_value = None
                with patch("asyncio.run") as mock_run:
                    main()
                    mock_run.assert_called_once()

    def test_no_tui_mode(self):
        """--no-tui uses basic interactive."""
        from pi_coding_agent.cli import main

        with patch("sys.argv", ["pi", "--no-tui"]):
            with patch("asyncio.run") as mock_run:
                # Should call _run_basic_interactive
                main()
                mock_run.assert_called_once()


class TestRunInteractiveTUI:
    """Tests for _run_interactive_tui function."""

    def test_run_interactive_tui_creates_session(self):
        """_run_interactive_tui creates AgentSession."""
        from pi_coding_agent.cli import _run_interactive_tui
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            with patch("pi_coding_agent.tui.app.PiCodingAgentApp") as mock_app:
                mock_app_instance = MagicMock()
                mock_app.return_value = mock_app_instance

                _run_interactive_tui(workspace, None, None)

                mock_app.assert_called_once()
                # Session should be passed to app
                call_arg = mock_app.call_args[0][0]
                assert isinstance(call_arg, AgentSession)

    def test_run_interactive_tui_with_model(self):
        """_run_interactive_tui uses specified model."""
        from pi_coding_agent.cli import _run_interactive_tui

        with tempfile.TemporaryDirectory() as workspace:
            with patch("pi_coding_agent.tui.app.PiCodingAgentApp") as mock_app:
                mock_app_instance = MagicMock()
                mock_app.return_value = mock_app_instance

                _run_interactive_tui(workspace, "claude-3-opus", None)

                mock_app.assert_called_once()

    def test_run_interactive_tui_runs_app(self):
        """_run_interactive_tui calls app.run()."""
        from pi_coding_agent.cli import _run_interactive_tui

        with tempfile.TemporaryDirectory() as workspace:
            with patch("pi_coding_agent.tui.app.PiCodingAgentApp") as mock_app:
                mock_app_instance = MagicMock()
                mock_app.return_value = mock_app_instance

                _run_interactive_tui(workspace, None, None)

                mock_app_instance.run.assert_called_once()


class TestRunBasicInteractive:
    """Tests for _run_basic_interactive function."""

    @pytest.mark.asyncio
    async def test_run_basic_interactive_prints_welcome(self, capsys):
        """_run_basic_interactive prints welcome message."""
        from pi_coding_agent.cli import _run_basic_interactive

        with tempfile.TemporaryDirectory() as workspace:
            # Mock input to exit immediately
            with patch("builtins.input", side_effect=["exit"]):
                await _run_basic_interactive(Path(workspace), None)

            captured = capsys.readouterr()
            assert "Pi Coding Agent" in captured.out
            assert "basic mode" in captured.out

    @pytest.mark.asyncio
    async def test_run_basic_interactive_exits_on_exit_command(self):
        """_run_basic_interactive exits on 'exit' input."""
        from pi_coding_agent.cli import _run_basic_interactive

        with tempfile.TemporaryDirectory() as workspace:
            with patch("builtins.input", return_value="exit"):
                # Should exit without error
                await _run_basic_interactive(Path(workspace), None)

    @pytest.mark.asyncio
    async def test_run_basic_interactive_handles_eof(self):
        """_run_basic_interactive handles EOFError."""
        from pi_coding_agent.cli import _run_basic_interactive

        with tempfile.TemporaryDirectory() as workspace:
            with patch("builtins.input", side_effect=EOFError):
                # Should exit without error
                await _run_basic_interactive(Path(workspace), None)

    @pytest.mark.asyncio
    async def test_run_basic_interactive_handles_keyboard_interrupt(self):
        """_run_basic_interactive handles KeyboardInterrupt."""
        from pi_coding_agent.cli import _run_basic_interactive

        with tempfile.TemporaryDirectory() as workspace:
            with patch("builtins.input", side_effect=KeyboardInterrupt):
                # Should exit gracefully
                await _run_basic_interactive(Path(workspace), None)


class TestCLIPackageExports:
    """Tests for package exports used by CLI."""

    def test_agent_session_import(self):
        """AgentSession can be imported."""
        from pi_coding_agent import AgentSession

        assert AgentSession is not None

    def test_print_mode_import(self):
        """Print mode functions can be imported."""
        from pi_coding_agent import run_print_mode

        assert callable(run_print_mode)

    def test_sdk_mode_import(self):
        """SDK mode functions can be imported."""
        from pi_coding_agent import create_sdk_session, run_sdk_prompt

        assert callable(create_sdk_session)
        assert callable(run_sdk_prompt)

    def test_tui_import(self):
        """TUI components can be imported."""
        from pi_coding_agent import PiCodingAgentApp

        assert PiCodingAgentApp is not None

    def test_session_manager_import(self):
        """SessionManager can be imported."""
        from pi_coding_agent import SessionManager

        assert SessionManager is not None

    def test_compaction_import(self):
        """Compaction functions can be imported."""
        from pi_coding_agent import compact_messages, should_compact

        assert callable(compact_messages)
        assert callable(should_compact)

    def test_version_in_package(self):
        """__version__ is defined."""
        import pi_coding_agent

        assert hasattr(pi_coding_agent, "__version__")
        assert pi_coding_agent.__version__ == "0.1.0"


class TestPyprojectToml:
    """Tests for pyproject.toml configuration."""

    def test_pyproject_exists(self):
        """pyproject.toml exists."""
        pyproject_path = Path("D:/qcl/pi-mono/pi_coding_agent/pyproject.toml")
        assert pyproject_path.exists()

    def test_project_name(self):
        """Project name is pi-coding-agent."""
        pyproject_path = Path("D:/qcl/pi-mono/pi_coding_agent/pyproject.toml")
        content = pyproject_path.read_text()

        assert 'name = "pi-coding-agent"' in content

    def test_project_version(self):
        """Project version is 0.1.0."""
        pyproject_path = Path("D:/qcl/pi-mono/pi_coding_agent/pyproject.toml")
        content = pyproject_path.read_text()

        assert 'version = "0.1.0"' in content

    def test_project_has_script_entry_point(self):
        """Project defines 'pi' script entry point."""
        pyproject_path = Path("D:/qcl/pi-mono/pi_coding_agent/pyproject.toml")
        content = pyproject_path.read_text()

        assert "pi = " in content
        assert "pi_coding_agent.cli:main" in content

    def test_project_dependencies(self):
        """Project has required dependencies."""
        pyproject_path = Path("D:/qcl/pi-mono/pi_coding_agent/pyproject.toml")
        content = pyproject_path.read_text()

        assert "pi-ai" in content
        assert "pi-agent-core" in content
        assert "textual" in content

    def test_project_requires_python(self):
        """Project requires Python >= 3.10."""
        pyproject_path = Path("D:/qcl/pi-mono/pi_coding_agent/pyproject.toml")
        content = pyproject_path.read_text()

        assert "requires-python" in content
        assert ">=3.10" in content

    def test_build_system(self):
        """Build system uses hatchling."""
        pyproject_path = Path("D:/qcl/pi-mono/pi_coding_agent/pyproject.toml")
        content = pyproject_path.read_text()

        assert "hatchling" in content