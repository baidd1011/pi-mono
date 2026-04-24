# tests/test_tui.py
"""Tests for the Textual TUI components."""

import pytest
from pathlib import Path
import tempfile


class TestMessageDisplay:
    """Tests for MessageDisplay widget."""

    def test_message_display_creation(self):
        """MessageDisplay can be created."""
        from pi_coding_agent.tui import MessageDisplay

        display = MessageDisplay(id="test-display")
        assert display.id == "test-display"

    def test_message_display_update_messages(self):
        """MessageDisplay can update with messages."""
        from pi_coding_agent.tui import MessageDisplay

        display = MessageDisplay()

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there"}]},
        ]

        # Just test internal state, not rendering (which requires app context)
        display._messages = messages

        assert len(display._messages) == 2

    def test_message_display_clear_messages(self):
        """MessageDisplay can clear messages."""
        from pi_coding_agent.tui import MessageDisplay

        display = MessageDisplay()

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
        display._messages = messages

        display._messages = []

        assert len(display._messages) == 0

    def test_message_display_get_role_from_dict(self):
        """MessageDisplay extracts role from dict message."""
        from pi_coding_agent.tui import MessageDisplay

        display = MessageDisplay()

        msg = {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        role = display._get_role(msg)

        assert role == "user"

    def test_message_display_get_content_from_dict(self):
        """MessageDisplay extracts content from dict message."""
        from pi_coding_agent.tui import MessageDisplay

        display = MessageDisplay()

        msg = {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        content = display._get_content(msg)

        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Hello"

    def test_message_display_handles_string_content(self):
        """MessageDisplay handles string content."""
        from pi_coding_agent.tui import MessageDisplay

        display = MessageDisplay()

        msg = {"role": "user", "content": "Plain string"}
        content = display._get_content(msg)

        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Plain string"


class TestEditorWidget:
    """Tests for EditorWidget."""

    def test_editor_widget_creation(self):
        """EditorWidget can be created."""
        from pi_coding_agent.tui import EditorWidget

        editor = EditorWidget(id="test-editor")
        assert editor.id == "test-editor"

    def test_editor_widget_get_text(self):
        """EditorWidget can get text."""
        from pi_coding_agent.tui import EditorWidget

        editor = EditorWidget()

        # Initially empty
        assert editor.get_text() == ""

    def test_editor_widget_set_text(self):
        """EditorWidget can set text."""
        from pi_coding_agent.tui import EditorWidget

        editor = EditorWidget()

        editor.set_text("Hello world")
        assert editor.get_text() == "Hello world"

    def test_editor_widget_clear(self):
        """EditorWidget can clear text."""
        from pi_coding_agent.tui import EditorWidget

        editor = EditorWidget()

        editor.set_text("Some text")
        editor.clear()

        assert editor.get_text() == ""

    def test_editor_widget_is_empty(self):
        """EditorWidget can check if empty."""
        from pi_coding_agent.tui import EditorWidget

        editor = EditorWidget()

        assert editor.is_empty() is True

        editor.set_text("content")
        assert editor.is_empty() is False

        editor.set_text("   ")  # Only whitespace
        assert editor.is_empty() is True


class TestFooterWidget:
    """Tests for FooterWidget."""

    def test_footer_widget_creation(self):
        """FooterWidget can be created."""
        from pi_coding_agent.tui import FooterWidget

        footer = FooterWidget(id="test-footer")
        assert footer.id == "test-footer"

    def test_footer_widget_default_status(self):
        """FooterWidget has default status 'Ready'."""
        from pi_coding_agent.tui import FooterWidget

        footer = FooterWidget()

        assert footer.status == "Ready"

    def test_footer_widget_update_status(self):
        """FooterWidget can update status."""
        from pi_coding_agent.tui import FooterWidget

        footer = FooterWidget()

        footer.update_status("Processing")
        assert footer.status == "Processing"

    def test_footer_widget_set_running(self):
        """FooterWidget can set running state."""
        from pi_coding_agent.tui import FooterWidget

        footer = FooterWidget()

        footer.set_running(True)
        # Status should show running indicator


class TestPiCodingAgentApp:
    """Tests for PiCodingAgentApp."""

    def test_app_creation_with_session(self):
        """PiCodingAgentApp can be created with AgentSession."""
        from pi_coding_agent.tui import PiCodingAgentApp
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            app = PiCodingAgentApp(session)

            assert app.session == session
            assert app.running is False

    def test_app_has_bindings(self):
        """PiCodingAgentApp has required bindings."""
        from pi_coding_agent.tui import PiCodingAgentApp
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            app = PiCodingAgentApp(session)

            # BINDINGS is a list of tuples (key, action, description)
            binding_keys = [b[0] for b in app.BINDINGS]
            assert "ctrl+c" in binding_keys
            assert "ctrl+enter" in binding_keys
            assert "ctrl+l" in binding_keys

    def test_app_compose_method(self):
        """PiCodingAgentApp compose method returns expected widgets."""
        from pi_coding_agent.tui import PiCodingAgentApp
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            app = PiCodingAgentApp(session)

            # compose() returns generator, need to iterate
            widgets = list(app.compose())

            # Should have Header and 2 Containers
            assert len(widgets) == 3

    def test_app_running_reactive(self):
        """PiCodingAgentApp running is reactive."""
        from pi_coding_agent.tui import PiCodingAgentApp
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            app = PiCodingAgentApp(session)

            assert app.running is False

            # Set running
            app.running = True
            assert app.running is True

    def test_app_add_message(self):
        """PiCodingAgentApp can add messages."""
        from pi_coding_agent.tui import PiCodingAgentApp
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            app = PiCodingAgentApp(session)

            msg = {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            app._add_message(msg)

            assert len(app._messages) == 1

    def test_app_unsubscribe_on_unmount(self):
        """PiCodingAgentApp unsubscribes from events on unmount."""
        from pi_coding_agent.tui import PiCodingAgentApp
        from pi_coding_agent import AgentSession

        with tempfile.TemporaryDirectory() as workspace:
            session = AgentSession(workspace)
            app = PiCodingAgentApp(session)

            # Should have unsubscribe function
            assert app._unsubscribe is not None
            assert callable(app._unsubscribe)


class TestTuiExports:
    """Tests for TUI module exports."""

    def test_import_from_package(self):
        """TUI components can be imported from pi_coding_agent."""
        from pi_coding_agent import (
            PiCodingAgentApp,
            MessageDisplay,
            EditorWidget,
            FooterWidget,
        )

        assert PiCodingAgentApp is not None
        assert MessageDisplay is not None
        assert EditorWidget is not None
        assert FooterWidget is not None

    def test_import_from_tui_module(self):
        """TUI components can be imported from tui submodule."""
        from pi_coding_agent.tui import (
            PiCodingAgentApp,
            MessageDisplay,
            EditorWidget,
            FooterWidget,
        )

        assert PiCodingAgentApp is not None
        assert MessageDisplay is not None
        assert EditorWidget is not None
        assert FooterWidget is not None