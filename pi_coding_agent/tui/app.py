"""
Main TUI application for pi-coding-agent.

Provides:
- Message display with role styling
- Multi-line input editor
- Status footer with running indicator
- Keyboard bindings for submit/quit/clear
"""

import asyncio
from typing import List, Any, Callable, Awaitable, Union

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Header

from ..core.agent_session import AgentSession
from .messages import MessageDisplay
from .editor import EditorWidget
from .footer import FooterWidget
from pi_agent_core import AbortSignal


class PiCodingAgentApp(App):
    """Main TUI for pi coding agent.

    Features:
    - Message display area with conversation history
    - Multi-line input editor with Ctrl+Enter submission
    - Status footer showing running state
    - Keyboard bindings: Ctrl+C (quit), Ctrl+Enter (submit), Ctrl+L (clear)
    """

    CSS = """
    Screen {
        layout: vertical;
    }

    #message-container {
        height: 70%;
        overflow-y: auto;
    }

    #input-container {
        height: 30%;
    }

    #editor {
        height: 80%;
    }

    #footer {
        height: 20%;
    }

    Header {
        background: $primary;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+enter", "submit", "Submit"),
        ("ctrl+l", "clear", "Clear"),
    ]

    running = reactive(False)

    def __init__(self, session: AgentSession):
        """Initialize the TUI app.

        Args:
            session: The AgentSession to use for prompts.
        """
        super().__init__()
        self.session = session
        self._messages: List[Any] = []

        # Subscribe to agent events
        self._unsubscribe: Callable[[], None] = session.subscribe(self._on_event)

    def compose(self) -> ComposeResult:
        """Compose the UI widgets."""
        yield Header()
        yield Container(
            MessageDisplay(id="message-display"),
            id="message-container",
        )
        yield Container(
            EditorWidget(id="editor"),
            FooterWidget(id="footer"),
            id="input-container",
        )

    def on_mount(self) -> None:
        """Called when app is mounted."""
        # Initial status
        footer = self.query_one("#footer", FooterWidget)
        footer.update_status("Ready")

    def action_submit(self) -> None:
        """Submit current input."""
        editor = self.query_one("#editor", EditorWidget)
        text = editor.get_text()
        if text and not self.running:
            # Add user message
            self._add_message({
                "role": "user",
                "content": [{"type": "text", "text": text}]
            })
            editor.clear()
            self.running = True

            # Update status
            footer = self.query_one("#footer", FooterWidget)
            footer.update_status("Processing...")

            # Process prompt asynchronously
            asyncio.create_task(self._process_prompt(text))

    def action_clear(self) -> None:
        """Clear input editor."""
        editor = self.query_one("#editor", EditorWidget)
        editor.clear()

    async def _process_prompt(self, text: str) -> None:
        """Process a user prompt through the agent session.

        Args:
            text: The user's input text.
        """
        try:
            await self.session.prompt(text)
            await self.session.wait_for_idle()
        except Exception as e:
            # Add error message
            self._add_message({
                "role": "assistant",
                "content": [{"type": "text", "text": f"Error: {str(e)}"}]
            })
        finally:
            self.running = False
            footer = self.query_one("#footer", FooterWidget)
            footer.update_status("Ready")

    def _on_event(self, event: Any, signal: AbortSignal) -> None:
        """Handle agent events.

        Args:
            event: The agent event.
            signal: The abort signal for the current run.
        """
        event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)

        if event_type == "message_end":
            msg = event.get("message") if isinstance(event, dict) else getattr(event, "message", None)
            if msg:
                self._add_message(msg)

        elif event_type == "tool_execution_start":
            tool_name = event.get("tool_name") if isinstance(event, dict) else getattr(event, "tool_name", "unknown")
            self._update_status(f"Running {tool_name}...")

        elif event_type == "turn_end":
            msg = event.get("message") if isinstance(event, dict) else getattr(event, "message", None)
            if msg:
                self._add_message(msg)

    def _add_message(self, msg: Any) -> None:
        """Add a message to the conversation display.

        Args:
            msg: The message to add.
        """
        self._messages.append(msg)

        # Update message display
        try:
            display = self.query_one("#message-display", MessageDisplay)
            display.update_messages(self._messages)
        except Exception:
            pass  # Widget might not be mounted yet

    def _update_status(self, text: str) -> None:
        """Update the status footer.

        Args:
            text: The status text to display.
        """
        try:
            footer = self.query_one("#footer", FooterWidget)
            footer.update_status(text)
        except Exception:
            pass  # Widget might not be mounted yet

    def on_unmount(self) -> None:
        """Called when app is unmounted."""
        # Unsubscribe from agent events
        if self._unsubscribe:
            self._unsubscribe()