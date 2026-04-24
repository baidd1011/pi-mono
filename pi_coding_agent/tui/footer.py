"""
Status footer widget for the TUI.

Displays:
- Current status (Ready, Processing, Running tool)
- Running indicator
"""

from textual.widgets import Static
from textual.reactive import reactive


class FooterWidget(Static):
    """Status footer widget.

    Features:
    - Reactive status text
    - Running indicator when agent is active
    """

    DEFAULT_CSS = """
    FooterWidget {
        background: $panel;
        color: $text;
        padding: 1;
        content-align: left middle;
    }
    """

    status = reactive("Ready")

    def __init__(self, id: str = None):
        """Initialize the footer widget.

        Args:
            id: Widget ID.
        """
        super().__init__(id=id)

    def update_status(self, text: str) -> None:
        """Update the status text.

        Args:
            text: The new status text.
        """
        self.status = text

    def watch_status(self, status: str) -> None:
        """React to status changes.

        Args:
            status: The new status value.
        """
        self.update(f"Status: {status}")

    def set_running(self, running: bool) -> None:
        """Set the running state.

        Args:
            running: True if agent is running.
        """
        if running:
            self.update("Status: Running...")
        else:
            self.update(f"Status: {self.status}")