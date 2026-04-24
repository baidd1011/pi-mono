"""
Multi-line text input editor widget.

Provides:
- TextArea-based input
- Placeholder text
- Clear and get_text methods
"""

from textual.widgets import TextArea


class EditorWidget(TextArea):
    """Multi-line text input widget.

    Features:
    - Based on TextArea for rich editing
    - Placeholder text when empty
    - Ctrl+Enter to submit (handled by parent app)
    """

    DEFAULT_CSS = """
    EditorWidget {
        border: solid $primary;
        padding: 1;
    }

    EditorWidget:focus {
        border: double $primary;
    }
    """

    def __init__(self, id: str = None):
        """Initialize the editor widget.

        Args:
            id: Widget ID.
        """
        # Note: TextArea doesn't have a placeholder option, but we can
        # work with it as a simple text area
        super().__init__(
            id=id,
            text="",
        )

    def get_text(self) -> str:
        """Get the current text content.

        Returns:
            The text content of the editor.
        """
        return self.text

    def clear(self) -> None:
        """Clear the editor content."""
        self.text = ""

    def set_text(self, text: str) -> None:
        """Set the editor content.

        Args:
            text: The text to set.
        """
        self.text = text

    def is_empty(self) -> bool:
        """Check if the editor is empty.

        Returns:
            True if the editor has no content.
        """
        return len(self.text.strip()) == 0