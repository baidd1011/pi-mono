"""
Message display widget for the TUI.

Displays conversation messages with role-based styling:
- User messages in blue
- Assistant messages in green
"""

from typing import List, Any

from textual.widgets import Static
from rich.text import Text
from rich.markdown import Markdown


class MessageDisplay(Static):
    """Display conversation messages with role-based styling.

    Features:
    - User messages shown in bold blue
    - Assistant messages shown in bold green
    - Text content rendered with markdown support
    """

    def __init__(self, id: str = None):
        """Initialize the message display.

        Args:
            id: Widget ID.
        """
        super().__init__(id=id)
        self._messages: List[Any] = []

    def update_messages(self, messages: List[Any]) -> None:
        """Update the displayed messages.

        Args:
            messages: List of message objects to display.
        """
        self._messages = messages
        self._render()

    def _render(self) -> None:
        """Render messages to the widget."""
        text = Text()

        for msg in self._messages:
            role = self._get_role(msg)
            content = self._get_content(msg)

            if role == "user":
                text.append("You: ", style="bold blue")
            elif role == "assistant":
                text.append("Assistant: ", style="bold green")
            else:
                text.append(f"{role}: ", style="bold yellow")

            # Render content blocks
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "text")
                    if block_type == "text":
                        block_text = block.get("text", "")
                        text.append(block_text)
                elif isinstance(block, str):
                    text.append(block)
                else:
                    text.append(str(block))

            text.append("\n\n")

        self.update(text)

    def _get_role(self, msg: Any) -> str:
        """Get the role from a message.

        Args:
            msg: The message object.

        Returns:
            The role string.
        """
        if isinstance(msg, dict):
            return msg.get("role", "unknown")
        return getattr(msg, "role", "unknown")

    def _get_content(self, msg: Any) -> List[Any]:
        """Get the content from a message.

        Args:
            msg: The message object.

        Returns:
            List of content blocks.
        """
        if isinstance(msg, dict):
            content = msg.get("content", [])
            if isinstance(content, list):
                return content
            return [{"type": "text", "text": str(content)}]

        # For Pydantic/message objects
        content = getattr(msg, "content", None)
        if content is None:
            return [{"type": "text", "text": ""}]
        if isinstance(content, list):
            return content
        return [{"type": "text", "text": str(content)}]

    def clear_messages(self) -> None:
        """Clear all displayed messages."""
        self._messages = []
        self.update(Text())