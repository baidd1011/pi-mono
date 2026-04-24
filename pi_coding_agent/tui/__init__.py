"""
Textual TUI for pi-coding-agent.

This module provides a rich terminal UI for interactive coding sessions.
"""

from .app import PiCodingAgentApp
from .messages import MessageDisplay
from .editor import EditorWidget
from .footer import FooterWidget


__all__ = [
    "PiCodingAgentApp",
    "MessageDisplay",
    "EditorWidget",
    "FooterWidget",
]