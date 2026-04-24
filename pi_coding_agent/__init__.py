"""
pi-coding-agent: Interactive coding agent CLI and SDK.

This package provides a coding agent that can:
- Read, write, and edit files
- Run shell commands
- Search file contents
- Manage sessions with persistence
"""

from .core import (
    AgentSession,
    DEFAULT_MODEL,
    DEFAULT_THINKING_LEVEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_MAX_LINES,
    DEFAULT_MAX_BYTES,
    create_coding_tools,
)


__version__ = "0.1.0"

__all__ = [
    "AgentSession",
    "DEFAULT_MODEL",
    "DEFAULT_THINKING_LEVEL",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_MAX_LINES",
    "DEFAULT_MAX_BYTES",
    "create_coding_tools",
]