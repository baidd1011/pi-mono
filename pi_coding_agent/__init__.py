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

from .core.session_manager import SessionManager
from .core.compaction import compact_messages, should_compact

from .core.extensions import (
    Extension,
    ExtensionCommand,
    ExtensionKeybinding,
    discover_extensions,
    load_extension,
    load_all_extensions,
)

from .tui import (
    PiCodingAgentApp,
    MessageDisplay,
    EditorWidget,
    FooterWidget,
)

from .modes.sdk import create_sdk_session, run_sdk_prompt
from .modes.print_mode import run_print_mode


__version__ = "0.1.0"

__all__ = [
    "AgentSession",
    "DEFAULT_MODEL",
    "DEFAULT_THINKING_LEVEL",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_MAX_LINES",
    "DEFAULT_MAX_BYTES",
    "create_coding_tools",
    # Session management
    "SessionManager",
    "compact_messages",
    "should_compact",
    # Extension system
    "Extension",
    "ExtensionCommand",
    "ExtensionKeybinding",
    "discover_extensions",
    "load_extension",
    "load_all_extensions",
    # TUI components
    "PiCodingAgentApp",
    "MessageDisplay",
    "EditorWidget",
    "FooterWidget",
    # SDK mode
    "create_sdk_session",
    "run_sdk_prompt",
    "run_print_mode",
]