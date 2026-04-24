"""
pi-coding-agent core module.

This module provides the AgentSession class - the core abstraction
shared by all modes (interactive, print, rpc, sdk).
"""

from .agent_session import AgentSession
from .defaults import (
    DEFAULT_MODEL,
    DEFAULT_THINKING_LEVEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_MAX_LINES,
    DEFAULT_MAX_BYTES,
)
from .tools import create_coding_tools
from .session_header import (
    SessionHeader,
    SessionVersion,
    SessionEntry,
    SessionMessageEntry,
    ThinkingLevelChangeEntry,
    ModelChangeEntry,
    CompactionEntry,
    BranchSummaryEntry,
    CustomEntry,
    LabelEntry,
    SessionInfoEntry,
    CURRENT_SESSION_VERSION,
)
from .session_manager import (
    SessionManager,
    SessionContext,
    SessionInfo,
    SessionTreeNode,
)


__all__ = [
    "AgentSession",
    "DEFAULT_MODEL",
    "DEFAULT_THINKING_LEVEL",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_MAX_LINES",
    "DEFAULT_MAX_BYTES",
    "create_coding_tools",
    # Session header types
    "SessionHeader",
    "SessionVersion",
    "SessionEntry",
    "SessionMessageEntry",
    "ThinkingLevelChangeEntry",
    "ModelChangeEntry",
    "CompactionEntry",
    "BranchSummaryEntry",
    "CustomEntry",
    "LabelEntry",
    "SessionInfoEntry",
    "CURRENT_SESSION_VERSION",
    # Session manager
    "SessionManager",
    "SessionContext",
    "SessionInfo",
    "SessionTreeNode",
]