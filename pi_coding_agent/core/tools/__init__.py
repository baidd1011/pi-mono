"""
Tool factory functions for pi-coding-agent.

This module provides the create_coding_tools function that creates
all 7 coding tools for the agent session.
"""

from typing import List
from pi_agent_core import AgentTool

from .read import create_read_tool
from .write import create_write_tool
from .edit import create_edit_tool
from .bash import create_bash_tool
from .grep import create_grep_tool
from .find import create_find_tool
from .ls import create_ls_tool


def create_coding_tools(workspace_dir: str) -> List[AgentTool]:
    """Create all 7 coding tools for the agent session.

    Args:
        workspace_dir: The workspace directory to constrain tool operations.

    Returns:
        A list of 7 AgentTool instances for coding operations:
        - read: Read file contents
        - write: Write file contents
        - edit: Make precise edits to files
        - bash: Run shell commands
        - grep: Search file contents
        - find: Find files by pattern
        - ls: List directory contents
    """
    return [
        create_read_tool(workspace_dir),
        create_write_tool(workspace_dir),
        create_edit_tool(workspace_dir),
        create_bash_tool(workspace_dir),
        create_grep_tool(workspace_dir),
        create_find_tool(workspace_dir),
        create_ls_tool(workspace_dir),
    ]


__all__ = [
    "create_coding_tools",
    "create_read_tool",
    "create_write_tool",
    "create_edit_tool",
    "create_bash_tool",
    "create_grep_tool",
    "create_find_tool",
    "create_ls_tool",
]