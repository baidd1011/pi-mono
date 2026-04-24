"""
List tool for directory listing.

This tool allows the agent to list directory contents.
"""

from typing import Any, Optional, Callable, Awaitable
from pi_agent_core import AgentTool, AgentToolResult
from pi_ai import TextContent


def create_ls_tool(workspace_dir: str) -> AgentTool:
    """Create the ls tool for directory listing.

    Args:
        workspace_dir: The workspace directory to list from.

    Returns:
        An AgentTool configured for directory listing.
    """
    async def execute(
        tool_call_id: str,
        args: Any,
        signal: Optional[Any] = None,
        update_cb: Optional[Callable[[AgentToolResult], None]] = None,
    ) -> AgentToolResult:
        """Execute the ls tool - stub implementation."""
        # TODO: Implement actual directory listing logic
        return AgentToolResult(
            content=[TextContent(text="Ls tool stub - not yet implemented")],
            details={"path": args.get("path", ""), "workspace_dir": workspace_dir}
        )

    class LsParameters:
        """Parameters for the ls tool."""
        path: Optional[str] = None

    return AgentTool(
        name="ls",
        label="List Directory",
        description="List the contents of a directory. Shows files and subdirectories.",
        parameters=LsParameters,
        execute=execute,
    )