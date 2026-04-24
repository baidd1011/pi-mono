"""
Grep tool for content search.

This tool allows the agent to search for patterns in file contents.
"""

from typing import Any, Optional, Callable, Awaitable
from pi_agent_core import AgentTool, AgentToolResult
from pi_ai import TextContent


def create_grep_tool(workspace_dir: str) -> AgentTool:
    """Create the grep tool for content search.

    Args:
        workspace_dir: The workspace directory to search in.

    Returns:
        An AgentTool configured for grep searching.
    """
    async def execute(
        tool_call_id: str,
        args: Any,
        signal: Optional[Any] = None,
        update_cb: Optional[Callable[[AgentToolResult], None]] = None,
    ) -> AgentToolResult:
        """Execute the grep tool - stub implementation."""
        # TODO: Implement actual grep search logic
        return AgentToolResult(
            content=[TextContent(text="Grep tool stub - not yet implemented")],
            details={"pattern": args.get("pattern", ""), "workspace_dir": workspace_dir}
        )

    class GrepParameters:
        """Parameters for the grep tool."""
        pattern: str
        path: Optional[str] = None
        glob: Optional[str] = None
        output_mode: str = "files_with_matches"

    return AgentTool(
        name="grep",
        label="Search Content",
        description="Search for patterns in file contents using regex. Returns matching files or lines.",
        parameters=GrepParameters,
        execute=execute,
    )