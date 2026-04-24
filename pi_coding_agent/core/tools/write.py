"""
Write tool for file writing operations.

This tool allows the agent to write content to files in the workspace.
"""

from typing import Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from pathlib import Path
import os

from pi_agent_core import AgentTool, AgentToolResult
from pi_ai import TextContent


@dataclass
class WriteToolDetails:
    """Details about write tool execution."""
    path: str
    bytes_written: int


def _resolve_path(path: str, workspace_dir: str) -> Path:
    """Resolve path relative to workspace and validate it's within workspace."""
    workspace = Path(workspace_dir).resolve()

    # Handle both relative and absolute paths
    target = Path(path)
    if not target.is_absolute():
        target = workspace / path

    target = target.resolve()

    # Security check: ensure path is within workspace
    try:
        target.relative_to(workspace)
    except ValueError:
        raise ValueError(f"Path '{path}' is outside workspace directory")

    return target


async def execute_write(
    tool_call_id: str,
    args: dict,
    signal: Optional[Any] = None,
    update_cb: Optional[Callable[[AgentToolResult], None]] = None,
) -> AgentToolResult[WriteToolDetails]:
    """Execute the write tool.

    Args:
        tool_call_id: Unique ID for this tool call
        args: Tool arguments with path, content
        signal: Optional abort signal
        update_cb: Optional callback for streaming updates

    Returns:
        AgentToolResult with confirmation and details
    """
    workspace_dir = args.get("_workspace_dir", ".")

    # Extract arguments
    path = args.get("path")
    content = args.get("content")

    if not path:
        return AgentToolResult(
            content=[TextContent(text="Error: 'path' argument is required")],
            details=WriteToolDetails(path="", bytes_written=0)
        )

    if content is None:
        return AgentToolResult(
            content=[TextContent(text="Error: 'content' argument is required")],
            details=WriteToolDetails(path=path, bytes_written=0)
        )

    # Validate and resolve path
    try:
        target = _resolve_path(path, workspace_dir)
    except ValueError as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error: {e}")],
            details=WriteToolDetails(path=path, bytes_written=0)
        )

    # Check for abort before creating directory
    if signal and hasattr(signal, 'aborted') and signal.aborted:
        return AgentToolResult(
            content=[TextContent(text="Operation aborted")],
            details=WriteToolDetails(path=path, bytes_written=0)
        )

    # Create parent directories if needed
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error creating directory: {e}")],
            details=WriteToolDetails(path=path, bytes_written=0)
        )

    # Check for abort before writing
    if signal and hasattr(signal, 'aborted') and signal.aborted:
        return AgentToolResult(
            content=[TextContent(text="Operation aborted")],
            details=WriteToolDetails(path=path, bytes_written=0)
        )

    # Write the file
    try:
        bytes_written = len(content.encode('utf-8'))
        with open(target, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error writing file: {e}")],
            details=WriteToolDetails(path=path, bytes_written=0)
        )

    return AgentToolResult(
        content=[TextContent(text=f"Successfully wrote {bytes_written} bytes to {path}")],
        details=WriteToolDetails(path=path, bytes_written=bytes_written)
    )


def create_write_tool(workspace_dir: str) -> AgentTool:
    """Create the write tool for file writing operations.

    Args:
        workspace_dir: The workspace directory to constrain file access.

    Returns:
        An AgentTool configured for file writing.
    """
    async def execute(
        tool_call_id: str,
        args: Any,
        signal: Optional[Any] = None,
        update_cb: Optional[Callable[[AgentToolResult], None]] = None,
    ) -> AgentToolResult:
        """Execute the write tool."""
        # Inject workspace_dir into args
        full_args = {**args, "_workspace_dir": workspace_dir}
        return await execute_write(tool_call_id, full_args, signal, update_cb)

    # JSON Schema for parameters
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write (relative or absolute)"
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file"
            }
        },
        "required": ["path", "content"]
    }

    return AgentTool(
        name="write",
        label="Write File",
        description="Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Creates parent directories automatically.",
        parameters=parameters,
        execute=execute,
    )