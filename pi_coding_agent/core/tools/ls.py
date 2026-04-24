"""
Ls tool for directory listing.

This tool allows the agent to list directory contents.
"""

from typing import Any, Optional, Callable, Awaitable, List
from dataclasses import dataclass
from pathlib import Path
import os

from pi_agent_core import AgentTool, AgentToolResult
from pi_ai import TextContent


# Default limits
DEFAULT_LIMIT = 500  # Maximum entries
DEFAULT_MAX_BYTES = 100 * 1024  # 100KB


@dataclass
class LsToolDetails:
    """Details about ls tool execution."""
    path: str
    entry_count: int
    entry_limit_reached: bool = False
    truncated: bool = False


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


async def execute_ls(
    tool_call_id: str,
    args: dict,
    signal: Optional[Any] = None,
    update_cb: Optional[Callable[[AgentToolResult], None]] = None,
) -> AgentToolResult[LsToolDetails]:
    """Execute the ls tool.

    Args:
        tool_call_id: Unique ID for this tool call
        args: Tool arguments with optional path
        signal: Optional abort signal
        update_cb: Optional callback for streaming updates

    Returns:
        AgentToolResult with directory listing and details
    """
    workspace_dir = args.get("_workspace_dir", ".")

    # Extract arguments
    path = args.get("path", ".")
    limit = args.get("limit", DEFAULT_LIMIT)

    # Validate and resolve path
    try:
        target = _resolve_path(path, workspace_dir)
    except ValueError as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error: {e}")],
            details=LsToolDetails(path=path, entry_count=0)
        )

    # Check if path exists
    if not target.exists():
        return AgentToolResult(
            content=[TextContent(text=f"Error: Path not found: {path}")],
            details=LsToolDetails(path=path, entry_count=0)
        )

    # Check if it's a directory
    if not target.is_dir():
        return AgentToolResult(
            content=[TextContent(text=f"Error: Not a directory: {path}")],
            details=LsToolDetails(path=path, entry_count=0)
        )

    # Check for abort
    if signal and hasattr(signal, 'aborted') and signal.aborted:
        return AgentToolResult(
            content=[TextContent(text="Operation aborted")],
            details=LsToolDetails(path=path, entry_count=0)
        )

    # Read directory entries
    try:
        entries = list(target.iterdir())
    except PermissionError:
        return AgentToolResult(
            content=[TextContent(text=f"Error: Cannot read directory: {path}")],
            details=LsToolDetails(path=path, entry_count=0)
        )
    except Exception as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error reading directory: {e}")],
            details=LsToolDetails(path=path, entry_count=0)
        )

    # Check for abort
    if signal and hasattr(signal, 'aborted') and signal.aborted:
        return AgentToolResult(
            content=[TextContent(text="Operation aborted")],
            details=LsToolDetails(path=path, entry_count=0)
        )

    # Format entries with directory indicators
    results: List[str] = []
    entry_limit_reached = False

    # Sort entries (case-insensitive)
    entries.sort(key=lambda e: e.name.lower())

    for entry in entries[:limit]:
        # Add suffix for directories
        suffix = "/" if entry.is_dir() else ""
        results.append(entry.name + suffix)

    # Check if we hit the limit
    if len(entries) > limit:
        entry_limit_reached = True

    # Build output
    if not results:
        return AgentToolResult(
            content=[TextContent(text="(empty directory)")],
            details=LsToolDetails(path=path, entry_count=0)
        )

    output_text = '\n'.join(results)

    # Truncate by bytes
    truncated = False
    if len(output_text.encode('utf-8')) > DEFAULT_MAX_BYTES:
        truncated = True
        output_text = output_text[:DEFAULT_MAX_BYTES]
        output_text = output_text[:output_text.rfind('\n')] if '\n' in output_text else output_text

    # Add notices
    notices = []
    if entry_limit_reached:
        notices.append(f"{limit} entries limit reached")
    if truncated:
        notices.append(f"{DEFAULT_MAX_BYTES // 1024}KB limit reached")

    if notices:
        output_text += f"\n\n[Truncated: {', '.join(notices)}]"

    return AgentToolResult(
        content=[TextContent(text=output_text)],
        details=LsToolDetails(
            path=path,
            entry_count=len(results),
            entry_limit_reached=entry_limit_reached,
            truncated=truncated
        )
    )


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
        """Execute the ls tool."""
        # Inject workspace_dir into args
        full_args = {**args, "_workspace_dir": workspace_dir}
        return await execute_ls(tool_call_id, full_args, signal, update_cb)

    # JSON Schema for parameters
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to list (default: current directory)"
            },
            "limit": {
                "type": "number",
                "description": f"Maximum number of entries (default: {DEFAULT_LIMIT})"
            }
        },
        "required": []
    }

    return AgentTool(
        name="ls",
        label="List Directory",
        description=f"List directory contents. Returns entries sorted alphabetically with '/' suffix for directories. Output is truncated to {DEFAULT_LIMIT} entries or {DEFAULT_MAX_BYTES // 1024}KB.",
        parameters=parameters,
        execute=execute,
    )