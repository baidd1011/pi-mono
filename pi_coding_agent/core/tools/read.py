"""
Read tool for file reading operations.

This tool allows the agent to read file contents from the workspace.
"""

from typing import Any, Optional, Callable, Awaitable, List, Union
from dataclasses import dataclass
from pathlib import Path
import os

from pi_agent_core import AgentTool, AgentToolResult
from pi_ai import TextContent, ImageContent


# Default limits matching TypeScript implementation
DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 100 * 1024  # 100KB


@dataclass
class ReadToolDetails:
    """Details about read tool execution."""
    path: str
    offset: Optional[int] = None
    limit: Optional[int] = None
    truncated: bool = False
    total_lines: int = 0
    output_lines: int = 0


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


def _format_output_with_line_numbers(lines: List[str], start_line: int = 1) -> str:
    """Format lines with line numbers like cat -n."""
    result = []
    for i, line in enumerate(lines):
        line_num = start_line + i
        result.append(f"{line_num}\t{line}")
    return "\n".join(result)


async def execute_read(
    tool_call_id: str,
    args: dict,
    signal: Optional[Any] = None,
    update_cb: Optional[Callable[[AgentToolResult], None]] = None,
) -> AgentToolResult[ReadToolDetails]:
    """Execute the read tool.

    Args:
        tool_call_id: Unique ID for this tool call
        args: Tool arguments with path, offset, limit
        signal: Optional abort signal
        update_cb: Optional callback for streaming updates

    Returns:
        AgentToolResult with file content and details
    """
    workspace_dir = args.get("_workspace_dir", ".")

    # Extract arguments
    path = args.get("path")
    if not path:
        return AgentToolResult(
            content=[TextContent(text="Error: 'path' argument is required")],
            details=ReadToolDetails(path="")
        )

    offset = args.get("offset")
    limit = args.get("limit", DEFAULT_MAX_LINES)

    # Validate and resolve path
    try:
        target = _resolve_path(path, workspace_dir)
    except ValueError as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error: {e}")],
            details=ReadToolDetails(path=path)
        )

    # Check if file exists
    if not target.exists():
        return AgentToolResult(
            content=[TextContent(text=f"Error: File not found: {path}")],
            details=ReadToolDetails(path=path)
        )

    # Check if it's a file
    if not target.is_file():
        return AgentToolResult(
            content=[TextContent(text=f"Error: Not a file: {path}")],
            details=ReadToolDetails(path=path)
        )

    # Check for image file (basic detection)
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
    ext = target.suffix.lower()

    if ext in image_extensions:
        # For images, we just return a note - actual image handling would require
        # reading binary and encoding as base64
        return AgentToolResult(
            content=[TextContent(text=f"[Image file: {ext[1:]} format. Binary content not displayed in text mode.]")],
            details=ReadToolDetails(path=path)
        )

    # Read file content
    try:
        with open(target, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return AgentToolResult(
            content=[TextContent(text=f"Error: Cannot read file as UTF-8 text: {path}")],
            details=ReadToolDetails(path=path)
        )
    except Exception as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error reading file: {e}")],
            details=ReadToolDetails(path=path)
        )

    # Split into lines
    all_lines = content.split('\n')
    total_lines = len(all_lines)

    # Apply offset (1-indexed input to 0-indexed array)
    start_idx = 0
    if offset is not None:
        start_idx = max(0, offset - 1)
        if start_idx >= total_lines:
            return AgentToolResult(
                content=[TextContent(text=f"Error: Offset {offset} is beyond end of file ({total_lines} lines total)")],
                details=ReadToolDetails(path=path, offset=offset, total_lines=total_lines)
            )

    # Apply limit
    end_idx = min(start_idx + limit, total_lines)
    selected_lines = all_lines[start_idx:end_idx]

    # Truncate by bytes if needed
    output_text = '\n'.join(selected_lines)
    truncated = False
    output_lines = len(selected_lines)

    if len(output_text.encode('utf-8')) > DEFAULT_MAX_BYTES:
        # Truncate to byte limit
        truncated = True
        byte_count = 0
        truncated_lines = []
        for line in selected_lines:
            line_bytes = len(line.encode('utf-8')) + 1  # +1 for newline
            if byte_count + line_bytes > DEFAULT_MAX_BYTES:
                break
            truncated_lines.append(line)
            byte_count += line_bytes
        selected_lines = truncated_lines
        output_lines = len(truncated_lines)

    # Format with line numbers
    output = _format_output_with_line_numbers(selected_lines, start_idx + 1)

    # Add truncation notice if needed
    if truncated:
        output += f"\n\n[Truncated: showing {output_lines} of {total_lines} lines ({DEFAULT_MAX_BYTES // 1024}KB limit). Use offset={start_idx + output_lines + 1} to continue.]"
    elif offset is None and limit == DEFAULT_MAX_LINES and end_idx < total_lines:
        remaining = total_lines - end_idx
        output += f"\n\n[{remaining} more lines in file. Use offset={end_idx + 1} to continue.]"

    return AgentToolResult(
        content=[TextContent(text=output)],
        details=ReadToolDetails(
            path=path,
            offset=offset,
            limit=limit,
            truncated=truncated,
            total_lines=total_lines,
            output_lines=output_lines
        )
    )


def create_read_tool(workspace_dir: str) -> AgentTool:
    """Create the read tool for file reading operations.

    Args:
        workspace_dir: The workspace directory to constrain file access.

    Returns:
        An AgentTool configured for file reading.
    """
    async def execute(
        tool_call_id: str,
        args: Any,
        signal: Optional[Any] = None,
        update_cb: Optional[Callable[[AgentToolResult], None]] = None,
    ) -> AgentToolResult:
        """Execute the read tool."""
        # Inject workspace_dir into args
        full_args = {**args, "_workspace_dir": workspace_dir}
        return await execute_read(tool_call_id, full_args, signal, update_cb)

    # JSON Schema for parameters
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read (relative or absolute)"
            },
            "offset": {
                "type": "number",
                "description": "Line number to start reading from (1-indexed)"
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of lines to read"
            }
        },
        "required": ["path"]
    }

    return AgentTool(
        name="read",
        label="Read File",
        description=f"Read the contents of a file. Output is truncated to {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB. Use offset/limit for large files.",
        parameters=parameters,
        execute=execute,
    )