"""
Find tool for file pattern matching.

This tool allows the agent to find files matching glob patterns.
"""

from typing import Any, Optional, Callable, Awaitable, List
from dataclasses import dataclass
from pathlib import Path
import os

from pi_agent_core import AgentTool, AgentToolResult
from pi_ai import TextContent


# Default limits
DEFAULT_LIMIT = 1000  # Maximum results
DEFAULT_MAX_BYTES = 100 * 1024  # 100KB


@dataclass
class FindToolDetails:
    """Details about find tool execution."""
    pattern: str
    path: str
    result_count: int
    result_limit_reached: bool = False
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


def _find_files_by_glob(
    directory: Path,
    pattern: str,
    limit: int,
    ignore_dirs: List[str] = None
) -> List[Path]:
    """Find files matching glob pattern in directory."""
    ignore_dirs = ignore_dirs or ["node_modules", ".git", "__pycache__", ".venv", ".claude"]
    files = []
    count = 0

    try:
        for item in directory.rglob(pattern):
            # Skip ignored directories
            if any(ignored in item.parts for ignored in ignore_dirs):
                continue

            if item.is_file():
                files.append(item)
                count += 1
                if count >= limit:
                    break
    except Exception:
        pass

    return files


def _find_files_by_name(
    directory: Path,
    name_pattern: str,
    limit: int,
    ignore_dirs: List[str] = None
) -> List[Path]:
    """Find files where the filename matches the pattern."""
    ignore_dirs = ignore_dirs or ["node_modules", ".git", "__pycache__", ".venv", ".claude"]

    # Convert name pattern to glob if it doesn't have wildcards
    if '*' not in name_pattern and '?' not in name_pattern:
        name_pattern = f"*{name_pattern}*"

    files = []
    count = 0

    try:
        for item in directory.rglob("*"):
            # Skip ignored directories
            if any(ignored in item.parts for ignored in ignore_dirs):
                continue

            if item.is_file():
                # Check if filename matches
                if Path(item.name).match(name_pattern):
                    files.append(item)
                    count += 1
                    if count >= limit:
                        break
    except Exception:
        pass

    return files


async def execute_find(
    tool_call_id: str,
    args: dict,
    signal: Optional[Any] = None,
    update_cb: Optional[Callable[[AgentToolResult], None]] = None,
) -> AgentToolResult[FindToolDetails]:
    """Execute the find tool.

    Args:
        tool_call_id: Unique ID for this tool call
        args: Tool arguments with pattern and optional path
        signal: Optional abort signal
        update_cb: Optional callback for streaming updates

    Returns:
        AgentToolResult with matching file paths and details
    """
    workspace_dir = args.get("_workspace_dir", ".")

    # Extract arguments
    pattern = args.get("pattern")
    if not pattern:
        return AgentToolResult(
            content=[TextContent(text="Error: 'pattern' argument is required")],
            details=FindToolDetails(pattern="", path="", result_count=0)
        )

    search_path = args.get("path", ".")
    limit = args.get("limit", DEFAULT_LIMIT)

    # Validate and resolve path
    try:
        target = _resolve_path(search_path, workspace_dir)
    except ValueError as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error: {e}")],
            details=FindToolDetails(pattern=pattern, path=search_path, result_count=0)
        )

    # Check if path exists
    if not target.exists():
        return AgentToolResult(
            content=[TextContent(text=f"Error: Path not found: {search_path}")],
            details=FindToolDetails(pattern=pattern, path=search_path, result_count=0)
        )

    # Check if it's a directory
    if not target.is_dir():
        return AgentToolResult(
            content=[TextContent(text=f"Error: Not a directory: {search_path}")],
            details=FindToolDetails(pattern=pattern, path=search_path, result_count=0)
        )

    # Check for abort
    if signal and hasattr(signal, 'aborted') and signal.aborted:
        return AgentToolResult(
            content=[TextContent(text="Operation aborted")],
            details=FindToolDetails(pattern=pattern, path=search_path, result_count=0)
        )

    # Find files
    ignore_dirs = ["node_modules", ".git", "__pycache__", ".venv", ".claude", "build", "dist"]

    # Determine search method based on pattern
    if '/' in pattern or pattern.startswith('**'):
        # Path pattern - use rglob directly
        files = _find_files_by_glob(target, pattern, limit, ignore_dirs)
    else:
        # Name pattern - match against filename
        files = _find_files_by_name(target, pattern, limit, ignore_dirs)

    # Check for abort after search
    if signal and hasattr(signal, 'aborted') and signal.aborted:
        return AgentToolResult(
            content=[TextContent(text="Operation aborted")],
            details=FindToolDetails(pattern=pattern, path=search_path, result_count=0)
        )

    # Build output
    if not files:
        return AgentToolResult(
            content=[TextContent(text="No files found matching pattern")],
            details=FindToolDetails(pattern=pattern, path=search_path, result_count=0)
        )

    # Convert to relative paths
    relative_paths = []
    for file_path in files:
        try:
            rel_path = file_path.relative_to(target)
        except ValueError:
            rel_path = file_path
        relative_paths.append(str(rel_path).replace('\\', '/'))

    # Sort results
    relative_paths.sort()

    # Check limit
    result_limit_reached = len(relative_paths) >= limit

    # Build output
    output_text = '\n'.join(relative_paths)

    # Truncate by bytes
    truncated = False
    if len(output_text.encode('utf-8')) > DEFAULT_MAX_BYTES:
        truncated = True
        output_text = output_text[:DEFAULT_MAX_BYTES]
        output_text = output_text[:output_text.rfind('\n')] if '\n' in output_text else output_text

    # Add notices
    notices = []
    if result_limit_reached:
        notices.append(f"{limit} results limit reached")
    if truncated:
        notices.append(f"{DEFAULT_MAX_BYTES // 1024}KB limit reached")

    if notices:
        output_text += f"\n\n[Truncated: {', '.join(notices)}]"

    return AgentToolResult(
        content=[TextContent(text=output_text)],
        details=FindToolDetails(
            pattern=pattern,
            path=search_path,
            result_count=len(relative_paths),
            result_limit_reached=result_limit_reached,
            truncated=truncated
        )
    )


def create_find_tool(workspace_dir: str) -> AgentTool:
    """Create the find tool for file pattern matching.

    Args:
        workspace_dir: The workspace directory to search in.

    Returns:
        An AgentTool configured for file finding.
    """
    async def execute(
        tool_call_id: str,
        args: Any,
        signal: Optional[Any] = None,
        update_cb: Optional[Callable[[AgentToolResult], None]] = None,
    ) -> AgentToolResult:
        """Execute the find tool."""
        # Inject workspace_dir into args
        full_args = {**args, "_workspace_dir": workspace_dir}
        return await execute_find(tool_call_id, full_args, signal, update_cb)

    # JSON Schema for parameters
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match files (e.g., '*.py', '**/*.json', 'src/**/*.ts')"
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: current directory)"
            },
            "limit": {
                "type": "number",
                "description": f"Maximum number of results (default: {DEFAULT_LIMIT})"
            }
        },
        "required": ["pattern"]
    }

    return AgentTool(
        name="find",
        label="Find Files",
        description=f"Find files matching glob patterns. Returns matching file paths. Output is truncated to {DEFAULT_LIMIT} results or {DEFAULT_MAX_BYTES // 1024}KB.",
        parameters=parameters,
        execute=execute,
    )