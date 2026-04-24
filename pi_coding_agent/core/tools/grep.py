"""
Grep tool for content search.

This tool allows the agent to search for patterns in file contents.
"""

from typing import Any, Optional, Callable, Awaitable, List
from dataclasses import dataclass
from pathlib import Path
import re
import os

from pi_agent_core import AgentTool, AgentToolResult
from pi_ai import TextContent


# Default limits
DEFAULT_LIMIT = 100  # Maximum matches
DEFAULT_MAX_BYTES = 100 * 1024  # 100KB
GREP_MAX_LINE_LENGTH = 500  # Truncate long lines


@dataclass
class GrepToolDetails:
    """Details about grep tool execution."""
    pattern: str
    path: str
    match_count: int
    match_limit_reached: bool = False
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


def _truncate_line(line: str, max_length: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]:
    """Truncate a long line if needed."""
    if len(line) <= max_length:
        return line, False
    return line[:max_length-3] + "...", True


def _find_files(directory: Path, pattern: str = "*", ignore_dirs: List[str] = None) -> List[Path]:
    """Find files matching pattern in directory."""
    ignore_dirs = ignore_dirs or ["node_modules", ".git", "__pycache__", ".venv"]
    files = []

    try:
        for item in directory.rglob(pattern):
            # Skip ignored directories
            if any(ignored in item.parts for ignored in ignore_dirs):
                continue
            if item.is_file():
                files.append(item)
    except Exception:
        pass

    return files


async def execute_grep(
    tool_call_id: str,
    args: dict,
    signal: Optional[Any] = None,
    update_cb: Optional[Callable[[AgentToolResult], None]] = None,
) -> AgentToolResult[GrepToolDetails]:
    """Execute the grep tool.

    Args:
        tool_call_id: Unique ID for this tool call
        args: Tool arguments with pattern, path, glob, etc.
        signal: Optional abort signal
        update_cb: Optional callback for streaming updates

    Returns:
        AgentToolResult with matching lines and details
    """
    workspace_dir = args.get("_workspace_dir", ".")

    # Extract arguments
    pattern = args.get("pattern")
    if not pattern:
        return AgentToolResult(
            content=[TextContent(text="Error: 'pattern' argument is required")],
            details=GrepToolDetails(pattern="", path="", match_count=0)
        )

    search_path = args.get("path", ".")
    glob_pattern = args.get("glob")
    case_insensitive = args.get("-i", args.get("ignore_case", False))
    literal = args.get("literal", False)
    context = args.get("context", 0)
    limit = args.get("limit", DEFAULT_LIMIT)
    output_mode = args.get("output_mode", "content")

    # Validate and resolve path
    try:
        target = _resolve_path(search_path, workspace_dir)
    except ValueError as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error: {e}")],
            details=GrepToolDetails(pattern=pattern, path=search_path, match_count=0)
        )

    # Check if path exists
    if not target.exists():
        return AgentToolResult(
            content=[TextContent(text=f"Error: Path not found: {search_path}")],
            details=GrepToolDetails(pattern=pattern, path=search_path, match_count=0)
        )

    # Check for abort
    if signal and hasattr(signal, 'aborted') and signal.aborted:
        return AgentToolResult(
            content=[TextContent(text="Operation aborted")],
            details=GrepToolDetails(pattern=pattern, path=search_path, match_count=0)
        )

    # Build regex pattern
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        if literal:
            regex = re.compile(re.escape(pattern), flags)
        else:
            regex = re.compile(pattern, flags)
    except re.error as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error: Invalid regex pattern: {e}")],
            details=GrepToolDetails(pattern=pattern, path=search_path, match_count=0)
        )

    # Get files to search
    is_directory = target.is_dir()
    files_to_search: List[Path] = []

    if is_directory:
        if glob_pattern:
            files_to_search = _find_files(target, glob_pattern)
        else:
            # Search all text files
            files_to_search = _find_files(target, "*")
    else:
        files_to_search = [target]

    # Check for abort
    if signal and hasattr(signal, 'aborted') and signal.aborted:
        return AgentToolResult(
            content=[TextContent(text="Operation aborted")],
            details=GrepToolDetails(pattern=pattern, path=search_path, match_count=0)
        )

    # Search files
    matches: List[str] = []
    match_count = 0
    match_limit_reached = False
    lines_truncated = False
    ignore_dirs = ["node_modules", ".git", "__pycache__", ".venv"]

    for file_path in files_to_search:
        # Skip binary files and ignored directories
        if any(ignored in file_path.parts for ignored in ignore_dirs):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except (UnicodeDecodeError, IOError):
            continue  # Skip binary or unreadable files

        # Check for abort during search
        if signal and hasattr(signal, 'aborted') and signal.aborted:
            return AgentToolResult(
                content=[TextContent(text="Operation aborted")],
                details=GrepToolDetails(pattern=pattern, path=search_path, match_count=match_count)
            )

        # Get relative path for display
        if is_directory:
            try:
                rel_path = file_path.relative_to(target)
            except ValueError:
                rel_path = file_path
            display_path = str(rel_path).replace('\\', '/')
        else:
            display_path = file_path.name

        # Search each line
        for line_num, line in enumerate(lines, 1):
            line_text = line.rstrip('\n\r')

            if regex.search(line_text):
                match_count += 1

                if match_count > limit:
                    match_limit_reached = True
                    break

                # Truncate long lines
                truncated_line, was_truncated = _truncate_line(line_text)
                if was_truncated:
                    lines_truncated = True

                if output_mode == "files_with_matches":
                    # Only show file path
                    if display_path not in matches:
                        matches.append(display_path)
                else:
                    # Show full match with line number
                    if context > 0:
                        # Add context lines
                        start = max(1, line_num - context)
                        end = min(len(lines), line_num + context)

                        for ctx_num in range(start, end + 1):
                            ctx_line = lines[ctx_num - 1].rstrip('\n\r')
                            ctx_truncated, ctx_was_truncated = _truncate_line(ctx_line)
                            if ctx_was_truncated:
                                lines_truncated = True

                            if ctx_num == line_num:
                                matches.append(f"{display_path}:{ctx_num}: {ctx_truncated}")
                            else:
                                matches.append(f"{display_path}-{ctx_num}- {ctx_truncated}")
                    else:
                        matches.append(f"{display_path}:{line_num}: {truncated_line}")

        if match_limit_reached:
            break

    # Build output
    if not matches:
        return AgentToolResult(
            content=[TextContent(text="No matches found")],
            details=GrepToolDetails(pattern=pattern, path=search_path, match_count=0)
        )

    # Dedupe for files_with_matches mode
    if output_mode == "files_with_matches":
        matches = list(set(matches))
        matches.sort()

    output_text = '\n'.join(matches)

    # Truncate by bytes
    truncated = False
    if len(output_text.encode('utf-8')) > DEFAULT_MAX_BYTES:
        truncated = True
        output_text = output_text[:DEFAULT_MAX_BYTES]
        output_text = output_text[:output_text.rfind('\n')] if '\n' in output_text else output_text

    # Add notices
    notices = []
    if match_limit_reached:
        notices.append(f"{limit} matches limit reached")
    if truncated:
        notices.append(f"{DEFAULT_MAX_BYTES // 1024}KB limit reached")
    if lines_truncated:
        notices.append(f"some lines truncated to {GREP_MAX_LINE_LENGTH} chars")

    if notices:
        output_text += f"\n\n[Truncated: {', '.join(notices)}]"

    return AgentToolResult(
        content=[TextContent(text=output_text)],
        details=GrepToolDetails(
            pattern=pattern,
            path=search_path,
            match_count=match_count,
            match_limit_reached=match_limit_reached,
            truncated=truncated
        )
    )


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
        """Execute the grep tool."""
        # Inject workspace_dir into args
        full_args = {**args, "_workspace_dir": workspace_dir}
        return await execute_grep(tool_call_id, full_args, signal, update_cb)

    # JSON Schema for parameters
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for"
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search (default: current directory)"
            },
            "glob": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g., '*.py')"
            },
            "-i": {
                "type": "boolean",
                "description": "Case-insensitive search (default: false)"
            },
            "ignore_case": {
                "type": "boolean",
                "description": "Alternative name for -i"
            },
            "literal": {
                "type": "boolean",
                "description": "Treat pattern as literal string (default: false)"
            },
            "context": {
                "type": "number",
                "description": "Lines of context to show around matches"
            },
            "limit": {
                "type": "number",
                "description": f"Maximum number of matches (default: {DEFAULT_LIMIT})"
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches"],
                "description": "Output format (default: content)"
            }
        },
        "required": ["pattern"]
    }

    return AgentTool(
        name="grep",
        label="Search Content",
        description=f"Search file contents for a pattern using regex. Returns matching lines with file paths and line numbers. Output is truncated to {DEFAULT_LIMIT} matches or {DEFAULT_MAX_BYTES // 1024}KB.",
        parameters=parameters,
        execute=execute,
    )