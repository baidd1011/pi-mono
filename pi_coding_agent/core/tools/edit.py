"""
Edit tool for file editing operations.

This tool allows the agent to make precise edits to existing files.
"""

from typing import Any, Optional, Callable, Awaitable, List
from dataclasses import dataclass
from pathlib import Path
import os
import re

from pi_agent_core import AgentTool, AgentToolResult
from pi_ai import TextContent


@dataclass
class EditToolDetails:
    """Details about edit tool execution."""
    path: str
    edits_count: int
    diff: str


@dataclass
class Edit:
    """A single edit operation."""
    old_text: str
    new_text: str


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


def _strip_bom(content: str) -> tuple[str, str]:
    """Strip BOM from content if present. Returns (bom, content_without_bom)."""
    bom = ""
    if content.startswith('﻿'):
        bom = '﻿'
        content = content[1:]
    return bom, content


def _detect_line_ending(content: str) -> str:
    """Detect the dominant line ending in content."""
    crlf_count = content.count('\r\n')
    lf_count = content.count('\n') - crlf_count

    if crlf_count > lf_count:
        return '\r\n'
    return '\n'


def _normalize_to_lf(content: str) -> str:
    """Normalize content to LF line endings."""
    return content.replace('\r\n', '\n').replace('\r', '\n')


def _restore_line_endings(content: str, line_ending: str) -> str:
    """Restore line endings to the detected format."""
    if line_ending == '\r\n':
        return content.replace('\n', '\r\n')
    return content


def _generate_diff(old_content: str, new_content: str) -> str:
    """Generate a simple unified diff between old and new content."""
    old_lines = old_content.split('\n')
    new_lines = new_content.split('\n')

    diff_lines = []

    # Simple approach: find first and last changed lines
    first_change = None
    last_change = None

    # Find changes
    max_len = max(len(old_lines), len(new_lines))
    changes = []
    for i in range(max_len):
        old_line = old_lines[i] if i < len(old_lines) else None
        new_line = new_lines[i] if i < len(new_lines) else None
        if old_line != new_line:
            changes.append(i)

    if not changes:
        return ""

    first_change = changes[0]
    last_change = changes[-1]

    # Add context (3 lines before and after)
    context_start = max(0, first_change - 3)
    context_end = min(max_len, last_change + 4)

    for i in range(context_start, context_end):
        old_line = old_lines[i] if i < len(old_lines) else None
        new_line = new_lines[i] if i < len(new_lines) else None

        if old_line != new_line:
            if old_line is not None:
                diff_lines.append(f"- {old_line}")
            if new_line is not None:
                diff_lines.append(f"+ {new_line}")
        else:
            if old_line is not None:
                diff_lines.append(f"  {old_line}")

    return "\n".join(diff_lines)


def _apply_edits(content: str, edits: List[Edit], path: str) -> tuple[str, str, str]:
    """Apply edits to content and return (new_content, base_content, diff).

    Raises ValueError if an edit cannot be applied.
    """
    bom, content = _strip_bom(content)
    original_ending = _detect_line_ending(content)
    normalized_content = _normalize_to_lf(content)

    base_content = normalized_content

    # Apply each edit
    for edit in edits:
        old_text = _normalize_to_lf(edit.old_text)
        new_text = _normalize_to_lf(edit.new_text)

        # Find the old text - must be unique
        occurrences = []
        start = 0
        while True:
            idx = normalized_content.find(old_text, start)
            if idx == -1:
                break
            occurrences.append(idx)
            start = idx + 1

        if len(occurrences) == 0:
            raise ValueError(f"Could not find text to replace in {path}. Make sure old_text matches exactly.")

        if len(occurrences) > 1:
            raise ValueError(f"Found {len(occurrences)} occurrences of old_text in {path}. Make old_text unique or use replace_all.")

        # Apply the edit
        normalized_content = normalized_content.replace(old_text, new_text, 1)

    # Restore line endings
    new_content = bom + _restore_line_endings(normalized_content, original_ending)

    # Generate diff
    diff = _generate_diff(base_content, normalized_content)

    return new_content, base_content, diff


async def execute_edit(
    tool_call_id: str,
    args: dict,
    signal: Optional[Any] = None,
    update_cb: Optional[Callable[[AgentToolResult], None]] = None,
) -> AgentToolResult[EditToolDetails]:
    """Execute the edit tool.

    Args:
        tool_call_id: Unique ID for this tool call
        args: Tool arguments with path/file_path and edits or oldText/newText
        signal: Optional abort signal
        update_cb: Optional callback for streaming updates

    Returns:
        AgentToolResult with confirmation and diff
    """
    workspace_dir = args.get("_workspace_dir", ".")

    # Extract path (accept both 'path' and 'file_path')
    path = args.get("path") or args.get("file_path")
    if not path:
        return AgentToolResult(
            content=[TextContent(text="Error: 'path' argument is required")],
            details=EditToolDetails(path="", edits_count=0, diff="")
        )

    # Extract edits (support both new and legacy formats)
    edits = []
    replace_all = args.get("replace_all", False)

    # New format: edits array
    if "edits" in args:
        raw_edits = args["edits"]
        if isinstance(raw_edits, str):
            # Some models send edits as JSON string
            try:
                import json
                raw_edits = json.loads(raw_edits)
            except:
                pass

        if isinstance(raw_edits, list):
            for edit in raw_edits:
                if isinstance(edit, dict):
                    old_text = edit.get("oldText") or edit.get("old_text")
                    new_text = edit.get("newText") or edit.get("new_text")
                    if old_text and new_text:
                        edits.append(Edit(old_text=old_text, new_text=new_text))

    # Legacy format: old_string/new_string
    old_string = args.get("old_string") or args.get("oldText")
    new_string = args.get("new_string") or args.get("newText")

    if old_string and new_string:
        edits.append(Edit(old_text=old_string, new_text=new_string))

    if not edits:
        return AgentToolResult(
            content=[TextContent(text="Error: No edits provided. Use 'edits' array or 'old_string'/'new_string' pair")],
            details=EditToolDetails(path=path, edits_count=0, diff="")
        )

    # Validate and resolve path
    try:
        target = _resolve_path(path, workspace_dir)
    except ValueError as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error: {e}")],
            details=EditToolDetails(path=path, edits_count=0, diff="")
        )

    # Check if file exists
    if not target.exists():
        return AgentToolResult(
            content=[TextContent(text=f"Error: File not found: {path}")],
            details=EditToolDetails(path=path, edits_count=0, diff="")
        )

    # Check if it's a file
    if not target.is_file():
        return AgentToolResult(
            content=[TextContent(text=f"Error: Not a file: {path}")],
            details=EditToolDetails(path=path, edits_count=0, diff="")
        )

    # Check for abort before reading
    if signal and hasattr(signal, 'aborted') and signal.aborted:
        return AgentToolResult(
            content=[TextContent(text="Operation aborted")],
            details=EditToolDetails(path=path, edits_count=0, diff="")
        )

    # Read file content
    try:
        with open(target, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error reading file: {e}")],
            details=EditToolDetails(path=path, edits_count=0, diff="")
        )

    # Check for abort before editing
    if signal and hasattr(signal, 'aborted') and signal.aborted:
        return AgentToolResult(
            content=[TextContent(text="Operation aborted")],
            details=EditToolDetails(path=path, edits_count=0, diff="")
        )

    # Apply edits
    try:
        # Handle replace_all for single edit
        if replace_all and len(edits) == 1:
            edit = edits[0]
            bom, clean_content = _strip_bom(content)
            original_ending = _detect_line_ending(clean_content)
            normalized_content = _normalize_to_lf(clean_content)
            base_content = normalized_content

            old_text = _normalize_to_lf(edit.old_text)
            new_text = _normalize_to_lf(edit.new_text)

            count = normalized_content.count(old_text)
            normalized_content = normalized_content.replace(old_text, new_text)

            new_content = bom + _restore_line_endings(normalized_content, original_ending)
            diff = _generate_diff(base_content, normalized_content)
            edits_count = count
        else:
            new_content, base_content, diff = _apply_edits(content, edits, path)
            edits_count = len(edits)
    except ValueError as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error: {e}")],
            details=EditToolDetails(path=path, edits_count=0, diff="")
        )

    # Check for abort before writing
    if signal and hasattr(signal, 'aborted') and signal.aborted:
        return AgentToolResult(
            content=[TextContent(text="Operation aborted")],
            details=EditToolDetails(path=path, edits_count=0, diff="")
        )

    # Write the modified file
    try:
        with open(target, 'w', encoding='utf-8') as f:
            f.write(new_content)
    except Exception as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error writing file: {e}")],
            details=EditToolDetails(path=path, edits_count=0, diff="")
        )

    # Build response
    response_text = f"Successfully replaced {edits_count} block(s) in {path}"
    if diff:
        response_text += f"\n\nDiff:\n{diff}"

    return AgentToolResult(
        content=[TextContent(text=response_text)],
        details=EditToolDetails(path=path, edits_count=edits_count, diff=diff)
    )


def create_edit_tool(workspace_dir: str) -> AgentTool:
    """Create the edit tool for file editing operations.

    Args:
        workspace_dir: The workspace directory to constrain file access.

    Returns:
        An AgentTool configured for file editing.
    """
    async def execute(
        tool_call_id: str,
        args: Any,
        signal: Optional[Any] = None,
        update_cb: Optional[Callable[[AgentToolResult], None]] = None,
    ) -> AgentToolResult:
        """Execute the edit tool."""
        # Inject workspace_dir into args
        full_args = {**args, "_workspace_dir": workspace_dir}
        return await execute_edit(tool_call_id, full_args, signal, update_cb)

    # JSON Schema for parameters
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit (relative or absolute)"
            },
            "file_path": {
                "type": "string",
                "description": "Alternative name for path argument"
            },
            "edits": {
                "type": "array",
                "description": "Array of edit operations",
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": {
                            "type": "string",
                            "description": "Exact text to find (must be unique)"
                        },
                        "newText": {
                            "type": "string",
                            "description": "Replacement text"
                        }
                    },
                    "required": ["oldText", "newText"]
                }
            },
            "old_string": {
                "type": "string",
                "description": "Text to replace (legacy format)"
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text (legacy format)"
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences (default false)"
            }
        },
        "required": ["path"]
    }

    return AgentTool(
        name="edit",
        label="Edit File",
        description="Make precise edits to a file using exact string replacement. old_text must be unique in the file. Use replace_all for multiple occurrences.",
        parameters=parameters,
        execute=execute,
    )