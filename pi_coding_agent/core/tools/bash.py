"""
Bash tool for command execution.

This tool allows the agent to run shell commands in the workspace.
"""

from typing import Any, Optional, Callable, Awaitable, List
from dataclasses import dataclass
from pathlib import Path
import asyncio
import subprocess
import sys

from pi_agent_core import AgentTool, AgentToolResult
from pi_ai import TextContent


# Default limits matching TypeScript implementation
DEFAULT_TIMEOUT = 120  # 2 minutes
DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 100 * 1024  # 100KB


@dataclass
class BashToolDetails:
    """Details about bash tool execution."""
    command: str
    exit_code: Optional[int] = None
    timeout: Optional[int] = None
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    truncated: bool = False
    output_lines: int = 0


async def execute_bash(
    tool_call_id: str,
    args: dict,
    signal: Optional[Any] = None,
    update_cb: Optional[Callable[[AgentToolResult], None]] = None,
) -> AgentToolResult[BashToolDetails]:
    """Execute the bash tool.

    Args:
        tool_call_id: Unique ID for this tool call
        args: Tool arguments with command and optional timeout
        signal: Optional abort signal
        update_cb: Optional callback for streaming updates

    Returns:
        AgentToolResult with command output and details
    """
    workspace_dir = args.get("_workspace_dir", ".")

    # Extract arguments
    command = args.get("command")
    if not command:
        return AgentToolResult(
            content=[TextContent(text="Error: 'command' argument is required")],
            details=BashToolDetails(command="", exit_code=None)
        )

    timeout = args.get("timeout", DEFAULT_TIMEOUT)
    run_in_background = args.get("run_in_background", False)

    # Validate workspace exists
    workspace = Path(workspace_dir).resolve()
    if not workspace.exists():
        return AgentToolResult(
            content=[TextContent(text=f"Error: Working directory does not exist: {workspace_dir}")],
            details=BashToolDetails(command=command, exit_code=None)
        )

    # Determine shell based on platform
    if sys.platform == "win32":
        # On Windows, use cmd.exe or PowerShell
        shell_cmd = ["cmd.exe", "/c", command]
    else:
        # On Unix, use bash
        shell_cmd = ["bash", "-c", command]

    # Initialize output collection
    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []
    total_stdout_bytes = 0
    total_stderr_bytes = 0

    # Check for initial abort
    if signal and hasattr(signal, 'aborted') and signal.aborted:
        return AgentToolResult(
            content=[TextContent(text="Operation aborted")],
            details=BashToolDetails(command=command, exit_code=None)
        )

    try:
        # Start the process
        process = await asyncio.create_subprocess_exec(
            *shell_cmd,
            cwd=workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Set up abort handling
        aborted = False

        async def read_stream(stream, chunks, is_stderr=False):
            """Read from stream and collect output."""
            while True:
                try:
                    line = await stream.readline()
                    if not line:
                        break

                    text = line.decode('utf-8', errors='replace')

                    chunks.append(text)
                    if is_stderr:
                        nonlocal total_stderr_bytes
                        total_stderr_bytes += len(text)
                    else:
                        nonlocal total_stdout_bytes
                        total_stdout_bytes += len(text)

                    # Stream updates if callback provided
                    if update_cb and not is_stderr:
                        # Send partial update
                        partial_output = ''.join(chunks[-50:])  # Last 50 chunks
                        truncated = len(partial_output.encode('utf-8')) > DEFAULT_MAX_BYTES
                        if truncated:
                            partial_output = partial_output[-DEFAULT_MAX_BYTES:]

                        update_cb(AgentToolResult(
                            content=[TextContent(text=partial_output)],
                            details=BashToolDetails(command=command, truncated=truncated)
                        ))

                except Exception:
                    break

        # Create tasks for reading both streams
        stdout_task = asyncio.create_task(read_stream(process.stdout, stdout_chunks, False))
        stderr_task = asyncio.create_task(read_stream(process.stderr, stderr_chunks, True))

        # Wait with timeout
        try:
            # Set up timeout
            async def wait_for_process():
                await asyncio.gather(stdout_task, stderr_task)
                return await process.wait()

            # Run with timeout and abort checking
            if timeout > 0:
                exit_code = await asyncio.wait_for(wait_for_process(), timeout=timeout)
            else:
                exit_code = await wait_for_process()

        except asyncio.TimeoutError:
            # Kill the process on timeout
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass

            stdout = ''.join(stdout_chunks)
            stderr = ''.join(stderr_chunks)

            output = stdout + stderr
            output += f"\n\nCommand timed out after {timeout} seconds"

            return AgentToolResult(
                content=[TextContent(text=output)],
                details=BashToolDetails(
                    command=command,
                    exit_code=None,
                    timeout=timeout,
                    stdout_bytes=total_stdout_bytes,
                    stderr_bytes=total_stderr_bytes
                )
            )

        # Check for abort during execution
        if signal and hasattr(signal, 'aborted') and signal.aborted:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass

            stdout = ''.join(stdout_chunks)
            stderr = ''.join(stderr_chunks)

            output = stdout + stderr
            output += "\n\nCommand aborted"

            return AgentToolResult(
                content=[TextContent(text=output)],
                details=BashToolDetails(
                    command=command,
                    exit_code=None,
                    stdout_bytes=total_stdout_bytes,
                    stderr_bytes=total_stderr_bytes
                )
            )

        # Combine output
        stdout = ''.join(stdout_chunks)
        stderr = ''.join(stderr_chunks)
        full_output = stdout + stderr

        # Truncate if needed
        truncated = False
        output_lines = 0
        output_text = full_output

        if len(full_output.encode('utf-8')) > DEFAULT_MAX_BYTES:
            truncated = True
            # Truncate from end
            output_text = full_output[-DEFAULT_MAX_BYTES:]
            output_text = output_text[output_text.find('\n')+1:] if '\n' in output_text else output_text

        output_lines = len(output_text.split('\n'))

        # Add truncation notice
        if truncated:
            output_text += f"\n\n[Output truncated to {DEFAULT_MAX_BYTES // 1024}KB limit. Showing last {output_lines} lines.]"

        # Add exit code info if non-zero
        if exit_code != 0 and exit_code is not None:
            output_text += f"\n\nCommand exited with code {exit_code}"

        # Return result
        return AgentToolResult(
            content=[TextContent(text=output_text if output_text else "(no output)")],
            details=BashToolDetails(
                command=command,
                exit_code=exit_code,
                timeout=timeout if timeout != DEFAULT_TIMEOUT else None,
                stdout_bytes=total_stdout_bytes,
                stderr_bytes=total_stderr_bytes,
                truncated=truncated,
                output_lines=output_lines
            )
        )

    except Exception as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error executing command: {e}")],
            details=BashToolDetails(command=command, exit_code=None)
        )


def create_bash_tool(workspace_dir: str) -> AgentTool:
    """Create the bash tool for command execution.

    Args:
        workspace_dir: The workspace directory to run commands in.

    Returns:
        An AgentTool configured for bash execution.
    """
    async def execute(
        tool_call_id: str,
        args: Any,
        signal: Optional[Any] = None,
        update_cb: Optional[Callable[[AgentToolResult], None]] = None,
    ) -> AgentToolResult:
        """Execute the bash tool."""
        # Inject workspace_dir into args
        full_args = {**args, "_workspace_dir": workspace_dir}
        return await execute_bash(tool_call_id, full_args, signal, update_cb)

    # JSON Schema for parameters
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute"
            },
            "timeout": {
                "type": "number",
                "description": f"Timeout in seconds (default {DEFAULT_TIMEOUT})"
            },
            "run_in_background": {
                "type": "boolean",
                "description": "Run command in background (not supported in Python implementation)"
            }
        },
        "required": ["command"]
    }

    return AgentTool(
        name="bash",
        label="Run Command",
        description=f"Execute a shell command in the workspace. Output is truncated to {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB. Use timeout argument for longer commands.",
        parameters=parameters,
        execute=execute,
    )