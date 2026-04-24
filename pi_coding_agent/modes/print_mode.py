"""
Print mode - single-shot output without TUI.

Provides a simple interface for running the agent and printing output,
suitable for CLI usage or scripting.
"""

import asyncio
from typing import Optional, Callable, List, Any

from ..core.agent_session import AgentSession
from ..core.defaults import DEFAULT_MODEL


async def run_print_mode(
    prompt: str,
    workspace_dir: str,
    model_id: Optional[str] = None,
    on_event: Optional[Callable] = None,
) -> str:
    """Run agent in print mode and return output.

    Args:
        prompt: The user's prompt to process.
        workspace_dir: Directory for file operations.
        model_id: Optional model ID to use (defaults to DEFAULT_MODEL).
        on_event: Optional callback for agent events.

    Returns:
        The assistant's text response.
    """
    session = AgentSession(workspace_dir)

    output_parts: List[str] = []

    def handle_event(event: Any, signal: Any) -> None:
        if on_event:
            on_event(event)

        if event.get("type") == "message_end":
            msg = event.get("message")
            if msg and msg.get("role") == "assistant":
                content = msg.get("content", [])
                for block in content if isinstance(content, list) else []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        output_parts.append(block.get("text", ""))

    session.subscribe(handle_event)

    await session.prompt(prompt)
    await session.wait_for_idle()

    return "".join(output_parts)


async def print_and_exit(
    prompt: str,
    workspace_dir: str,
    model_id: Optional[str] = None,
) -> None:
    """Run agent and print output to stdout.

    Args:
        prompt: The user's prompt to process.
        workspace_dir: Directory for file operations.
        model_id: Optional model ID to use.
    """
    output = await run_print_mode(prompt, workspace_dir, model_id)
    print(output)