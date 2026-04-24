"""
SDK mode - embedding API for Python applications.

Provides a simple interface for embedding the coding agent
in other Python applications.
"""

from typing import Optional, Callable, List, Union, Any

from ..core.agent_session import AgentSession
from ..core.defaults import DEFAULT_MODEL


def create_sdk_session(
    workspace_dir: str,
    model_id: Optional[str] = None,
    on_event: Optional[Callable] = None,
) -> AgentSession:
    """Create SDK session for embedding.

    Args:
        workspace_dir: Directory for file operations.
        model_id: Optional model ID to use (defaults to DEFAULT_MODEL).
        on_event: Optional callback for agent events.

    Returns:
        Configured AgentSession ready for use.
    """
    session = AgentSession(workspace_dir)

    if on_event:
        session.subscribe(lambda e, s: on_event(e))

    return session


async def run_sdk_prompt(
    session: AgentSession,
    prompt: str,
    wait: bool = True,
) -> Union[str, None]:
    """Run prompt in SDK session.

    Args:
        session: The AgentSession to use.
        prompt: The user's prompt to process.
        wait: If True, wait for agent to finish and return output.
              If False, return immediately.

    Returns:
        Assistant's text response if wait=True, otherwise None.
    """
    output_parts: List[str] = []

    def collect_output(event: Any, signal: Any) -> None:
        if event.get("type") == "message_end":
            msg = event.get("message")
            if msg and msg.get("role") == "assistant":
                content = msg.get("content", [])
                for block in content if isinstance(content, list) else []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        output_parts.append(block.get("text", ""))

    unsub = session.subscribe(collect_output)

    await session.prompt(prompt)

    if wait:
        await session.wait_for_idle()

    unsub()
    return "".join(output_parts) if wait else None