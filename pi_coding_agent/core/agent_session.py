"""
AgentSession - the core abstraction for pi-coding-agent.

AgentSession wraps the Agent class with session persistence,
tool initialization, and mode-independent core functionality.
"""

import asyncio
from pathlib import Path
from typing import Callable, Optional, Awaitable, List, Any

from pi_agent_core import Agent, AgentOptions, AgentEvent, AbortSignal, AgentState
from pi_ai import Model, stream_simple, UserMessage

from .defaults import (
    DEFAULT_MODEL,
    DEFAULT_THINKING_LEVEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_SESSION_DIR_NAME,
    DEFAULT_SESSIONS_SUBDIR,
)
from .tools import create_coding_tools


class AgentSession:
    """Core session abstraction shared by all modes.

    AgentSession wraps pi-agent-core's Agent class with:
    - Session persistence (to be implemented later)
    - Tool initialization with workspace constraints
    - Default model and system prompt configuration
    - Mode-independent core API
    """

    def __init__(
        self,
        workspace_dir: str,
        model: Optional[Model] = None,
        session_dir: Optional[str] = None,
    ):
        """Initialize an AgentSession.

        Args:
            workspace_dir: The workspace directory for file operations.
            model: Optional model configuration (defaults to DEFAULT_MODEL).
            session_dir: Optional directory for session files.
        """
        self.workspace_dir = Path(workspace_dir)
        self._model = model or DEFAULT_MODEL

        # Create agent with stream_simple and parallel tool execution
        self._agent = Agent(AgentOptions(
            stream_fn=stream_simple,
            tool_execution="parallel",
        ))

        # Initialize tools for the workspace
        self._tools = create_coding_tools(str(self.workspace_dir))
        self._agent.set_tools(self._tools)

        # Set default model and system prompt
        self._agent.set_model(self._model)
        self._agent.set_system_prompt(DEFAULT_SYSTEM_PROMPT)

        # Session state
        if session_dir:
            self._session_dir = Path(session_dir)
        else:
            self._session_dir = self.workspace_dir / DEFAULT_SESSION_DIR_NAME / DEFAULT_SESSIONS_SUBDIR

        self._session_id: Optional[str] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def model(self) -> Model:
        """Get the current model configuration."""
        return self._model

    @property
    def tools(self) -> List[Any]:
        """Get the list of tools available to the agent."""
        return self._agent.state.tools

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self._session_id

    @property
    def state(self) -> AgentState:
        """Get read-only state snapshot from the underlying agent."""
        return self._agent.state

    # -------------------------------------------------------------------------
    # Agent Delegation Methods
    # -------------------------------------------------------------------------

    def set_model(self, model: Model) -> None:
        """Set the model configuration."""
        self._model = model
        self._agent.set_model(model)

    def set_thinking_level(self, level: str) -> None:
        """Set the thinking level for reasoning models."""
        self._agent.set_thinking_level(level)

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        self._agent.set_system_prompt(prompt)

    async def prompt(self, input: str, images: Optional[List[Any]] = None) -> None:
        """Process a user prompt.

        Args:
            input: The user's text input.
            images: Optional list of image content blocks.
        """
        # Build content array
        content: List[Any] = [{"type": "text", "text": input}]
        if images:
            content.extend(images)

        # Create user message with timestamp
        message = UserMessage(
            role="user",
            content=content,
            timestamp=int(asyncio.get_running_loop().time() * 1000),
        )

        # Call agent.prompt
        await self._agent.prompt(message)

    async def continue_(self) -> None:
        """Continue the agent's current turn."""
        await self._agent.continue_()

    def subscribe(self, listener: Callable[[AgentEvent, AbortSignal], Awaitable[None] | None]) -> Callable[[], None]:
        """Subscribe to agent events. Returns unsubscribe function.

        Args:
            listener: Callback function for agent events.

        Returns:
            Unsubscribe function to remove the listener.
        """
        return self._agent.subscribe(listener)

    def abort(self) -> None:
        """Abort the current agent run."""
        self._agent.abort()

    async def wait_for_idle(self) -> None:
        """Wait for the agent to finish its current run."""
        await self._agent.wait_for_idle()

    # -------------------------------------------------------------------------
    # Message Queue Methods
    # -------------------------------------------------------------------------

    def steer(self, message: Any) -> None:
        """Queue a message to inject after current assistant turn."""
        self._agent.steer(message)

    def follow_up(self, message: Any) -> None:
        """Queue a message for after agent would otherwise stop."""
        self._agent.follow_up(message)

    def clear_steering_queue(self) -> None:
        """Clear the steering queue."""
        self._agent.clear_steering_queue()

    def clear_follow_up_queue(self) -> None:
        """Clear the follow-up queue."""
        self._agent.clear_follow_up_queue()

    def clear_all_queues(self) -> None:
        """Clear both steering and follow-up queues."""
        self._agent.clear_all_queues()

    def has_queued_messages(self) -> bool:
        """Check if there are any queued messages."""
        return self._agent.has_queued_messages()

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the agent state (clear messages and queues)."""
        self._agent.reset()