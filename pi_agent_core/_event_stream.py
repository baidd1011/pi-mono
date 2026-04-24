"""
AgentEventStream implementation for pi-agent-core.

This module provides the AgentEventStream class - a specialization
of EventStream for agent lifecycle events.
"""

from typing import List, Any
from pi_ai.stream import EventStream
from pi_ai import Message

from .types import AgentEvent

# Alias for AgentMessage
AgentMessage = Message


class AgentEventStream(EventStream[AgentEvent, List[AgentMessage]]):
    """
    Event stream specialization for agent events.

    Provides typed streaming for agent lifecycle events and collects
    the final list of messages.

    Example:
        stream = AgentEventStream()
        stream.push({"type": "agent_start"})
        stream.push({"type": "turn_start"})
        stream.push({"type": "message_start", "message": user_message})
        stream.end([user_message, assistant_message])
        async for event in stream:
            print(event)
        messages = await stream.result()
    """

    def __init__(self) -> None:
        """Initialize the agent event stream."""
        super().__init__()
        self._messages: List[AgentMessage] = []

    def end(self, messages: List[AgentMessage]) -> None:
        """
        End the stream with the final message list.

        Args:
            messages: The final list of agent messages.
        """
        self._messages = messages
        super().end(messages)

    @property
    def messages(self) -> List[AgentMessage]:
        """Get the collected messages (available after end is called)."""
        return self._messages


def create_agent_stream() -> AgentEventStream:
    """Create a new agent event stream."""
    return AgentEventStream()