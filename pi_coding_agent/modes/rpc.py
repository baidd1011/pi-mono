"""
RPC mode - JSON-RPC over stdin/stdout.

Provides a programmatic interface for controlling the agent
via JSON-RPC protocol, suitable for IDE integration or
other process-based communication.
"""

import asyncio
import json
import sys
from typing import Optional, Any, Dict

from ..core.agent_session import AgentSession


class RPCHandler:
    """Handle JSON-RPC protocol for agent communication.

    Methods supported:
    - prompt: Send a prompt to the agent
    - continue: Continue agent's current turn
    - abort: Abort the current agent run
    - getStatus: Get current agent status
    """

    def __init__(self, session: AgentSession):
        """Initialize RPC handler.

        Args:
            session: The AgentSession to control.
        """
        self.session = session
        self._pending_requests: Dict[str, Any] = {}

    async def handle_request(self, request: dict) -> dict:
        """Process a JSON-RPC request.

        Args:
            request: The JSON-RPC request object.

        Returns:
            A JSON-RPC response object.
        """
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        try:
            if method == "prompt":
                result = await self._handle_prompt(params)
                return {"jsonrpc": "2.0", "result": result, "id": request_id}
            elif method == "continue":
                result = await self._handle_continue(params)
                return {"jsonrpc": "2.0", "result": result, "id": request_id}
            elif method == "abort":
                self.session.abort()
                return {"jsonrpc": "2.0", "result": {"aborted": True}, "id": request_id}
            elif method == "getStatus":
                return {"jsonrpc": "2.0", "result": self._get_status(), "id": request_id}
            else:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                    "id": request_id,
                }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": str(e)},
                "id": request_id,
            }

    async def _handle_prompt(self, params: dict) -> dict:
        """Handle prompt method.

        Args:
            params: Method parameters containing 'prompt' key.

        Returns:
            Result dict with status.
        """
        prompt = params.get("prompt", "")
        await self.session.prompt(prompt)
        return {"status": "processing"}

    async def _handle_continue(self, params: dict) -> dict:
        """Handle continue method.

        Args:
            params: Method parameters (unused).

        Returns:
            Result dict with status.
        """
        await self.session.continue_()
        return {"status": "processing"}

    def _get_status(self) -> dict:
        """Get current agent status.

        Returns:
            Status dict with streaming state, message count, and queue status.
        """
        state = self.session.state
        return {
            "isStreaming": state.is_streaming,
            "messageCount": len(state.messages),
            "hasQueuedMessages": self.session.has_queued_messages(),
        }


async def run_rpc_mode(workspace_dir: str) -> None:
    """Run agent in RPC mode.

    Reads JSON-RPC requests from stdin and writes responses to stdout.
    Also emits event notifications for agent events.

    Args:
        workspace_dir: Directory for file operations.
    """
    session = AgentSession(workspace_dir)
    handler = RPCHandler(session)

    # Subscribe to events and emit notifications
    def emit_notification(event: Any, signal: Any) -> None:
        notification = {
            "jsonrpc": "2.0",
            "method": "event",
            "params": event,
        }
        print(json.dumps(notification), flush=True)

    session.subscribe(emit_notification)

    # Read requests from stdin
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break

        try:
            request = json.loads(line.decode())
            response = await handler.handle_request(request)
            print(json.dumps(response), flush=True)
        except json.JSONDecodeError:
            print(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "error": {"code": -32700, "message": "Parse error"},
                        "id": None,
                    }
                ),
                flush=True,
            )