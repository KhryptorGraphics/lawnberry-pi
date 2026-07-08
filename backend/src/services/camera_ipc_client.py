"""Thin IPC client for the dedicated lawnberry-camera.service.

Per the constitution, camera-stream.service exclusively owns the camera;
everything else (the main backend process included) consumes frames via its
Unix socket IPC protocol rather than opening the camera itself. This module
is that consumer for the HTTP-facing camera routes.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

SOCKET_PATH = "/apps/lawnberry-pi/data/lawnberry-camera.sock"
_TIMEOUT = 5.0
# A base64-encoded JPEG frame line comfortably exceeds asyncio's 64KB
# default readline() limit; frame lines share this connection (see below).
_LIMIT = 8 * 1024 * 1024


async def send_command(command: str, **extra: Any) -> dict[str, Any]:
    """Open a connection, send one command, read one response, close."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(SOCKET_PATH, limit=_LIMIT), timeout=_TIMEOUT
        )
    except (OSError, TimeoutError) as exc:
        return {"status": "error", "error": f"Camera service unavailable: {exc}"}

    try:
        writer.write(json.dumps({"command": command, **extra}).encode() + b"\n")
        await writer.drain()
        # Connecting adds us to the broadcast set immediately, so a
        # streamed {"type":"frame",...} push can race ahead of the actual
        # command response on the same connection -- skip past any.
        deadline = asyncio.get_running_loop().time() + _TIMEOUT
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return {"status": "error", "error": "Camera IPC timed out"}
            line = await asyncio.wait_for(reader.readline(), timeout=remaining)
            if not line:
                return {"status": "error", "error": "Camera service closed the connection"}
            message = json.loads(line.decode())
            if message.get("type") == "frame":
                continue
            return message
    except (OSError, TimeoutError, ValueError) as exc:
        return {"status": "error", "error": f"Camera IPC error: {exc}"}
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass


async def frame_stream() -> AsyncIterator[dict[str, Any]]:
    """Connect once and yield each frame the service pushes, until disconnected.

    Connecting is itself the "subscribe" action: the service adds any
    connected client to its broadcast set and pushes {"type":"frame",...}
    messages without the client needing to send a command.
    """
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(SOCKET_PATH, limit=_LIMIT), timeout=_TIMEOUT
    )
    try:
        while True:
            try:
                line = await reader.readline()
            except ValueError:
                continue
            if not line:
                break
            try:
                message = json.loads(line.decode())
            except json.JSONDecodeError:
                continue
            if message.get("type") == "frame":
                yield message.get("data", {})
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
