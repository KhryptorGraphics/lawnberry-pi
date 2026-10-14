#!/usr/bin/env python3
"""Control the camera service's on-board Hailo detector over its IPC socket.

    pi_detector_ctl.py {status|release|reload|frame|stats} [--socket PATH] [--timeout S]

``release`` frees /dev/hailo0 (e.g. for ``eval_detector.py --backend hef``);
``reload`` re-reads the ``detector.hef`` symlink and its sidecar spec without a
service restart. Exits 0 on success, 1 on error or timeout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

DEFAULT_SOCKET = "/apps/lawnberry-pi/data/lawnberry-camera.sock"
COMMANDS = {
    "status": "detector_status",
    "release": "release_detector",
    "reload": "reload_detector",
    "frame": "get_frame",
    "stats": "get_status",
}
# Frame broadcasts share the socket and are large single-line JSON messages.
STREAM_LIMIT = 32 * 1024 * 1024


async def request(socket_path: str, command: str) -> dict:
    reader, writer = await asyncio.open_unix_connection(socket_path, limit=STREAM_LIMIT)
    try:
        writer.write(json.dumps({"command": command}).encode() + b"\n")
        await writer.drain()
        while True:
            line = await reader.readline()
            if not line:
                raise ConnectionError("camera service closed the connection")
            msg = json.loads(line)
            if isinstance(msg, dict) and msg.get("type") != "frame" and "status" in msg:
                return msg
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


def render(cmd: str, resp: dict) -> object:
    if resp.get("status") != "success":
        return resp
    data = resp.get("data") or {}
    if cmd == "frame":
        return {k: v for k, v in data.items() if k != "data"}
    if cmd == "stats":
        return data.get("statistics")
    return resp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=sorted(COMMANDS))
    ap.add_argument("--socket", default=DEFAULT_SOCKET)
    ap.add_argument("--timeout", type=float, default=30.0)
    a = ap.parse_args(argv)
    try:
        resp = asyncio.run(asyncio.wait_for(request(a.socket, COMMANDS[a.cmd]), a.timeout))
    except TimeoutError:
        print(f"timeout after {a.timeout}s waiting for {COMMANDS[a.cmd]}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(render(a.cmd, resp), indent=2))
    return 0 if resp.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
