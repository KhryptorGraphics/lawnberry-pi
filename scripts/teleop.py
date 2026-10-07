#!/usr/bin/env python3
"""Laptop teleoperation + W1 capture control for the mower Pi (NOT a service).

Run from a laptop on the LAN with the Pi reachable. Requires --confirm-supervised:
the operator attests a human watches the machine; without it the tool refuses to
start. Nothing here auto-starts anything on the Pi.

    python3 scripts/teleop.py --host 192.168.50.20 --session yard001 --confirm-supervised

Keys (20 Hz command loop, deadman: a key stops commanding 0.25 s after its last
auto-repeat event, so releasing = neutral without needing key-release events):
    W/S      both levers forward / back (reverse)
    A/D      pivot: left or right lever forward
    Space    emergency stop (tractor e-stop; capture keeps recording)
    C        start/stop the capture session
    B        toggle blade (interlock-gated server-side)
    Q        quit (stops capture if running, then neutral)

Commands go through POST /api/v2/tractor/command (TractorControlService interlocks);
capture through /api/v2/capture/*. Live state + RTK fix print each second. If the
backend requires operator auth (OPERATOR_AUTH_REQUIRED=1), pass --token <bearer>
(from /api/v2/auth/login) or export LAWNBERRY_TOKEN.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time
import urllib.error
import urllib.request

RATE_HZ = 20.0
DEADMAN_S = 0.25  # a key counts as held this long after its last repeat event


def api(
    host: str, port: int, path: str, body: dict | None = None, token: str | None = None
) -> dict:
    url = f"http://{host}:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            return {"_error": exc.code, "_detail": exc.read().decode()[:200]}
        except Exception:
            return {"_error": exc.code, "_detail": str(exc)}
    except Exception as exc:
        return {"_error": -1, "_detail": f"{type(exc).__name__}: {exc}"}


class Keys:
    """Termios raw-mode keyboard; poll() returns the keys seen since last call."""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._old = None

    def __enter__(self) -> Keys:
        import termios

        self._old = termios.tcgetattr(self._fd)
        new = termios.tcgetattr(self._fd)
        new[3] &= ~termios.ICANON & ~termios.ECHO  # lflags: raw, no echo
        new[6][termios.VMIN] = 0  # read returns immediately
        new[6][termios.VTIME] = 0
        termios.tcsetattr(self._fd, termios.TCSADRAIN, new)
        return self

    def __exit__(self, *exc: object) -> None:
        import termios

        if self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self) -> set[str]:
        """Consume everything in the tty buffer; return the keys seen."""
        got: set[str] = set()
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if not r:
                return got
            ch = sys.stdin.read(1)
            if not ch:  # EOF
                return got
            if ch == " ":
                got.add("space")
            elif ch in ("\n", "\r"):
                got.add("enter")
            else:
                got.add(ch.lower())


def levers_from_keys(held: set[str], speed: float) -> tuple[float, float]:
    left = right = 0.0
    if "w" in held:
        left = right = speed
    if "s" in held:
        left = right = -speed
    if "a" in held:
        left = max(left, speed)
    if "d" in held:
        right = max(right, speed)
    return left, right


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="Pi address (e.g. 192.168.50.20)")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--session", help="capture session name (C key starts/stops)")
    ap.add_argument("--speed", type=float, default=0.5, help="lever magnitude, 0..1")
    ap.add_argument("--token", default=None, help="bearer token if auth is required")
    ap.add_argument(
        "--confirm-supervised",
        action="store_true",
        required=True,
        help="attest a human is watching the machine (refuses without)",
    )
    a = ap.parse_args()

    token = a.token or os.getenv("LAWNBERRY_TOKEN")
    host, port = a.host, a.port

    st = api(host, port, "/api/v2/tractor/state", token=token)
    if "_error" in st:
        print(f"backend unreachable at {host}:{port}: {st}", file=sys.stderr)
        return 2
    print(
        f"connected; tractor: engine={st.get('engine')} blade={st.get('blade_engaged')} "
        f"estop={st.get('emergency_stop_active')}"
    )
    if st.get("emergency_stop_active"):
        print(
            "e-stop is active on the Pi; clear it first (dashboard or "
            "POST /api/v2/tractor/clear-emergency)",
            file=sys.stderr,
        )
        return 3

    blade = False
    capturing = False
    last_print = 0.0
    next_tick = time.monotonic()
    last_seen: dict[str, float] = {}
    session = a.session
    print("keys: WASD drive (deadman) | Space e-stop | C capture | B blade | Q quit")

    with Keys() as keys:
        while True:
            now = time.monotonic()
            pressed = keys.poll()
            for k in pressed:
                last_seen[k] = now
            held = {k for k, t in last_seen.items() if now - t <= DEADMAN_S}

            if "q" in pressed:
                break
            if "c" in pressed and session:
                if not capturing:
                    r = api(host, port, "/api/v2/capture/start", {"name": session}, token)
                    capturing = "_error" not in r
                    print(f"capture start: {r}")
                else:
                    r = api(host, port, "/api/v2/capture/stop", {}, token)
                    capturing = False
                    print(f"capture stop: ok={r.get('ok')} problems={r.get('problems')}")
                time.sleep(0.3)  # debounce the keypress
                continue
            if "b" in pressed:
                blade = not blade
                time.sleep(0.3)

            if "space" in pressed:
                r = api(host, port, "/api/v2/tractor/emergency-stop", {}, token)
                print(f"E-STOP -> {r.get('status')}")
                break

            left, right = levers_from_keys(held, a.speed)
            body = {
                "left_lever": left,
                "right_lever": right,
                "throttle": a.speed if (left or right) else 0.0,
                "blade_engaged": blade,
            }
            r = api(host, port, "/api/v2/tractor/command", body, token)
            if "_error" in r and now - last_print > 1.0:
                print(f"command failed: {r}", file=sys.stderr)

            if now - last_print > 1.0:
                st = api(host, port, "/api/v2/tractor/state", token=token)
                gps = api(host, port, "/api/v2/sensors/gps/status", token=token)
                fix = gps.get("rtk_fix_type") or gps.get("rtk_status") or "?"
                print(
                    f"L={left:+.2f} R={right:+.2f} blade={blade} "
                    f"capturing={capturing} rtk={fix} moving={st.get('moving')}"
                )
                last_print = now

            next_tick += 1.0 / RATE_HZ
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()  # fell behind; reset the schedule

    # Safe exit: neutral, blade off, capture stopped.
    api(
        host,
        port,
        "/api/v2/tractor/command",
        {"left_lever": 0.0, "right_lever": 0.0, "throttle": 0.0, "blade_engaged": False},
        token,
    )
    if capturing:
        r = api(host, port, "/api/v2/capture/stop", {}, token)
        print(f"capture stop: ok={r.get('ok')} problems={r.get('problems')}")
    print("quit (levers neutral)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
