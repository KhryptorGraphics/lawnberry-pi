#!/usr/bin/env python3
"""Minimal Thor-side endpoint of the Pi <-> Thor link (protocol v2). Not a service by itself.

Run it on the Thor from the repository root. It authenticates every uplink, answers each
one with the next waypoint window (or a pause), and echoes the uplink's ``sid``/``seq``:

    THOR_LINK_KEY=<hex> python3 scripts/thor_link_client.py --plan plan.json
    python3 scripts/thor_link_client.py --key-file /etc/lawnberry/thor_link.key --boundary yard.json
    SIM_MODE=1 python3 scripts/thor_link_client.py --sim --plan plan.json   # public sim key

Plans:
    --plan FILE       JSON ``{"plan_id": "bench-1", "waypoints": [{"lat": 45.0, "lon": -93.0,
                      "blade": false}, ...]}`` (or a bare waypoint list), served in order in
                      windows of at most 16.
    --boundary FILE   JSON ``{"boundary": [[lat, lon], ...], "no_go": [[[lat, lon], ...]]}``: the
                      coverage strategist (``nav/thor_strategist.py``) plans a mow of it.

Key: ``--key-file`` (hex, mode 0600; wins), else ``THOR_LINK_KEY`` (hex, >= 32 bytes; generate
with ``openssl rand -hex 32``), the same value as on the Pi. Without a key it refuses to start;
``--sim`` takes the published sim key and only with ``SIM_MODE=1``.

What this endpoint does so the Pi can trust it (the Thor-side obligations):
  * drops, without any answer, every datagram whose size, HMAC tag or schema is wrong;
  * drops uplinks whose ``seq`` is not newer than the last seen for that ``sid`` (a new ``sid``
    means the Pi restarted: its counters start over);
  * answers each accepted uplink exactly once, from the uplink's own source address, echoing
    ``sid`` and ``seq`` (as ``ack``) on pauses too, since the Pi ignores anything unechoed;
  * pauses (never a plan) while the Pi has no ``rtk_fixed`` fix, reports an e-stop or a hold
    fault, or the plan is finished; the Pi keeps every safety decision regardless;
  * never sends anything that does not answer an authentic uplink.

SIGUSR1 pauses (every answer becomes a pause), SIGUSR2 resumes; SIGINT/SIGTERM stop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import stat
import sys
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.src.models.thor_link import (  # noqa: E402
    MAX_WAYPOINTS,
    SIM_LINK_KEY,
    Downlink,
    LinkAuthError,
    LinkBudgetError,
    LinkKeyError,
    Uplink,
    Waypoint,
    decode_uplink,
    encode,
    parse_key,
)

log = logging.getLogger("thor_link_client")

KEY_ENV = "THOR_LINK_KEY"
DEFAULT_PORT = 47101
# Pi sessions remembered for replay protection; a Pi restart is one new sid, so a handful
# is plenty and the table cannot grow without bound.
MAX_SESSIONS = 8

Planner = Callable[[Uplink], Downlink]


# ------------------------------------ key ------------------------------------


def load_key(
    *,
    key_file: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
    sim: bool = False,
) -> bytes:
    """The shared link key: ``key_file``, else ``THOR_LINK_KEY``, else the public one (``sim``).

    Raises ``LinkKeyError`` instead of falling back to anything weaker: a Thor that cannot
    authenticate must not talk. The key value is never put in an error message.
    """
    env = os.environ if env is None else env
    if key_file is not None:
        path = Path(key_file)
        try:
            mode = path.stat().st_mode
            if mode & (stat.S_IRWXG | stat.S_IRWXO):
                raise LinkKeyError(f"{path} is readable by group or others: chmod 600 it")
            return parse_key(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise LinkKeyError(f"cannot read key file {path}: {exc.strerror}") from None
    if env.get(KEY_ENV):
        return parse_key(env[KEY_ENV])
    if sim:
        if env.get("SIM_MODE") != "1":
            raise LinkKeyError("--sim needs SIM_MODE=1: the sim key is public, never for hardware")
        log.warning("using the public SIM_LINK_KEY (SIM_MODE only)")
        return SIM_LINK_KEY
    raise LinkKeyError(f"no link key: set {KEY_ENV} (hex, openssl rand -hex 32) or pass --key-file")


# ----------------------------------- planners -----------------------------------


class StaticPlanner:
    """Serves a fixed waypoint list in windows: the smallest planner that satisfies the Pi.

    Progress follows the Pi's own ``waypoint_index`` (the first waypoint it has not reached)
    and never goes backwards, so a restarted Thor resumes where the Pi says it is.
    """

    def __init__(
        self,
        waypoints: list[Waypoint],
        plan_id: str = "static",
        require_fix: str = "rtk_fixed",
    ) -> None:
        if not waypoints:
            raise ValueError("a plan needs at least one waypoint")
        self.waypoints = list(waypoints)
        self.plan_id = plan_id
        self.require_fix = require_fix
        self.next_index = 0

    def on_uplink(self, up: Uplink) -> Downlink:
        if up.waypoint_index is not None:
            self.next_index = max(self.next_index, min(up.waypoint_index, len(self.waypoints)))
        pause = (
            up.lat is None
            or up.lon is None
            or up.fix != self.require_fix
            or up.estop
            or up.hold_fault
            or self.next_index >= len(self.waypoints)
        )
        window = [] if pause else self.waypoints[self.next_index : self.next_index + MAX_WAYPOINTS]
        return Downlink(
            sid=up.sid,
            ack=up.seq,
            plan_id=self.plan_id,
            start_index=self.next_index,
            waypoints=window,
            pause=pause,
        )


class StrategistPlanner:
    """Adapts ``ThorStrategist``: a coverage plan of a boundary, replanned from reports.

    The strategist plans the first leg from the first stripe's start. The first time a
    session reports a usable fix with nothing reached yet, the plan is rebuilt so its first
    transit starts from where the mower really is.
    """

    def __init__(self, strategist: Any, require_fix: str = "rtk_fixed") -> None:
        self.strategist = strategist
        self.require_fix = require_fix
        self._replanned_for: int | None = None

    def on_uplink(self, up: Uplink) -> Downlink:
        s = self.strategist
        if (
            self._replanned_for != up.sid
            and up.lat is not None
            and up.lon is not None
            and up.fix == self.require_fix
            and (up.waypoint_index or 0) == 0
            and s.state.next_index == 0
        ):
            self._replanned_for = up.sid
            try:
                s.replan(start=s.frame.xy((up.lat, up.lon)))
            except Exception:
                log.exception("replan from the reported position failed; keeping the plan")
        return s.on_uplink(up)


def _waypoints_from(items: Iterable[dict[str, Any]]) -> list[Waypoint]:
    return [Waypoint.model_validate(item) for item in items]


def load_planner(*, plan: str | None, boundary: str | None, require_fix: str) -> Planner:
    """Build the planner from ``--plan`` or ``--boundary`` (exactly one)."""
    if (plan is None) == (boundary is None):
        raise ValueError("give exactly one of --plan and --boundary")
    if plan is not None:
        data = json.loads(Path(plan).read_text(encoding="utf-8"))
        items, plan_id = (data, "static") if isinstance(data, list) else (data["waypoints"], None)
        if plan_id is None:
            plan_id = str(data.get("plan_id", "static"))
        return StaticPlanner(_waypoints_from(items), plan_id, require_fix).on_uplink
    from backend.src.nav.thor_strategist import StrategyConfig, ThorStrategist

    data = json.loads(Path(boundary).read_text(encoding="utf-8"))  # type: ignore[arg-type]
    strategist = ThorStrategist(
        [tuple(p) for p in data["boundary"]],
        [[tuple(p) for p in zone] for zone in data.get("no_go", [])],
        StrategyConfig(require_fix=require_fix),
        plan_id=str(data.get("plan_id", "plan")),
    )
    return StrategistPlanner(strategist, require_fix).on_uplink


# ------------------------------------ endpoint ------------------------------------


class ThorLinkClient(asyncio.DatagramProtocol):
    """UDP endpoint: authenticate an uplink, answer it, never answer anything else.

    ``handle`` is the whole decision (bytes in, reply bytes or None out), so it can be tested
    without a socket; ``datagram_received`` only moves bytes. ``paused`` turns every answer
    into a pause (the operator's hold).
    """

    def __init__(
        self,
        key: bytes,
        planner: Planner,
        *,
        allowed_pi: Iterable[str] | None = None,
    ) -> None:
        self._key = key
        self._planner = planner
        self._allowed = {str(a) for a in allowed_pi} if allowed_pi is not None else None
        self._last_seq: OrderedDict[int, int] = OrderedDict()
        self._transport: asyncio.DatagramTransport | None = None
        self.paused = False
        self.counters: Counter[str] = Counter()
        self.last_uplink: Uplink | None = None

    # -- decision ----------------------------------------------------------------

    def handle(self, data: bytes, addr: tuple) -> bytes | None:
        self.counters["heard"] += 1
        if self._allowed is not None and addr[0] not in self._allowed:
            self.counters["source"] += 1
            return None
        try:
            up = decode_uplink(data, self._key)
        except LinkBudgetError:
            self.counters["oversize"] += 1
            return None
        except LinkAuthError:
            self.counters["auth"] += 1
            return None
        except ValueError:  # authentic bytes, invalid content
            self.counters["malformed"] += 1
            return None
        last = self._last_seq.get(up.sid)
        if last is not None and up.seq <= last:
            self.counters["replay"] += 1
            return None
        self._last_seq[up.sid] = up.seq
        self._last_seq.move_to_end(up.sid)
        while len(self._last_seq) > MAX_SESSIONS:
            self._last_seq.popitem(last=False)
        self.last_uplink = up
        try:
            down = self._pause(up, "paused") if self.paused else self._planner(up)
            reply = encode(down, self._key)
        except Exception:
            # No planner failure may leave the Pi waiting: say "pause" instead of staying silent.
            log.exception("planner failed; answering with a pause")
            self.counters["planner_error"] += 1
            reply = encode(self._pause(up, "error"), self._key)
        self.counters["answered"] += 1
        return reply

    @staticmethod
    def _pause(up: Uplink, plan_id: str) -> Downlink:
        return Downlink(sid=up.sid, ack=up.seq, plan_id=plan_id, pause=True)

    # -- asyncio plumbing ----------------------------------------------------------

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        reply = self.handle(data, addr)
        if reply is not None and self._transport is not None:
            self._transport.sendto(reply, addr)

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()

    @classmethod
    async def serve(
        cls,
        key: bytes,
        planner: Planner,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        *,
        allowed_pi: Iterable[str] | None = None,
    ) -> ThorLinkClient:
        client = cls(key, planner, allowed_pi=allowed_pi)
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(lambda: client, local_addr=(host, port))
        return client

    @property
    def port(self) -> int:
        assert self._transport is not None
        return self._transport.get_extra_info("sockname")[1]

    def summary(self) -> str:
        up = self.last_uplink
        pi = (
            f"fix={up.fix} hold={up.safe_hold} fault={up.hold_fault} estop={up.estop} "
            f"index={up.waypoint_index}"
            if up
            else "no uplink yet"
        )
        return f"{dict(self.counters)} paused={self.paused} pi: {pi}"


# --------------------------------------- CLI ---------------------------------------


def _parse_listen(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    return (host or "0.0.0.0", int(port))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--listen", default=f"0.0.0.0:{DEFAULT_PORT}", help="HOST:PORT to bind")
    ap.add_argument("--plan", help="waypoint list JSON served in windows")
    ap.add_argument("--boundary", help="boundary JSON planned by the coverage strategist")
    ap.add_argument("--key-file", help="hex key file, mode 0600 (default: $THOR_LINK_KEY)")
    ap.add_argument("--sim", action="store_true", help="use the public sim key (SIM_MODE=1 only)")
    ap.add_argument("--pi", action="append", help="accept uplinks only from this Pi IP (repeat)")
    ap.add_argument("--require-fix", default="rtk_fixed", help="fix the Pi needs to get a plan")
    ap.add_argument("--status-every", type=float, default=5.0, help="seconds between status lines")
    return ap


async def run(args: argparse.Namespace) -> int:
    try:
        key = load_key(key_file=args.key_file, sim=args.sim)
        planner = load_planner(plan=args.plan, boundary=args.boundary, require_fix=args.require_fix)
    except (LinkKeyError, ValueError, KeyError, OSError) as exc:
        print(f"thor_link_client: {exc}", file=sys.stderr)
        return 2
    host, port = _parse_listen(args.listen)
    client = await ThorLinkClient.serve(key, planner, host, port, allowed_pi=args.pi)
    log.info("listening on %s:%d; Pi allowlist %s", host, client.port, args.pi or "off (HMAC only)")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig_name, action in (
        ("SIGINT", stop.set),
        ("SIGTERM", stop.set),
        ("SIGUSR1", lambda: setattr(client, "paused", True)),
        ("SIGUSR2", lambda: setattr(client, "paused", False)),
    ):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            loop.add_signal_handler(sig, action)
    try:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=args.status_every)
            except TimeoutError:
                log.info("%s", client.summary())
    finally:
        client.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
