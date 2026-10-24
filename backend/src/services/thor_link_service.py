"""Pi-side slow network loop to the Thor strategist (W7).

Decoupled from the fast safety loop (``TractorSafetyMonitor`` + ``Watchdog``):
this task never feeds the safety watchdog and the safety loop never waits on
it. Its only actuation authority is to put the mower into **safe hold** --
both levers neutral and blade off through ``TractorControlService``'s
interlocked methods, without revoking authorization (the acceptance
criteria's soft-stop path). It never moves the mower; waypoints are only
stored for the navigation layer, and only while the plan is fresh.

Trust model (protocol v2, ``models/thor_link.py``): a downlink is acted on only
if it passes, in order, the optional source allowlist, the size cap, the
HMAC-SHA256 tag, schema validation (waypoint count, ranges), the session
check (it echoes this run's ``sid``), the ack check (it answers an uplink this
run actually sent, newer than any already-accepted answer), the age check (that
uplink was sent no more than ``plan_max_age_s`` ago) and, for plans, the
optional geofence. Only a packet that passes all of them refreshes the
heartbeat or replaces the plan; everything else is counted in ``rejects`` and
logged at a bounded rate. A ``pause`` is never refused for its *content*.

Safe hold is entered when no downlink has been accepted within ``timeout_s``,
the accepted plan's information is older than ``plan_max_age_s``, or the Thor
sends ``pause``. Leaving hold requires a fresh, non-paused plan.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import math
import secrets
import time
from collections import Counter
from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import TYPE_CHECKING

from ..core.secrets_manager import SecretsManager
from ..core.simulation import is_simulation_mode
from ..models.geofence import LatLng
from ..models.thor_link import (
    MAX_SESSION_ID,
    SIM_LINK_KEY,
    UPLINK_HZ,
    Downlink,
    LinkAuthError,
    LinkBudgetError,
    LinkKeyError,
    Uplink,
    Waypoint,
    decode_downlink,
    encode,
    parse_key,
    validate_key,
)
from ..nav.geofence_validator import GeofenceShape, contains

if TYPE_CHECKING:
    from .tractor_service import TractorControlService

logger = logging.getLogger(__name__)

UplinkBuilder = Callable[[], Uplink]

# Secret name looked up through SecretsManager: environment variable first, then
# config/secrets.json (0600). Hex-encoded, at least 32 bytes.
KEY_SECRET_NAME = "THOR_LINK_KEY"
# Label on the actuator calls so capture logs do not attribute holds to "operator".
_SOURCE = "thor_link"
# Floor between repeated log lines for one cause, so a flood or a persistent
# fault cannot turn a 10 Hz loop into a 10 Hz log.
_LOG_INTERVAL_S = 5.0


class HoldReason(StrEnum):
    """Why the plan is not usable (``ThorLinkService.hold_reason``)."""

    NO_LINK = "no_link"  # nothing accepted since start
    LINK_TIMEOUT = "link_timeout"  # no accepted downlink within timeout_s
    PLAN_STALE = "plan_stale"  # plan information older than plan_max_age_s
    PAUSED = "paused"  # the Thor asked for a pause


def resolve_link_key(secrets_manager: SecretsManager | None = None) -> bytes:
    """Return the shared link key, or raise ``LinkKeyError`` when none may be used.

    ``THOR_LINK_KEY`` (hex, >= 32 bytes; ``openssl rand -hex 32``) is read
    through ``SecretsManager``: environment first, then ``config/secrets.json``.
    With no key configured, SIM_MODE gets the published ``SIM_LINK_KEY`` (logged);
    anywhere else the missing key is fatal, because an unauthenticated link
    would let any host on the network steer the mower.
    """
    sm = secrets_manager or SecretsManager()
    raw = sm.get(KEY_SECRET_NAME, default=None, purpose="thor_link")
    if raw:
        return parse_key(raw)
    if is_simulation_mode():
        logger.warning(
            "%s not configured: using the public SIM_LINK_KEY (SIM_MODE only)", KEY_SECRET_NAME
        )
        return SIM_LINK_KEY
    raise LinkKeyError(
        f"{KEY_SECRET_NAME} is not configured: refusing to run the Thor link unauthenticated "
        "(generate a key with `openssl rand -hex 32` and store it in the environment or "
        "config/secrets.json)"
    )


def _parse_sources(
    entries: Iterable[str] | None,
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    if entries is None:
        return ()
    nets = []
    for entry in entries:
        try:
            nets.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            raise ValueError(
                f"allowed_sources entry {entry!r} is not an IP address or CIDR network"
            ) from None
    if not nets:
        raise ValueError("allowed_sources is empty; pass None to disable the allowlist")
    return tuple(nets)


class _Protocol(asyncio.DatagramProtocol):
    def __init__(self, owner: ThorLinkService) -> None:
        self._owner = owner

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        self._owner._on_datagram(data, addr)


class ThorLinkService:
    """Authenticated Pi-side endpoint of the Pi <-> Thor link.

    Args:
        tractor: actuation service used for the safe-hold neutral commands.
        thor_addr: where uplinks go.
        timeout_s: link heartbeat timeout (no accepted downlink -> hold).
        plan_max_age_s: hard cap on the age of the information a plan was built
            from, measured from the send time of the uplink it answers (an upper
            bound on its age, with no Thor clock involved).
        key: shared HMAC key (>= 32 bytes). ``None`` resolves ``THOR_LINK_KEY``
            via ``resolve_link_key``; with none configured, only SIM_MODE starts.
        allowed_sources: optional IPs/CIDRs datagrams must come from. HMAC is
            the real control; this only sheds traffic early.
        geofence: optional ``GeofenceShape``; a plan with any waypoint outside
            it is rejected whole. Without one, waypoints are only range-checked
            here and the navigation consumer must bound them.

    Raises:
        LinkKeyError: no usable key (or the public sim key outside SIM_MODE).
        ValueError: a non-positive timing parameter or a bad allowlist entry.
    """

    def __init__(
        self,
        tractor: TractorControlService,
        thor_addr: tuple[str, int],
        bind_addr: tuple[str, int] = ("0.0.0.0", 47101),
        timeout_s: float = 0.5,
        uplink_hz: float = UPLINK_HZ,
        build_uplink: UplinkBuilder | None = None,
        clock: Callable[[], float] = time.monotonic,
        *,
        key: bytes | None = None,
        plan_max_age_s: float = 1.0,
        allowed_sources: Iterable[str] | None = None,
        geofence: GeofenceShape | None = None,
    ) -> None:
        if min(timeout_s, uplink_hz, plan_max_age_s) <= 0:
            raise ValueError("timeout_s, uplink_hz, plan_max_age_s must be > 0")
        key = resolve_link_key() if key is None else validate_key(key)
        if key == SIM_LINK_KEY and not is_simulation_mode():
            raise LinkKeyError("SIM_LINK_KEY is public: it is only accepted with SIM_MODE=1")
        self._key = key
        self._allowed_sources = _parse_sources(allowed_sources)
        self._geofence = geofence
        self._tractor = tractor
        self._thor_addr = thor_addr
        self._bind_addr = bind_addr
        self._timeout_s = float(timeout_s)
        self._plan_max_age_s = float(plan_max_age_s)
        self._period = 1.0 / float(uplink_hz)
        self._build_uplink = build_uplink
        self._clock = clock

        # Random per service run: downlinks must echo it, so nothing captured in an
        # earlier run (or aimed at another Pi) can be replayed into this one.
        self.session_id = secrets.randbelow(MAX_SESSION_ID) + 1
        self._seq = 0  # next uplink seq; every smaller one has been sent
        self._sent: dict[int, float] = {}  # recent uplink seq -> send time
        self._sent_cap = max(16, math.ceil(plan_max_age_s * uplink_hz) + 8)
        self._last_ack = -1  # newest uplink seq answered by an accepted downlink
        self._last_rx: float | None = None  # when that downlink arrived
        self._plan_basis = 0.0  # send time of the uplink the active plan answers
        self._paused = False

        self.plan: Downlink | None = None
        self.safe_hold = True  # no link yet == hold
        self.hold_reason: HoldReason | None = HoldReason.NO_LINK
        self.bytes_sent = 0
        self.bytes_received = 0
        self.rejected = 0  # total dropped datagrams
        self.rejects: Counter[str] = Counter()  # ... by reason
        self.uplink_errors = 0
        self._log_state: dict[str, tuple[float, int]] = {}
        self._transport: asyncio.DatagramTransport | None = None
        self._task: asyncio.Task | None = None
        self._stop_evt = asyncio.Event()

    # ------------------------------ lifecycle ------------------------------

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _Protocol(self), local_addr=self._bind_addr
        )
        self._stop_evt.clear()
        self._task = asyncio.create_task(self._run())
        logger.info(
            "Thor link up: listening %s:%d, Thor %s:%d, HMAC-SHA256 on, sources=%s, geofence=%s",
            *self._bind_addr,
            *self._thor_addr,
            [str(n) for n in self._allowed_sources] or "any",
            "on" if self._geofence is not None else "off",
        )
        if self._geofence is None and not is_simulation_mode():
            logger.warning("Thor link: no geofence configured, plan waypoints are not bounded here")

    async def stop(self) -> None:
        self._stop_evt.set()
        if self._task:
            await self._task
        if self._transport:
            self._transport.close()

    # ------------------------------- inbound -------------------------------

    def _on_datagram(self, data: bytes, addr: tuple) -> None:
        self.bytes_received += len(data)
        if not self._source_allowed(addr):
            self._reject("source", addr)
            return
        try:
            msg = decode_downlink(data, self._key)
        except LinkBudgetError:
            self._reject("oversize", addr)
            return
        except LinkAuthError:
            self._reject("auth", addr)
            return
        except ValueError:  # authentic bytes, invalid content (count, range, shape)
            self._reject("malformed", addr)
            return
        now = self._clock()
        reason = self._admit(msg, now)
        if reason is not None:
            self._reject(reason, addr)
            return
        # Accepted: the only place the heartbeat, the replay baseline and the plan move.
        self._last_ack = msg.ack
        self._last_rx = now
        self._plan_basis = self._sent[msg.ack]
        self._paused = msg.pause
        self.plan = None if msg.pause else msg

    def _admit(self, msg: Downlink, now: float) -> str | None:
        """Why an authentic ``msg`` must be dropped, or None. Never changes link state."""
        if msg.sid != self.session_id:
            return "session"  # another run's (or another Pi's) traffic
        if msg.ack >= self._seq:
            return "future_ack"  # answers an uplink this run never sent
        if msg.ack <= self._last_ack:
            return "replay"  # duplicate or reordered
        sent_at = self._sent.get(msg.ack)
        if sent_at is None or now - sent_at > self._plan_max_age_s:
            return "stale"  # too slow, or older than the uplinks we still remember
        if not msg.pause and not self._waypoints_in_geofence(msg.waypoints):
            return "geofence"
        return None

    def _source_allowed(self, addr: tuple) -> bool:
        if not self._allowed_sources:
            return True
        try:
            ip = ipaddress.ip_address(addr[0])
        except (ValueError, IndexError, TypeError):
            return False
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        return any(ip in net for net in self._allowed_sources)

    def _waypoints_in_geofence(self, waypoints: list[Waypoint]) -> bool:
        if self._geofence is None:
            return True
        try:
            return all(
                contains(self._geofence, LatLng(latitude=w.lat, longitude=w.lon)) for w in waypoints
            )
        except Exception:
            self._log_limited(
                "geofence", logging.ERROR, "Thor link: geofence check failed", exc_info=True
            )
            return False  # cannot prove it is inside: reject

    def _reject(self, reason: str, addr: tuple) -> None:
        self.rejected += 1
        self.rejects[reason] += 1
        self._log_limited(
            f"reject:{reason}",
            logging.WARNING,
            "Thor link: dropped downlink from %s (%s)",
            addr[:2],
            reason,
        )

    def link_fresh(self) -> bool:
        return self._last_rx is not None and (self._clock() - self._last_rx) <= self._timeout_s

    def _hold_reason_now(self) -> HoldReason | None:
        """Why the plan is unusable at this instant, from the clock alone (None = usable)."""
        if self._last_rx is None:
            return HoldReason.NO_LINK
        now = self._clock()
        if now - self._last_rx > self._timeout_s:
            return HoldReason.LINK_TIMEOUT
        if self._paused:
            return HoldReason.PAUSED
        if self.plan is None or now - self._plan_basis > self._plan_max_age_s:
            return HoldReason.PLAN_STALE
        return None

    def current_plan(self) -> Downlink | None:
        """The active plan, or None while held or once it has aged out.

        Navigation must poll this each step. It re-checks the clock on every
        call, so a plan expires on time even if ``tick`` is late or has stopped.
        """
        if self.safe_hold or self._hold_reason_now() is not None:
            return None
        return self.plan

    # ------------------------------- loop ---------------------------------

    async def tick(self) -> None:
        """One slow-loop iteration: evaluate hold, then send an uplink. Never raises."""
        try:
            await self._evaluate_hold()
        except Exception:
            logger.exception("Thor link: safe-hold evaluation failed")
        try:
            self._send_uplink()
        except Exception:
            self.uplink_errors += 1
            self._log_limited(
                "uplink", logging.ERROR, "Thor link: uplink send failed", exc_info=True
            )

    async def _evaluate_hold(self) -> None:
        reason = self._hold_reason_now()
        if reason is not None:
            self.hold_reason = reason
            if not self.safe_hold:
                logger.warning("Thor link: entering safe hold (%s)", reason.value)
                self.plan = None  # a stale plan must not outlive the link that vouched for it
                self.safe_hold = True
                await self._tractor.engage_blade(False, source=_SOURCE)
                await self._tractor.set_levers(0.0, 0.0, source=_SOURCE)
            elif self._tractor.state.moving:
                # Something else moved the mower while held: re-assert neutral.
                await self._tractor.set_levers(0.0, 0.0, source=_SOURCE)
            return
        if self.safe_hold:
            logger.info("Thor link: fresh plan received, leaving safe hold")
            self.safe_hold = False
            self.hold_reason = None

    def _send_uplink(self) -> None:
        if self._transport is None:
            return
        msg = self._build_uplink() if self._build_uplink else Uplink(seq=0, t_ms=0)
        state = self._tractor.state
        now = self._clock()
        seq = self._seq
        msg = msg.model_copy(
            update={
                "sid": self.session_id,
                "seq": seq,
                "t_ms": int(now * 1000),
                "estop": state.emergency_stop_active,
                "safe_hold": self.safe_hold,
                "blade": state.blade_engaged,
            }
        )
        data = encode(msg, self._key)
        # Remember the send time before sending: the reply is judged against it.
        self._sent[seq] = now
        while len(self._sent) > self._sent_cap:
            del self._sent[next(iter(self._sent))]
        self._transport.sendto(data, self._thor_addr)
        self._seq = seq + 1
        self.bytes_sent += len(data)

    async def _run(self) -> None:
        while not self._stop_evt.is_set():
            await self.tick()
            try:
                await asyncio.wait_for(self._stop_evt.wait(), timeout=self._period)
            except TimeoutError:
                pass

    def _log_limited(
        self, key: str, level: int, msg: str, *args: object, exc_info: bool = False
    ) -> None:
        """Log at most once per ``_LOG_INTERVAL_S`` per ``key``, counting what was skipped."""
        now = self._clock()
        last, skipped = self._log_state.get(key, (-math.inf, 0))
        if now - last < _LOG_INTERVAL_S:
            self._log_state[key] = (last, skipped + 1)
            return
        if skipped:
            msg, args = msg + " (+%d similar suppressed)", (*args, skipped)
        logger.log(level, msg, *args, exc_info=exc_info)
        self._log_state[key] = (now, 0)
