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
sends ``pause``. Leaving hold requires a fresh, non-paused plan *and* a
confirmed neutral: if the neutral commands fail the service stays held, keeps
retrying every tick and reports the failure (``hold_fault`` / ``hold_error``,
an ERROR log, and ``hold_fault`` in the uplink) instead of swallowing it.
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
    MAX_UPLINK_HZ,
    SIM_LINK_KEY,
    UPLINK_HZ,
    DatagramBudget,
    Downlink,
    LinkAuthError,
    LinkBudgetError,
    LinkKeyError,
    Uplink,
    Waypoint,
    decode_downlink,
    encode,
    fit_uplink,
    parse_key,
    refuse_published_key,
    validate_key,
)
from ..models.tractor_control import LEVER_NEUTRAL_EPS
from ..nav.geofence_validator import GeofenceShape, path_within

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
# Memory bound on the table of recent uplinks (see ``ThorLinkService._sent_cap``): 1,024 uplinks
# is 20 s at the 50 Hz ceiling, far beyond any plan age worth having.
MAX_SENT_UPLINKS = 1024
# Least datagrams per second one source may have processed, whatever the uplink rate.
MIN_RX_RATE_HZ = 20.0
# Narrowest allowlist network accepted without acknowledgement, by IP version. An entry with a
# shorter prefix names more than one LAN (a /8 is 16 million IPv4 addresses): that is a typo
# far more often than a decision, so it needs ``allow_wide_sources``.
_WIDE_PREFIX = {4: 16, 6: 64}


class HoldReason(StrEnum):
    """Why the plan is not usable (``ThorLinkService.hold_reason``)."""

    NO_LINK = "no_link"  # nothing accepted since start
    LINK_TIMEOUT = "link_timeout"  # no accepted downlink within timeout_s
    PLAN_STALE = "plan_stale"  # plan information older than plan_max_age_s
    PAUSED = "paused"  # the Thor asked for a pause
    NEUTRAL_UNCONFIRMED = "neutral_unconfirmed"  # plan usable, but neutral not confirmed


def resolve_link_key(secrets_manager: SecretsManager | None = None) -> bytes:
    """Return the shared link key, or raise ``LinkKeyError`` when none may be used.

    ``THOR_LINK_KEY`` (hex, >= 32 bytes, random-looking; ``openssl rand -hex 32``) is read
    through ``SecretsManager``: environment first, then ``config/secrets.json``. A weak key
    (see ``validate_key``) or the published ``SIM_LINK_KEY`` outside SIM_MODE is refused.
    With no key configured, SIM_MODE gets the published ``SIM_LINK_KEY`` (logged);
    anywhere else the missing key is fatal, because an unauthenticated link
    would let any host on the network steer the mower.
    """
    sm = secrets_manager or SecretsManager()
    raw = sm.get(KEY_SECRET_NAME, default=None, purpose="thor_link")
    if raw:
        return parse_key(raw, simulation=is_simulation_mode())
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


def parse_allowed_sources(
    entries: Iterable[str] | None, *, allow_wide: bool = False
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Parse a source allowlist strictly. ``None`` means no allowlist (an empty tuple).

    Every entry is one IP address or one CIDR network, and must say exactly what it means:

    * host bits must be zero: ``192.168.1.64/24`` is refused, not quietly widened to the whole
      ``192.168.1.0/24`` (write the network, or the host as ``192.168.1.64``);
    * ``/0`` is refused always: it matches everyone, which is "no allowlist" and is spelled
      ``allowed_sources: any`` (``None`` here);
    * a network wider than one LAN (IPv4 shorter than /16, IPv6 shorter than /64) is refused
      unless ``allow_wide`` acknowledges it: ``10.0.0.0/8`` for ``10.0.0.0/28`` is a typo that
      silently admits 16 million addresses;
    * an empty list is an error, not "allow nobody" or "allow everybody".
    """
    if entries is None:
        return ()
    nets = []
    for entry in entries:
        if not isinstance(entry, str):
            raise ValueError(
                f"allowed_sources entry {entry!r} is not an IP address or CIDR network"
            )
        try:
            net = ipaddress.ip_network(entry, strict=True)
        except ValueError:
            try:
                widened = ipaddress.ip_network(entry, strict=False)
                host = ipaddress.ip_interface(entry).ip
            except ValueError:
                raise ValueError(
                    f"allowed_sources entry {entry!r} is not an IP address or CIDR network"
                ) from None
            raise ValueError(
                f"allowed_sources entry {entry!r} has host bits set, so it would silently become "
                f"{widened}: write {widened} for the whole network or {host} for the one host"
            ) from None
        if net.prefixlen == 0:
            raise ValueError(
                f"allowed_sources entry {entry!r} matches every address: "
                "use 'any' to switch the allowlist off"
            )
        if net.prefixlen < _WIDE_PREFIX[net.version] and not allow_wide:
            raise ValueError(
                f"allowed_sources entry {entry!r} matches {net.num_addresses} addresses, more than "
                "one LAN: that is usually a typo; set allow_wide_sources to accept it"
            )
        nets.append(net)
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
        bind_addr: where the Pi listens. Loopback by default: a forgotten setting hears nobody.
        timeout_s: link heartbeat timeout (no accepted downlink -> hold).
        uplink_hz: uplink rate, at most ``MAX_UPLINK_HZ``.
        plan_max_age_s: hard cap on the age of the information a plan was built
            from, measured from the send time of the uplink it answers (an upper
            bound on its age, with no Thor clock involved).
        neutral_timeout_s: bound on each neutral command so a hung actuator bus
            shows up as a reported fault instead of stalling the slow loop.
        key: shared HMAC key (>= 32 bytes, random-looking: see ``validate_key``). ``None``
            resolves ``THOR_LINK_KEY`` via ``resolve_link_key``; with none configured, only
            SIM_MODE starts.
        allowed_sources: optional IPs/CIDRs datagrams must come from, parsed strictly by
            ``parse_allowed_sources``. HMAC is the real control; this only sheds traffic early.
        allow_wide_sources: accept allowlist entries wider than one LAN (see
            ``parse_allowed_sources``); a typo that widens the list must be acknowledged.
        geofence: optional ``GeofenceShape``; a plan with any waypoint outside it, or any leg
            between two waypoints leaving it or crossing a no-go zone, is rejected whole.
            Without one, waypoints are only range-checked here and the navigation consumer must
            bound them.

    Datagrams are admitted through a ``DatagramBudget`` (a token bucket per source, one for the
    port) before any HMAC work, so a flood cannot make the loop pay for each packet; what it
    sheds is counted in ``rejects["rate"]``.

    Raises:
        LinkKeyError: no usable key, a weak one, or the public sim key outside SIM_MODE.
        ValueError: a non-finite or non-positive timing parameter, ``uplink_hz`` above
            ``MAX_UPLINK_HZ``, or a bad allowlist entry.
    """

    def __init__(
        self,
        tractor: TractorControlService,
        thor_addr: tuple[str, int],
        bind_addr: tuple[str, int] = ("127.0.0.1", 47101),
        timeout_s: float = 0.5,
        uplink_hz: float = UPLINK_HZ,
        build_uplink: UplinkBuilder | None = None,
        clock: Callable[[], float] = time.monotonic,
        *,
        key: bytes | None = None,
        plan_max_age_s: float = 1.0,
        neutral_timeout_s: float = 1.0,
        allowed_sources: Iterable[str] | None = None,
        allow_wide_sources: bool = False,
        geofence: GeofenceShape | None = None,
    ) -> None:
        timings = (timeout_s, uplink_hz, plan_max_age_s, neutral_timeout_s)
        if not all(math.isfinite(t) and t > 0 for t in timings):
            # A NaN or infinite timeout would mean "never time out": refuse, do not run.
            raise ValueError(
                "timeout_s, uplink_hz, plan_max_age_s, neutral_timeout_s must be > 0 and finite"
            )
        if uplink_hz > MAX_UPLINK_HZ:
            raise ValueError(f"uplink_hz must be <= {MAX_UPLINK_HZ:g}")
        key = resolve_link_key() if key is None else validate_key(key)
        refuse_published_key(key, simulation=is_simulation_mode())
        self._key = key
        self._allowed_sources = parse_allowed_sources(
            allowed_sources, allow_wide=allow_wide_sources
        )
        self._geofence = geofence
        self._tractor = tractor
        self._thor_addr = thor_addr
        self._bind_addr = bind_addr
        self._timeout_s = float(timeout_s)
        self._plan_max_age_s = float(plan_max_age_s)
        self._neutral_timeout_s = float(neutral_timeout_s)
        self._period = 1.0 / float(uplink_hz)
        self._build_uplink = build_uplink
        self._clock = clock

        # Random per service run: downlinks must echo it, so nothing captured in an
        # earlier run (or aimed at another Pi) can be replayed into this one.
        self.session_id = secrets.randbelow(MAX_SESSION_ID) + 1
        self._seq = 0  # next uplink seq; every smaller one has been sent
        self._sent: dict[int, float] = {}  # recent uplink seq -> send time
        # Memory bound as well as a replay window: an answer to an uplink older than the last
        # ``_sent_cap`` is "stale" whatever ``plan_max_age_s`` says, so a huge age setting
        # cannot grow this table without limit.
        self._sent_cap = min(MAX_SENT_UPLINKS, max(16, math.ceil(plan_max_age_s * uplink_hz) + 8))
        # The legitimate Thor sends one datagram per uplink, so twice the uplink rate (and a
        # burst for the answers a radio stall queues up) is generous for it and tight for a flood.
        rx_rate = max(2.0 * uplink_hz, MIN_RX_RATE_HZ)
        self._budget = DatagramBudget(
            rx_rate,
            self._sent_cap,
            total_rate_hz=4.0 * rx_rate,
            total_burst=4 * self._sent_cap,
        )
        self._trusted_source: str | None = None  # source of the last accepted downlink
        self._last_ack = -1  # newest uplink seq answered by an accepted downlink
        self._last_rx: float | None = None  # when that downlink arrived
        self._plan_basis = 0.0  # send time of the uplink the active plan answers
        self._paused = False

        self.plan: Downlink | None = None
        self.safe_hold = True  # no link yet == hold
        self.hold_reason: HoldReason | None = HoldReason.NO_LINK
        # Set when a hold could not confirm blade off and both levers neutral.
        # Blocks leaving hold and is reported to the Thor in every uplink.
        self.hold_fault = False
        self.hold_error: str | None = None
        self.hold_failures = 0  # failed neutral attempts, lifetime
        self.bytes_sent = 0
        self.bytes_received = 0
        self.rejected = 0  # total dropped datagrams
        self.rejects: Counter[str] = Counter()  # ... by reason
        self.uplink_errors = 0  # uplinks that could not be encoded or sent at all
        self.builder_errors = 0  # uplinks sent bare because the state builder failed
        self.uplinks_degraded = 0  # uplinks thinned by ``fit_uplink`` to fit one datagram
        self.detections_dropped = 0  # detections left out of uplinks to make them fit, lifetime
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
        try:
            self._receive(data, addr)
        except Exception:
            # A bug here must not become one asyncio "Exception in callback" traceback per
            # datagram, nor leave a half-accepted downlink: drop it, count it, say so (bounded).
            self.rejected += 1
            self.rejects["internal"] += 1
            self._log_limited(
                "rx_internal", logging.ERROR, "Thor link: datagram handler failed", exc_info=True
            )

    def _receive(self, data: bytes, addr: tuple) -> None:
        self.bytes_received += len(data)
        # Cheapest checks first, each before any work a sender can cause: the source allowlist
        # (no state), then the datagram budget (a bucket per source and one for the port), and
        # only then the size check, HMAC and parse. A flood costs a few dict operations per
        # datagram, never an HMAC or a log line.
        if not self._source_allowed(addr):
            self._reject("source", addr)
            return
        source = str(addr[0]) if addr else ""
        # The peer that last delivered an accepted downlink is exempt from the shared bucket
        # (still bounded by its own), so strangers flooding the port cannot starve it.
        if not self._budget.admit(
            source, self._clock(), exempt_total=source == self._trusted_source
        ):
            self._reject("rate", addr)
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
        # Accepted: the only place the heartbeat, the replay baseline and the plan move. The one
        # lookup that could fail comes first, so a failure cannot leave it half done.
        basis = self._sent[msg.ack]
        self._last_ack = msg.ack
        self._last_rx = now
        self._plan_basis = basis
        self._paused = msg.pause
        self.plan = None if msg.pause else msg
        self._trusted_source = source

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
        """Every waypoint inside the fence (margin applied, no-go zones clear) and every leg
        between consecutive ones too: two inside points can still be joined across a notch."""
        if self._geofence is None:
            return True
        try:
            return path_within(
                self._geofence, [LatLng(latitude=w.lat, longitude=w.lon) for w in waypoints]
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
        except Exception as exc:
            # A bug here must not leave a plan live or hide itself: fail closed and report.
            self.plan = None
            self.safe_hold = True
            self.hold_fault = True
            self.hold_error = f"hold evaluation failed: {exc!r}"
            self.hold_failures += 1
            self._log_limited(
                "evaluate", logging.ERROR, "Thor link: safe-hold evaluation failed", exc_info=True
            )
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
                await self._neutralise()
                return
            state = self._tractor.state
            # The engine flag is only a commanded value (it reads OFF after any backend restart
            # whatever the engine is doing), so "is it moving" must not decide whether the
            # neutral is re-asserted: any lever out of neutral, or a blade that is on, does.
            levers_out = (
                abs(state.left_lever) > LEVER_NEUTRAL_EPS
                or abs(state.right_lever) > LEVER_NEUTRAL_EPS
            )
            if self.hold_fault or levers_out or state.blade_engaged:
                # Retry a failed neutral, or re-assert it: something moved the mower while held.
                # Blade off is attempted on every such tick until the relay write is confirmed.
                await self._neutralise()
            return
        if not self.safe_hold:
            return
        if self.hold_fault:
            # The plan is fine but neutral was never confirmed: keep holding and keep trying.
            self.hold_reason = HoldReason.NEUTRAL_UNCONFIRMED
            if not await self._neutralise():
                return
        logger.info("Thor link: fresh plan received, leaving safe hold")
        self.safe_hold = False
        self.hold_reason = None

    async def _neutralise(self) -> bool:
        """Blade off, then levers neutral; True if the tractor state confirms both.

        Both are always attempted. A raised error, a timeout or a refusal is
        recorded (``hold_fault``/``hold_error``/``hold_failures``) and logged, never
        swallowed. "Confirmed" is read from ``tractor.state``, which the tractor
        service updates only after an actuator write succeeds, so a latched e-stop
        that already neutralised the mower counts as confirmed while one whose
        neutral write failed does not. It is the last *commanded* state, not a
        measurement of the physical levers (PCA9685 holds its last pulse).
        """
        steps = (
            ("blade_off", lambda: self._tractor.engage_blade(False, source=_SOURCE)),
            ("levers_neutral", lambda: self._tractor.set_levers(0.0, 0.0, source=_SOURCE)),
        )
        errors: list[str] = []
        for name, command in steps:
            try:
                result = await asyncio.wait_for(command(), timeout=self._neutral_timeout_s)
            except Exception as exc:
                # A TimeoutError with no message is wait_for expiring, not a driver error.
                expired = isinstance(exc, TimeoutError) and not exc.args
                detail = f"timed out after {self._neutral_timeout_s:g}s" if expired else repr(exc)
                errors.append(f"{name} failed: {detail}")
                self._log_limited(
                    f"neutral:{name}",
                    logging.ERROR,
                    "Thor link: safe-hold %s failed (%s)",
                    name,
                    detail,
                    exc_info=not expired,
                )
            else:
                if isinstance(result, dict) and result.get("status") == "rejected":
                    errors.append(f"{name} rejected: {result.get('reason')}")
        state = self._tractor.state
        confirmed = (
            not state.blade_engaged
            and abs(state.left_lever) <= LEVER_NEUTRAL_EPS
            and abs(state.right_lever) <= LEVER_NEUTRAL_EPS
        )
        if confirmed:
            if self.hold_fault:
                logger.warning(
                    "Thor link: safe-hold neutral confirmed after %d failed attempt(s)",
                    self.hold_failures,
                )
            self.hold_fault = False
            self.hold_error = None
            return True
        self.hold_fault = True
        self.hold_failures += 1
        self.hold_error = "; ".join(errors) or "tractor state still shows blade on or levers moving"
        self._log_limited(
            "neutral_fault",
            logging.ERROR,
            "Thor link: SAFE HOLD NOT CONFIRMED NEUTRAL (%s), attempt %d",
            self.hold_error,
            self.hold_failures,
        )
        return False

    def _compose_uplink(self, seq: int, now: float) -> Uplink:
        """The uplink numbered ``seq``: the builder's state plus this service's own fields, fitted.

        The heartbeat must not depend on the state builder. If the builder raises, or what it
        returned cannot be made to fit one datagram, a bare uplink (this service's own fields
        only: session, counter, clock, e-stop, hold and blade flags) goes out instead and the
        failure is counted and logged, so the Thor keeps hearing the Pi and can pause it
        deliberately. A builder that merely returns too many or too long detections is thinned
        by ``fit_uplink`` and still gets its position through.
        """
        state = self._tractor.state
        own = {
            "sid": self.session_id,
            "seq": seq,
            "t_ms": int(now * 1000),
            "estop": state.emergency_stop_active,
            "safe_hold": self.safe_hold,
            "hold_fault": self.hold_fault,
            "blade": state.blade_engaged,
        }
        try:
            base = self._build_uplink() if self._build_uplink else Uplink(seq=0, t_ms=0)
            msg = base.model_copy(update=own)
            fitted = fit_uplink(msg)
        except Exception:
            self.builder_errors += 1
            self._log_limited(
                "builder",
                logging.ERROR,
                "Thor link: uplink state unusable, sending a bare heartbeat",
                exc_info=True,
            )
            return Uplink(seq=0, t_ms=0).model_copy(update=own)
        if fitted is not msg:
            dropped = (fitted.detections_dropped or 0) - (msg.detections_dropped or 0)
            self.uplinks_degraded += 1
            self.detections_dropped += dropped
            self._log_limited(
                "degraded",
                logging.WARNING,
                "Thor link: uplink thinned to fit one datagram (class names clipped, "
                "%d detection(s) dropped)",
                dropped,
            )
        return fitted

    def _send_uplink(self) -> None:
        if self._transport is None:
            return
        now = self._clock()
        seq = self._seq
        data = encode(self._compose_uplink(seq, now), self._key)
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
