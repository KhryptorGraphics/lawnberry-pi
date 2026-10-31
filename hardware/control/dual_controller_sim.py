"""Offline model of the Pi 5 primary / Libre Alta standby stop architecture.

Pure Python. Imports nothing from ``backend/``, and no GPIO/I2C/network library, so no
test can reach hardware. The clock is injected as integer milliseconds and every actuator
sink is a modelled object, so results are deterministic.

Scope: LOGIC AND MODELLED ELECTRICAL STATE ONLY.

A requested OFF is a request. ``modeled_clutch_feed_permitted`` is a fact about this
model's contact graph, not a measurement of a clutch. Nothing here reports that steering
bars moved, that blades stopped, or that an engine stopped: ``physical_observations`` stays
``None`` unless a caller injects a separately labelled simulated sensor fixture.

Hardware assumptions that are NOT verified are marked where they are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

HEARTBEAT_PERIOD_MS = 200
MISSING_LIMIT = 3
PEER_TIMEOUT_MS = HEARTBEAT_PERIOD_MS * MISSING_LIMIT
PWM_PERIOD_MS = 20
PWM_FREQUENCY_HZ = 1000 // PWM_PERIOD_MS

# Pololu 2806 defaults, from the manufacturer guide (section 7-8).
SEL_MASTER_US = 1000
SEL_SLAVE_US = 2000
SEL_THRESHOLD_US = 1696
SEL_HYSTERESIS_US = 64
SEL_QUALIFY_PULSES = 3
SEL_INVALID_AFTER_MS = 130

PROTOCOL_VERSION = 1


class HostRole(StrEnum):
    PI = "pi"
    ALTA = "alta"


class ControllerState(StrEnum):
    BOOT_DISARMED = "boot_disarmed"
    PEER_VALIDATED = "peer_validated"
    RUN_AUTHORIZED = "run_authorized"
    FAULT_LATCHED = "fault_latched"


class Transport(StrEnum):
    HALOW = "halow"
    ETHERNET = "ethernet"


class MuxSource(StrEnum):
    PI = "pi"
    ALTA = "alta"
    OUTPUTS_LOW = "outputs_low"


class ContactFault(StrEnum):
    HEALTHY = "healthy"
    STUCK_OPEN = "stuck_open"
    STUCK_CLOSED = "stuck_closed"


NEUTRAL_NOT_COMMANDABLE = "neutral_not_commandable"
OUTPUT_NOT_OBSERVED = "requested_not_observed"


@dataclass(frozen=True, slots=True)
class PwmCalibration:
    """Servo pulse endpoints. UNCALIBRATED defaults; override with measured values."""

    left_neutral_us: int = 1500
    right_neutral_us: int = 1500
    throttle_idle_us: int = 1100
    physically_verified: bool = False


@dataclass(frozen=True, slots=True)
class LocalOutputs:
    """Which electrical functions this host can actually reach.

    ``steering_neutral`` is False for Alta's own M-bus because Alta is never a normal
    steering authority; it reaches steering only through the mux's slave inputs, which is
    modelled by ``slave_steering_neutral``.
    """

    pto_contact: bool = True
    engine_stop: bool = True
    steering_neutral: bool = True
    throttle_idle: bool = True


@dataclass
class PermissionContact:
    """A modelled relay contact: commanded coil != actual contact state.

    ``kind`` is ``no`` (energize-to-allow, PTO) or ``nc`` (energize-to-run, engine stop).
    Faults are independent of the command, so a welded contact persists across stop
    requests -- which is exactly the case the architecture cannot recover from.
    """

    name: str
    kind: str = "no"
    coil_energized: bool = False
    coil_power: bool = True
    fault: ContactFault = ContactFault.HEALTHY

    @property
    def closed(self) -> bool:
        """Whether the modelled contact is conducting."""
        energized = self.coil_energized and self.coil_power
        if self.kind == "no":
            resting = False
            actuated = True
        else:  # "nc": closed at rest, opens when energized
            resting = True
            actuated = False
        if self.fault is ContactFault.STUCK_OPEN:
            return False
        if self.fault is ContactFault.STUCK_CLOSED:
            return True
        return actuated if energized else resting


@dataclass
class PtoPermissionChain:
    """Two independent NO, energize-to-allow contacts in series with an OEM-authorized feed."""

    pi: PermissionContact = field(
        default_factory=lambda: PermissionContact("K_PTO_PI", "no")
    )
    alta: PermissionContact = field(
        default_factory=lambda: PermissionContact("K_PTO_ALTA", "no")
    )
    oem_pto_allowed: bool = False

    @property
    def feed_permitted(self) -> bool:
        return self.oem_pto_allowed and self.pi.closed and self.alta.closed


@dataclass
class EngineStopPair:
    """Two parallel candidate stop contacts to the engine kill node.

    Parallel because EITHER host must be able to stop the engine. Series contacts would
    require both hosts to close, so one living host would defeat the other's stop.
    """

    pi: PermissionContact = field(
        default_factory=lambda: PermissionContact("K_KILL_PI", "nc")
    )
    alta: PermissionContact = field(
        default_factory=lambda: PermissionContact("K_KILL_ALTA", "nc")
    )

    @property
    def kill_ground_requested(self) -> bool:
        return self.pi.closed or self.alta.closed

    @property
    def run_permitted(self) -> bool:
        """Energize-to-run: the kill node is grounded while either NC contact is closed."""
        return not self.kill_ground_requested


@dataclass
class PwmBoard:
    """A PCA9685 channel bank that keeps its last pulses while it has power.

    This is the property that makes a host crash different from a host power loss.
    """

    name: str
    powered: bool = True
    channels_us: dict[int, int] = field(default_factory=dict)
    driven_by_live_host: bool = True

    def write(self, channel: int, pulse_us: int) -> bool:
        """Write a channel. Returns False when it could not reach the board."""
        if not self.powered or not self.driven_by_live_host:
            return False
        self.channels_us[channel] = pulse_us
        return True

    def read(self, channel: int, default_us: int) -> int:
        return self.channels_us.get(channel, default_us)


@dataclass
class ServoMux:
    """Pulse-based model of the Pololu 2806 four-channel RC servo multiplexer.

    Not a boolean "alive" flag: valid SEL needs consecutive good pulses, and losing SEL
    takes nominally 130 ms to register. FAILMODE ON drives outputs low, which is not
    neutral.
    """

    failmode_on: bool = False
    last_source: MuxSource = MuxSource.PI
    good_pulses: int = 0
    sel_last_valid_us: int = 0
    last_pulse_ms: int | None = None
    powered: bool = True

    def drive_sel(self, now_ms: int, pulse_us: int) -> None:
        """Apply one SEL pulse at ``now_ms``."""
        if not self.powered:
            return
        self.last_pulse_ms = now_ms
        if not 500 <= pulse_us <= 2500:
            self.good_pulses = 0
            return
        self.good_pulses += 1
        self.sel_last_valid_us = pulse_us
        if self.good_pulses < SEL_QUALIFY_PULSES:
            return
        above = pulse_us > SEL_THRESHOLD_US
        below = pulse_us < SEL_THRESHOLD_US - SEL_HYSTERESIS_US
        if not (above or below):
            return  # inside the hysteresis band: hold the previous source
        self.last_source = MuxSource.ALTA if above else MuxSource.PI

    def sel_valid(self, now_ms: int) -> bool:
        if self.last_pulse_ms is None or self.good_pulses < SEL_QUALIFY_PULSES:
            return False
        return (now_ms - self.last_pulse_ms) < SEL_INVALID_AFTER_MS

    def source(self, now_ms: int) -> MuxSource:
        if not self.powered:
            return MuxSource.OUTPUTS_LOW
        if self.sel_valid(now_ms):
            return self.last_source
        return MuxSource.OUTPUTS_LOW if self.failmode_on else MuxSource.PI


@dataclass(frozen=True, slots=True)
class Heartbeat:
    version: int
    sender: HostRole
    boot_id: str
    sequence: int
    peer_boot_id: str
    state: ControllerState
    fault_bits: int = 0
    authenticated: bool = True


@dataclass
class PeerRecord:
    boot_id: str | None = None
    last_sequence: int = -1
    last_rx_ms: int | None = None
    retired_boot_ids: set[str] = field(default_factory=set)
    reciprocal: bool = False


@dataclass
class FaultLatchStore:
    """Persistence model for latched faults and retired peer sessions.

    In-memory checkpoint, not a deployed durability implementation. A shared store makes a
    reconstructed host stay latched; a brand-new store still boots disarmed.
    """

    reasons: dict[HostRole, str] = field(default_factory=dict)
    retired_boot_ids: dict[HostRole, set[str]] = field(default_factory=dict)

    def latch(self, role: HostRole, reason: str) -> None:
        self.reasons.setdefault(role, reason)

    def is_latched(self, role: HostRole) -> bool:
        return role in self.reasons

    def retire(self, role: HostRole, boot_id: str) -> None:
        self.retired_boot_ids.setdefault(role, set()).add(boot_id)

    def retired(self, role: HostRole) -> set[str]:
        return self.retired_boot_ids.get(role, set())


@dataclass
class HostRequests:
    """What this host last asked for. Requests are not physical outcomes."""

    pto_off: bool = False
    engine_off: bool = False
    steering_neutral: bool = False
    throttle_idle: bool = False
    unavailable: list[str] = field(default_factory=list)


class DualControllerHost:
    """One controller's supervision and stop logic."""

    def __init__(
        self,
        role: HostRole,
        boot_id: str,
        outputs: LocalOutputs,
        latch_store: FaultLatchStore,
        transport: Transport = Transport.HALOW,
        calibration: PwmCalibration | None = None,
    ) -> None:
        self.role = role
        self.boot_id = boot_id
        self.outputs = outputs
        self.latch_store = latch_store
        self.transport = transport
        self.calibration = calibration or PwmCalibration()
        self.peer_role = HostRole.ALTA if role is HostRole.PI else HostRole.PI
        self.peer = PeerRecord()
        self.state = ControllerState.BOOT_DISARMED
        self.sequence = 0
        self.requests = HostRequests()
        self.fault_reason: str | None = None
        self.alive = True
        self.pto_contact = PermissionContact(f"K_PTO_{role.upper()}", "no")
        self.engine_contact = PermissionContact(f"K_KILL_{role.upper()}", "nc")
        self.neutral_only_blocked = 0
        if self.latch_store.is_latched(role):
            self.state = ControllerState.FAULT_LATCHED
            self.fault_reason = self.latch_store.reasons[role]

    # -- supervision -----------------------------------------------------------------
    def heartbeat(self, now_ms: int) -> Heartbeat:
        self.sequence += 1
        return Heartbeat(
            version=PROTOCOL_VERSION,
            sender=self.role,
            boot_id=self.boot_id,
            sequence=self.sequence,
            peer_boot_id=self.peer.boot_id or "",
            state=self.state,
        )

    def receive_heartbeat(self, message: Heartbeat, now_ms: int) -> bool:
        """Accept a peer heartbeat. False means it did not refresh freshness."""
        if not self.alive:
            return False
        if message.version != PROTOCOL_VERSION or message.sender != self.peer_role:
            return False
        if not message.authenticated:
            return False
        if message.peer_boot_id not in ("", self.boot_id):
            return False  # addressed to a different local session
        if message.boot_id in self.latch_store.retired(self.peer_role):
            return False
        if message.boot_id != self.peer.boot_id:
            # A peer whose session changed cannot be trusted as the same authorized peer.
            if self.peer.boot_id is not None and self.state is ControllerState.RUN_AUTHORIZED:
                self.enter_latched_stop("peer_session_changed", now_ms)
                return False
            if self.peer.boot_id is not None:
                self.peer.retired_boot_ids.add(self.peer.boot_id)
            self.peer.boot_id = message.boot_id
            self.peer.last_sequence = -1
            self.peer.reciprocal = False
        if message.sequence <= self.peer.last_sequence:
            return False  # duplicate or replayed
        self.peer.last_sequence = message.sequence
        self.peer.last_rx_ms = now_ms
        if message.peer_boot_id == self.boot_id:
            # The peer named this local session, so validation is reciprocal.
            self.peer.reciprocal = True
        if self.state is ControllerState.BOOT_DISARMED and self.peer.reciprocal:
            self.state = ControllerState.PEER_VALIDATED
        if message.state is ControllerState.FAULT_LATCHED:
            self.enter_latched_stop("peer_reported_fault", now_ms)
        return True

    def peer_fresh(self, now_ms: int) -> bool:
        if self.peer.last_rx_ms is None:
            return False
        return (now_ms - self.peer.last_rx_ms) < PEER_TIMEOUT_MS

    def tick(self, now_ms: int) -> None:
        if not self.alive or self.state is ControllerState.FAULT_LATCHED:
            return
        if self.peer.last_rx_ms is not None and not self.peer_fresh(now_ms):
            self.enter_latched_stop("peer_timeout", now_ms)

    # -- authority -------------------------------------------------------------------
    def authorize_run(self, *, operator_confirmed: bool, now_ms: int) -> bool:
        if not operator_confirmed or not self.alive:
            return False
        if self.state is not ControllerState.PEER_VALIDATED:
            return False
        if not self.peer_fresh(now_ms):
            return False
        self.state = ControllerState.RUN_AUTHORIZED
        return True

    def rearm(self, *, operator_confirmed: bool, now_ms: int) -> bool:
        """Deliberate human re-arm. Returns to PEER_VALIDATED, never RUN_AUTHORIZED."""
        if not operator_confirmed or not self.alive:
            return False
        if self.state is not ControllerState.FAULT_LATCHED:
            return False
        if not self.peer_fresh(now_ms):
            return False
        if not (self.outputs.pto_contact and self.outputs.engine_stop):
            return False
        self.latch_store.reasons.pop(self.role, None)
        self.fault_reason = None
        self.requests = HostRequests()
        self.state = ControllerState.PEER_VALIDATED
        return True

    # -- stop path -------------------------------------------------------------------
    def request_pto_off(self) -> bool:
        """De-energize this host's own PTO permission contact."""
        if not self.outputs.pto_contact:
            if OUTPUT_NOT_OBSERVED not in self.requests.unavailable:
                self.requests.unavailable.append(OUTPUT_NOT_OBSERVED)
            return False
        self.pto_contact.coil_energized = False
        self.requests.pto_off = True
        return True

    def request_engine_off(self) -> bool:
        """De-energize this host's own engine-stop contact (closing an NC kill path)."""
        if not self.outputs.engine_stop:
            if OUTPUT_NOT_OBSERVED not in self.requests.unavailable:
                self.requests.unavailable.append(OUTPUT_NOT_OBSERVED)
            return False
        self.engine_contact.coil_energized = False
        self.requests.engine_off = True
        return True

    def request_neutral(self) -> bool:
        if self.role is HostRole.ALTA:
            if not self.outputs.steering_neutral:
                if NEUTRAL_NOT_COMMANDABLE not in self.requests.unavailable:
                    self.requests.unavailable.append(NEUTRAL_NOT_COMMANDABLE)
                return False
        elif not self.outputs.steering_neutral:
            if NEUTRAL_NOT_COMMANDABLE not in self.requests.unavailable:
                self.requests.unavailable.append(NEUTRAL_NOT_COMMANDABLE)
            return False
        self.requests.steering_neutral = True
        return True

    def request_throttle_idle(self) -> bool:
        if not self.outputs.throttle_idle:
            if OUTPUT_NOT_OBSERVED not in self.requests.unavailable:
                self.requests.unavailable.append(OUTPUT_NOT_OBSERVED)
            return False
        self.requests.throttle_idle = True
        return True

    def enter_latched_stop(self, reason: str, now_ms: int) -> None:
        """Latch the fault, then issue every OFF request in the same tick.

        A missing or failed capability is recorded and must not suppress the other,
        independent request. Nothing here waits for servo motion or peer acknowledgement.
        """
        if self.state is ControllerState.FAULT_LATCHED and self.fault_reason == reason:
            return  # idempotent
        self.state = ControllerState.FAULT_LATCHED
        self.fault_reason = reason
        self.latch_store.latch(self.role, reason)
        self.request_pto_off()
        self.request_engine_off()
        self.request_neutral()
        self.request_throttle_idle()


@dataclass
class Simulation:
    """Two hosts plus the modelled plant, advanced on explicit integer time."""

    transport: Transport = Transport.HALOW
    calibration: PwmCalibration = field(default_factory=PwmCalibration)
    latch_store: FaultLatchStore = field(default_factory=FaultLatchStore)
    mux: ServoMux = field(default_factory=ServoMux)
    pto_chain: PtoPermissionChain = field(default_factory=PtoPermissionChain)
    engine_pair: EngineStopPair = field(default_factory=EngineStopPair)
    pi_pwm: PwmBoard = field(
        default_factory=lambda: PwmBoard(
            "pi",
            channels_us={0: 1600, 2: 1400, 1: 1100},
        )
    )
    alta_pwm: PwmBoard = field(
        default_factory=lambda: PwmBoard(
            "alta",
            channels_us={0: 1500, 2: 1500, 1: 1400, 3: SEL_MASTER_US},
        )
    )
    now_ms: int = 0
    pi_alive: bool = True
    alta_alive: bool = True
    link_up: bool = True
    throttle_command_us: int = 1600
    servo_power: bool = True
    neutral_mechanically_confirmed: bool | None = None
    events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.pi = DualControllerHost(
            HostRole.PI,
            "pi-boot-1",
            LocalOutputs(steering_neutral=True),
            self.latch_store,
            self.transport,
            self.calibration,
        )
        self.alta = DualControllerHost(
            HostRole.ALTA,
            "alta-boot-1",
            LocalOutputs(steering_neutral=True),
            self.latch_store,
            self.transport,
            self.calibration,
        )
        self.pi.pto_contact = self.pto_chain.pi
        self.alta.pto_contact = self.pto_chain.alta
        self.pi.engine_contact = self.engine_pair.pi
        self.alta.engine_contact = self.engine_pair.alta
        self.pi.pto_contact.coil_energized = True
        self.alta.pto_contact.coil_energized = True
        self.pi.engine_contact.coil_energized = True
        self.alta.engine_contact.coil_energized = True

    # -- messaging -------------------------------------------------------------------
    def exchange(self) -> None:
        """One heartbeat exchange at ``now_ms``."""
        for sender, receiver in ((self.pi, self.alta), (self.alta, self.pi)):
            sender.alive = self.pi_alive if sender.role is HostRole.PI else self.alta_alive
            if not sender.alive:
                continue
            message = sender.heartbeat(self.now_ms)
            if self.link_up:
                receiver.receive_heartbeat(message, self.now_ms)

    def step(self, dt_ms: int = HEARTBEAT_PERIOD_MS) -> None:
        self.now_ms += dt_ms
        self.exchange()
        for host in (self.pi, self.alta):
            host.tick(self.now_ms)
        self.sync_plant()

    # -- plant -----------------------------------------------------------------------
    def sync_plant(self) -> None:
        """Apply each host's stop requests to the boards, contacts and mux."""
        pi_dead = not self.pi_alive
        alta_dead = not self.alta_alive

        if pi_dead:
            self.pi_pwm.driven_by_live_host = False
        if alta_dead:
            self.alta_pwm.driven_by_live_host = False

        if self.pi.requests.steering_neutral and not pi_dead:
            self.pi_pwm.write(0, self.calibration.left_neutral_us)
            self.pi_pwm.write(2, self.calibration.right_neutral_us)

        if self.alta.requests.steering_neutral and not alta_dead:
            # Alta is neutral-only on its slave channels, always.
            self.alta_pwm.write(0, self.calibration.left_neutral_us)
            self.alta_pwm.write(2, self.calibration.right_neutral_us)
        if self.alta.requests.throttle_idle and not alta_dead:
            self.alta_pwm.write(1, self.calibration.throttle_idle_us)

        sel_us = SEL_SLAVE_US if self.alta.state is ControllerState.FAULT_LATCHED else SEL_MASTER_US
        if not alta_dead and self.alta_pwm.powered:
            self.alta_pwm.write(3, sel_us)
        if self.alta_pwm.powered:
            # A powered PCA9685 keeps emitting its held pulse even after its host dies;
            # an unpowered one emits nothing, so SEL goes invalid after ~130 ms.
            self.mux.drive_sel(self.now_ms, self.alta_pwm.read(3, SEL_MASTER_US))

    def throttle_channel_write(self, pulse_us: int) -> bool:
        """Attempt a throttle write; Alta owns the channel, so the command crosses the link."""
        if not self.alta_alive or not self.link_up:
            return False
        return self.alta_pwm.write(1, pulse_us)

    def steering_channel_write(self, role: HostRole, channel: int, pulse_us: int) -> bool:
        """Write a steering channel, refusing Alta non-neutral slave writes."""
        if role is HostRole.ALTA:
            neutral = (
                self.calibration.left_neutral_us
                if channel == 0
                else self.calibration.right_neutral_us
            )
            if pulse_us != neutral:
                self.alta.neutral_only_blocked += 1
                return False
            return self.alta_pwm.write(channel, pulse_us)
        return self.pi_pwm.write(channel, pulse_us)

    def selected_steering(self) -> dict:
        source = self.mux.source(self.now_ms)
        if source is MuxSource.PI:
            pwm = self.pi_pwm if self.pi_pwm.powered else None
        elif source is MuxSource.ALTA:
            pwm = self.alta_pwm if self.alta_pwm.powered else None
        else:
            pwm = None
        if pwm is None:
            return {
                "selected_source": source.value,
                "left_us": None,
                "right_us": None,
                "signal_present": False,
            }
        return {
            "selected_source": source.value,
            "left_us": pwm.read(0, self.calibration.left_neutral_us),
            "right_us": pwm.read(2, self.calibration.right_neutral_us),
            "signal_present": self.servo_power,
        }

    def snapshot(self) -> dict:
        steering = self.selected_steering()
        left = steering["left_us"]
        right = steering["right_us"]
        neutral_selected = (
            left == self.calibration.left_neutral_us
            and right == self.calibration.right_neutral_us
            and steering["signal_present"]
        )
        return {
            "now_ms": self.now_ms,
            "pi_state": self.pi.state.value,
            "alta_state": self.alta.state.value,
            "pi_fault": self.pi.fault_reason,
            "alta_fault": self.alta.fault_reason,
            "neutral_requested": self.pi.requests.steering_neutral
            or self.alta.requests.steering_neutral,
            "selected_source": steering["selected_source"],
            "selected_pwm_us": [left, right],
            "neutral_pulse_selected": neutral_selected,
            "pto_off_requested": self.pi.requests.pto_off or self.alta.requests.pto_off,
            "engine_off_requested": self.pi.requests.engine_off
            or self.alta.requests.engine_off,
            "modeled_clutch_feed_permitted": self.pto_chain.feed_permitted,
            "modeled_kill_ground_requested": self.engine_pair.kill_ground_requested,
            "capability_failures": sorted(
                set(self.pi.requests.unavailable) | set(self.alta.requests.unavailable)
            ),
            # Never a claim of bar position, clutch release, blade RPM or engine state.
            "physical_observations": self.neutral_mechanically_confirmed,
        }


def fresh_pair(transport: Transport = Transport.HALOW, **kwargs) -> Simulation:
    """Two hosts past reciprocal validation and authorized to run."""
    sim = Simulation(transport=transport, **kwargs)
    for _ in range(4):
        sim.step()
    sim.pi.authorize_run(operator_confirmed=True, now_ms=sim.now_ms)
    sim.alta.authorize_run(operator_confirmed=True, now_ms=sim.now_ms)
    return sim
