"""Shared harness for the Thor link tests: the service on a fake clock and a fake socket.

``Rig.answer`` plays the Thor: it answers an uplink the way ``ThorStrategist`` + the Thor's
sender do (echo ``sid``/``seq`` as ``sid``/``ack``, sign with the shared key) and keeps no
state, which is exactly what makes a Thor restart a non-event.
"""

from backend.src.models.thor_link import Downlink, Uplink, Waypoint, decode_uplink, encode
from backend.src.models.tractor_control import EngineState
from backend.src.services.thor_link_service import ThorLinkService
from backend.src.services.tractor_service import TractorControlService

KEY = b"thorlink-test-key/0123456789abcdef!"
OTHER_KEY = b"someone-elses-key/0123456789abcdef!!"
THOR = ("10.0.0.2", 47101)
WP = Waypoint(lat=45.0, lon=-93.0, blade=True)
LOGGER = "backend.src.services.thor_link_service"


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class FakeWire:
    """Stands in for the UDP socket: records every datagram the Pi sends."""

    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple]] = []

    def sendto(self, data: bytes, addr: tuple) -> None:
        self.sent.append((data, addr))

    def close(self) -> None:
        pass


class Rig:
    """A ThorLinkService on a fake clock and socket, plus the Thor's side of the wire."""

    def __init__(self, *, clock: FakeClock | None = None, key: bytes = KEY, **kw) -> None:
        kw.setdefault("timeout_s", 0.5)
        self.clock = clock or FakeClock()
        self.tractor = TractorControlService(config={})
        self.link = ThorLinkService(self.tractor, THOR, clock=self.clock, key=key, **kw)
        self.wire = FakeWire()
        self.link._transport = self.wire

    @property
    def uplinks(self) -> list[Uplink]:
        return [decode_uplink(data, KEY) for data, _ in self.wire.sent]

    async def tick(self) -> None:
        await self.link.tick()

    def answer(
        self,
        up: Uplink | None = None,
        *,
        waypoints: list[Waypoint] | None = None,
        pause: bool = False,
        plan_id: str = "p1",
        key: bytes = KEY,
        **override,
    ) -> bytes:
        """What a healthy Thor sends for ``up`` (default: the newest uplink)."""
        up = up or self.uplinks[-1]
        down = Downlink(
            sid=up.sid,
            ack=up.seq,
            plan_id=plan_id,
            waypoints=[WP] if waypoints is None else waypoints,
            pause=pause,
        ).model_copy(update=override)  # model_copy skips validation: crafts out-of-range values
        return encode(down, key)

    def deliver(self, data: bytes, addr: tuple = THOR) -> None:
        self.link._on_datagram(data, addr)

    async def connect(self) -> bytes:
        """Uplink #0, the Thor's answer to it, then a tick that leaves hold. Returns the answer."""
        await self.tick()
        answer = self.answer()
        self.deliver(answer)
        await self.tick()
        assert not self.link.safe_hold
        return answer

    async def run_to_neutral_failure(self) -> None:
        """Engine on, moving, blade on, then the Thor goes silent past the timeout."""
        self.tractor.state.engine = EngineState.RUNNING
        self.tractor.authorize()
        await self.connect()
        await self.tractor.set_levers(0.6, 0.6)
        await self.tractor.engage_blade(True)
