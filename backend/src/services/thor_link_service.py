"""Pi-side slow network loop to the Thor strategist (W7).

Decoupled from the fast safety loop (``TractorSafetyMonitor`` + ``Watchdog``):
this task never feeds the safety watchdog and the safety loop never waits on
it. Its only actuation authority is to put the mower into **safe hold** --
both levers neutral and blade off through ``TractorControlService``'s
interlocked methods, without revoking authorization (the acceptance
criteria's soft-stop path). It never moves the mower; waypoints are only
stored for the navigation layer, and only while the link is fresh.

Safe hold is entered when no valid downlink has arrived within
``timeout_s`` or the Thor sends ``pause``. Leaving hold requires a fresh,
non-paused downlink; the stale plan is discarded on entry.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..models.thor_link import (
    UPLINK_HZ,
    Downlink,
    LinkBudgetError,
    Uplink,
    decode_downlink,
    encode,
)

if TYPE_CHECKING:
    from .tractor_service import TractorControlService

logger = logging.getLogger(__name__)

UplinkBuilder = Callable[[], Uplink]


class _Protocol(asyncio.DatagramProtocol):
    def __init__(self, owner: ThorLinkService) -> None:
        self._owner = owner

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        self._owner._on_datagram(data, addr)


class ThorLinkService:
    def __init__(
        self,
        tractor: TractorControlService,
        thor_addr: tuple[str, int],
        bind_addr: tuple[str, int] = ("0.0.0.0", 47101),
        timeout_s: float = 0.5,
        uplink_hz: float = UPLINK_HZ,
        build_uplink: UplinkBuilder | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tractor = tractor
        self._thor_addr = thor_addr
        self._bind_addr = bind_addr
        self._timeout_s = float(timeout_s)
        self._period = 1.0 / float(uplink_hz)
        self._build_uplink = build_uplink
        self._clock = clock
        self._seq = 0
        self._last_rx: float | None = None
        self._last_rx_seq = -1
        self.plan: Downlink | None = None
        self.safe_hold = True  # no link yet == hold
        self.bytes_sent = 0
        self.bytes_received = 0
        self.rejected = 0
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

    async def stop(self) -> None:
        self._stop_evt.set()
        if self._task:
            await self._task
        if self._transport:
            self._transport.close()

    # ------------------------------- inbound -------------------------------

    def _on_datagram(self, data: bytes, addr: tuple) -> None:
        self.bytes_received += len(data)
        try:
            msg = decode_downlink(data)
        except (LinkBudgetError, ValueError):
            self.rejected += 1
            logger.warning("Thor link: rejected malformed/oversized downlink from %s", addr)
            return
        if msg.seq <= self._last_rx_seq:
            return  # duplicate or reordered; ignore
        self._last_rx_seq = msg.seq
        self._last_rx = self._clock()
        self.plan = None if msg.pause else msg

    def link_fresh(self) -> bool:
        return self._last_rx is not None and (self._clock() - self._last_rx) <= self._timeout_s

    def current_plan(self) -> Downlink | None:
        """The active plan, or None while in safe hold. Navigation must poll this each step."""
        return None if self.safe_hold else self.plan

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
            logger.exception("Thor link: uplink send failed")

    async def _evaluate_hold(self) -> None:
        should_hold = not self.link_fresh() or self.plan is None
        if should_hold and not self.safe_hold:
            logger.warning("Thor link: entering safe hold (fresh=%s)", self.link_fresh())
            self.plan = None
            self.safe_hold = True
            await self._tractor.engage_blade(False)
            await self._tractor.set_levers(0.0, 0.0)
        elif should_hold and self.safe_hold and self._tractor.state.moving:
            # Something else moved the mower while held: re-assert neutral.
            await self._tractor.set_levers(0.0, 0.0)
        elif not should_hold and self.safe_hold:
            logger.info("Thor link: fresh plan received, leaving safe hold")
            self.safe_hold = False

    def _send_uplink(self) -> None:
        if self._transport is None:
            return
        msg = self._build_uplink() if self._build_uplink else Uplink(seq=0, t_ms=0)
        state = self._tractor.state
        msg = msg.model_copy(
            update={
                "seq": self._seq,
                "t_ms": int(self._clock() * 1000),
                "estop": state.emergency_stop_active,
                "safe_hold": self.safe_hold,
                "blade": state.blade_engaged,
            }
        )
        data = encode(msg)
        self._transport.sendto(data, self._thor_addr)
        self._seq += 1
        self.bytes_sent += len(data)

    async def _run(self) -> None:
        while not self._stop_evt.is_set():
            await self.tick()
            try:
                await asyncio.wait_for(self._stop_evt.wait(), timeout=self._period)
            except TimeoutError:
                pass
