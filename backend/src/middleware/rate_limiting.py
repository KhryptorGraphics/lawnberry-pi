"""Global rate limiting middleware.

Token bucket per (TCP peer, bucket group) with path-based overrides and exemptions. Defaults
are conservative and configurable via environment variables so production can tune without
code changes.

Notes:
- The client key is the TCP peer address (``core/peer_address.py``), never ``X-Client-Id`` or
  any other header the client writes: rotating such a header used to mint a fresh bucket per
  request and let ``_buckets`` grow without bound. Behind a same-host proxy (the shipped web UI
  server) every browser shares the proxy's one key; raise ``GLOBAL_RATE_LIMIT_RATE`` /
  ``GLOBAL_RATE_LIMIT_BURST`` if that default budget is too tight.
- Each override prefix (or named group of prefixes) has its OWN bucket per peer, so a joystick
  stream or a lever re-send cannot be starved by UI polling in the default bucket.
- The three E-stop POSTs are never limited (``middleware/emergency_paths.py``).
- The bucket table is bounded: a bucket idle long enough to be full again is indistinguishable
  from a new one and is dropped, and the table never holds more than ``MAX_BUCKETS``.
- In-memory store suitable for single-process deployment on Pi.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from ..core.peer_address import request_peer_key
from .emergency_paths import is_emergency_stop_path

# Tractor command POSTs share one bucket per peer, sized like /control/drive: the control page
# re-sends the lever and throttle commands every ~hold/3 (about 18 requests/s) while a control is
# off neutral. GET polling (``/tractor/state``) stays in the default bucket.
TRACTOR_COMMAND_GROUP = "tractor-command"
TRACTOR_COMMAND_PATHS = (
    "/api/v2/tractor/left-lever",
    "/api/v2/tractor/right-lever",
    "/api/v2/tractor/throttle",
    "/api/v2/tractor/command",
    "/api/v2/tractor/blade",
    "/api/v2/tractor/starter",
    "/api/v2/tractor/stop-engine",
    "/api/v2/tractor/authorize",
    "/api/v2/tractor/revoke",
)
TRACTOR_COMMAND_RATE = 30.0
TRACTOR_COMMAND_BURST = 90


@dataclass
class Bucket:
    tokens: float
    last_refill: float
    # The parameters of the last request that used it: what "full again" means for pruning.
    rate: float = 1.0
    burst: float = 1.0


# (prefix, rate, burst) -> its own bucket; (prefix, rate, burst, group) -> buckets shared by
# every prefix with the same group.
Override = tuple[str, float, int] | tuple[str, float, int, str]


class GlobalRateLimiter(BaseHTTPMiddleware):
    MAX_BUCKETS = 4096
    PRUNE_INTERVAL_SECONDS = 30.0

    def __init__(
        self,
        app: FastAPI,
        *,
        refill_rate_per_sec: float = 2.0,  # tokens per second
        burst: int = 20,
        exempt_prefixes: Sequence[str] | None = None,
        strict_prefix_overrides: Sequence[Override] | None = None,
    ) -> None:
        super().__init__(app)
        self._rate = max(0.1, float(refill_rate_per_sec))
        self._burst = max(1, int(burst))
        self._exempt = tuple(exempt_prefixes or ("/health", "/metrics", "/docs", "/openapi.json"))
        # (prefix, rate, burst, group) to override defaults per path
        self._overrides = tuple(
            (o[0], max(0.1, float(o[1])), max(1, int(o[2])), o[3] if len(o) > 3 else o[0])
            for o in (strict_prefix_overrides or ())
        )
        self._buckets: dict[str, Bucket] = {}
        self._last_prune = time.monotonic()

    async def dispatch(self, request: Request, call_next) -> Response:
        # The routed path, so a root_path prefix cannot hide an E-stop from the exemption.
        if is_emergency_stop_path(request.scope.get("path", "")):
            return await call_next(request)

        path = request.url.path
        if self._is_exempt(path):
            return await call_next(request)

        rate, burst, group = self._rate, self._burst, ""
        override = self._match_override(path)
        if override is not None:
            rate, burst, group = override

        allowed, retry_after = await self._consume_token(
            f"{request_peer_key(request)}|{group}", rate, burst
        )
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"},
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)

    def _is_exempt(self, path: str) -> bool:
        return any(path.startswith(p) for p in self._exempt)

    def _match_override(self, path: str) -> tuple[float, int, str] | None:
        for prefix, rate, burst, group in self._overrides:
            if path.startswith(prefix):
                return (rate, burst, group)
        return None

    def _prune(self, now: float) -> None:
        """Drop buckets that are full again, then the least recently used beyond the cap."""
        if (
            len(self._buckets) <= self.MAX_BUCKETS
            and now - self._last_prune < self.PRUNE_INTERVAL_SECONDS
        ):
            return
        self._last_prune = now
        for key, bucket in list(self._buckets.items()):
            if bucket.tokens + (now - bucket.last_refill) * bucket.rate >= bucket.burst:
                del self._buckets[key]
        excess = len(self._buckets) - int(self.MAX_BUCKETS * 0.9)
        if len(self._buckets) > self.MAX_BUCKETS and excess > 0:
            for key in sorted(self._buckets, key=lambda k: self._buckets[k].last_refill)[:excess]:
                del self._buckets[key]

    async def _consume_token(self, key: str, rate: float, burst: int) -> tuple[bool, int]:
        # No await between the read and the write of a bucket, so the event loop already makes
        # this atomic; there is no lock to hold.
        now = time.monotonic()
        self._prune(now)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = Bucket(tokens=float(burst), last_refill=now)
            self._buckets[key] = bucket
        bucket.rate, bucket.burst = rate, float(burst)

        # Refill
        elapsed = now - bucket.last_refill
        if elapsed > 0:
            bucket.tokens = min(float(burst), bucket.tokens + elapsed * rate)
            bucket.last_refill = now

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True, 0

        # Compute Retry-After seconds
        needed = 1.0 - bucket.tokens
        retry_after = int(max(1, needed / rate))
        return False, retry_after


def register_global_rate_limiter(app: FastAPI) -> None:
    """Register the global rate limiter middleware with env-based config."""
    rate = float(os.getenv("GLOBAL_RATE_LIMIT_RATE", "2"))
    burst = int(os.getenv("GLOBAL_RATE_LIMIT_BURST", "20"))
    # Exempt core health/docs and low-risk settings endpoints by default to avoid
    # test flakiness and allow bursty UI saves; can be overridden via env. The E-stop
    # endpoints are exempt whatever this says (middleware/emergency_paths.py).
    exempt_default = (
        "/health,/metrics,/docs,/openapi.json,/api/v2/settings/maps,/api/v2/camera/stream.mjpeg"
    )
    exempt = os.getenv("GLOBAL_RATE_LIMIT_EXEMPT", exempt_default).split(",")
    override_str = os.getenv("GLOBAL_RATE_LIMIT_OVERRIDES", "")
    overrides: list[Override] = []
    if override_str.strip():
        # format: "/api/v2/auth:1:5;/api/v2/ws:5:50"
        parts = [p.strip() for p in override_str.split(";") if p.strip()]
        for p in parts:
            try:
                prefix, rate_s, burst_s = p.split(":")
                overrides.append((prefix, float(rate_s), int(burst_s)))
            except Exception:
                continue

    # Ensure manual control endpoints are permissive enough for continuous joystick inputs.
    # (``/control/emergency`` also matches the authenticated ``/control/emergency_clear``; the
    # E-stop POSTs themselves never reach this table.)
    manual_overrides: list[Override] = [
        ("/api/v2/control/drive", 30.0, 90),
        ("/api/v2/control/blade", 8.0, 20),
        ("/api/v2/control/emergency", 5.0, 12),
    ]
    manual_overrides += [
        (path, TRACTOR_COMMAND_RATE, TRACTOR_COMMAND_BURST, TRACTOR_COMMAND_GROUP)
        for path in TRACTOR_COMMAND_PATHS
    ]
    for override in manual_overrides:
        if not any(existing[0] == override[0] for existing in overrides):
            overrides.append(override)

    camera_overrides: list[Override] = [
        ("/api/v2/camera/status", 12.0, 36),
        ("/api/v2/camera/frame", 12.0, 36),
        ("/api/v2/camera/start", 4.0, 12),
        ("/api/v2/camera/stop", 4.0, 12),
        ("/api/v2/camera/configuration", 4.0, 12),
        ("/api/v2/camera/statistics", 4.0, 12),
    ]
    for override in camera_overrides:
        if not any(existing[0] == override[0] for existing in overrides):
            overrides.append(override)

    app.add_middleware(
        GlobalRateLimiter,
        refill_rate_per_sec=rate,
        burst=burst,
        exempt_prefixes=[s for s in (e.strip() for e in exempt) if s],
        strict_prefix_overrides=overrides,
    )
