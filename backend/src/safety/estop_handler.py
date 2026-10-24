"""Emergency stop handler (T030).

Immediate, thread-safe revocation of motor authorization. Hardware GPIO
integration will be added later; this provides core behavior for tests.

``auth`` is anything with a ``revoke()``: ``MotorAuthorization`` in the unit tests, and the
``TractorControlService`` itself in production, so a watchdog trip revokes the authorization the
actuation routes actually check (audit N-04: a private ``MotorAuthorization`` nothing read).
"""

from __future__ import annotations

from typing import Protocol


class Revocable(Protocol):
    def revoke(self) -> None: ...


class EstopHandler:
    def __init__(self, auth: Revocable) -> None:
        self._auth = auth
        self._last_reason: str | None = None

    def trigger_estop(self, reason: str = "unknown") -> None:
        self._last_reason = reason
        # Immediate disable
        self._auth.revoke()

    @property
    def last_reason(self) -> str | None:
        return self._last_reason
