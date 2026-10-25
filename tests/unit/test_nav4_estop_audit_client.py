"""The open E-stop routes write the caller's address to their audit row (SEC3-13, rest.py half).

``/control/emergency`` and ``/control/emergency-stop`` need no credential, so the address the
connection came from is the only identity their audit rows can carry. The rows were written with
``client_id`` None. The address comes from ``core.peer_address.client_host`` (the connection's peer,
or the client behind a trusted forwarder), never from a client-written header: the request below
claims another identity in ``X-Client-Id`` and the row must still say who connected.
"""

from __future__ import annotations

import httpx

from backend.src.core.persistence import persistence


async def test_the_open_rest_stop_routes_audit_the_peer_address():
    from backend.src.main import app

    persistence.flush_audit_coalesced()  # an earlier test may have left a coalescing window open
    forged = {"X-Client-Id": "someone-else@10.9.9.9"}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        assert (
            await http.post("/api/v2/control/emergency-stop", headers=forged)
        ).status_code == 200
        assert (
            await http.post("/api/v2/control/emergency", json={}, headers=forged)
        ).status_code == 202
    persistence.flush_audit_coalesced()

    rows = [
        row
        for row in persistence.load_audit_logs(limit=500)
        if row["action"] in ("control.emergency_stop", "control.emergency.triggered")
    ]
    assert {row["action"] for row in rows} >= {
        "control.emergency_stop",
        "control.emergency.triggered",
    }
    newest = {row["action"]: row for row in reversed(rows)}  # the newest row of each action
    for action in ("control.emergency_stop", "control.emergency.triggered"):
        assert newest[action]["client_id"] == "127.0.0.1", (action, newest[action])
