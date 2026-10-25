"""SEC3-08 / SEC3-13: who cleared an E-stop, and who stopped the machine, in the audit trail.

* SEC3-08: the clear route's audit row took its client from the ``X-Client-Id`` header the caller
  writes (``someone-else@10.9.9.9``); its twin ``/tractor/clear-emergency`` already used the peer.
* SEC3-13: the three E-stop routes are open by design, so their audit row is the only record of who
  stopped the mower. The coalesced writer kept the first event's client only (and the tractor route
  wrote no row at all): a window that merged stops from several clients said nothing about them.

The two REST stop routes (``api/rest.py``) pass ``client_host(request)`` through
``_audit_estop_coalesced``: that hunk belongs to the owner of ``rest.py`` and is pinned by a test in
their file; what is tested here is the writer (``AuditCoalescer``), the tractor route and the clear.
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest

from backend.src.core.persistence import AuditCoalescer, persistence

CLEAR = {"confirmation": True, "reason": "bench"}


def rows(action: str) -> list[dict]:
    persistence.flush_audit_coalesced()
    found = []
    for row in persistence.load_audit_logs(limit=1000):
        if row["action"] == action:
            details = row["details"]
            row = {**row, "details": json.loads(details) if isinstance(details, str) else details}
            found.append(row)
    return found


def client() -> httpx.AsyncClient:
    from backend.src.main import app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_the_clear_audit_row_names_the_peer_not_the_header_the_caller_wrote():
    async with client() as http:
        reply = await http.post(
            "/api/v2/control/emergency_clear",
            json={**CLEAR, "reason": f"sec3-08-{uuid.uuid4().hex}"},
            headers={"X-Client-Id": "someone-else@10.9.9.9"},
        )
    assert reply.status_code == 200
    found = [r for r in rows("control.emergency_clear") if "sec3-08-" in str(r["details"])]
    assert found, "the clear was not audited"
    assert found[0]["client_id"] == "127.0.0.1"  # the ASGI test client's TCP peer
    assert "someone-else" not in str(found[0])


async def test_the_tractor_stop_route_writes_an_audit_row_with_the_peer_address():
    async with client() as http:
        reply = await http.post(
            "/api/v2/tractor/emergency-stop", headers={"X-Client-Id": "someone-else@10.9.9.9"}
        )
    assert reply.status_code == 200
    found = rows("tractor.emergency_stop")
    assert found, "the open tractor E-stop left no audit row at all"
    assert found[0]["client_id"] == "127.0.0.1"
    assert found[0]["resource"] == "tractor_emergency_stop"
    assert "response" in found[0]["details"]


def test_a_coalesced_window_names_every_client_it_merged():
    action = f"test.sec3_13.{uuid.uuid4().hex}"
    for peer in ("192.168.1.20", "192.168.1.21", "192.168.1.20", "192.168.1.22"):
        persistence.add_audit_log_coalesced(action, client_id=peer, window_s=30.0)
    first = rows(action)
    assert [r["client_id"] for r in first] == ["192.168.1.20"]  # the first event, written at once
    summary = rows(action + ".coalesced")
    assert len(summary) == 1 and summary[0]["details"]["suppressed"] == 3
    assert summary[0]["details"]["peers"] == ["192.168.1.20", "192.168.1.21", "192.168.1.22"]
    assert summary[0]["details"]["peers_truncated"] is False


def test_the_peer_list_of_a_window_is_bounded_and_says_when_it_was_cut():
    action = f"test.sec3_13.{uuid.uuid4().hex}"
    total = AuditCoalescer.MAX_PEERS + 9
    for i in range(total):
        persistence.add_audit_log_coalesced(action, client_id=f"10.0.0.{i}", window_s=30.0)
    summary = rows(action + ".coalesced")[0]["details"]
    assert len(summary["peers"]) == AuditCoalescer.MAX_PEERS
    assert summary["peers_truncated"] is True
    assert summary["suppressed"] == total - 1


@pytest.mark.parametrize("client_id", [None, ""])
def test_events_without_a_client_add_no_peer(client_id):
    action = f"test.sec3_13.{uuid.uuid4().hex}"
    persistence.add_audit_log_coalesced(action, client_id="10.1.1.1", window_s=30.0)
    persistence.add_audit_log_coalesced(action, client_id=client_id, window_s=30.0)
    assert rows(action + ".coalesced")[0]["details"]["peers"] == ["10.1.1.1"]
