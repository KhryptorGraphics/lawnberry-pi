"""Integration test for manual control gated by MFA authentication."""

import httpx
import pytest


@pytest.mark.asyncio
async def test_manual_control_requires_authentication(monkeypatch, operator_credential):
    """/control/drive needs the operator's bearer token: 401 without it, accepted with it."""
    from backend.src.main import app

    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        drive_command = {"mode": "arcade", "throttle": 0.5, "turn": 0.2}

        response = await client.post("/api/v2/control/drive", json=drive_command)
        assert response.status_code == 401, "Should require authentication"

        login = await client.post("/api/v2/auth/login", json={"credential": operator_credential})
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.post("/api/v2/control/drive", json=drive_command, headers=headers)
        assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_manual_control_with_valid_mfa(operator_credential):
    """Test that manual control works with valid MFA token."""
    from backend.src.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # First, authenticate with MFA
        auth_payload = {"credential": operator_credential}
        auth_response = await client.post("/api/v2/auth/login", json=auth_payload)
        assert auth_response.status_code == 200

        auth_data = auth_response.json()
        token = auth_data.get("access_token") or auth_data.get("token")

        # Use token for manual control
        headers = {"Authorization": f"Bearer {token}"}

        drive_command = {"mode": "arcade", "throttle": 0.3, "turn": -0.1}

        # This test currently passes because auth isn't enforced yet
        # When implemented, this should verify proper auth flow
        response = await client.post("/api/v2/control/drive", json=drive_command, headers=headers)

        # Should succeed with valid auth
        assert response.status_code in [200, 202], "Should accept command with valid auth"


@pytest.mark.asyncio
async def test_blade_control_requires_mfa(monkeypatch, operator_credential):
    """The blade needs the bearer token (401 without it); an ON also needs the manual session.

    The manual-control session is the additional factor of the blade and drive routes
    (``api/routers/auth.py``): a blade ON with the bearer alone is 401 "Manual control session
    required" (fail closed, audit HW-02: a body without a ``session_id`` used to be answered
    without touching the blade, so it never needed the factor). A blade OFF is the safe direction
    and needs only the bearer.
    """
    from backend.src.main import app

    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        blade_command = {"active": True}

        response = await client.post("/api/v2/control/blade", json=blade_command)
        assert response.status_code == 401, "Blade control should require auth"

        login = await client.post("/api/v2/auth/login", json={"credential": operator_credential})
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        response = await client.post("/api/v2/control/blade", json=blade_command, headers=headers)
        assert response.status_code == 401, response.text
        assert "Manual control session required" in response.text

        off = await client.post("/api/v2/control/blade", json={"active": False}, headers=headers)
        assert off.status_code != 401, off.text  # an OFF needs no manual session

        unlock = await client.post(
            "/api/v2/control/manual-unlock",
            json={"method": "password", "password": operator_credential},
            headers=headers,
        )
        assert unlock.status_code == 200, unlock.text
        with_session = {**blade_command, "session_id": unlock.json()["session_id"]}
        response = await client.post("/api/v2/control/blade", json=with_session, headers=headers)
        assert response.status_code != 401, response.text


@pytest.mark.asyncio
async def test_emergency_stop_always_accessible():
    """Test that emergency stop is accessible even without full auth."""
    from backend.src.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Emergency stop should work without authentication for safety
        response = await client.post("/api/v2/control/emergency-stop")

        # Should always succeed for safety reasons
        assert response.status_code == 200

        data = response.json()
        assert "emergency_stop_active" in data
        assert data["emergency_stop_active"] is True


@pytest.mark.asyncio
async def test_mfa_token_expiration(operator_credential):
    """Test that MFA tokens expire and require refresh."""
    from backend.src.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Get a token
        auth_payload = {"credential": operator_credential}
        auth_response = await client.post("/api/v2/auth/login", json=auth_payload)
        assert auth_response.status_code == 200

        auth_data = auth_response.json()
        expires_in = auth_data.get("expires_in", 3600)

        # Verify token has reasonable expiration
        assert expires_in > 0, "Token should have positive expiration"
        assert expires_in <= 86400, "Token should not be valid for more than 24 hours"

        # Refresh needs the bearer token (a refresh without one is 401), and returns a new one
        assert (await client.post("/api/v2/auth/refresh", json={})).status_code == 401
        headers = {"Authorization": f"Bearer {auth_data['access_token']}"}
        refresh_response = await client.post("/api/v2/auth/refresh", json={}, headers=headers)
        assert refresh_response.status_code == 200, refresh_response.text
        refresh_data = refresh_response.json()
        assert refresh_data["access_token"] and refresh_data["expires_in"] > 0


@pytest.mark.asyncio
async def test_concurrent_manual_control_sessions(operator_credential):
    """Test behavior with multiple concurrent manual control sessions."""
    from backend.src.main import app

    transport = httpx.ASGITransport(app=app)

    # Create two client sessions
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client1:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client2:
            # Both authenticate
            auth_payload = {"credential": operator_credential}

            auth1 = await client1.post("/api/v2/auth/login", json=auth_payload)
            auth2 = await client2.post("/api/v2/auth/login", json=auth_payload)

            assert auth1.status_code == 200
            assert auth2.status_code == 200

            # Both try to send control commands simultaneously
            drive_cmd = {"mode": "arcade", "throttle": 0.2, "turn": 0.0}

            response1 = await client1.post("/api/v2/control/drive", json=drive_cmd)
            response2 = await client2.post("/api/v2/control/drive", json=drive_cmd)

            # Both should be accepted (or system should handle conflicts gracefully)
            assert response1.status_code in [200, 202, 409]  # 409 = conflict
            assert response2.status_code in [200, 202, 409]
