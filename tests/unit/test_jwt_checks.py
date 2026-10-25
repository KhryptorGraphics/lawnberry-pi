"""The checks that decide whether a bearer is a login: signature, expiry, scheme (SEC3-15).

The reviewer's mutants S5 (``verify_signature=False``), S6 (``verify_exp=False``) and U20 (any
``Authorization`` scheme is a bearer) survived every committed test: nothing presented a forged,
an expired or a mis-labelled token to the path that authenticates. Each test below presents one
to the REAL checks and fails if that check is removed.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import jwt
import pytest
from fastapi.testclient import TestClient

from backend.src.api.routers import auth as auth_router
from backend.src.api.routers.auth import _extract_bearer_token
from backend.src.main import app
from backend.src.services.auth_service import AuthService

CRED = "correct-horse-battery-staple"
PEER = "203.0.113.9"


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.delenv("LAWN_BERRY_SIM_OPEN_AUTH", raising=False)
    monkeypatch.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", CRED)
    svc = AuthService()
    monkeypatch.setattr(auth_router, "primary_auth_service", svc)
    monkeypatch.setattr("backend.src.api.deps.primary_auth_service", svc)
    monkeypatch.setattr("backend.src.middleware.rate_limiting.primary_auth_service", svc)
    return svc


def claims(session_id: str, *, exp_in: float = 3600.0) -> dict:
    return {
        "sub": "operator",
        "sid": session_id,
        "role": "operator",
        "exp": int(time.time() + exp_in),
    }


async def test_a_token_signed_with_another_key_is_not_a_login(service):
    live = (await service.authenticate(CRED, client_ip=PEER)).session
    genuine = jwt.encode(claims(live.session_id), service.jwt_manager.secret_key, "HS256")
    forged = jwt.encode(claims(live.session_id), "not-the-signing-key" * 3, "HS256")
    assert (await service.verify_token(genuine)).session_id == live.session_id
    assert await service.verify_token(forged) is None
    assert await service.verify_token(forged, quiet=True) is None
    assert service.peek_session(forged) is None
    # the signature is the only thing that differs: flip one character of the genuine one
    flipped = genuine[:-2] + ("A" if genuine[-2] != "A" else "B") + genuine[-1]
    assert await service.verify_token(flipped) is None
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(claims(live.session_id)).encode()).rstrip(b"=")
    assert await service.verify_token(f"{header}.{body.decode()}.") is None  # alg none


async def test_an_expired_token_of_a_live_session_is_not_a_login(service):
    """A refresh extends the SESSION but leaves the old token to die at its own ``exp``."""
    live = (await service.authenticate(CRED, client_ip=PEER)).session
    key = service.jwt_manager.secret_key
    expired = jwt.encode(claims(live.session_id, exp_in=-30.0), key, "HS256")
    fresh = jwt.encode(claims(live.session_id, exp_in=30.0), key, "HS256")
    assert live.session_id in service.active_sessions and not live.is_expired()
    assert await service.verify_token(fresh) is not None
    assert await service.verify_token(expired) is None
    assert await service.verify_token(expired, quiet=True) is None
    assert service.peek_session(expired) is None
    assert live.session_id in service.active_sessions, "a dead token does not end the session"


async def test_the_http_routes_refuse_a_forged_and_an_expired_bearer(service):
    live = (await service.authenticate(CRED, client_ip=PEER)).session
    key = service.jwt_manager.secret_key
    client = TestClient(app, client=(PEER, 50000))
    for label, token in (
        ("forged", jwt.encode(claims(live.session_id), "x" * 40, "HS256")),
        ("expired", jwt.encode(claims(live.session_id, exp_in=-30.0), key, "HS256")),
    ):
        headers = {"Authorization": f"Bearer {token}"}
        assert client.get("/api/v2/auth/profile", headers=headers).status_code == 401, label
        assert client.post("/api/v2/auth/ws-ticket", headers=headers, json={}).status_code == 401
    good = {"Authorization": "Bearer " + jwt.encode(claims(live.session_id), key, "HS256")}
    assert client.get("/api/v2/auth/profile", headers=good).status_code == 200


@pytest.mark.parametrize("scheme", ["Basic", "Digest", "Token", "ApiKey", "bearer-x", ""])
def test_only_the_bearer_scheme_is_a_bearer(scheme):
    assert _extract_bearer_token(f"{scheme} some.jwt.value") is None
    assert _extract_bearer_token("Bearer some.jwt.value") == "some.jwt.value"
    assert _extract_bearer_token("bEaReR some.jwt.value") == "some.jwt.value"
    assert _extract_bearer_token("Bearer ") is None and _extract_bearer_token(None) is None


async def test_a_valid_token_under_another_scheme_authenticates_nothing(service):
    live = await service.authenticate(CRED, client_ip=PEER)
    client = TestClient(app, client=(PEER, 50000))
    for scheme in ("Basic", "Token", "ApiKey"):
        headers = {"Authorization": f"{scheme} {live.token}"}
        assert client.get("/api/v2/auth/profile", headers=headers).status_code == 401, scheme
        assert client.post("/api/v2/auth/ws-ticket", headers=headers, json={}).status_code == 401
    assert (
        client.get(
            "/api/v2/auth/profile", headers={"Authorization": f"Bearer {live.token}"}
        ).status_code
        == 200
    )


# ---- the same checks against the REAL PyJWT (SEC3-15: S5, S6)
#
# Under pytest ``jwt`` is ``backend/src/compat_stubs/jwt.py`` (tests/conftest.py puts the stub
# directory first on ``sys.path`` when another optional package is missing), and the stub ignores
# every key of ``options`` but ``require``: the reviewer's mutants ``"verify_signature": False`` and
# ``"verify_exp": False`` are no-ops there and can never fail an in-process test. The product runs
# on PyJWT, so the checks are exercised in a clean interpreter that imports it.

ROOT = Path(__file__).resolve().parents[2]
REAL_JWT_SCRIPT = textwrap.dedent(
    """
    import asyncio, base64, json, os, time
    import jwt
    assert "compat_stubs" not in jwt.__file__, jwt.__file__
    from backend.src.services.auth_service import AuthService

    async def main():
        service = AuthService()
        live = (await service.authenticate(os.environ["LB_CRED"], client_ip="203.0.113.9")).session
        key = service.jwt_manager.secret_key
        def claims(exp_in):
            return {"sub": "operator", "sid": live.session_id, "exp": int(time.time() + exp_in)}
        genuine = jwt.encode(claims(300), key, "HS256")
        head = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').rstrip(b"=").decode()
        body = base64.urlsafe_b64encode(json.dumps(claims(300)).encode()).rstrip(b"=").decode()
        cases = {
            "genuine": genuine,
            "wrong_key": jwt.encode(claims(300), "k" * 40, "HS256"),
            "flipped_signature": genuine[:-3] + ("AAA" if not genuine.endswith("AAA") else "BBB"),
            "alg_none": f"{head}.{body}.",
            "hs512_with_the_right_key": jwt.encode(claims(300), key, "HS512"),
            "expired_token_of_a_live_session": jwt.encode(claims(-60), key, "HS256"),
            "no_sid": jwt.encode({"sub": "o", "exp": int(time.time() + 300)}, key, "HS256"),
        }
        out = {}
        for name, token in cases.items():
            out[name] = [
                (await service.verify_token(token)) is not None,
                (await service.verify_token(token, quiet=True)) is not None,
                service.peek_session(token) is not None,
            ]
        print(json.dumps(out))

    asyncio.run(main())
    """
)


def test_the_real_pyjwt_refuses_forged_unsigned_wrong_algorithm_and_expired_tokens():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST", "LAWN_BERRY"))}
    env.update(
        PYTHONPATH=str(ROOT),
        SIM_MODE="1",
        LB_CRED=secrets.token_urlsafe(18),
        LAWN_BERRY_AUTH_SECRET=secrets.token_hex(32),
    )
    env["LAWN_BERRY_OPERATOR_CREDENTIAL"] = env["LB_CRED"]
    done = subprocess.run(
        [sys.executable, "-c", REAL_JWT_SCRIPT],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    verdicts = json.loads(done.stdout.strip().splitlines()[-1])
    assert verdicts.pop("genuine") == [True, True, True]
    assert verdicts == {name: [False, False, False] for name in verdicts}, verdicts
    assert set(verdicts) == {
        "wrong_key",
        "flipped_signature",
        "alg_none",
        "hs512_with_the_right_key",
        "expired_token_of_a_live_session",
        "no_sid",
    }
