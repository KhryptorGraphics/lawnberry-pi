"""The settings routes never return a secret, and the operator-only reads need a login (SEC20-01).

Before: ``GET /settings/remote-access`` and ``GET /settings/maps`` (and ``GET /settings`` through
its ``categories.maps``) answered ANY caller with the stored Cloudflare tunnel token, ngrok
authtoken and Google Maps key. Now a secret is write-only: a value stored under a secret name is
in no response, the two secret-bearing stores are operator-only reads that say which secrets are
set (and the last four characters of a long one), and the PUT semantics keep a form that echoes
its fields back from wiping or corrupting them.
"""

from __future__ import annotations

import json
import os
import stat

import httpx
import pytest
import pytest_asyncio

from backend.src.main import app

CF_TOKEN = "eyJhIjoiQ0YtVFVOTkVMLVRPS0VOLVNFQ1JFVCJ9-CLOUDFLARE-TUNNEL-TOKEN"
NGROK_TOKEN = "2NGROK-AUTHTOKEN-SECRET-1234567890abcdef"
MAPS_KEY = "AIzaSyD-MAPS-KEY-SECRET-0123456789abcdef"
SECRETS = (CF_TOKEN, NGROK_TOKEN, MAPS_KEY)

REMOTE = "/api/v2/settings/remote-access"
MAPS = "/api/v2/settings/maps"
SECURITY = "/api/v2/settings/security"
SYSTEM = "/api/v2/settings/system"
PROFILE = "/api/v2/settings"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LAWNBERRY_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def enforced(monkeypatch):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")


@pytest_asyncio.fixture
async def client(data_dir):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


@pytest_asyncio.fixture
async def operator(client, operator_credential):
    login = await client.post("/api/v2/auth/login", json={"credential": operator_credential})
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


async def _store_secrets(client, operator):
    remote = await client.put(
        REMOTE,
        headers=operator,
        json={"method": "ngrok", "cloudflare_token": CF_TOKEN, "ngrok_token": NGROK_TOKEN},
    )
    maps = await client.put(
        MAPS, headers=operator, json={"provider": "google", "google_api_key": MAPS_KEY}
    )
    assert remote.status_code == 200 and maps.status_code == 200
    return remote, maps


def _stored(data_dir, name):
    return json.loads((data_dir / name).read_text())


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


# ---- reads -------------------------------------------------------------------------------------


async def test_a_stored_secret_is_never_in_any_response_but_is_kept_on_disk(
    client, operator, data_dir
):
    remote_put, maps_put = await _store_secrets(client, operator)
    responses = [
        remote_put,
        maps_put,
        await client.get(REMOTE, headers=operator),
        await client.get(MAPS, headers=operator),
        await client.get(PROFILE),
        await client.get(PROFILE, headers=operator),
        await client.get(SYSTEM),
        await client.get(SECURITY),
    ]
    for response in responses:
        assert response.status_code == 200, (response.request.url, response.text)
        for secret in SECRETS:
            assert secret not in response.text, (str(response.request.url), secret)
            assert secret not in str(dict(response.headers))

    # "writes keep working": the remote-access daemon still gets the real values from the store.
    assert _stored(data_dir, "settings_remote_access_v2.json")["cloudflare_token"] == CF_TOKEN
    assert _stored(data_dir, "settings_remote_access_v2.json")["ngrok_token"] == NGROK_TOKEN
    assert _stored(data_dir, "settings_maps_v2.json")["google_api_key"] == MAPS_KEY


async def test_the_operator_view_says_which_secrets_are_set(client, operator):
    await _store_secrets(client, operator)

    remote = (await client.get(REMOTE, headers=operator)).json()
    assert remote["method"] == "ngrok"
    assert remote["cloudflare_token_set"] is True and remote["ngrok_token_set"] is True
    assert remote["cloudflare_token_last4"] == CF_TOKEN[-4:]
    assert remote["ngrok_token_last4"] == NGROK_TOKEN[-4:]
    assert "cloudflare_token" not in remote and "ngrok_token" not in remote

    maps = (await client.get(MAPS, headers=operator)).json()
    assert maps["provider"] == "google"
    assert maps["google_api_key_set"] is True
    assert maps["google_api_key_last4"] == MAPS_KEY[-4:]
    assert "google_api_key" not in maps


async def test_a_short_secret_is_reported_as_set_without_any_of_its_characters(client, operator):
    await client.put(MAPS, headers=operator, json={"google_api_key": "short-key"})
    maps = (await client.get(MAPS, headers=operator)).json()
    assert maps["google_api_key_set"] is True
    assert "google_api_key_last4" not in maps and "short-key" not in json.dumps(maps)


@pytest.mark.parametrize("path", [REMOTE, MAPS])
async def test_the_secret_bearing_reads_need_the_operators_login(client, operator, path):
    await _store_secrets(client, operator)
    anonymous = await client.get(path)
    forged = await client.get(path, headers={"Authorization": "Bearer not-a-real-token"})
    for response in (anonymous, forged):
        assert response.status_code == 401, (path, response.status_code, response.text)
        assert response.json()["detail"] == "Operator authentication required"
        assert not any(secret in response.text for secret in SECRETS)
    assert (await client.get(path, headers=operator)).status_code == 200


async def test_the_open_reads_say_nothing_at_all_about_secrets(client, operator):
    await _store_secrets(client, operator)
    await client.put(
        SECURITY,
        headers=operator,
        json={"level": "password_totp", "totp_secret": "JBSWY3DPEHPK3PXP-CANARY", "keep": 1},
    )
    await client.put(
        SYSTEM,
        headers=operator,
        json={
            "webhook_url": "https://hooks.example/T000/B000/CANARY",
            "ui": {"slack_token": "x" * 20},
        },
    )

    for path in (PROFILE, SYSTEM, SECURITY):
        response = await client.get(path)
        assert response.status_code == 200, (path, response.text)
        body = response.json()
        assert not any(
            "token" in str(k).lower() or "key" in str(k).lower() for k in _walk_keys(body)
        )
        assert "CANARY" not in response.text
        assert not any(str(k).endswith(("_set", "_last4")) for k in _walk_keys(body)), path

    categories = (await client.get(PROFILE)).json()["categories"]
    assert categories["maps"]["provider"] == "google"  # the non-secret part is still there
    assert (await client.get(SECURITY)).json()["level"] == "password_totp"


# ---- writes ------------------------------------------------------------------------------------


async def test_an_empty_secret_field_keeps_the_stored_value_and_null_removes_it(
    client, operator, data_dir
):
    await _store_secrets(client, operator)
    path = data_dir / "settings_remote_access_v2.json"

    # an untouched password input arrives empty (or blank): the stored token stays
    for empty in ("", "   "):
        kept = await client.put(
            REMOTE, headers=operator, json={"method": "cloudflare", "ngrok_token": empty}
        )
        assert kept.status_code == 200
        assert json.loads(path.read_text())["ngrok_token"] == NGROK_TOKEN
    assert (await client.get(REMOTE, headers=operator)).json()["method"] == "cloudflare"

    replaced = await client.put(REMOTE, headers=operator, json={"ngrok_token": "brand-new-ngrok"})
    assert replaced.json()["ngrok_token_set"] is True
    assert json.loads(path.read_text())["ngrok_token"] == "brand-new-ngrok"

    removed = await client.put(REMOTE, headers=operator, json={"ngrok_token": None})
    assert removed.json()["ngrok_token_set"] is False
    stored = json.loads(path.read_text())
    assert stored["ngrok_token"] is None and stored["cloudflare_token"] == CF_TOKEN
    assert (await client.get(REMOTE, headers=operator)).json()["ngrok_token_set"] is False


async def test_a_client_that_echoes_the_read_view_back_corrupts_nothing(client, operator, data_dir):
    await _store_secrets(client, operator)
    view = (await client.get(REMOTE, headers=operator)).json()
    assert any(key.endswith("_set") for key in view)

    echoed = await client.put(REMOTE, headers=operator, json=view)
    assert echoed.status_code == 200
    stored = _stored(data_dir, "settings_remote_access_v2.json")
    assert stored["cloudflare_token"] == CF_TOKEN and stored["ngrok_token"] == NGROK_TOKEN
    assert not any(key.endswith(("_set", "_last4")) for key in stored), stored

    maps_view = (await client.get(MAPS, headers=operator)).json()
    assert (await client.put(MAPS, headers=operator, json=maps_view)).status_code == 200
    assert _stored(data_dir, "settings_maps_v2.json")["google_api_key"] == MAPS_KEY


async def test_secrets_in_the_profile_stores_are_write_only_too(client, operator, data_dir):
    put = await client.put(
        SYSTEM, headers=operator, json={"integration": {"x": 1}, "webhook_url": "https://x/CANARY"}
    )
    assert put.status_code == 200 and "CANARY" not in put.text
    assert _stored(data_dir, "settings_profile_v2.json")["webhook_url"] == "https://x/CANARY"

    keep = await client.put(SYSTEM, headers=operator, json={"webhook_url": ""})
    assert keep.status_code == 200
    assert _stored(data_dir, "settings_profile_v2.json")["webhook_url"] == "https://x/CANARY"


async def test_the_store_files_are_private_and_written_atomically(client, operator, data_dir):
    legacy = data_dir / "settings_maps_v2.json"
    legacy.write_text(json.dumps({"provider": "osm", "google_api_key": MAPS_KEY}))
    legacy.chmod(0o644)  # what the old code left behind: a world-readable file of secrets

    await _store_secrets(client, operator)

    for name in ("settings_maps_v2.json", "settings_remote_access_v2.json"):
        mode = stat.S_IMODE(os.stat(data_dir / name).st_mode)
        assert mode == 0o600, (name, oct(mode))
    assert not list(data_dir.glob("*.tmp")), "an atomic write leaves no temp file behind"
    assert _stored(data_dir, "settings_maps_v2.json")["google_api_key"] == MAPS_KEY


async def test_a_bad_provider_is_still_rejected_before_anything_is_stored(client, operator):
    bad = await client.put(MAPS, headers=operator, json={"provider": "nope", "google_api_key": "x"})
    assert bad.status_code == 422
    assert (await client.get(MAPS, headers=operator)).json().get("google_api_key_set") is not True


async def test_writes_still_need_the_login(client):
    for path in (REMOTE, MAPS, SECURITY, SYSTEM):
        denied = await client.put(path, json={"google_api_key": MAPS_KEY})
        assert denied.status_code == 401, path
