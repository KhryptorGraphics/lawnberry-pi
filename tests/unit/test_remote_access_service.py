from pathlib import Path

import pytest

from backend.src.models.remote_access_config import (
    CloudflareConfig,
    NgrokConfig,
    RemoteAccessConfig,
)
from backend.src.services.remote_access_service import (
    RemoteAccessError,
    RemoteAccessService,
)


@pytest.fixture()
def temp_paths(tmp_path: Path):
    config_path = tmp_path / "remote_access.json"
    status_path = tmp_path / "remote_access_status.json"
    return config_path, status_path


def test_cloudflare_validation_requires_tunnel(temp_paths):
    config_path, status_path = temp_paths
    service = RemoteAccessService(config_path=config_path, status_path=status_path)
    cfg = RemoteAccessConfig(provider="cloudflare", enabled=True)
    with pytest.raises(RemoteAccessError):
        service.configure(cfg, persist=False)


@pytest.mark.asyncio
async def test_ngrok_fallback_when_cloudflare_start_fails(monkeypatch, temp_paths):
    config_path, status_path = temp_paths
    service = RemoteAccessService(config_path=config_path, status_path=status_path)
    creds = config_path.parent / "cf.json"
    creds.write_text("{}")
    cfg = RemoteAccessConfig(
        provider="cloudflare",
        enabled=True,
        cloudflare=CloudflareConfig(tunnel_name="lawnberry", credentials_file=str(creds)),
        ngrok=NgrokConfig(authtoken="unit-test-token"),
    )
    service.configure(cfg, persist=False)

    calls = {"ngrok": 0}

    async def fail_cloudflare(self, *_args, **_kwargs):
        raise RemoteAccessError("boom")

    async def fake_ngrok(self, *_args, fallback: bool, **_kwargs):
        calls["ngrok"] += 1
        assert fallback is True
        self._status.provider = "ngrok"
        self._status.active = True
        self._status.health = "starting"
        self._status.message = "fallback"
        self._status.fallback_provider = "ngrok"

    monkeypatch.setattr(RemoteAccessService, "_start_cloudflare", fail_cloudflare, raising=False)
    monkeypatch.setattr(RemoteAccessService, "_start_ngrok", fake_ngrok, raising=False)

    await service.enable()
    assert calls["ngrok"] == 1
    status = RemoteAccessService.load_status_from_disk(status_path)
    assert status.provider == "ngrok"
    assert status.fallback_provider == "ngrok"
    assert status.enabled is True


@pytest.mark.asyncio
async def test_disable_updates_status_on_disk(temp_paths):
    config_path, status_path = temp_paths
    service = RemoteAccessService(config_path=config_path, status_path=status_path)
    cfg = RemoteAccessConfig(provider="ngrok", enabled=True, ngrok=NgrokConfig(authtoken="token"))
    service.configure(cfg, persist=False)

    # Mark status as active before disabling to ensure fields flip correctly
    service.get_status().active = True
    await service.disable()
    persisted = RemoteAccessService.load_status_from_disk(status_path)
    assert persisted.active is False
    assert persisted.health == "stopped"
    assert persisted.url is None


# ---- SEC3-10: what a tunnel agent inherits ------------------------------------------------------

import asyncio  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import secrets  # noqa: E402
import stat  # noqa: E402
import time  # noqa: E402

from backend.src.models.remote_access_config import CustomTunnelConfig  # noqa: E402
from backend.src.services import remote_access_service as ras  # noqa: E402
from backend.src.services.remote_access_service import tunnel_environment  # noqa: E402

BACKEND_SECRETS = (
    "LAWN_BERRY_OPERATOR_CREDENTIAL",
    "LAWN_BERRY_AUTH_SECRET",
    "NTRIP_PASSWORD",
    "MANUAL_CONTROL_PASSWORD",
    "API_KEY_SECRET",
    "JWT_SECRET",
    "GOOGLE_MAPS_API_KEY",
    "CLOUDFLARE_API_TOKEN",
)


def _backend_environment(monkeypatch, tmp_path: Path) -> dict[str, str]:
    """The backend's environment as the unit builds it: its secrets (random, in memory only),
    something no agent needs, and what an agent does need."""
    values = {name: secrets.token_hex(16) for name in BACKEND_SECRETS}
    values["LB_TEST_NOT_ALLOWED"] = secrets.token_hex(16)  # not a secret by name: still not passed
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LANG", "C.UTF-8")
    return values


def test_the_environment_is_the_allow_list_and_nothing_else(monkeypatch):
    environ = {
        "PATH": "/usr/bin",
        "HOME": "/home/x",
        "USER": "x",
        "LOGNAME": "x",
        "SHELL": "/bin/bash",
        "LANG": "C",
        "LANGUAGE": "en",
        "LC_ALL": "C",
        "TZ": "UTC",
        "TMPDIR": "/tmp",
        "XDG_RUNTIME_DIR": "/run/user/1",
        "SSL_CERT_FILE": "/etc/ssl/c.pem",
        "SSL_CERT_DIR": "/etc/ssl",
        "HTTPS_PROXY": "http://proxy:3128",
        "https_proxy": "http://proxy:3128",
        "NO_PROXY": "localhost",
        "ALL_PROXY": "socks5://p",
        "TUNNEL_TOKEN": "cloudflared-own",
        "NGROK_AUTHTOKEN": "ngrok-own",
        # everything below belongs to the backend
        "LAWN_BERRY_AUTH_SECRET": "s",
        "LAWN_BERRY_OPERATOR_CREDENTIAL": "c",
        "NTRIP_PASSWORD": "p",
        "MANUAL_CONTROL_PASSWORD": "m",
        "OPERATOR_AUTH_REQUIRED": "1",
        "PYTHONPATH": "/root",
        "LAWNBERRY_DATA_DIR": "/d",
        "SOME_OTHER": "1",
    }
    env = tunnel_environment(environ=environ)
    assert env == {
        k: v
        for k, v in environ.items()
        if k
        in (
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "SHELL",
            "LANG",
            "LANGUAGE",
            "LC_ALL",
            "TZ",
            "TMPDIR",
            "XDG_RUNTIME_DIR",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "HTTPS_PROXY",
            "https_proxy",
            "NO_PROXY",
            "ALL_PROXY",
            "TUNNEL_TOKEN",
            "NGROK_AUTHTOKEN",
        )
    }


@pytest.mark.parametrize(
    "name",
    [
        "TUNNEL_SECRET_KEY",
        "NGROK_PASSWORD",
        "XDG_CREDENTIAL",
        "LC_PASSWD",  # look like secrets
        "LAWN_BERRY_ANYTHING",
        "NTRIP_USER",
        "GOOGLE_MAPS_API_KEY",
        "JWT_SECRET",
        "API_KEY_SECRET",
        "MANUAL_CONTROL_PASSWORD",
        "CLOUDFLARE_API_TOKEN",
    ],
)
def test_a_name_that_looks_like_a_backend_secret_is_not_inherited_even_under_an_allowed_prefix(
    name,
):
    assert name not in tunnel_environment(environ={name: "v", "PATH": "/bin"})


def test_custom_env_is_the_operators_explicit_choice_and_wins():
    environ = {"PATH": "/bin", "HOME": "/h", "LAWN_BERRY_AUTH_SECRET": "backend"}
    env = tunnel_environment(
        {"HOME": "/tunnel-home", "FRP_TOKEN": "t", "LAWN_BERRY_AUTH_SECRET": "chosen"},
        environ=environ,
    )
    assert env == {
        "PATH": "/bin",
        "HOME": "/tunnel-home",
        "FRP_TOKEN": "t",
        "LAWN_BERRY_AUTH_SECRET": "chosen",
    }, "what the operator writes under custom.env is passed as written, nothing else is added"
    for bad in ({"": "x"}, {"A=B": "x"}, {"A\0B": "x"}, {"A": "x\0y"}):
        with pytest.raises(RemoteAccessError):
            tunnel_environment(bad, environ=environ)


async def _wait_for_file(path: Path, *, within: float = 10.0) -> bytes:
    end = time.monotonic() + within
    while time.monotonic() < end:
        if path.exists() and path.stat().st_size > 0:
            await asyncio.sleep(0.05)  # let the writer finish
            return path.read_bytes()
        await asyncio.sleep(0.05)
    raise AssertionError(f"the tunnel child wrote nothing to {path} in {within} s")


def _assert_clean(dump: bytes, values: dict[str, str]) -> None:
    names = {entry.split(b"=", 1)[0].decode() for entry in dump.split(b"\0") if entry}
    text = dump.decode(errors="replace")
    for name, value in values.items():
        assert name not in names, f"{name} reached the tunnel child"
        assert value not in text, f"the value of {name} reached the tunnel child"
    assert {"PATH", "HOME", "LANG"} <= names, "the child lost what it needs to run"


async def test_a_custom_tunnel_child_gets_the_allow_list_and_custom_env_only(monkeypatch, tmp_path):
    """The REAL child (``bash -lc`` through the real ``_launch_process``): none of the backend's
    secrets, nor any other name outside the allow-list, reaches it; ``custom.env`` does."""
    values = _backend_environment(monkeypatch, tmp_path)
    out = tmp_path / "custom-env.bin"
    service = RemoteAccessService(config_path=tmp_path / "c.json", status_path=tmp_path / "s.json")
    wanted = secrets.token_hex(8)
    await service._start_custom(
        '/usr/bin/env -0 > "$LB_OUT"', {"LB_OUT": str(out), "FRP_CHOSEN": wanted}
    )
    try:
        dump = await _wait_for_file(out)
    finally:
        await service.disable()
    _assert_clean(dump, values)
    names = {entry.split(b"=", 1)[0].decode() for entry in dump.split(b"\0") if entry}
    assert {"LB_OUT", "FRP_CHOSEN"} <= names and wanted in dump.decode()


@pytest.mark.parametrize("agent", ["cloudflared", "ngrok"])
async def test_the_built_in_agents_get_the_allow_list_only(monkeypatch, tmp_path, agent):
    values = _backend_environment(monkeypatch, tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    prefix = "TUNNEL" if agent == "cloudflared" else "NGROK"
    out = tmp_path / f"{agent}-env.bin"
    script = bin_dir / agent
    script.write_text(f'#!/bin/sh\nexec /usr/bin/env -0 > "${prefix}_TEST_OUT"\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv(f"{prefix}_TEST_OUT", str(out))  # the agent's own prefix: inherited
    monkeypatch.setattr(ras, "NGROK_CONFIG_DIR", tmp_path / "ngrok")
    monkeypatch.setattr(ras, "NGROK_CONFIG_FILE", tmp_path / "ngrok" / "lawnberry.yml")
    service = RemoteAccessService(config_path=tmp_path / "c.json", status_path=tmp_path / "s.json")
    try:
        if agent == "cloudflared":
            await service._start_cloudflare(CloudflareConfig(tunnel_name="lawnberry"))
        else:
            await service._start_ngrok(NgrokConfig(authtoken="unit-test-token"), fallback=False)
        dump = await _wait_for_file(out)
    finally:
        await service.disable()
    _assert_clean(dump, values)
    assert f"{prefix}_TEST_OUT".encode() in dump, "the agent's own variables are inherited"


async def test_a_token_in_a_custom_command_is_not_logged(monkeypatch, tmp_path, caplog):
    token = secrets.token_urlsafe(24)
    seen: list[tuple[tuple, dict]] = []

    class Child:
        stdout = stderr = None
        returncode = None
        pid = 1

        def __init__(self):
            self.stopped = asyncio.Event()

        async def wait(self):
            await self.stopped.wait()
            return 0

        def terminate(self):
            self.stopped.set()

        def kill(self):
            self.stopped.set()

    async def factory(*args, **kwargs):
        seen.append((args, kwargs))
        return Child()

    service = RemoteAccessService(
        config_path=tmp_path / "c.json", status_path=tmp_path / "s.json", process_factory=factory
    )
    with caplog.at_level(logging.DEBUG):
        await service._start_custom(f"frpc --token {token} --server tunnel.example", {})
        await service.disable()
    assert token not in caplog.text and "frpc" not in caplog.text
    assert "bash" in caplog.text, "the program is still named"
    assert seen[0][0][-1].endswith(token + " --server tunnel.example"), (
        "the command itself is intact"
    )
    assert set(seen[0][1]["env"]) <= set(os.environ), "never an env that inherits everything"


async def test_a_launch_without_an_env_still_gets_the_allow_list(monkeypatch, tmp_path):
    values = _backend_environment(monkeypatch, tmp_path)
    seen: dict = {}

    class Child:
        stdout = stderr = None
        returncode = None
        pid = 1

        def __init__(self):
            self.stopped = asyncio.Event()

        async def wait(self):
            await self.stopped.wait()
            return 0

        def terminate(self):
            self.stopped.set()

        def kill(self):
            self.stopped.set()

    async def factory(*args, **kwargs):
        seen.update(kwargs)
        return Child()

    service = RemoteAccessService(
        config_path=tmp_path / "c.json", status_path=tmp_path / "s.json", process_factory=factory
    )
    await service._launch_process(ras.RemoteAccessProvider.CUSTOM, ["/bin/true"], None)
    await service.disable()
    assert seen["env"] is not None
    assert not (set(values) & set(seen["env"])), "env=None would hand the child everything"


def test_the_custom_config_model_still_carries_its_env_map():
    assert CustomTunnelConfig(command="x", env={"A": "b"}).env == {"A": "b"}
