"""The shipped units, nginx templates and UI server agree on who forwards client addresses.

SEC20-07: the unit used to trust the whole loopback (default ``127.0.0.0/8,::1``) and left uvicorn's
own proxy handling on (``FORWARDED_ALLOW_IPS``, any string as the client). Now the backend trusts
exactly the two forwarders the repository itself configures, each on its own source address and in
its own shape, and uvicorn's handling is off. These tests read the real files, so a drifting
address in one of them fails here instead of silently collapsing every browser into one bucket.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.src.core.peer_address import parse_trusted_proxies, peer_key, resolve_client_host

REPO = Path(__file__).resolve().parents[2]
BACKEND_UNIT = (REPO / "systemd" / "lawnberry-backend.service").read_text()
FRONTEND_UNIT = (REPO / "systemd" / "lawnberry-frontend.service").read_text()
NGINX_FILES = {
    "config/nginx.conf": (REPO / "config" / "nginx.conf").read_text(),
    "scripts/setup_https.sh": (REPO / "scripts" / "setup_https.sh").read_text(),
}


def _env(unit: str, name: str) -> str:
    match = re.search(rf"^Environment={re.escape(name)}=(.*)$", unit, re.MULTILINE)
    assert match, f"{name} is not set by the unit"
    return match.group(1).strip()


def _exec_start(unit: str) -> str:
    joined = re.sub(r"\\\n\s*", " ", unit)
    return re.search(r"^ExecStart=(.*)$", joined, re.MULTILINE).group(1)


def test_the_backend_unit_trusts_exactly_the_ui_server_and_the_repo_nginx():
    proxies = parse_trusted_proxies(_env(BACKEND_UNIT, "LAWN_BERRY_TRUSTED_PROXIES"))
    assert [str(p) for p in proxies] == ["127.0.0.2/32=single", "127.0.0.3/32=last"]


def test_no_loopback_forwarder_other_than_the_declared_ones_can_pick_its_identity():
    raw = _env(BACKEND_UNIT, "LAWN_BERRY_TRUSTED_PROXIES")
    proxies = parse_trusted_proxies(raw)
    for peer in ("127.0.0.1", "127.0.0.4", "::1"):  # ssh -R, socat, a tunnel agent, ...
        keys = {
            peer_key(resolve_client_host(peer, [f"198.51.100.{n}"], proxies)) for n in range(12)
        }
        assert len(keys) == 1, peer
    # the UI server hop gives each browser its own key
    assert (
        len(
            {
                peer_key(resolve_client_host("127.0.0.2", [f"198.51.100.{n}"], proxies))
                for n in range(12)
            }
        )
        == 12
    )


def test_the_ui_unit_connects_from_the_address_the_backend_trusts_as_single():
    source = _env(FRONTEND_UNIT, "BACKEND_SOURCE_ADDRESS")
    backend = {
        str(p.network.network_address): p.shape
        for p in parse_trusted_proxies(_env(BACKEND_UNIT, "LAWN_BERRY_TRUSTED_PROXIES"))
    }
    assert backend[source] == "single"


def test_the_ui_unit_trusts_the_nginx_address_the_templates_bind_to():
    declared = parse_trusted_proxies(_env(FRONTEND_UNIT, "UI_TRUSTED_PROXIES"))
    assert [str(p) for p in declared] == ["127.0.0.3/32=last"]
    for name, text in NGINX_FILES.items():
        upstreams = re.findall(r"proxy_pass http://127\.0\.0\.1:", text)
        binds = re.findall(r"proxy_bind 127\.0\.0\.3;", text)
        assert upstreams and len(binds) == len(upstreams), (
            f"{name}: every upstream needs proxy_bind"
        )


def test_every_proxying_nginx_location_sends_the_client_address_including_the_websocket_ones():
    """SEC20-02 repro 4: ``/ws`` and ``/api/v2/ws/`` sent no X-Forwarded-For at all."""
    for name, text in NGINX_FILES.items():
        blocks = [b for b in re.split(r"\n\s*location ", text)[1:] if "proxy_pass" in b]
        assert len(blocks) >= 4, name
        for block in blocks:
            assert re.search(
                r"proxy_set_header X-Forwarded-For \\?\$proxy_add_x_forwarded_for;", block
            ), f"{name}: location {block.splitlines()[0]} sends no X-Forwarded-For"


def test_uvicorn_does_not_decide_the_client_address():
    command = _exec_start(BACKEND_UNIT)
    assert "--no-proxy-headers" in command
    assert "--proxy-headers" not in command.replace("--no-proxy-headers", "")
    assert not re.search(r"^Environment=FORWARDED_ALLOW_IPS=", BACKEND_UNIT, re.MULTILINE)


def test_the_connection_limits_are_in_the_exec_start_and_concurrency_limit_is_not():
    command = _exec_start(BACKEND_UNIT)
    assert "--http backend.src.core.uvicorn_protocol:LawnBerryHttpProtocol" in command
    assert "--ws-max-size 65536" in command
    assert "--timeout-keep-alive 5" in command
    assert "--backlog 128" in command
    assert "--limit-concurrency" not in command, "it answers the E-stop POST with 503"


def test_the_exec_start_flags_load_in_the_installed_uvicorn():
    """A bad flag or protocol path would stop the backend (and its E-stop) at boot."""
    import uvicorn

    from backend.src.core.uvicorn_protocol import LawnBerryHttpProtocol

    config = uvicorn.Config(
        "backend.src.main:app",
        http="backend.src.core.uvicorn_protocol:LawnBerryHttpProtocol",
        ws_max_size=65536,
        timeout_keep_alive=5,
        backlog=128,
        proxy_headers=False,
        lifespan="off",
    )
    config.load()
    assert config.http_protocol_class is LawnBerryHttpProtocol
    assert config.proxy_headers is False
    assert config.limit_concurrency is None


def test_the_declared_uvicorn_floor_can_load_the_guarded_protocol():
    """Before uvicorn 0.36 ``--http`` is a Choice of auto|h11|httptools: an import string there
    makes the backend (and its E-stop) refuse to start, so the pin must not allow those."""
    floor = re.search(
        r'"uvicorn\[standard\]>=(\d+)\.(\d+)\.(\d+)"', (REPO / "pyproject.toml").read_text()
    )
    assert floor, "pyproject.toml must pin uvicorn[standard] with a lower bound"
    assert tuple(int(part) for part in floor.groups()) >= (0, 36, 0)
    assert re.search(r"--http \S+:\S+", _exec_start(BACKEND_UNIT))


@pytest.mark.parametrize("unit", [BACKEND_UNIT, FRONTEND_UNIT], ids=["backend", "frontend"])
def test_the_descriptor_budget_is_explicit(unit):
    assert int(re.search(r"^LimitNOFILE=(\d+)$", unit, re.MULTILINE).group(1)) >= 4096
