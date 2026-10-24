"""Thor link wiring: the config flag, the fail-closed start-up rules, and the real ``lifespan``.

The link is OFF unless ``config/thor_link.yaml`` says so, and once asked for it starts only
if everything it relies on is present: the tractor platform with its safety monitor, a real
key (no public-key fallback outside SIM_MODE), and outside SIM_MODE a geofence. Every refusal
is a ``ThorLinkStartError`` that leaves the backend up, with no socket bound and no task left.
"""

from __future__ import annotations

import asyncio
import socket
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backend.src.models.navigation_state import NavigationState
from backend.src.models.thor_link import SIM_LINK_KEY
from backend.src.models.tractor_control import EngineState
from backend.src.services.thor_link_wiring import (
    CONFIG_FILE,
    MAX_UPLINK_HZ,
    FenceSettings,
    ThorLinkConfigError,
    ThorLinkSettings,
    ThorLinkStartError,
    build_fence,
    load_thor_link_settings,
    start_thor_link,
    stop_thor_link,
)
from backend.src.services.thor_plan_follower import GnssSnapshot
from backend.src.services.tractor_service import TractorControlService

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
GOOD_KEY = "6d" * 40
SQUARE = [(45.0, -93.0), (45.0, -92.999), (45.0009, -92.999), (45.0009, -93.0)]


def _fence() -> FenceSettings:
    return FenceSettings(boundary=SQUARE)


def _settings(**kw) -> ThorLinkSettings:
    base = {
        "enabled": True,
        "thor_host": "127.0.0.1",
        "thor_port": 47999,
        "bind_host": "127.0.0.1",
        "bind_port": 0,
        "geofence": _fence(),
    }
    return ThorLinkSettings(**{**base, **kw})


def _gnss() -> GnssSnapshot:
    return GnssSnapshot(None, None, "none", None)


NAV = SimpleNamespace(navigation_state=NavigationState())  # start_thor_link only stores it


async def _start(tractor, settings, **kw):
    return await start_thor_link(
        tractor, safety_monitor_running=True, gnss=_gnss, nav=NAV, settings=settings, **kw
    )


@pytest.fixture
def tractor():
    t = TractorControlService(config={"enabled": True})
    t.authorize()
    t.state.engine = EngineState.RUNNING
    return t


@pytest.fixture
def no_key(monkeypatch, tmp_path):
    """No key anywhere: not in the environment and not in a secrets file."""
    monkeypatch.delenv("THOR_LINK_KEY", raising=False)
    monkeypatch.setenv("LAWN_SECRETS_PATH", str(tmp_path / "no-secrets.json"))


@pytest.fixture
def real_key(monkeypatch, tmp_path):
    monkeypatch.setenv("THOR_LINK_KEY", GOOD_KEY)
    monkeypatch.setenv("LAWN_SECRETS_PATH", str(tmp_path / "no-secrets.json"))


# ------------------------------------ the flag ------------------------------------


def test_the_shipped_config_leaves_the_link_off():
    shipped = REPO_CONFIG / CONFIG_FILE
    assert yaml.safe_load(shipped.read_text())["enabled"] is False
    assert load_thor_link_settings(REPO_CONFIG).enabled is False


def test_every_default_is_the_closed_one(tmp_path):
    cfg = load_thor_link_settings(tmp_path)  # no file at all
    assert cfg == ThorLinkSettings()
    assert cfg.enabled is False and cfg.allow_blade is False
    assert cfg.bind_host == "127.0.0.1", "a forgotten bind_host must hear nobody, not everybody"
    assert cfg.require_fix == ["rtk_fixed"] and cfg.geofence is None
    assert cfg.sources() is None  # nothing to allow when there is no Thor configured


async def test_disabled_means_nothing_is_built_bound_or_started(tractor, no_key):
    before = set(asyncio.all_tasks())
    runtime = await _start(tractor, ThorLinkSettings())
    assert runtime is None
    assert set(asyncio.all_tasks()) == before


def test_a_full_config_file_round_trips(tmp_path):
    (tmp_path / CONFIG_FILE).write_text(
        textwrap.dedent(
            """
            enabled: true
            thor_host: 10.1.2.3
            thor_port: 5000
            bind_host: 10.1.2.9
            bind_port: 5001
            allowed_sources: [10.1.2.0/24, 10.9.9.9]
            timeout_s: 0.4
            plan_max_age_s: 0.8
            uplink_hz: 20
            neutral_timeout_s: 0.5
            allow_blade: true
            speed_percent: 30
            require_fix: [rtk_fixed, rtk_float]
            max_position_age_s: 0.7
            geofence:
              margin_m: 1.5
              boundary: [[45.0, -93.0], [45.0, -92.999], [45.0009, -92.999]]
              no_go:
                - [[45.0002, -92.9995], [45.0002, -92.9994], [45.0003, -92.9994]]
            """
        )
    )
    cfg = load_thor_link_settings(tmp_path)
    assert (cfg.thor_host, cfg.thor_port, cfg.bind_host, cfg.bind_port) == (
        "10.1.2.3",
        5000,
        "10.1.2.9",
        5001,
    )
    assert cfg.sources() == ["10.1.2.0/24", "10.9.9.9"]
    assert (cfg.timeout_s, cfg.plan_max_age_s, cfg.uplink_hz) == (0.4, 0.8, 20)
    assert cfg.allow_blade is True and cfg.speed_percent == 30
    assert cfg.require_fix == ["rtk_fixed", "rtk_float"]
    assert cfg.geofence.margin_m == 1.5 and len(cfg.geofence.no_go) == 1


def test_the_allowlist_defaults_to_the_thor_alone_and_any_switches_it_off():
    assert _settings().sources() == ["127.0.0.1"]
    assert _settings(allowed_sources="any").sources() is None
    assert _settings(allowed_sources=["10.0.0.0/8"]).sources() == ["10.0.0.0/8"]


BAD_CONFIGS = {
    "unknown key": "enabled: false\nallow_blades: true\n",
    "enabled without a thor": "enabled: true\n",
    "hostname for thor": "enabled: true\nthor_host: thor.local\n",
    "ipv6 bind": "bind_host: '::1'\n",
    "port zero": "thor_port: 0\n",
    "uplink flood": f"uplink_hz: {MAX_UPLINK_HZ + 1}\n",
    "zero timeout": "timeout_s: 0\n",
    "empty allowlist": "allowed_sources: []\n",
    "bad allowlist word": "allowed_sources: everyone\n",
    "bad allowlist entry": "allowed_sources: [not-an-ip]\n",
    "none is no fix": "require_fix: [none]\n",
    "no fix at all": "require_fix: []\n",
    "speed zero": "speed_percent: 0\n",
    "two-vertex fence": "geofence: {boundary: [[45.0, -93.0], [45.1, -93.0]]}\n",
    "two-vertex no-go": (
        "geofence:\n"
        "  boundary: [[45, -93], [45, -92.9], [45.1, -92.9]]\n"
        "  no_go: [[[45.01, -92.99], [45.02, -92.99]]]\n"
    ),
    "negative margin": "geofence: {margin_m: -1, boundary: [[45,-93],[45,-92.9],[45.1,-92.9]]}\n",
    "not yaml": "enabled: [unclosed\n",
    "not a mapping": "- enabled\n- true\n",
}


@pytest.mark.parametrize("name", sorted(BAD_CONFIGS))
def test_a_config_that_does_not_validate_is_an_error_never_a_silent_default(tmp_path, name):
    (tmp_path / CONFIG_FILE).write_text(BAD_CONFIGS[name])
    with pytest.raises(ThorLinkConfigError):
        load_thor_link_settings(tmp_path)


async def test_a_broken_config_file_keeps_the_link_off_with_the_reason(
    tractor, tmp_path, monkeypatch
):
    (tmp_path / CONFIG_FILE).write_text("enabled: true\nthor_hots: 1.2.3.4\n")
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(tmp_path))
    with pytest.raises(ThorLinkStartError, match="thor_hots"):
        await start_thor_link(tractor, safety_monitor_running=True, gnss=_gnss)


def test_an_unusable_fence_is_refused_not_silently_widened():
    one_metre_square = [(45.0, -93.0), (45.0, -92.99998), (45.00001, -92.99998), (45.00001, -93.0)]
    with pytest.raises(ThorLinkStartError, match="geofence"):
        build_fence(FenceSettings(boundary=one_metre_square, margin_m=5.0))
    with pytest.raises(ThorLinkStartError, match="geofence"):
        build_fence(FenceSettings(boundary=[(45.0, -93.0), (45.0, -93.0), (95.0, -93.0)]))


# ------------------------------- refusals at start-up -------------------------------


async def _refused(tractor, match, *, monitor=True, **kw):
    before = set(asyncio.all_tasks())
    with pytest.raises(ThorLinkStartError, match=match):
        await start_thor_link(
            tractor, safety_monitor_running=monitor, gnss=_gnss, nav=NAV, settings=_settings(**kw)
        )
    assert set(asyncio.all_tasks()) == before, "a refused start leaves no task behind"


async def test_it_does_not_start_on_a_platform_without_the_tractor(real_key):
    await _refused(TractorControlService(config={"enabled": False}), "tractor.enabled")


async def test_it_does_not_start_without_the_tilt_and_watchdog_monitor(tractor, real_key):
    await _refused(tractor, "safety monitor", monitor=False)


async def test_outside_sim_there_is_no_link_without_a_key(tractor, monkeypatch, no_key):
    monkeypatch.setenv("SIM_MODE", "0")
    await _refused(tractor, "THOR_LINK_KEY")


async def test_outside_sim_the_public_sim_key_is_refused_even_when_configured(
    tractor, monkeypatch, no_key
):
    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.setenv("THOR_LINK_KEY", SIM_LINK_KEY.hex())
    await _refused(tractor, "SIM_LINK_KEY|SIM_MODE")


@pytest.mark.parametrize("bad", ["not-hex", "ab" * 8, ""])
async def test_a_malformed_or_short_key_never_starts_a_link(tractor, monkeypatch, no_key, bad):
    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.setenv("THOR_LINK_KEY", bad)
    await _refused(tractor, "THOR_LINK_KEY|hex|32 bytes")


async def test_outside_sim_a_plan_is_never_accepted_without_a_geofence(
    tractor, monkeypatch, real_key
):
    monkeypatch.setenv("SIM_MODE", "0")
    await _refused(tractor, "geofence", geofence=None)


async def test_a_port_that_is_already_taken_is_a_refusal_not_a_half_started_link(tractor, real_key):
    taken = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    taken.bind(("127.0.0.1", 0))
    try:
        await _refused(tractor, "cannot bind", bind_port=taken.getsockname()[1])
    finally:
        taken.close()


# --------------------------------- a successful start ---------------------------------


async def test_sim_mode_without_a_key_starts_on_the_public_key_and_says_so(
    tractor, monkeypatch, no_key, caplog
):
    monkeypatch.setenv("SIM_MODE", "1")
    runtime = await _start(tractor, _settings(geofence=None))
    try:
        assert runtime is not None and runtime.link._key == SIM_LINK_KEY
        assert "public SIM_LINK_KEY" in caplog.text
    finally:
        await stop_thor_link(runtime)


async def test_a_configured_key_is_the_one_the_link_uses(tractor, real_key):
    runtime = await _start(tractor, _settings())
    try:
        assert runtime.link._key == bytes.fromhex(GOOD_KEY)
        assert runtime.link.safe_hold, "a fresh link holds until the Thor sends a plan"
        st = runtime.status()
        assert st["safe_hold"] and st["hold_reason"] == "no_link" and st["rejected"] == 0
        assert st["follower"]["state"] == "idle" and st["session_id"] == runtime.link.session_id
    finally:
        await stop_thor_link(runtime)


async def test_the_policy_in_the_config_reaches_the_link_and_the_follower(tractor, real_key):
    runtime = await _start(
        tractor,
        _settings(
            allow_blade=True,
            speed_percent=25,
            require_fix=["rtk_fixed", "rtk_float"],
            timeout_s=0.7,
            plan_max_age_s=0.9,
            uplink_hz=20,
            allowed_sources=["10.0.0.0/8"],
        ),
    )
    try:
        link, follower = runtime.link, runtime.follower
        assert (link._timeout_s, link._plan_max_age_s, link._period) == (0.7, 0.9, 0.05)
        assert [str(n) for n in link._allowed_sources] == ["10.0.0.0/8"]
        assert link._geofence is not None and link._geofence.no_go == ()
        assert follower._cfg.allow_blade is True and follower._cfg.speed_percent == 25
        assert follower._cfg.require_fix == ("rtk_fixed", "rtk_float")
        assert follower._fence is link._geofence
        assert link._build_uplink == follower.build_uplink and follower._link is link
    finally:
        await stop_thor_link(runtime)


async def test_stopping_is_safe_to_repeat_and_to_skip(tractor, real_key):
    await stop_thor_link(None)
    runtime = await _start(tractor, _settings())
    transport = runtime.link._transport
    await stop_thor_link(runtime)
    await stop_thor_link(runtime)
    assert transport.is_closing() and runtime.follower._task.done()


# ------------------------------- the real lifespan -------------------------------


class _Stub:
    def __init__(self, *_a, **_k) -> None:
        pass

    async def start(self):
        pass

    async def stop(self):
        pass

    async def initialize(self):
        pass

    async def start_streaming(self):
        pass

    async def shutdown(self):
        pass


async def _noop(*_a, **_k):
    return None


def _stub_the_rest_of_the_backend(monkeypatch, main, tractor, order: list[str]) -> None:
    async def park(source="shutdown"):
        order.append("park")
        return {"faults": [], "complete": True}

    # ``main.app`` is one object for the whole test session: forget what an earlier run set.
    for name in ("tractor_safety_monitor", "tractor_watchdog", "thor_link", "thor_link_error"):
        main.app.state._state.pop(name, None)
    monkeypatch.setattr(main, "get_tractor_service", lambda: tractor)
    monkeypatch.setattr(main, "initialize_robohat_service", _noop)
    monkeypatch.setattr(main, "shutdown_robohat_service", _noop)
    monkeypatch.setattr(main, "park_actuators", park)
    monkeypatch.setattr(main, "GPSDegradationMonitor", _Stub)
    monkeypatch.setattr(main, "Watchdog", _Stub)
    monkeypatch.setattr(main, "TractorSafetyMonitor", _Stub)
    monkeypatch.setattr(main, "camera_service", _Stub())
    monkeypatch.setattr(main.websocket_hub, "start_telemetry_loop", _noop)
    monkeypatch.setattr(main.websocket_hub, "stop_telemetry_loop", _noop)


def _config_dir(tmp_path: Path, body: str | None) -> Path:
    if body is not None:
        (tmp_path / CONFIG_FILE).write_text(textwrap.dedent(body))
    return tmp_path


async def test_lifespan_without_the_flag_has_no_link(monkeypatch, tmp_path):
    import backend.src.main as main

    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(_config_dir(tmp_path, None)))
    tractor = TractorControlService(config={"enabled": True})
    _stub_the_rest_of_the_backend(monkeypatch, main, tractor, [])
    async with main.lifespan(main.app):
        assert main.app.state.thor_link is None and main.app.state.thor_link_error is None
    await tractor.shutdown()


async def test_lifespan_with_the_flag_runs_the_link_and_stops_it_before_the_park(
    monkeypatch, tmp_path
):
    import backend.src.main as main
    from backend.src.models.thor_link import Downlink, Waypoint, decode_uplink, encode

    key = bytes.fromhex(GOOD_KEY)
    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.setenv("THOR_LINK_KEY", GOOD_KEY)
    monkeypatch.setenv("LAWN_SECRETS_PATH", str(tmp_path / "no-secrets.json"))
    heard: list = []

    class Thor(asyncio.DatagramProtocol):
        def connection_made(self, transport):
            self.transport = transport

        def datagram_received(self, data, addr):
            up = decode_uplink(data, key)
            heard.append(up)
            plan = Downlink(
                sid=up.sid, ack=up.seq, plan_id="life", waypoints=[Waypoint(lat=45.0, lon=-93.0)]
            )
            self.transport.sendto(encode(plan, key), addr)

    loop = asyncio.get_running_loop()
    thor, _ = await loop.create_datagram_endpoint(Thor, local_addr=("127.0.0.1", 0))
    cfg_dir = _config_dir(
        tmp_path,
        f"""
        enabled: true
        thor_host: 127.0.0.1
        thor_port: {thor.get_extra_info("sockname")[1]}
        bind_host: 127.0.0.1
        bind_port: 0
        timeout_s: 0.3
        uplink_hz: 40
        """,
    )
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(cfg_dir))
    tractor = TractorControlService(config={"enabled": True})
    order: list[str] = []
    _stub_the_rest_of_the_backend(monkeypatch, main, tractor, order)
    real_stop = main.stop_thor_link

    async def recording_stop(runtime):
        order.append("thor_link_stop")
        await real_stop(runtime)

    monkeypatch.setattr(main, "stop_thor_link", recording_stop)
    try:
        async with main.lifespan(main.app):
            runtime = main.app.state.thor_link
            assert runtime is not None and main.app.state.thor_link_error is None
            for _ in range(150):
                await asyncio.sleep(0.02)
                if not runtime.link.safe_hold:
                    break
            assert heard and not runtime.link.safe_hold, "Thor and Pi talk through lifespan"
            transport = runtime.link._transport
        assert order == ["thor_link_stop", "park"], "the link stops before the park"
        assert transport.is_closing() and runtime.follower._task.done()
    finally:
        thor.close()
        await tractor.shutdown()


async def test_lifespan_survives_a_refused_link_and_records_why(monkeypatch, tmp_path, no_key):
    import backend.src.main as main

    monkeypatch.setenv("SIM_MODE", "1")
    cfg_dir = _config_dir(tmp_path, "enabled: true\nthor_host: 127.0.0.1\nbind_port: 0\n")
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(cfg_dir))
    tractor = TractorControlService(config={"enabled": False})  # the platform is off
    _stub_the_rest_of_the_backend(monkeypatch, main, tractor, [])
    async with main.lifespan(main.app):
        assert main.app.state.thor_link is None
        assert "tractor.enabled" in main.app.state.thor_link_error
    await tractor.shutdown()


async def test_no_safety_monitor_means_no_link_even_with_everything_else_in_place(
    monkeypatch, tmp_path, real_key
):
    """If tractor initialisation fails there is no tilt/watchdog monitor, and so no Thor link."""
    import backend.src.main as main

    monkeypatch.setenv("SIM_MODE", "1")
    cfg_dir = _config_dir(tmp_path, "enabled: true\nthor_host: 127.0.0.1\nbind_port: 0\n")
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(cfg_dir))
    tractor = TractorControlService(config={"enabled": True})
    _stub_the_rest_of_the_backend(monkeypatch, main, tractor, [])

    async def broken_initialize():
        raise RuntimeError("PCA9685 on fire")

    tractor.initialize = broken_initialize
    async with main.lifespan(main.app):
        assert main.app.state.thor_link is None
        assert "safety monitor" in main.app.state.thor_link_error
    await tractor.shutdown()
