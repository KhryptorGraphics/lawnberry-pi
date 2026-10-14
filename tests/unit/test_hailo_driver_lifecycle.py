"""HailoDriver hardware lifecycle against a fake ``hailo_platform``.

The real VDevice has no scheduler by default, so inference fails with
NETWORK_GROUP_NOT_ACTIVATED unless the network group is activated while the
vstreams are open; a failed init must also release /dev/hailo0.
"""

from __future__ import annotations

import sys
import types

import pytest

from backend.src.drivers.ai.hailo_driver import HailoDriver


class _Info:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


def _fake_platform(events: list[str], fail_configure: bool = False) -> types.ModuleType:
    class _Ctx:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            events.append(f"enter:{self.name}")
            return self

        def __exit__(self, *exc):
            events.append(f"exit:{self.name}")

    class NetworkGroup:
        def get_input_vstream_infos(self):
            return [_Info("in", (640, 640, 3))]

        def get_output_vstream_infos(self):
            return [_Info("out", (80, 80, 4))]

        def create_params(self):
            return "params"

        def activate(self, params):
            events.append(f"activate({params})")
            return _Ctx("activation")

    class VDevice:
        def __init__(self):
            events.append("vdevice")

        def configure(self, hef, params):
            if fail_configure:
                raise RuntimeError("configure failed")
            return [NetworkGroup()]

        def release(self):
            events.append("release")

    class InferVStreams(_Ctx):
        def __init__(self, ng, inp, out):
            super().__init__("vstreams")
            events.append(f"formats:{inp}/{out}")

    class _Params:
        @staticmethod
        def make(ng, format_type):
            return format_type

    mod = types.ModuleType("hailo_platform")
    mod.__version__ = "4.24.0"
    mod.HEF = lambda path: types.SimpleNamespace(get_network_group_names=lambda: ["net"])
    mod.ConfigureParams = types.SimpleNamespace(create_from_hef=lambda hef, interface: None)
    mod.HailoStreamInterface = types.SimpleNamespace(PCIe="pcie")
    mod.FormatType = types.SimpleNamespace(UINT8="u8", FLOAT32="f32")
    mod.InferVStreams = InferVStreams
    mod.InputVStreamParams = _Params
    mod.OutputVStreamParams = _Params
    mod.VDevice = VDevice
    return mod


@pytest.fixture
def hef(tmp_path, monkeypatch):
    monkeypatch.delenv("SIM_MODE", raising=False)
    monkeypatch.setattr("backend.src.drivers.ai.hailo_driver.is_simulation_mode", lambda: False)
    path = tmp_path / "m.hef"
    path.write_bytes(b"")
    return path


async def test_start_activates_network_group_and_stop_unwinds(hef, monkeypatch):
    events: list[str] = []
    monkeypatch.setitem(sys.modules, "hailo_platform", _fake_platform(events))
    driver = HailoDriver({"hef_path": str(hef), "input_format": "uint8"})

    await driver.initialize()
    await driver.start()
    await driver.stop()

    assert driver.hardware_available is True
    assert events == [
        "vdevice",
        "formats:u8/f32",
        "enter:vstreams",
        "activate(params)",
        "enter:activation",
        "exit:activation",
        "exit:vstreams",
        "release",
    ]


async def test_failed_init_releases_device_and_reports_error(hef, monkeypatch):
    events: list[str] = []
    monkeypatch.setitem(sys.modules, "hailo_platform", _fake_platform(events, fail_configure=True))
    driver = HailoDriver({"hef_path": str(hef)})

    await driver.initialize()

    assert driver.hardware_available is False
    assert driver.hardware_error == "configure failed"
    assert events == ["vdevice", "release"]


async def test_missing_hef_never_opens_device(tmp_path, hef, monkeypatch):
    events: list[str] = []
    monkeypatch.setitem(sys.modules, "hailo_platform", _fake_platform(events))
    driver = HailoDriver({"hef_path": str(tmp_path / "absent.hef")})

    await driver.initialize()

    assert driver.hardware_available is False
    assert "absent.hef" in driver.hardware_error
    assert events == []


def test_rejects_unknown_input_format():
    with pytest.raises(ValueError):
        HailoDriver({"hef_path": "x.hef", "input_format": "int8"})
