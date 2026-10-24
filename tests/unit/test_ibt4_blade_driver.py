import pytest

from backend.src.drivers.blade.ibt4_gpio import IBT4BladeDriver, _GPIOAdapter


class DummyGPIO(_GPIOAdapter):
    def __init__(self, in1: int, in2: int):
        super().__init__(in1, in2)
        self.setup_called = False
        self.cleanup_called = False

    def setup(self) -> None:
        self.setup_called = True
        super().setup()

    def cleanup(self) -> None:
        self.cleanup_called = True
        super().cleanup()


class FlakyGPIO(DummyGPIO):
    """An adapter that raises on demand, like a real one that cannot claim or confirm."""

    def __init__(self, in1: int, in2: int):
        super().__init__(in1, in2)
        self.fail_setup = False
        self.fail_set = False
        self.fail_cleanup = False
        self.writes: list[bool] = []

    def setup(self) -> None:
        self.setup_called = True
        if self.fail_setup:
            raise OSError("claim failed")
        super().setup()

    def set_active(self, active: bool) -> None:
        self.writes.append(bool(active))
        if self.fail_set:
            raise OSError("write not confirmed")
        super().set_active(active)

    def cleanup(self) -> None:
        self.cleanup_called = True
        if self.fail_cleanup:
            raise OSError("free failed")
        super().cleanup()


@pytest.mark.asyncio
async def test_ibt4_initialize_and_toggle_in_sim_mode(monkeypatch):
    # Force SIM_MODE for test safety
    monkeypatch.setenv("SIM_MODE", "1")

    gpio = DummyGPIO(24, 25)
    drv = IBT4BladeDriver(config={"pins": {"in1": 24, "in2": 25}}, gpio_adapter=gpio)

    await drv.initialize()
    assert drv.initialized

    await drv.start()
    assert drv.running

    # Should be off by default
    hc = await drv.health_check()
    assert not hc["active"]

    # Toggle on
    ok = await drv.set_active(True)
    assert ok
    hc = await drv.health_check()
    assert hc["active"]

    # E-stop blocks activation
    await drv.set_estop(True)
    ok = await drv.set_active(True)
    assert not ok

    # Stop driver
    await drv.stop()
    assert not (await drv.health_check())["active"]


@pytest.mark.asyncio
async def test_ibt4_sim_mode_never_sets_up_the_adapter(monkeypatch):
    """SIM_MODE must not reach the adapter's setup (a real one would claim GPIO lines)."""
    monkeypatch.setenv("SIM_MODE", "1")
    gpio = FlakyGPIO(24, 25)
    gpio.fail_setup = True  # would raise if SIM_MODE called it
    drv = IBT4BladeDriver(gpio_adapter=gpio)

    await drv.initialize()

    assert drv.initialized is True
    assert gpio.setup_called is False


@pytest.mark.asyncio
async def test_ibt4_initialize_outside_sim_sets_up_the_adapter(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    gpio = DummyGPIO(24, 25)
    drv = IBT4BladeDriver(gpio_adapter=gpio)

    await drv.initialize()

    assert gpio.setup_called is True and drv.initialized is True


@pytest.mark.asyncio
async def test_ibt4_failed_setup_fails_closed_instead_of_falling_back_to_simulation(monkeypatch):
    """The old driver logged a warning, set ``initialized = True`` and kept 'working' on flags."""
    monkeypatch.setenv("SIM_MODE", "0")
    gpio = FlakyGPIO(24, 25)
    gpio.fail_setup = True
    drv = IBT4BladeDriver(gpio_adapter=gpio)

    await drv.initialize()  # must not raise

    health = await drv.health_check()
    assert drv.initialized is False and health["initialized"] is False
    assert "OSError" in health["last_error"] and "claim failed" in health["last_error"]
    assert await drv.set_active(True) is False
    assert await drv.set_active(False) is False
    assert gpio.writes == [], "an uninitialised driver never reaches the adapter"


@pytest.mark.asyncio
async def test_ibt4_unconfirmed_write_returns_false_and_reads_as_possibly_on(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    gpio = FlakyGPIO(24, 25)
    drv = IBT4BladeDriver(gpio_adapter=gpio)
    await drv.initialize()
    assert await drv.set_active(True) is True
    gpio.fail_set = True

    assert await drv.set_active(False) is False

    health = await drv.health_check()
    assert health["active"] is True, "an unconfirmed OFF must read as 'may be on'"
    assert health["state_confirmed"] is False and drv.state_confirmed is False
    assert "write not confirmed" in health["last_error"]

    gpio.fail_set = False
    assert await drv.set_active(False) is True
    health = await drv.health_check()
    assert health["active"] is False and health["state_confirmed"] is True
    assert health["last_error"] is None


@pytest.mark.asyncio
async def test_ibt4_unconfirmed_on_does_not_report_the_blade_as_confirmed_on(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    gpio = FlakyGPIO(24, 25)
    gpio.fail_set = True
    drv = IBT4BladeDriver(gpio_adapter=gpio)
    await drv.initialize()

    assert await drv.set_active(True) is False

    health = await drv.health_check()
    assert health["state_confirmed"] is False
    assert health["active"] is True, "conservative: the output state is unknown"
    assert gpio.is_active() is False, "the adapter never recorded a confirmed ON"


@pytest.mark.asyncio
async def test_ibt4_set_estop_returns_the_off_result_and_always_latches(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    gpio = FlakyGPIO(24, 25)
    drv = IBT4BladeDriver(gpio_adapter=gpio)
    await drv.initialize()
    await drv.set_active(True)

    assert await drv.set_estop(True) is True, "confirmed OFF"
    assert drv.estop_latched is True and gpio.is_active() is False
    assert await drv.set_estop(False) is True
    assert drv.estop_latched is False

    await drv.set_active(True)
    gpio.fail_set = True
    assert await drv.set_estop(True) is False, "unconfirmed OFF is reported, not hidden"
    assert drv.estop_latched is True, "the latch is set even though the write failed"
    writes_before = len(gpio.writes)
    assert await drv.set_active(True) is False
    assert len(gpio.writes) == writes_before, "a latched E-stop never reaches the adapter"


@pytest.mark.asyncio
async def test_ibt4_stop_never_raises_and_still_releases_the_adapter(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    gpio = FlakyGPIO(24, 25)
    drv = IBT4BladeDriver(gpio_adapter=gpio)
    await drv.initialize()
    await drv.start()
    await drv.set_active(True)
    gpio.fail_set = True
    gpio.fail_cleanup = True

    await drv.stop()  # must not raise

    assert drv.running is False and drv.initialized is False
    assert gpio.cleanup_called is True
    assert (await drv.health_check())["active"] is True, "the OFF was never confirmed"


@pytest.mark.asyncio
async def test_ibt4_health_check_reports_the_gate_and_confirmation_state(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "1")
    drv = IBT4BladeDriver(config={"pins": {"in1": 20, "in2": 21}})

    health = await drv.health_check()

    assert health == {
        "driver": "ibt4",
        "initialized": False,
        "running": False,
        "active": False,
        "pins": {"in1": 20, "in2": 21},
        "estop_latched": False,
        "hardware_backed": False,
        "gpio_enabled": False,
        "state_confirmed": True,
        "last_error": None,
    }
