"""Unit tests for scripts/check_hardware_pin_conflicts.py.

The real-repo tests pin the REVIEWED state (spec/known_conflicts.yaml). Moving a pin (owner action
O-5) must update them together with the allow-list; that is the point of the gate.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_hardware_pin_conflicts.py"

# Import the checker module with proper registration
spec = importlib.util.spec_from_file_location("scripts.check_hardware_pin_conflicts", SCRIPT)
checker = importlib.util.module_from_spec(spec)
sys.modules["scripts.check_hardware_pin_conflicts"] = checker
spec.loader.exec_module(checker)

# Everything the checker reads (the registry, the config and the backend sources it compares)
COPIED = (
    "spec/hardware.yaml",
    "spec/known_conflicts.yaml",
    "config/tractor.yaml",
    "config/limits.yaml",
    "config/hardware.yaml",
    "backend/src/drivers/sensors/ultrasonic_driver.py",
    "backend/src/services/hw_selftest.py",
    "backend/src/services/motor_service.py",
    "backend/src/drivers/blade/ibt4_gpio.py",
)
ULTRASONIC_DRIVER = "backend/src/drivers/sensors/ultrasonic_driver.py"
HW_SELFTEST = "backend/src/services/hw_selftest.py"
MOTOR_SERVICE = "backend/src/services/motor_service.py"
IBT4_DRIVER = "backend/src/drivers/blade/ibt4_gpio.py"


def _copy(tmp_path: Path) -> Path:
    for rel in COPIED:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text((REPO / rel).read_text())
    return tmp_path


def _mutate(path: Path, fn) -> None:
    data = yaml.safe_load(path.read_text())
    fn(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _edit_text(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert text.count(old) == 1, f"{old!r} must occur exactly once in {path.name}"
    path.write_text(text.replace(old, new))


def _gpio_row(role: str):
    return lambda d: next(e for e in d["pins"]["gpio_used"] if e["role"] == role)


def _drift(report) -> list[str]:
    return [e for e in report.errors if "registry drift" in e]


class TestCheckerPureFunctions:
    """Test pure functions with synthetic data."""

    def test_collect_gpio_claims_basic_conflict(self):
        gpio_claims, _ = checker.collect_gpio_claims(
            {"pins": {"gpio_used": [{"gpio": 6, "role": "A"}]}},
            {"relays": {"x": {"gpio": 6}}},
            {},
        )
        detected = checker.find_conflicts(gpio_claims)
        assert ("gpio", 6) in detected
        assert len(detected[("gpio", 6)]) == 2

    def test_rxd_txd_rows_map_via_header_map(self):
        gpio_claims, _ = checker.collect_gpio_claims(
            {
                "pins": {
                    "gpio_used": [
                        {"pin": 33, "rxd": 4, "role": "RX"},
                        {"pin": 32, "txd": 4, "role": "TX"},
                    ]
                }
            },
            {},
            {},
        )
        rows = [c for c in gpio_claims if c.source == "spec/hardware.yaml"]
        assert {c.detail: c.key for c in rows} == {"RX": 13, "TX": 12}
        assert checker.find_conflicts(rows) == {}

    def test_the_self_test_fails_when_the_header_map_collides_rx_and_tx(self, monkeypatch):
        """The old assertion (`13 in detected or 12 in detected or not detected`) was true for the
        correct result and for a false collision alike, so --self-test stayed green."""
        monkeypatch.setitem(checker.HEADER_PIN_TO_GPIO, 33, 12)  # BNO085 RX now collides with TX
        with pytest.raises(AssertionError):
            checker.self_test()

    def test_the_self_test_passes_on_the_real_header_map(self):
        assert checker.self_test() == 0

    def test_txd_collides_with_gpio_row(self):
        gpio_claims, _ = checker.collect_gpio_claims(
            {
                "pins": {
                    "gpio_used": [
                        {"pin": 32, "gpio": 12, "role": "ToF IRQ"},
                        {"pin": 32, "txd": 4, "role": "UART4 TX"},
                    ]
                }
            },
            {},
            {},
        )
        detected = checker.find_conflicts(gpio_claims)
        assert ("gpio", 12) in detected
        assert len(detected[("gpio", 12)]) == 2

    def test_i2c_string_vs_int_normalization(self):
        i2c_claims, _ = checker.collect_i2c_claims(
            {"pins": {"i2c_addresses": {"0x40": "INA3221"}}},
            {"pca9685": {"address": 0x40}},
        )
        detected = checker.find_conflicts(i2c_claims)
        assert ("i2c", "bus1:0x40") in detected

    def test_allow_list_reconciliation_matching(self):
        allowed = [
            checker.AllowEntry(
                kind="gpio",
                pin=6,
                claimants=["A", "B"],
                gap=["H-01"],
                reason="test",
                owner_action="test",
            ),
        ]
        detected = {
            ("gpio", 6): [
                checker.Claim("gpio", 6, "A", "s", "A"),
                checker.Claim("gpio", 6, "B", "s", "B"),
            ]
        }
        r = checker.reconcile(detected, allowed)
        assert len(r.known) == 1
        assert len(r.new) == 0
        assert len(r.stale) == 0

    def test_allow_list_reconciliation_changed_claimants(self):
        allowed = [
            checker.AllowEntry(
                kind="gpio",
                pin=6,
                claimants=["A", "B"],
                gap=["H-01"],
                reason="test",
                owner_action="test",
            ),
        ]
        detected = {
            ("gpio", 6): [
                checker.Claim("gpio", 6, "A", "s", "A"),
                checker.Claim("gpio", 6, "B", "s", "B"),
                checker.Claim("gpio", 6, "C", "s", "C"),
            ]
        }
        r = checker.reconcile(detected, allowed)
        assert len(r.new) == 1
        assert len(r.stale) == 1

    def test_allow_list_reconciliation_stale(self):
        allowed = [
            checker.AllowEntry(
                kind="gpio",
                pin=6,
                claimants=["A", "B"],
                gap=["H-01"],
                reason="test",
                owner_action="test",
            ),
        ]
        r = checker.reconcile({}, allowed)
        assert len(r.stale) == 1

    def test_unverified_claims_excluded_from_conflicts(self):
        gpio_claims, _ = checker.collect_gpio_claims(
            {
                "pins": {
                    "gpio_used": [
                        {"gpio": 6, "role": "ToF IRQ", "status": "verified"},
                        {"gpio": 6, "role": "HaLow RGB", "status": "unverified"},
                    ]
                }
            },
            {},
            {},
        )
        conflict_claims = [c for c in gpio_claims if c.status != "unverified"]
        detected = checker.find_conflicts(conflict_claims)
        assert ("gpio", 6) not in detected


class TestOwnerGating:
    """BLOCKED follows the owner of a claim and the owner's arming switch, not a label."""

    ENTRY = checker.AllowEntry(
        kind="gpio",
        pin=5,
        claimants=["HaLow", "Starter (a label without the config file name)"],
        gap=["H-02b"],
        reason="test",
        owner_action="test",
    )

    def _detected(self, owner):
        return {
            ("gpio", 5): [
                checker.Claim("gpio", 5, "HaLow", "s", "H"),
                checker.Claim(
                    "gpio",
                    5,
                    "Starter (a label without the config file name)",
                    "s",
                    "T",
                    owner=owner,
                ),
            ]
        }

    def test_an_armed_owner_blocks_whatever_its_claim_is_called(self):
        r = checker.reconcile(self._detected("tractor"), [self.ENTRY], {"tractor"})
        assert len(r.blocked) == 1 and r.blocked[0].blocked_by == ["tractor"]
        assert not r.known

    def test_another_owner_being_armed_does_not_block(self):
        r = checker.reconcile(self._detected("tractor"), [self.ENTRY], {"ibt4"})
        assert len(r.known) == 1 and not r.blocked

    def test_nothing_armed_blocks_nothing(self):
        r = checker.reconcile(self._detected("tractor"), [self.ENTRY], set())
        assert len(r.known) == 1 and not r.blocked

    def test_a_claim_without_an_owner_never_blocks(self):
        r = checker.reconcile(self._detected(None), [self.ENTRY], {"tractor", "ibt4"})
        assert len(r.known) == 1 and not r.blocked

    def test_arming_switches_follow_the_runtime_truthiness(self):
        assert checker.armed_owners({"enabled": True}, None) == {"tractor"}
        assert checker.armed_owners({"enabled": "false"}, None) == {"tractor"}  # fail closed
        assert checker.armed_owners({"enabled": False}, {"ibt4": {"gpio_enabled": True}}) == {
            "ibt4"
        }
        assert checker.armed_owners({}, {"ibt4": {"gpio_enabled": False}}) == set()
        assert checker.armed_owners({}, None) == set()


class TestCheckerRepoIntegration:
    """Integration tests using the real repo files."""

    def test_real_repo_known_conflicts_detected(self):
        report = checker.run_check(REPO)
        known_gpios = {c.key for c in report.known}
        assert known_gpios == {3, 5, 12, 25}

    def test_real_repo_no_new_conflicts(self):
        report = checker.run_check(REPO)
        assert len(report.new) == 0

    def test_real_repo_no_stale_allow_list(self):
        report = checker.run_check(REPO)
        assert len(report.stale) == 0

    def test_real_repo_tractor_disabled_no_blocked(self):
        report = checker.run_check(REPO)
        assert len(report.blocked) == 0 and report.blocked_deps == []

    def test_real_repo_has_no_errors(self):
        report = checker.run_check(REPO)
        assert report.errors == []
        assert checker.print_report(report) == 0

    def test_real_repo_ultrasonic_disabled_no_ultrasonic_conflicts(self):
        report = checker.run_check(REPO)
        conflict_gpios = {c.key for c in report.conflicts if c.kind == "gpio"}
        ultrasonic_gpios = {4, 17, 27, 10, 11, 9}
        assert not (conflict_gpios & ultrasonic_gpios)

    def test_real_repo_warnings_emitted(self):
        report = checker.run_check(REPO)
        warning_text = " ".join(report.warnings)
        assert "I2C" in warning_text and "disabled" in warning_text
        assert "UART4" in warning_text

    def test_real_repo_every_source_is_found_and_cross_checked(self):
        """A moved file or symbol is only a warning at the CLI; here it must not go unnoticed."""
        report = checker.run_check(REPO)
        assert not [w for w in report.warnings if "not cross-checked" in w]
        assert not [w for w in report.warnings if "not found" in w]

    def test_the_unclaimed_gpios_are_really_unclaimed(self):
        report = checker.run_check(REPO)
        hw = yaml.safe_load((REPO / "spec/hardware.yaml").read_text())
        claimed = {
            c.key for c in checker.collect_gpio_claims(hw, {}, {"ultrasonic_enabled": True})[0]
        }
        assert report.unclaimed and not set(report.unclaimed) & claimed
        assert set(report.unclaimed) <= checker.USABLE_GPIOS


class TestCheckerMutations:
    """Mutate copies of the real files: the gate must fail on new/stale/blocked entries."""

    def test_unlisted_conflict_fails(self, tmp_path):
        root = _copy(tmp_path)
        # PTO relay moved onto GPIO17: not in any allow-list entry
        _mutate(
            root / "spec/hardware.yaml",
            lambda d: d["pins"]["gpio_used"].append(
                {
                    "pin": 11,
                    "gpio": 17,
                    "role": "Test Claimant",
                    "status": "verified",
                    "evidence": ["test"],
                }
            ),
        )
        _mutate(root / "config/tractor.yaml", lambda d: d["relays"]["blade_pto"].update(gpio=17))
        report = checker.run_check(root)
        assert [c.key for c in report.new] == [17]
        assert checker.print_report(report) == 1

    def test_stale_allow_list_entry_fails(self, tmp_path):
        root = _copy(tmp_path)
        # starter moved off GPIO5: the allow-listed GPIO5 conflict no longer exists
        _mutate(root / "config/tractor.yaml", lambda d: d["relays"]["starter"].update(gpio=16))
        report = checker.run_check(root)
        assert [a.pin for a in report.stale] == [5]
        assert not report.new
        assert checker.print_report(report) == 1

    def test_extra_claimant_on_listed_pin_fails(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(
            root / "spec/hardware.yaml",
            lambda d: d["pins"]["gpio_used"].append(
                {
                    "pin": 29,
                    "gpio": 5,
                    "role": "Third Claimant",
                    "status": "verified",
                    "evidence": ["test"],
                }
            ),
        )
        report = checker.run_check(root)
        assert [c.key for c in report.new] == [5]
        assert [a.pin for a in report.stale] == [5]
        assert checker.print_report(report) == 1

    def test_two_relays_on_one_gpio_fail(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(root / "config/tractor.yaml", lambda d: d["relays"]["blade_pto"].update(gpio=5))
        report = checker.run_check(root)
        assert [c.key for c in report.new] == [
            5
        ]  # starter + blade_pto + HaLow: not the reviewed set
        assert checker.print_report(report) == 1

    def test_enabled_tractor_blocks_listed_tractor_conflict(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(root / "config/tractor.yaml", lambda d: d.update(enabled=True))
        report = checker.run_check(root)
        assert [c.key for c in report.blocked] == [5]
        assert report.blocked[0].blocked_by == ["tractor"]
        assert checker.print_report(report) == 1

    def test_the_blocked_report_names_the_switch(self, tmp_path, capsys):
        root = _copy(tmp_path)
        _mutate(root / "config/tractor.yaml", lambda d: d.update(enabled=True))
        checker.print_report(checker.run_check(root))
        err = capsys.readouterr().err
        assert "BLOCKED (config/tractor.yaml enabled: true): GPIO 5" in err

    def test_enabled_tractor_with_the_starter_moved_is_still_blocked(self, tmp_path):
        """Review T-03: with the starter off GPIO5 the allow-listed conflicts that remain (25, 12,
        3) have no tractor-labelled claimant, so `enabled: true` exited 0 although the tractor's
        PCA9685 sits on the header I2C bus that does not exist on the deployed Pi."""
        root = _copy(tmp_path)

        def tractor(d):
            d.update(enabled=True)
            d["relays"]["starter"].update(gpio=16)

        _mutate(root / "config/tractor.yaml", tractor)
        _mutate(  # the GPIO5 entry is stale once the starter moved: drop it, as the review did
            root / "spec/known_conflicts.yaml",
            lambda d: d.update(conflicts=[c for c in d["conflicts"] if c["pin"] != 5]),
        )
        report = checker.run_check(root)
        assert not report.new and not report.stale and not report.blocked
        assert [owner for owner, _ in report.blocked_deps] == ["tractor"]
        assert "header I2C bus" in report.blocked_deps[0][1]
        assert checker.print_report(report) == 1

    def test_the_bus_gate_is_clear_on_a_bus_that_exists(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(
            root / "config/tractor.yaml",
            lambda d: (d.update(enabled=True), d["pca9685"].update(bus=13)),
        )
        assert checker.run_check(root).blocked_deps == []

    def test_a_bus_the_pi_does_not_have_blocks_an_enabled_tractor(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(root / "config/tractor.yaml", lambda d: d["pca9685"].update(bus=5))
        warned = checker.run_check(root)
        assert warned.blocked_deps == []  # disabled: a warning only
        assert any("pca9685.bus=5 is not among the I2C buses present" in w for w in warned.warnings)
        _mutate(root / "config/tractor.yaml", lambda d: d.update(enabled=True))
        assert "does not exist" in checker.run_check(root).blocked_deps[0][1]

    def test_ibt4_armed_blocks_the_blade_in2_conflict(self, tmp_path):
        """GPIO25 is the IBT-4's (the differential-drive mower's), not the tractor's: its own
        arming switch gates it."""
        root = _copy(tmp_path)
        _mutate(root / "config/hardware.yaml", lambda d: d["ibt4"].update(gpio_enabled=True))
        report = checker.run_check(root)
        assert [(c.key, c.blocked_by) for c in report.blocked] == [(25, ["ibt4"])]
        assert checker.print_report(report) == 1

    def test_the_tractor_being_enabled_does_not_block_the_ibt4_conflict(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(root / "config/tractor.yaml", lambda d: d.update(enabled=True))
        assert 25 not in {c.key for c in checker.run_check(root).blocked}

    def test_ultrasonic_enabled_adds_halow_spi_conflicts(self, tmp_path):
        root = _copy(tmp_path)
        _with_hardware(root, {"backend": "gpio_hcsr04"})
        _mutate(root / "config/limits.yaml", lambda d: d.update(ultrasonic_enabled=True))
        report = checker.run_check(root)
        assert {c.key for c in report.new} >= {9, 10, 11}
        assert checker.print_report(report) == 1

    def test_missing_ultrasonic_key_fails_closed(self, tmp_path):
        root = _copy(tmp_path)
        _with_hardware(root, {"backend": "gpio_hcsr04"})
        _mutate(root / "config/limits.yaml", lambda d: d.pop("ultrasonic_enabled"))
        report = checker.run_check(root)
        assert {c.key for c in report.new} >= {9, 10, 11}

    def test_mcu_backend_claims_no_pi_pin_even_when_the_array_is_enabled(self, tmp_path):
        root = _copy(tmp_path)
        _with_hardware(root, {"backend": "mcu_usb"})
        _mutate(root / "config/limits.yaml", lambda d: d.update(ultrasonic_enabled=True))
        report = checker.run_check(root)
        assert not report.new and not report.blocked and not report.stale
        assert checker.print_report(report) == 0
        hw = yaml.safe_load((root / "spec/hardware.yaml").read_text())
        limits = yaml.safe_load((root / "config/limits.yaml").read_text())
        hwcfg = yaml.safe_load((root / "config/hardware.yaml").read_text())
        claims, _ = checker.collect_gpio_claims(hw, {}, limits, hwcfg)
        assert not [c for c in claims if c.detail.startswith("ultrasonic")]

    def test_legacy_gpio_backend_claims_the_halow_pins_when_enabled(self, tmp_path):
        root = _copy(tmp_path)
        _with_hardware(root, {"backend": "gpio_hcsr04"})
        _mutate(root / "config/limits.yaml", lambda d: d.update(ultrasonic_enabled=True))
        report = checker.run_check(root)
        assert {c.key for c in report.new} >= {9, 10, 11}
        assert checker.print_report(report) == 1

    def test_absent_ultrasonic_block_means_the_runtime_default_mcu_backend(self, tmp_path):
        root = _copy(tmp_path)
        _with_hardware(root, None)
        _mutate(root / "config/limits.yaml", lambda d: d.update(ultrasonic_enabled=True))
        assert not checker.run_check(root).new

    def test_an_unknown_backend_is_treated_as_the_pin_claiming_one(self, tmp_path):
        root = _copy(tmp_path)
        _with_hardware(root, {"backend": "i2c_magic"})
        _mutate(root / "config/limits.yaml", lambda d: d.update(ultrasonic_enabled=True))
        assert {c.key for c in checker.run_check(root).new} >= {9, 10, 11}

    def test_i2c_address_collision_detected(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(root / "config/tractor.yaml", lambda d: d["pca9685"].update(address=0x40))
        report = checker.run_check(root)
        assert any(c.kind == "i2c" and c.key == "bus1:0x40" for c in report.new)

    def test_a_missing_config_hardware_yaml_is_a_warning_and_claims_the_legacy_pins(self, tmp_path):
        root = _copy(tmp_path)
        (root / "config/hardware.yaml").unlink()
        _mutate(root / "config/limits.yaml", lambda d: d.update(ultrasonic_enabled=True))
        report = checker.run_check(root)
        assert any("config/hardware.yaml not found" in w for w in report.warnings)
        assert {c.key for c in report.new} >= {9, 10, 11}

    def test_a_malformed_config_hardware_yaml_is_an_error(self, tmp_path):
        root = _copy(tmp_path)
        (root / "config/hardware.yaml").write_text("ibt4: [unclosed\n")
        with pytest.raises(checker.RegistryError, match="hardware.yaml"):
            checker.run_check(root)


def _with_hardware(root: Path, ultrasonic: dict | None) -> None:
    cfg = yaml.safe_load((REPO / "config/hardware.yaml").read_text())
    cfg.pop("ultrasonic", None)
    if ultrasonic is not None:
        cfg["ultrasonic"] = ultrasonic
    (root / "config/hardware.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


class TestTractorValidation:
    """Review T-04: tractor values are validated like the registry's own rows."""

    @pytest.mark.parametrize("gpio", [31, 32, 99, 28, 40, 0, 1, -1, True, "5", None, 5.0])
    def test_a_relay_on_something_that_is_not_a_usable_header_gpio_is_an_error(
        self, tmp_path, gpio
    ):
        root = _copy(tmp_path)
        _mutate(root / "config/tractor.yaml", lambda d: d["relays"]["blade_pto"].update(gpio=gpio))
        with pytest.raises(checker.RegistryError, match="relays.blade_pto"):
            checker.run_check(root)

    def test_a_physical_pin_number_is_called_out(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(root / "config/tractor.yaml", lambda d: d["relays"]["blade_pto"].update(gpio=31))
        with pytest.raises(checker.RegistryError, match="physical pin 31 is GPIO6"):
            checker.run_check(root)

    @pytest.mark.parametrize("gpio", [2, 6, 14, 27])
    def test_a_usable_header_gpio_is_accepted(self, gpio):
        checker.validate_tractor({"relays": {"starter": {"gpio": gpio}}})

    def test_two_servos_on_one_pca9685_channel_are_an_error(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(
            root / "config/tractor.yaml", lambda d: d["actuators"]["throttle"].update(channel=0)
        )
        with pytest.raises(checker.RegistryError, match="channel 0"):
            checker.run_check(root)

    def test_a_default_channel_collides_with_an_explicit_one(self):
        # left_lever omitted: its runtime default is channel 0, which the throttle now claims
        with pytest.raises(checker.RegistryError, match="channel 0"):
            checker.validate_tractor({"actuators": {"throttle": {"channel": 0}}})

    @pytest.mark.parametrize("channel", [16, 99, -1, True, "1"])
    def test_a_channel_outside_0_to_15_is_an_error(self, channel):
        with pytest.raises(checker.RegistryError, match="0-15"):
            checker.validate_tractor({"actuators": {"throttle": {"channel": channel}}})

    @pytest.mark.parametrize("address", [0x400, 0x00, 0x78, "xyz", True])
    def test_an_address_outside_the_i2c_range_is_an_error(self, address):
        with pytest.raises(checker.RegistryError, match="I2C address"):
            checker.validate_tractor({"pca9685": {"address": address}})

    def test_a_negative_bus_is_an_error(self):
        with pytest.raises(checker.RegistryError, match="pca9685.bus"):
            checker.validate_tractor({"pca9685": {"bus": -1}})

    @pytest.mark.parametrize("edit", ["remove_address", "remove_section"])
    def test_a_deleted_pca9685_address_is_the_driver_default_that_collides(self, tmp_path, edit):
        """Review T-04: `PCA9685Driver({})` runs at 0x40, the INA3221's address, so deleting the
        key created the collision and CI stayed green."""
        root = _copy(tmp_path)
        if edit == "remove_address":
            _mutate(root / "config/tractor.yaml", lambda d: d["pca9685"].pop("address"))
        else:
            _mutate(root / "config/tractor.yaml", lambda d: d.pop("pca9685"))
        report = checker.run_check(root)
        assert [(c.kind, c.key) for c in report.new] == [("i2c", "bus1:0x40")]
        assert any("pca9685.address not set" in w for w in report.warnings)
        assert checker.print_report(report) == 1

    def test_a_deleted_relay_still_claims_its_runtime_default_pin(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(root / "config/tractor.yaml", lambda d: d["relays"].pop("starter"))
        report = checker.run_check(root)
        assert {c.key for c in report.known} == {3, 5, 12, 25}
        assert not report.new and not report.stale

    def test_a_relay_without_a_gpio_that_has_no_default_is_an_error(self):
        with pytest.raises(checker.RegistryError, match="relays.horn"):
            checker.validate_tractor({"relays": {"horn": {"active_high": True}}})

    def test_an_empty_registry_does_not_pass(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(root / "spec/hardware.yaml", lambda d: d["pins"].update(gpio_used=[]))
        with pytest.raises(checker.RegistryError, match="empty registry"):
            checker.run_check(root)
        root = _copy(tmp_path)
        _mutate(root / "spec/hardware.yaml", lambda d: d["pins"].update(i2c_addresses={}))
        with pytest.raises(checker.RegistryError, match="empty registry"):
            checker.run_check(root)

    def test_a_renamed_pins_key_does_not_pass(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(root / "spec/hardware.yaml", lambda d: d.update(pin=d.pop("pins")))
        with pytest.raises(checker.RegistryError, match="'pins' mapping is missing"):
            checker.run_check(root)

    def test_a_registry_row_on_a_gpio_that_does_not_exist_is_an_error(self):
        with pytest.raises(checker.RegistryError, match="not a Raspberry Pi header GPIO"):
            checker.collect_gpio_claims(
                {"pins": {"gpio_used": [{"gpio": 99, "role": "X"}]}}, {}, {}
            )

    def test_the_cli_exits_non_zero_for_a_bad_tractor_value(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(root / "config/tractor.yaml", lambda d: d["relays"]["blade_pto"].update(gpio=31))
        done = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(root)], capture_output=True, text=True
        )
        assert done.returncode == 1
        assert "relays.blade_pto" in done.stderr


class TestRegistryValidation:
    """Review T-05/T-11: the registry is checked against itself, as the docstring promised."""

    def test_the_real_registry_is_consistent(self):
        hw = yaml.safe_load((REPO / "spec/hardware.yaml").read_text())
        checker.validate_registry(hw)

    def test_a_bno085_row_on_a_pin_the_overlay_does_not_use_is_rejected(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(root / "spec/hardware.yaml", lambda d: _gpio_row("BNO085 RX")(d).update(pin=35))
        with pytest.raises(checker.RegistryError, match="uart4.pi5.pin_rxd"):
            checker.run_check(root)

    def test_an_overlay_entry_whose_gpio_disagrees_with_its_pin_is_rejected(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(
            root / "spec/hardware.yaml",
            lambda d: d["pins"]["uart_overlays"]["uart4"]["pi5"].update(rxd_gpio=19),
        )
        with pytest.raises(checker.RegistryError, match="physical pin 33 is GPIO13"):
            checker.run_check(root)

    def test_the_pi4_pins_are_checked_too(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(
            root / "spec/hardware.yaml",
            lambda d: d["pins"]["uart_overlays"]["uart4"]["pi4"].update(txd_gpio=11),
        )
        with pytest.raises(checker.RegistryError, match="uart4.pi4 txd"):
            checker.run_check(root)

    def test_an_rpi4_alt_pin_that_is_not_the_pi4_uart_pin_is_rejected(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(
            root / "spec/hardware.yaml", lambda d: _gpio_row("BNO085 RX")(d).update(rpi4_alt_pin=24)
        )
        with pytest.raises(checker.RegistryError, match="rpi4_alt_pin 24"):
            checker.run_check(root)

    def test_a_row_naming_an_undefined_uart_is_rejected(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(
            root / "spec/hardware.yaml",
            lambda d: next(e for e in d["pins"]["gpio_used"] if e["role"] == "BNO085 TX").update(
                txd=9
            ),
        )
        with pytest.raises(checker.RegistryError, match="uart9"):
            checker.run_check(root)

    def test_an_ultrasonic_pin_number_that_is_not_its_gpio_is_rejected(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(
            root / "spec/hardware.yaml",
            lambda d: d["pins"]["ultrasonic"]["sensors"]["front_left"].update(pin_trig=99),
        )
        with pytest.raises(checker.RegistryError, match="front_left"):
            checker.run_check(root)

    def test_an_ultrasonic_sensor_without_a_gpio_is_rejected_not_skipped(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(
            root / "spec/hardware.yaml",
            lambda d: d["pins"]["ultrasonic"]["sensors"]["front_left"].pop("echo_gpio"),
        )
        with pytest.raises(checker.RegistryError, match="echo_gpio"):
            checker.run_check(root)

    def test_an_unknown_owner_is_rejected(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(root / "spec/hardware.yaml", lambda d: _gpio_row("Blade IN1")(d).update(owner="x"))
        with pytest.raises(checker.RegistryError, match="owner must be one of"):
            checker.run_check(root)


class TestRegistryDrift:
    """Review T-05: a pin changed in config/ or backend/ must not go unseen."""

    def test_unmodified_copies_have_no_drift(self, tmp_path):
        report = checker.run_check(_copy(tmp_path))
        assert report.errors == []
        assert not [w for w in report.warnings if "not cross-checked" in w]

    @pytest.mark.parametrize(
        ("rel", "old", "new"),
        [
            (
                ULTRASONIC_DRIVER,
                '"front_left": {"trig": 4, "echo": 17}',
                '"front_left": {"trig": 4, "echo": 18}',
            ),
            (
                HW_SELFTEST,
                '"front_left": {"trig": 4, "echo": 17}',
                '"front_left": {"trig": 4, "echo": 18}',
            ),
            (MOTOR_SERVICE, '"blade_in2": 25,', '"blade_in2": 26,'),
            (IBT4_DRIVER, 'pins.get("in2", 25)', 'pins.get("in2", 26)'),
        ],
        ids=["ultrasonic_driver", "hw_selftest", "motor_service", "ibt4_driver"],
    )
    def test_a_pin_changed_in_a_backend_source_is_drift(self, tmp_path, rel, old, new):
        root = _copy(tmp_path)
        _edit_text(root / rel, old, new)
        report = checker.run_check(root)
        assert len(_drift(report)) == 1 and rel in _drift(report)[0]
        assert checker.print_report(report) == 1

    @pytest.mark.parametrize(
        "edit",
        [
            lambda d: d["ibt4"]["pins"].update(in2=26),
            lambda d: d["sensors"]["tof_config"].update(left_shutdown_gpio=27),
            lambda d: d["sensors"]["tof_config"].update(right_shutdown_gpio=18),
            lambda d: d["sensors"]["tof_config"].update(left_address=0x2A),
            lambda d: d["sensors"]["tof_config"].update(bus=13),
        ],
        ids=["ibt4_in2", "tof_left_xshut", "tof_right_xshut", "tof_left_address", "tof_bus"],
    )
    def test_a_value_changed_in_config_hardware_yaml_is_drift(self, tmp_path, edit):
        root = _copy(tmp_path)
        _mutate(root / "config/hardware.yaml", edit)
        report = checker.run_check(root)
        assert len(_drift(report)) == 1 and "config/hardware.yaml" in _drift(report)[0]
        assert checker.print_report(report) == 1

    def test_a_gpio_the_config_uses_but_the_registry_lacks_is_drift(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(
            root / "spec/hardware.yaml",
            lambda d: d["pins"].update(
                gpio_used=[e for e in d["pins"]["gpio_used"] if e["role"] != "ToF Left Shutdown"]
            ),
        )
        errors = _drift(checker.run_check(root))
        assert len(errors) == 1 and "has no 'ToF Left Shutdown' row" in errors[0]

    def test_the_specs_tractor_pca9685_address_must_match_the_config(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(root / "spec/hardware.yaml", lambda d: d["tractor"]["pca9685"].update(address=0x42))
        errors = _drift(checker.run_check(root))
        assert len(errors) == 1 and "pca9685.address" in errors[0]

    def test_the_specs_blade_driver_block_must_match_its_rows(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(
            root / "spec/hardware.yaml",
            lambda d: d["cutting_system"]["blade_driver"].update(gpio_in2=26),
        )
        errors = _drift(checker.run_check(root))
        assert len(errors) == 1 and "cutting_system.blade_driver.gpio_in2" in errors[0]

    def test_a_source_that_moved_is_a_warning_not_an_error(self, tmp_path):
        root = _copy(tmp_path)
        (root / HW_SELFTEST).unlink()
        report = checker.run_check(root)
        assert report.errors == []
        assert any(
            f"not cross-checked against {HW_SELFTEST}: file not found" in w for w in report.warnings
        )

    def test_a_symbol_that_moved_is_a_warning_not_an_error(self, tmp_path):
        root = _copy(tmp_path)
        _edit_text(root / HW_SELFTEST, "ULTRASONIC_GPIO_PINS = {", "RENAMED_PINS = {")
        text = (root / HW_SELFTEST).read_text().replace("ULTRASONIC_GPIO_PINS", "RENAMED_PINS")
        (root / HW_SELFTEST).write_text(text)
        report = checker.run_check(root)
        assert report.errors == []
        assert any("ULTRASONIC_GPIO_PINS not found" in w for w in report.warnings)

    def test_a_source_that_does_not_parse_is_an_error(self, tmp_path):
        root = _copy(tmp_path)
        (root / HW_SELFTEST).write_text("ULTRASONIC_GPIO_PINS = {\n")
        assert any("cannot parse" in e for e in checker.run_check(root).errors)


class TestDeploymentChecks:
    """Review T-06: the deployment warnings read the right keys."""

    def test_tof_on_the_disabled_header_bus_is_reported(self):
        """The old branch read `sensors.tof_config` from the spec, which has no such key (the ToF
        bus is in config/hardware.yaml), so the ToF never appeared in the warning."""
        text = " ".join(checker.run_check(REPO).warnings)
        assert "config/hardware.yaml sensors.tof_config (bus=1)" in text

    def test_a_tof_on_another_bus_is_not_reported(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(root / "config/hardware.yaml", lambda d: d["sensors"]["tof_config"].update(bus=13))
        text = " ".join(w for w in checker.run_check(root).warnings if "DEPLOYMENT" in w)
        assert "sensors.tof_config" not in text

    def test_the_uart4_warning_clears_once_uart4_is_recorded_enabled(self, tmp_path):
        """`uart_overlays_enabled` lists UART names (`uart0`); the old code compared it with the
        overlay string `uart4-pi5`, so `["uart0", "uart4"]` still warned."""
        root = _copy(tmp_path)
        assert "UART4 overlay" in " ".join(checker.run_check(root).warnings)
        _mutate(
            root / "spec/hardware.yaml",
            lambda d: d["deployment"].update(uart_overlays_enabled=["uart0", "uart4"]),
        )
        assert "UART4 overlay" not in " ".join(checker.run_check(root).warnings)

    def test_an_overlay_name_in_the_enabled_list_is_rejected(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(
            root / "spec/hardware.yaml",
            lambda d: d["deployment"].update(uart_overlays_enabled=["uart0", "uart4-pi5"]),
        )
        with pytest.raises(checker.RegistryError, match="uart4-pi5"):
            checker.run_check(root)

    def test_a_missing_deployment_block_is_a_warning(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(root / "spec/hardware.yaml", lambda d: d.pop("deployment"))
        report = checker.run_check(root)
        assert any("no 'deployment' block" in w for w in report.warnings)


class TestBusAwareI2C:
    """Review T-09: an address only conflicts with the same address on the same bus."""

    def test_the_same_address_on_another_bus_is_no_conflict(self, tmp_path):
        """Moving the PCA9685 to another bus (audit option B) at its default 0x40 reported a NEW
        CONFLICT against the INA3221 on the header bus."""
        root = _copy(tmp_path)
        _mutate(
            root / "config/tractor.yaml",
            lambda d: d["pca9685"].update(address=0x40, bus=13),
        )
        report = checker.run_check(root)
        assert not report.new

    def test_the_same_address_on_the_same_bus_still_conflicts(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(
            root / "config/tractor.yaml",
            lambda d: d["pca9685"].update(address=0x40, bus=1),
        )
        assert [c.key for c in checker.run_check(root).new] == ["bus1:0x40"]

    def test_a_registry_row_can_name_its_own_bus(self):
        claims, _ = checker.collect_i2c_claims(
            {
                "pins": {"i2c_addresses": {"0x40": {"device": "INA3221", "bus": 4}}},
                "deployment": {"header_i2c_bus": 1},
            },
            {"pca9685": {"address": 0x40, "bus": 1}},
        )
        assert checker.find_conflicts(claims) == {}
        assert {c.key for c in claims} == {"bus4:0x40", "bus1:0x40"}

    def test_allow_list_entries_for_an_i2c_address_need_a_bus(self, tmp_path):
        root = _copy(tmp_path)
        entry = {
            "kind": "i2c_address",
            "i2c_address": "0x40",
            "claimants": ["a", "b"],
            "gap": ["H-17"],
            "reason": "r",
            "owner_action": "o",
        }
        _mutate(root / "spec/known_conflicts.yaml", lambda d: d["conflicts"].append(entry))
        with pytest.raises(checker.RegistryError, match="requires integer 'bus'"):
            checker.load_allow_list(root)

    def test_an_i2c_allow_list_entry_matches_by_bus(self, tmp_path):
        root = _copy(tmp_path)
        _mutate(root / "config/tractor.yaml", lambda d: d["pca9685"].update(address=0x40))
        entry = {
            "kind": "i2c_address",
            "i2c_address": "0x40",
            "bus": 1,
            "claimants": ["config/tractor.yaml pca9685.address", "spec/hardware.yaml: INA3221"],
            "gap": ["H-17"],
            "reason": "r",
            "owner_action": "o",
        }
        _mutate(root / "spec/known_conflicts.yaml", lambda d: d["conflicts"].append(entry))
        _mutate(root / "spec/hardware.yaml", lambda d: d["tractor"]["pca9685"].update(address=0x40))
        report = checker.run_check(root)
        assert not report.new and not report.stale
        assert "bus1:0x40" in [c.key for c in report.known]
        entry["bus"] = 13  # the same address on another bus is another conflict: stale + new
        _mutate(root / "spec/known_conflicts.yaml", lambda d: d["conflicts"].pop())
        _mutate(root / "spec/known_conflicts.yaml", lambda d: d["conflicts"].append(entry))
        report = checker.run_check(root)
        assert [c.key for c in report.new] == ["bus1:0x40"]
        assert [a.bus for a in report.stale] == [13]


class TestOwnerActions:
    """Review T-10: following an owner action must not create a new conflict."""

    def test_the_shipped_owner_actions_name_no_claimed_gpio(self):
        assert checker.run_check(REPO).errors == []

    def _with_action(self, tmp_path, pin, text):
        root = _copy(tmp_path)

        def edit(d):
            next(c for c in d["conflicts"] if c["pin"] == pin).update(owner_action=text)

        _mutate(root / "spec/known_conflicts.yaml", edit)
        return checker.run_check(root)

    @pytest.mark.parametrize(
        ("pin", "text", "named"),
        [
            (25, "move blade IN2 to GPIO26 if not used for PTO", 26),  # the PTO relay
            (3, "use software I2C on GPIO12/13", 12),  # BNO085 UART4 / ToF IRQ
            (12, "move the ToF IRQ to GPIO17/27 if available", 17),  # legacy ultrasonic
        ],
    )
    def test_an_owner_action_naming_a_claimed_gpio_is_an_error(self, tmp_path, pin, text, named):
        report = self._with_action(tmp_path, pin, text)
        assert any(f"names GPIO{named}," in e for e in report.errors), report.errors
        assert checker.print_report(report) == 1

    def test_naming_its_own_pin_or_an_unclaimed_gpio_is_fine(self, tmp_path):
        report = self._with_action(tmp_path, 25, "move GPIO25's user to GPIO19")
        assert report.errors == []

    def test_the_report_lists_the_gpios_a_move_may_use(self, capsys):
        checker.print_report(checker.run_check(REPO))
        out = capsys.readouterr().out
        assert "Header GPIOs no registry row claims" in out


class TestRuntimeDefaultsMirror:
    """The checker's mirror constants must track the runtime classes they copy."""

    @pytest.fixture(autouse=True)
    def _sim(self, monkeypatch):
        monkeypatch.setenv("SIM_MODE", "1")

    def test_relay_and_channel_defaults_match_the_tractor_service(self):
        from backend.src.services.tractor_service import TractorControlService

        service = TractorControlService(config={})
        assert service.starter.gpio == checker.TRACTOR_RELAY_DEFAULTS["starter"]
        assert service.blade_pto.gpio == checker.TRACTOR_RELAY_DEFAULTS["blade_pto"]
        for name, channel in checker.ACTUATOR_CHANNEL_DEFAULTS.items():
            assert getattr(service, name).cal.channel == channel
        assert checker.effective_relays({}) == checker.TRACTOR_RELAY_DEFAULTS
        assert checker.effective_channels({}) == checker.ACTUATOR_CHANNEL_DEFAULTS

    def test_pca9685_defaults_match_the_driver(self):
        from backend.src.drivers.actuators.pca9685_driver import PCA9685Driver

        cfg = PCA9685Driver({})._cfg
        assert (cfg.address, cfg.bus) == (
            checker.PCA9685_DEFAULT_ADDRESS,
            checker.PCA9685_DEFAULT_BUS,
        )
        address, bus, configured = checker.effective_pca9685({})
        assert (address, bus, configured) == (cfg.address, cfg.bus, False)

    def test_the_header_map_is_a_forty_pin_header(self):
        assert sorted(checker.HEADER_PIN_TO_GPIO) == list(range(1, 41))
        assert len(checker.HEADER_GPIOS) == 28  # GPIO0-27


class TestCheckerConfigValidation:
    """Tests for config file validation."""

    def test_tractor_yaml_loads_enabled_false(self):
        tractor = yaml.safe_load((REPO / "config/tractor.yaml").read_text())
        assert tractor.get("enabled") is False
        assert isinstance(tractor.get("enabled"), bool)

    def test_tractor_yaml_has_required_sections(self):
        tractor = yaml.safe_load((REPO / "config/tractor.yaml").read_text())
        assert "relays" in tractor
        assert "pca9685" in tractor
        assert "actuators" in tractor
        assert tractor["pca9685"]["address"] == 0x41

    def test_tractor_relays_have_gpio_ints(self):
        tractor = yaml.safe_load((REPO / "config/tractor.yaml").read_text())
        for _name, relay in tractor["relays"].items():
            assert isinstance(relay["gpio"], int)

    def test_tractor_relay_gpio_values_match_registry(self):
        tractor = yaml.safe_load((REPO / "config/tractor.yaml").read_text())
        assert tractor["relays"]["starter"]["gpio"] == 5
        assert tractor["relays"]["blade_pto"]["gpio"] == 26

    def test_tractor_pca9685_address_not_0x40(self):
        tractor = yaml.safe_load((REPO / "config/tractor.yaml").read_text())
        assert tractor["pca9685"]["address"] != 0x40
        assert tractor["pca9685"]["address"] == 0x41

    def test_limits_yaml_ultrasonic_disabled(self):
        limits = yaml.safe_load((REPO / "config/limits.yaml").read_text())
        assert limits.get("ultrasonic_enabled") is False


class TestCheckerCLI:
    """CLI behavior tests via subprocess, on the interpreter that runs the suite."""

    @staticmethod
    def _run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO
        )

    def test_self_test_passes(self):
        result = self._run("--self-test")
        assert result.returncode == 0, result.stderr
        assert "self-test OK" in result.stdout

    def test_real_repo_exits_zero_with_known_conflicts(self):
        result = self._run()
        assert result.returncode == 0, result.stderr
        assert "Known conflicts (allow-listed):" in result.stdout
        assert "GPIO 25" in result.stdout
        assert "GPIO 12" in result.stdout
        assert "GPIO 5" in result.stdout
        assert "GPIO 3" in result.stdout

    def test_a_directory_without_the_registry_exits_one(self, tmp_path):
        result = self._run("--root", str(tmp_path))
        assert result.returncode == 1
        assert "ERROR" in result.stderr
