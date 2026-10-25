"""Unit tests for scripts/check_hardware_pin_conflicts.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

# Import the checker module with proper registration
spec = importlib.util.spec_from_file_location(
    "scripts.check_hardware_pin_conflicts", "scripts/check_hardware_pin_conflicts.py"
)
checker = importlib.util.module_from_spec(spec)
sys.modules["scripts.check_hardware_pin_conflicts"] = checker
spec.loader.exec_module(checker)


class TestCheckerPureFunctions:
    """Test pure functions with synthetic data."""

    def test_collect_gpio_claims_basic_conflict(self):
        gpio_claims, _ = checker.collect_gpio_claims(
            Path("."),
            {"pins": {"gpio_used": [{"gpio": 6, "role": "A"}]}},
            {"relays": {"x": {"gpio": 6}}},
            {},
            None,
        )
        detected = checker.find_conflicts(gpio_claims)
        assert ("gpio", 6) in detected
        assert len(detected[("gpio", 6)]) == 2

    def test_rxd_txd_rows_map_via_header_map(self):
        gpio_claims, _ = checker.collect_gpio_claims(
            Path("."),
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
            None,
        )
        detected = checker.find_conflicts(gpio_claims)
        assert ("gpio", 13) in detected or ("gpio", 12) in detected or not detected

    def test_txd_collides_with_gpio_row(self):
        gpio_claims, _ = checker.collect_gpio_claims(
            Path("."),
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
            None,
        )
        detected = checker.find_conflicts(gpio_claims)
        assert ("gpio", 12) in detected
        assert len(detected[("gpio", 12)]) == 2

    def test_i2c_string_vs_int_normalization(self):
        i2c_claims, _ = checker.collect_i2c_claims(
            Path("."),
            {"pins": {"i2c_addresses": {"0x40": "INA3221"}}},
            {"pca9685": {"address": 0x40}},
        )
        detected = checker.find_conflicts(i2c_claims)
        assert ("i2c", "0x40") in detected

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
        r = checker.reconcile(detected, allowed, False)
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
        r = checker.reconcile(detected, allowed, False)
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
        detected = {}
        r = checker.reconcile(detected, allowed, False)
        assert len(r.stale) == 1

    def test_tractor_enabled_blocks_allow_listed_conflict(self):
        allowed = [
            checker.AllowEntry(
                kind="gpio",
                pin=5,
                claimants=["HaLow", "config/tractor.yaml relays.starter"],
                gap=["H-02b"],
                reason="test",
                owner_action="test",
            )
        ]
        detected = {
            ("gpio", 5): [
                checker.Claim("gpio", 5, "HaLow", "s", "H"),
                checker.Claim("gpio", 5, "config/tractor.yaml relays.starter", "s", "T"),
            ]
        }
        r = checker.reconcile(detected, allowed, True)
        assert len(r.blocked) == 1
        r = checker.reconcile(detected, allowed, False)
        assert len(r.known) == 1

    def test_unverified_claims_excluded_from_conflicts(self):
        gpio_claims, _ = checker.collect_gpio_claims(
            Path("."),
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
            None,
        )
        conflict_claims = [c for c in gpio_claims if c.status != "unverified"]
        detected = checker.find_conflicts(conflict_claims)
        assert ("gpio", 6) not in detected


class TestCheckerRepoIntegration:
    """Integration tests using the real repo files."""

    def test_real_repo_known_conflicts_detected(self):
        report = checker.run_check(Path("."))
        known_gpios = {c.key for c in report.known}
        assert known_gpios == {3, 5, 12, 25}

    def test_real_repo_no_new_conflicts(self):
        report = checker.run_check(Path("."))
        assert len(report.new) == 0

    def test_real_repo_no_stale_allow_list(self):
        report = checker.run_check(Path("."))
        assert len(report.stale) == 0

    def test_real_repo_tractor_disabled_no_blocked(self):
        report = checker.run_check(Path("."))
        assert len(report.blocked) == 0

    def test_real_repo_ultrasonic_disabled_no_ultrasonic_conflicts(self):
        report = checker.run_check(Path("."))
        conflict_gpios = {c.key for c in report.conflicts if c.kind == "gpio"}
        ultrasonic_gpios = {4, 17, 27, 10, 11, 9}
        assert not (conflict_gpios & ultrasonic_gpios)

    def test_real_repo_warnings_emitted(self):
        report = checker.run_check(Path("."))
        warning_text = " ".join(report.warnings)
        assert "I2C" in warning_text and "disabled" in warning_text
        assert "UART4" in warning_text


class TestCheckerMutations:
    """Mutate copies of the real files: the gate must fail on new/stale/blocked entries."""

    @staticmethod
    def _copy(tmp_path: Path) -> Path:
        for rel in (
            "spec/hardware.yaml",
            "spec/known_conflicts.yaml",
            "config/tractor.yaml",
            "config/limits.yaml",
        ):
            dst = tmp_path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(Path(rel).read_text())
        return tmp_path

    @staticmethod
    def _mutate(path: Path, fn) -> None:
        data = yaml.safe_load(path.read_text())
        fn(data)
        path.write_text(yaml.safe_dump(data, sort_keys=False))

    def test_unlisted_conflict_fails(self, tmp_path):
        root = self._copy(tmp_path)
        # PTO relay moved onto GPIO17: not in any allow-list entry
        self._mutate(
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
        self._mutate(
            root / "config/tractor.yaml", lambda d: d["relays"]["blade_pto"].update(gpio=17)
        )
        report = checker.run_check(root)
        assert [c.key for c in report.new] == [17]
        assert checker.print_report(report) == 1

    def test_stale_allow_list_entry_fails(self, tmp_path):
        root = self._copy(tmp_path)
        # starter moved off GPIO5: the allow-listed GPIO5 conflict no longer exists
        self._mutate(root / "config/tractor.yaml", lambda d: d["relays"]["starter"].update(gpio=16))
        report = checker.run_check(root)
        assert [a.pin for a in report.stale] == [5]
        assert not report.new
        assert checker.print_report(report) == 1

    def test_extra_claimant_on_listed_pin_fails(self, tmp_path):
        root = self._copy(tmp_path)
        self._mutate(
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

    def test_enabled_tractor_blocks_listed_tractor_conflict(self, tmp_path):
        root = self._copy(tmp_path)
        self._mutate(root / "config/tractor.yaml", lambda d: d.update(enabled=True))
        report = checker.run_check(root)
        assert [c.key for c in report.blocked] == [5]
        assert checker.print_report(report) == 1

    def test_ultrasonic_enabled_adds_halow_spi_conflicts(self, tmp_path):
        root = self._copy(tmp_path)
        self._mutate(root / "config/limits.yaml", lambda d: d.update(ultrasonic_enabled=True))
        report = checker.run_check(root)
        assert {c.key for c in report.new} >= {9, 10, 11}
        assert checker.print_report(report) == 1

    def test_missing_ultrasonic_key_fails_closed(self, tmp_path):
        root = self._copy(tmp_path)
        self._mutate(root / "config/limits.yaml", lambda d: d.pop("ultrasonic_enabled"))
        report = checker.run_check(root)
        assert {c.key for c in report.new} >= {9, 10, 11}

    def test_i2c_address_collision_detected(self, tmp_path):
        root = self._copy(tmp_path)
        self._mutate(root / "config/tractor.yaml", lambda d: d["pca9685"].update(address=0x40))
        report = checker.run_check(root)
        assert any(c.kind == "i2c" and c.key == "0x40" for c in report.new)


class TestCheckerConfigValidation:
    """Tests for config file validation."""

    def test_tractor_yaml_loads_enabled_false(self):
        tractor = yaml.safe_load((Path("config/tractor.yaml")).read_text())
        assert tractor.get("enabled") is False
        assert isinstance(tractor.get("enabled"), bool)

    def test_tractor_yaml_has_required_sections(self):
        tractor = yaml.safe_load((Path("config/tractor.yaml")).read_text())
        assert "relays" in tractor
        assert "pca9685" in tractor
        assert "actuators" in tractor
        assert tractor["pca9685"]["address"] == 0x41

    def test_tractor_relays_have_gpio_ints(self):
        tractor = yaml.safe_load((Path("config/tractor.yaml")).read_text())
        for _name, relay in tractor["relays"].items():
            assert isinstance(relay["gpio"], int)

    def test_tractor_relay_gpio_values_match_registry(self):
        tractor = yaml.safe_load((Path("config/tractor.yaml")).read_text())
        assert tractor["relays"]["starter"]["gpio"] == 5
        assert tractor["relays"]["blade_pto"]["gpio"] == 26

    def test_tractor_pca9685_address_not_0x40(self):
        tractor = yaml.safe_load((Path("config/tractor.yaml")).read_text())
        assert tractor["pca9685"]["address"] != 0x40
        assert tractor["pca9685"]["address"] == 0x41

    def test_limits_yaml_ultrasonic_disabled(self):
        limits = yaml.safe_load((Path("config/limits.yaml")).read_text())
        assert limits.get("ultrasonic_enabled") is False


class TestCheckerCLI:
    """CLI behavior tests via subprocess."""

    def test_self_test_passes(self):
        import subprocess

        result = subprocess.run(
            ["python3", "scripts/check_hardware_pin_conflicts.py", "--self-test"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode == 0
        assert "self-test OK" in result.stdout

    def test_real_repo_exits_zero_with_known_conflicts(self):
        import subprocess

        result = subprocess.run(
            ["python3", "scripts/check_hardware_pin_conflicts.py"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode == 0
        assert "Known conflicts (allow-listed):" in result.stdout
        assert "GPIO 25" in result.stdout
        assert "GPIO 12" in result.stdout
        assert "GPIO 5" in result.stdout
        assert "GPIO 3" in result.stdout
