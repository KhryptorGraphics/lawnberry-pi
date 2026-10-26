"""Every manifest check in hardware-guard.yml must really fail its step.

GitHub runs a step's ``run:`` body with ``bash -e {0}``. Under ``-e`` a negated command
(``! grep ...``) and every element of an ``&&`` list except the last cannot fail the script, so
those checks only gated when they happened to be the final command (audit review T-02). The
steps are executed here exactly as GitHub runs them, against crafted bad copies of
``spec/hardware.yaml``; the steps that need ``uv`` and the network are skipped.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "hardware-guard.yml"
SPEC_TEXT = (REPO / "spec" / "hardware.yaml").read_text()

BLOCK_STEP = "Block irrelevant hardware in manifest"
INA_STEP = "Verify INA3221 documented channel mapping is present in spec"
REQUIRED_STEP = "Ensure required hardware appears in spec/hardware.yaml"
ETHERNET_STEP = "Ensure bench-only ethernet note exists"
I2C_STEP = "Confirm I2C addresses listed (sanity)"


def _manifest_steps() -> list[tuple[str, str]]:
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["manifest-check"]
    return [(s["name"], s["run"]) for s in job["steps"] if "run" in s and "uv " not in s["run"]]


def _failing_steps(tmp_path: Path, spec_text: str) -> set[str]:
    """Names of the manifest steps that exit non-zero on `spec_text`."""
    (tmp_path / "spec").mkdir(exist_ok=True)
    (tmp_path / "spec" / "hardware.yaml").write_text(spec_text)
    failing = set()
    for i, (name, body) in enumerate(_manifest_steps()):
        script = tmp_path / f"step{i}.sh"
        script.write_text(body)
        done = subprocess.run(
            ["bash", "-e", str(script)], cwd=tmp_path, capture_output=True, text=True
        )
        if done.returncode != 0:
            failing.add(name)
    return failing


def _remove(text: str, term: str) -> str:
    return re.sub(re.escape(term), "REDACTED", text, flags=re.IGNORECASE)


def test_the_harness_runs_the_steps_it_claims_to():
    names = {name for name, _ in _manifest_steps()}
    assert {BLOCK_STEP, INA_STEP, REQUIRED_STEP, ETHERNET_STEP, I2C_STEP} <= names


def test_the_shipped_manifest_passes_every_step(tmp_path):
    assert _failing_steps(tmp_path, SPEC_TEXT) == set()


@pytest.mark.parametrize("term", ["soil", "weather station", "jetson", "movidius", "Jetson"])
def test_each_forbidden_term_fails_the_block_step(tmp_path, term):
    # soil / weather station / jetson were `! grep` lines before the movidius line: never fatal
    assert _failing_steps(tmp_path, SPEC_TEXT + f"\n# {term}\n") == {BLOCK_STEP}


def test_several_forbidden_terms_fail_the_block_step(tmp_path):
    bad = SPEC_TEXT + "\n# soil\n# weather station\n# jetson\n"
    assert _failing_steps(tmp_path, bad) == {BLOCK_STEP}


@pytest.mark.parametrize("key", ["channel_1", "channel_2", "channel_3"])
def test_a_missing_ina3221_channel_key_fails_its_step(tmp_path, key):
    # channel_1 and channel_2 were `A && B` lines that were not the last: never fatal
    assert _failing_steps(tmp_path, _remove(SPEC_TEXT, key)) == {INA_STEP}


@pytest.mark.parametrize("name", ["Battery", "Unused", "Solar"])
def test_a_missing_ina3221_channel_name_fails_its_step(tmp_path, name):
    assert _failing_steps(tmp_path, _remove(SPEC_TEXT, name)) == {INA_STEP}


def _rename(text: str, channel: str, old: str, new: str) -> str:
    """Change the ``name:`` that follows ``channel:`` in the INA3221 block of the spec."""
    pattern = rf'({channel}:\s*\n\s*name: ")({re.escape(old)})'
    assert re.search(pattern, text), f"{channel} no longer carries {old!r}"
    return re.sub(pattern, rf"\g<1>{new}", text, count=1)


def test_swapping_which_channel_is_battery_and_solar_fails_its_step(tmp_path):
    # The step only checked that `channel_N` and the words appeared somewhere in the file, so a
    # spec with ch1 = Solar and ch3 = Battery passed (audit review PR #20, mutant G3).
    swapped = _rename(SPEC_TEXT, "channel_1", "Battery", "Solar Input")
    swapped = _rename(swapped, "channel_3", "Solar Input", "Battery")

    assert _failing_steps(tmp_path, swapped) == {INA_STEP}


@pytest.mark.parametrize(
    "channel, old, new",
    [
        ("channel_1", "Battery", "Unused"),
        ("channel_2", "Unused", "Battery"),
        ("channel_3", "Solar Input", "Unused"),
    ],
)
def test_a_channel_carrying_another_channels_name_fails_its_step(tmp_path, channel, old, new):
    # `Battery` etc. are still present elsewhere in the file; only the pairing is wrong.
    assert _failing_steps(tmp_path, _rename(SPEC_TEXT, channel, old, new)) == {INA_STEP}


def test_the_ina3221_pairing_is_read_from_the_line_after_its_key(tmp_path):
    # The key and the word may not be satisfied by two unrelated places in the file.
    moved = SPEC_TEXT.replace("channel_2:", "channel_9:") + "\n# channel_2:\n# Unused\n"

    assert _failing_steps(tmp_path, moved) == {INA_STEP}


@pytest.mark.parametrize(
    "term",
    [
        "GPS-RTK-SMA",
        "Neo-8M",
        "BNO085",
        "VL53L0X",
        "BME280",
        "SSD1306",
        "INA3221",
        "IBT-4",
        "MDDRC10",
        "RoboHAT",
        "Pi Camera v2",
        "Hailo",
    ],
)
def test_each_required_hardware_term_fails_the_required_step(tmp_path, term):
    assert REQUIRED_STEP in _failing_steps(tmp_path, _remove(SPEC_TEXT, term))


def test_removing_zed_f9p_fails_the_required_step(tmp_path):
    # the spec spells it "ZED-F9P"; the step accepts that or "ZEDF9P" (`A || B`)
    assert REQUIRED_STEP in _failing_steps(tmp_path, _remove(SPEC_TEXT, "ZED-F9P"))


@pytest.mark.parametrize("term", ["ethernet", "bench_only"])
def test_each_ethernet_note_term_fails_its_step(tmp_path, term):
    assert ETHERNET_STEP in _failing_steps(tmp_path, _remove(SPEC_TEXT, term))


@pytest.mark.parametrize("address", ["0x29", "0x30", "0x3C", "0x40", "0x76"])
def test_each_listed_i2c_address_fails_its_step(tmp_path, address):
    assert I2C_STEP in _failing_steps(tmp_path, _remove(SPEC_TEXT, address))
