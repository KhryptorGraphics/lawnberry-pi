"""One strict reader of ``SIM_MODE`` (audit SAF-07 / BL-06).

``SIM_MODE=true`` used to be simulation for ``core.simulation`` (the safety stack) and "not
simulation" for the posture check and "not '0'" for navigation, in one process. Now
``core.simulation.is_simulation_mode`` is the only reader: exactly ``1`` is simulation, nothing
else is, and an unrecognised value is reported once. An AST scan over ``backend/`` fails on any
other read of the variable.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from backend.src.core import auth_posture, simulation
from backend.src.core.simulation import is_simulation_mode, parse_sim_mode

BACKEND = Path(__file__).resolve().parents[2] / "backend"

#: Files another track is rewriting in the same wave still carry their raw reads here until they
#: are converted on their branches: a CEILING per file, not a permission (a raw read may not be
#: added to one). Every file has been converted (the last two, navigation_service and
#: calibration_service, in fix3/nav), so the table is empty and any raw read fails the scan.
PENDING_OTHER_TRACKS: dict[str, int] = {}


@pytest.mark.parametrize(
    ("raw", "simulated", "recognised"),
    [
        ("1", True, True),
        (" 1 ", True, True),
        ("0", False, True),
        ("", False, True),
        (None, False, True),
        ("false", False, True),
        ("FALSE", False, True),
        ("no", False, True),
        ("off", False, True),
        # fail closed: none of these is simulation, and each is reported as unrecognised
        ("true", False, False),
        ("True", False, False),
        ("yes", False, False),
        ("on", False, False),
        ("2", False, False),
        ("11", False, False),
        ("1 1", False, False),
        ("sim", False, False),
    ],
)
def test_only_the_exact_value_1_is_simulation(raw, simulated, recognised):
    assert parse_sim_mode(raw) == (simulated, recognised)


@pytest.mark.parametrize("raw", ["true", "True", "yes", "on", "2", "enabled"])
def test_every_reader_agrees_that_an_ambiguous_value_is_not_simulation(monkeypatch, raw):
    monkeypatch.setenv("SIM_MODE", raw)
    assert is_simulation_mode() is False
    assert auth_posture.sim_mode() is False
    assert (
        auth_posture.sim_open_auth_enabled({"SIM_MODE": raw, "LAWN_BERRY_SIM_OPEN_AUTH": "1"})
        is False
    )
    assert is_simulation_mode({"SIM_MODE": raw}) is False


def test_the_value_1_is_simulation_for_every_reader(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "1")
    assert is_simulation_mode() is True and auth_posture.sim_mode() is True
    monkeypatch.delenv("SIM_MODE")
    assert is_simulation_mode() is False and auth_posture.sim_mode() is False


def test_an_unrecognised_value_is_logged_once_and_says_what_to_write(monkeypatch, caplog):
    monkeypatch.setattr(simulation, "_warned", set())
    monkeypatch.setenv("SIM_MODE", "tru")
    with caplog.at_level(logging.WARNING, logger=simulation.logger.name):
        for _ in range(5):
            assert is_simulation_mode() is False
    messages = [r.getMessage() for r in caplog.records if r.name == simulation.logger.name]
    assert len(messages) == 1
    assert "'tru'" in messages[0] and "SIM_MODE=1" in messages[0] and "SIM_MODE=0" in messages[0]
    caplog.clear()
    monkeypatch.setenv("SIM_MODE", "0")  # a recognised value is never reported
    with caplog.at_level(logging.WARNING, logger=simulation.logger.name):
        is_simulation_mode()
    assert not caplog.records


def _is_environ(node: ast.expr) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "environ") or (
        isinstance(node, ast.Name) and node.id == "environ"
    )


def _raw_sim_mode_reads(path: Path) -> int:
    """Reads of the ``SIM_MODE`` variable from the environment in ``path`` (an AST walk)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = 0
    for node in ast.walk(tree):
        first: ast.expr | None = None
        if isinstance(node, ast.Call) and node.args and isinstance(node.func, ast.Attribute):
            func = node.func
            if func.attr == "getenv" or (
                func.attr in {"get", "pop", "setdefault"} and _is_environ(func.value)
            ):
                first = node.args[0]
        elif isinstance(node, ast.Subscript) and _is_environ(node.value):
            first = node.slice
        if isinstance(first, ast.Constant) and first.value == "SIM_MODE":
            found += 1
    return found


def test_nothing_in_backend_reads_sim_mode_except_through_the_shared_parser():
    offenders: dict[str, int] = {}
    stale: list[str] = []
    for path in sorted(BACKEND.rglob("*.py")):
        rel = path.relative_to(BACKEND.parent).as_posix()
        reads = _raw_sim_mode_reads(path)
        ceiling = PENDING_OTHER_TRACKS.get(rel, 0)
        if reads > ceiling:
            offenders[rel] = reads
        if rel in PENDING_OTHER_TRACKS and reads == 0:
            stale.append(rel)
    assert not offenders, (
        "raw reads of the SIM_MODE environment variable (use "
        f"backend.src.core.simulation.is_simulation_mode): {offenders}"
    )
    # An entry that has been converted is a hole in the scan until it is removed from the table.
    if stale:
        pytest.skip(f"converted on their branch: remove from PENDING_OTHER_TRACKS: {stale}")


def test_the_scan_sees_the_shapes_it_is_there_to_catch(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import os\n"
        "a = os.getenv('SIM_MODE', '0') == '1'\n"
        "b = os.environ.get('SIM_MODE')\n"
        "c = os.environ['SIM_MODE']\n"
        "d = os.environ.get('OTHER')\n"
        "e = {}.get('SIM_MODE')\n"
    )
    assert _raw_sim_mode_reads(sample) == 3
