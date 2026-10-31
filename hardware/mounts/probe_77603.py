"""Run every CSG probe and report pass/fail, treating any WARNING as failure.

A probe that renders EMPTY because variables were undefined is a FALSE PASS. This script
therefore fails a check if openscad emits any WARNING or ERROR, regardless of emptiness.

Usage: python probe_77603.py [--only substring]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hardware_77603_catalog import PROBES_77603  # noqa: E402

PROBE_SOURCE = HERE / "throttle_77603" / "fit_checks.scad"
TIMEOUT_S = 300


def run_probe(name: str, check: str, expect: str, params: dict) -> tuple[str, bool, str]:
    args = ["openscad", "--export-format", "binstl", "-o", "/dev/null"]
    merged = {"check": check, **params}
    for key, value in merged.items():
        args += ["-D", f'{key}="{value}"' if isinstance(value, str) else f"{key}={value}"]
    args.append(str(PROBE_SOURCE))
    try:
        result = subprocess.run(
            args, cwd=HERE, capture_output=True, text=True, timeout=TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired:
        return name, False, "timeout"
    text = result.stdout + result.stderr
    warnings = [
        line for line in text.splitlines()
        if "WARNING" in line or "ERROR" in line
    ]
    if warnings:
        # Undefined variables make a probe meaningless even if it renders empty.
        return name, False, f"{len(warnings)} warning(s): {warnings[0][:110]}"
    empty = "Current top level object is empty" in text
    ok = empty if expect == "empty" else not empty
    return name, ok, f"expect={expect} empty={empty}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None)
    args = parser.parse_args(argv)

    total = failures = 0
    for probe in PROBES_77603:
        if args.only and args.only not in probe.check:
            continue
        for params in probe.parameter_sets:
            total += 1
            name, ok, detail = run_probe(probe.check, probe.check, probe.expect, params)
            status = "PASS" if ok else "FAIL"
            if not ok:
                failures += 1
            print(f"{status} {name:<34} {detail}", flush=True)

    print(f"\n{total - failures}/{total} probes pass (warnings are failures)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
