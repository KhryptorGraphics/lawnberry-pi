"""Preflight: render every 77603 piece and report ALL failures in one pass.

The full build stops at the first bad mesh, so fixing geometry one build cycle per piece is
slow. This script renders each catalogued piece independently, applies the same
``mesh_report`` gate the builder uses, and prints a complete pass/fail table. It publishes
nothing; it only diagnoses.

Usage:
    python hardware/mounts/preflight_77603.py [--only substring] [--jobs N]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_printables import defines, mesh_report  # noqa: E402
from hardware_77603_catalog import PARTS_77603, USABLE_VOLUME_MM  # noqa: E402

TIMEOUT_S = 300


def render(piece, staging: Path) -> tuple[str, str]:
    """Render one piece. Returns (status, detail)."""
    target = staging / f"{piece.name}.stl"
    command = [
        "openscad",
        "--export-format",
        "binstl",
        "-o",
        str(target),
        *defines(piece.parameters),
        str(HERE / piece.source),
    ]
    try:
        result = subprocess.run(
            command, cwd=HERE, capture_output=True, text=True, timeout=TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired:
        return piece.name, "FAIL timeout"
    text = result.stdout + result.stderr
    # WARNINGs matter: an ignored unknown -D variable silently renders the wrong part.
    if result.returncode or "ERROR:" in text or "WARNING:" in text:
        first = next(
            (ln for ln in text.splitlines() if "ERROR:" in ln or "WARNING:" in ln),
            text.splitlines()[:1],
        )
        return piece.name, f"FAIL render: {first}"
    try:
        report = mesh_report(target)
    except ValueError as error:
        return piece.name, f"FAIL gate: {error}"
    size = report["size_mm"]
    if (
        size[0] > USABLE_VOLUME_MM[0]
        or size[1] > USABLE_VOLUME_MM[1]
        or size[2] > USABLE_VOLUME_MM[2]
    ):
        return piece.name, f"FAIL envelope: {size} > {USABLE_VOLUME_MM}"
    return piece.name, f"PASS {size}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="substring filter on piece name")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args(argv)

    pieces = [p for p in PARTS_77603 if args.only is None or args.only in p.name]
    if not pieces:
        print(f"no catalogued piece matches {args.only!r}")
        return 2

    failures = []
    with tempfile.TemporaryDirectory(prefix=".preflight-77603-", dir=HERE) as temp:
        staging = Path(temp)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for name, detail in pool.map(lambda p: render(p, staging), pieces):
                print(f"{detail:<70} {name}", flush=True)
                if detail.startswith("FAIL"):
                    failures.append((name, detail))

    print(f"\n{len(pieces) - len(failures)}/{len(pieces)} pieces pass")
    if failures:
        print("FAILURES:")
        for name, detail in failures:
            print(f"  - {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
