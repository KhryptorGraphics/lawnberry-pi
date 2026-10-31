"""Check one rendered STL against the builder's mesh gate.

Usage: python check_mesh.py <file.stl> [label]

Prints OK/FAIL with size, solid count and bed-zero offset. Used during geometry bring-up so
shell quoting can never corrupt the check.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_printables import mesh_report  # noqa: E402
from hardware_77603_catalog import USABLE_VOLUME_MM  # noqa: E402


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: check_mesh.py <file.stl> [label]")
        return 2
    path = Path(argv[0])
    label = argv[1] if len(argv) > 1 else path.stem
    try:
        report = mesh_report(path)
    except Exception as error:  # noqa: BLE001 - diagnostic tool
        print(f"FAIL {label}: {error}")
        return 1
    size = report["size_mm"]
    over = (
        size[0] > USABLE_VOLUME_MM[0]
        or size[1] > USABLE_VOLUME_MM[1]
        or size[2] > USABLE_VOLUME_MM[2]
    )
    if over:
        print(f"FAIL {label}: size {size} exceeds usable {USABLE_VOLUME_MM}")
        return 1
    print(
        f"OK {label}: size={size} solids={report['connected_solids']} "
        f"zmin={report['bounds_mm'][0][2]:.3f} vol={report['volume_mm3']:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
