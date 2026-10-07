"""W2 human-review gate.

``review.jsonl`` rows: ``{"image", "cls", "yolo", "reviewed", "verdict"?}``
where verdict is ``ok`` | ``fixed`` | ``deleted`` | ``added``. Spot checks
(``spot_checks.jsonl``) record a second reviewer's audit of a random sample:
``{"image", "errors": int, "boxes": int}``.

    python -m workshop.review DATASET_DIR   # exits non-zero unless releasable
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def release_check(
    ds: Path, min_spot_fraction: float = 0.05, max_spot_error_rate: float = 0.02
) -> list[str]:
    problems = []
    rows = [json.loads(x) for x in (ds / "review.jsonl").read_text().splitlines() if x.strip()]
    pending = sum(not r.get("reviewed") for r in rows)
    if pending:
        problems.append(f"{pending} boxes not human-reviewed")
    images = {r["image"] for r in rows} | {p.name for p in (ds / "images").glob("*")}
    spot = ds / "spot_checks.jsonl"
    checks = []
    if spot.exists():
        checks = [json.loads(x) for x in spot.read_text().splitlines() if x.strip()]
    if len(checks) < max(1, int(min_spot_fraction * len(images))):
        problems.append(
            f"{len(checks)} spot checks < {min_spot_fraction:.0%} of {len(images)} images"
        )
    boxes = sum(c["boxes"] for c in checks)
    errors = sum(c["errors"] for c in checks)
    if boxes and errors / boxes > max_spot_error_rate:
        problems.append(f"spot-check error rate {errors / boxes:.1%} > {max_spot_error_rate:.0%}")
    return problems


if __name__ == "__main__":
    p = release_check(Path(sys.argv[1]))
    print("\n".join(p) or "releasable")
    sys.exit(1 if p else 0)
