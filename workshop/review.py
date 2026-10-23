"""W2 human-review gate.

``review.jsonl`` rows: ``{"image", "cls", "yolo", "reviewed", "verdict"?}``
where verdict is ``ok`` | ``fixed`` | ``deleted`` | ``added``. Spot checks
(``spot_checks.jsonl``) record a second reviewer's audit of a random sample:
``{"image", "errors": int, "boxes": int}``.

    python -m workshop.review DATASET_DIR   # exits non-zero unless releasable
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path


def _label_line_counts(ds: Path, images: set[str]) -> dict[str, int | None]:
    """Boxes per image from ``labels/<stem>.txt``; ``None`` when the file is missing."""
    counts: dict[str, int | None] = {}
    labels_dir = ds / "labels"
    for name in images:
        path = labels_dir / f"{Path(name).stem}.txt"
        if not path.exists():
            counts[name] = None
            continue
        counts[name] = sum(1 for line in path.read_text().splitlines() if line.strip())
    return counts


def release_check(
    ds: Path, min_spot_fraction: float = 0.05, max_spot_error_rate: float = 0.02
) -> list[str]:
    """Return the reasons this dataset may not be released; empty means releasable.

    Checks, in order: provenance (manifest), that every image has a label file, that
    ``review.jsonl`` and the label files agree box-for-box (minus rows whose verdict is
    ``deleted``), that every box was human-reviewed, and that the spot-check sample is at
    least ``min_spot_fraction`` of the images with an error rate under the limit.
    """
    problems: list[str] = []
    review_path = ds / "review.jsonl"
    if not review_path.exists():
        return ["no review.jsonl (nothing has been labelled or reviewed)"]
    if not (ds / "manifest.json").exists():
        problems.append("no manifest.json (the run's model, revision and input hashes are unknown)")
    rows = [json.loads(x) for x in review_path.read_text().splitlines() if x.strip()]
    pending = sum(not r.get("reviewed") for r in rows)
    if pending:
        problems.append(f"{pending} boxes not human-reviewed")

    on_disk = {p.name for p in (ds / "images").glob("*") if p.is_file()}
    referenced = {r["image"] for r in rows}
    unknown = sorted(referenced - on_disk)
    if unknown:
        problems.append(f"{len(unknown)} review rows reference images that do not exist")

    images = referenced | on_disk
    counts = _label_line_counts(ds, images)
    missing_labels = sorted(name for name, n in counts.items() if n is None)
    if missing_labels:
        problems.append(f"{len(missing_labels)} images have no label file")
    kept = Counter(r["image"] for r in rows if r.get("verdict") != "deleted")
    mismatched = sorted(
        name for name, n in counts.items() if n is not None and n != kept.get(name, 0)
    )
    if mismatched:
        problems.append(
            f"{len(mismatched)} images: label file and review.jsonl disagree on box count"
        )

    spot = ds / "spot_checks.jsonl"
    checks = []
    if spot.exists():
        checks = [json.loads(x) for x in spot.read_text().splitlines() if x.strip()]
    required = max(1, math.ceil(min_spot_fraction * len(images)))
    if len(checks) < required:
        problems.append(
            f"{len(checks)} spot checks < {required} "
            f"({min_spot_fraction:.0%} of {len(images)} images)"
        )
    boxes = sum(c.get("boxes", 0) for c in checks)
    errors = sum(c.get("errors", 0) for c in checks)
    if boxes and errors / boxes > max_spot_error_rate:
        problems.append(f"spot-check error rate {errors / boxes:.1%} > {max_spot_error_rate:.0%}")
    return problems


if __name__ == "__main__":
    p = release_check(Path(sys.argv[1]))
    print("\n".join(p) or "releasable")
    sys.exit(1 if p else 0)
