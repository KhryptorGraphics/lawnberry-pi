"""Dataset audit for YOLO-format detector sets (W2/W5).

    python -m workshop.dataset_stats DIR [DIR ...] [--json OUT] [--near-dupes]

Answers the questions that decide whether a training run is worth starting:

- how many images carry no label at all (a silent recall hole, or harmless background)
- which classes are thin (the tier-1 hard-stop classes need the most instances, not the fewest)
- how box sizes are distributed (tiny boxes are the ones quantization eats first)
- whether exact or near-duplicate frames have leaked across a split
- whether a label file is malformed or names a class outside `--names`

Reads only; writes nothing unless `--json` is given. No GPU, no extra dependencies
(PIL is already in workshop/requirements.txt for the auto-labeler).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .classes import NAMES

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _dhash(path: Path, size: int = 8) -> int | None:
    """Difference hash: cheap, dependency-free near-duplicate signature."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - PIL is in workshop requirements
        return None
    try:
        with Image.open(path) as im:
            small = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
            px = small.tobytes()  # 'L' mode: one byte per pixel, row-major
    except Exception:
        return None
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def parse_label_file(
    path: Path, n_classes: int
) -> tuple[list[tuple[int, float, float, float, float]], int]:
    """Return (boxes, malformed_line_count). Boxes are (cls, cx, cy, w, h), normalized."""
    boxes: list[tuple[int, float, float, float, float]] = []
    bad = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            bad += 1
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            bad += 1
            continue
        if not 0 <= cls < n_classes:
            bad += 1
            continue
        boxes.append((cls, cx, cy, w, h))
    return boxes, bad


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _near_duplicate_pairs(images: list[Path], root: Path, max_near: int) -> list[list[str]]:
    pairs: list[list[str]] = []
    sigs = [(str(p.relative_to(root)), _dhash(p)) for p in images]
    sigs = [(name, s) for name, s in sigs if s is not None]
    if len(sigs) > 4000:
        return pairs
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            if _hamming(sigs[i][1], sigs[j][1]) <= max_near:
                pairs.append([sigs[i][0], sigs[j][0]])
                if len(pairs) >= 200:
                    return pairs
    return pairs


def scan(root: Path, names: list[str], near_dupes: bool = False, max_near: int = 5) -> dict:
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    per_image_boxes: list[int] = []
    per_class_images: Counter[int] = Counter()
    per_class_boxes: Counter[int] = Counter()
    areas: list[float] = []
    unlabeled: list[str] = []
    malformed_total = 0
    hashes: dict[str, list[str]] = defaultdict(list)
    missing_labels: list[str] = []

    for img in images:
        label = img.with_suffix(".txt")
        if not label.exists():
            # YOLO layout: images/<split>/x.jpg -> labels/<split>/x.txt
            parts = img.relative_to(root).parts
            if parts and parts[0] == "images":
                label = root / "labels" / Path(*parts[1:]).with_suffix(".txt")
        if not label.exists():
            missing_labels.append(str(img.relative_to(root)))
            per_image_boxes.append(0)
        else:
            boxes, bad = parse_label_file(label, len(names))
            malformed_total += bad
            per_image_boxes.append(len(boxes))
            if not boxes:
                unlabeled.append(str(img.relative_to(root)))
            seen = set()
            for cls, _cx, _cy, w, h in boxes:
                per_class_boxes[cls] += 1
                areas.append(max(0.0, w) * max(0.0, h))
                seen.add(cls)
            for cls in seen:
                per_class_images[cls] += 1
        try:
            digest = hashlib.sha256(img.read_bytes()).hexdigest()
            hashes[digest].append(str(img.relative_to(root)))
        except OSError:
            pass

    exact_dupes = [paths for paths in hashes.values() if len(paths) > 1]
    near_pairs = _near_duplicate_pairs(images, root, max_near) if near_dupes else []

    total_boxes = sum(per_class_boxes.values())
    classes = []
    for idx, name in enumerate(names):
        classes.append(
            {
                "class": idx,
                "name": name,
                "images": per_class_images.get(idx, 0),
                "instances": per_class_boxes.get(idx, 0),
                "share": (
                    round(per_class_boxes.get(idx, 0) / total_boxes, 4) if total_boxes else 0.0
                ),
            }
        )
    classes.sort(key=lambda c: c["instances"])

    return {
        "root": str(root),
        "images": len(images),
        "images_without_label_file": len(missing_labels),
        "images_with_zero_boxes": len(unlabeled),
        "boxes": total_boxes,
        "malformed_lines": malformed_total,
        "boxes_per_image": {
            "min": min(per_image_boxes) if per_image_boxes else 0,
            "median": _percentile([float(v) for v in per_image_boxes], 0.5),
            "max": max(per_image_boxes) if per_image_boxes else 0,
        },
        "box_area_norm": {
            "p10": round(_percentile(areas, 0.10), 5),
            "p50": round(_percentile(areas, 0.50), 5),
            "p90": round(_percentile(areas, 0.90), 5),
        },
        "classes": classes,
        "exact_duplicate_groups": exact_dupes[:20],
        "exact_duplicate_count": len(exact_dupes),
        "near_duplicate_pairs": near_pairs[:20],
        "near_duplicate_count": len(near_pairs),
        "unlabeled_examples": unlabeled[:10],
        "missing_label_examples": missing_labels[:10],
    }


def render(report: dict) -> str:
    lines = []
    pct_ok = (
        100.0 * report["images_with_zero_boxes"] / report["images"] if report["images"] else 0.0
    )
    lines.append(f"root: {report['root']}")
    lines.append(
        f"images {report['images']}  boxes {report['boxes']}  "
        f"zero-box {report['images_with_zero_boxes']} ({pct_ok:.1f}%)  "
        f"missing-label {report['images_without_label_file']}  "
        f"malformed {report['malformed_lines']}"
    )
    bpi = report["boxes_per_image"]
    area = report["box_area_norm"]
    lines.append(
        f"boxes/image min {bpi['min']} median {bpi['median']} max {bpi['max']}  |  "
        f"box area p10 {area['p10']} p50 {area['p50']} p90 {area['p90']}"
    )
    lines.append(
        f"duplicates: exact {report['exact_duplicate_count']}  "
        f"near {report['near_duplicate_count']}"
    )
    lines.append("")
    lines.append(f"{'class':>4}  {'name':<18} {'images':>7} {'inst':>7} {'share':>7}")
    for c in report["classes"]:
        share = f"{c['share']:>7.2%}"
        lines.append(
            f"{c['class']:>4}  {c['name']:<18} {c['images']:>7} {c['instances']:>7} {share}"
        )
    if report["exact_duplicate_count"]:
        lines.append("")
        lines.append("first exact-duplicate groups:")
        for group in report["exact_duplicate_groups"][:5]:
            lines.append("  " + " == ".join(group))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None, help="write the full report here")
    ap.add_argument(
        "--near-dupes",
        action="store_true",
        help="also scan for near-duplicates (dHash)",
    )
    ap.add_argument("--max-hamming", type=int, default=5)
    ap.add_argument("--names", default="workshop.classes", help="module exposing NAMES")
    a = ap.parse_args(argv)

    names = list(NAMES)
    if a.names != "workshop.classes":
        mod = __import__(a.names, fromlist=["NAMES"])
        names = list(mod.NAMES)

    reports = [scan(d, names, a.near_dupes, a.max_hamming) for d in a.dirs]
    for report in reports:
        print(render(report))
        print()
    if a.json:
        a.json.write_text(json.dumps(reports if len(reports) > 1 else reports[0], indent=2))
        print(f"wrote {a.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
