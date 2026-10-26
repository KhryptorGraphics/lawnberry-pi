"""Per-class detection matching for compiled-model gates (pure Python, no numpy).

Predictions are detection dicts (``class_id``, ``confidence``, ``bbox`` = xyxy in
source pixels); ground truth is ``(class_id, xyxy)`` from YOLO label files.

A recall gate decides whether a compiled detector may replace the live one, so everything that
feeds it fails closed: a label file that is missing or cannot be read in full, an image without a
label (or a label without an image), a class id the spec does not have, too little ground truth
for a tier-1 class, and a ``--baseline`` report that is not a comparable detector-eval report all
raise instead of shrinking the evidence until nothing is left to fail. ``DatasetError`` and
``BaselineError`` carry the reason; ``scripts/eval_detector.py`` maps them to distinct exit codes.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from workshop.classes import CLASSES, NAMES, RECALL_GATE

Box = Sequence[float]
Counts = dict[int, dict[str, int]]

# A report this module writes carries one of these in ``schema``; older reports of the same
# shape (``per_class`` with ``gt``/``recall``, ``iou``, ``conf``, ``images``) are still readable.
REPORT_SCHEMA = "detector-eval/2"
BASELINE_SCHEMAS = ("detector-eval/1", REPORT_SCHEMA)

# Zero misses in 150 boxes is the smallest sample that shows recall >= 0.98 (the tier-1 gate) at
# 95 % one-sided confidence (0.98 ** 149 < 0.05 < 0.98 ** 148): a smaller tier-1 sample cannot
# support the decision however well the model did on it.
GATE_MIN_GT = 150
# Fewer images than this is a smoke test, not a validation set.
MIN_IMAGES = 100
# A label-directory file that is not an annotation (LabelImg / Label Studio write it).
NON_LABEL_FILES = frozenset({"classes"})
SHOW = 5  # names quoted per problem


class EvidenceError(ValueError):
    """The evaluation cannot support a verdict; ``problems`` lists every reason found."""

    def __init__(self, problems: str | Iterable[str]):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


class DatasetError(EvidenceError):
    """The images and labels cannot support a verdict (missing, malformed or too little)."""


class BaselineError(EvidenceError):
    """The ``--baseline`` report cannot be compared with this run."""


def _names(items: Iterable, limit: int = SHOW) -> str:
    items = [str(i) for i in items]
    return ", ".join(items[:limit]) + (f", ... ({len(items)} in all)" if len(items) > limit else "")


def parse_label_file(
    path: str | Path, n_classes: int
) -> list[tuple[int, float, float, float, float]]:
    """Read one YOLO label file: ``(cls, cx, cy, bw, bh)`` rows, normalized. Never skips a row.

    A missing file is an error (an image without objects has an empty file). So is any row that is
    not exactly ``cls cx cy bw bh`` with a class id in ``[0, n_classes)``, finite numbers and a box
    of positive size: a row that is skipped is ground truth that silently stops counting.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"label file missing: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"cannot read label file {path}: {exc}") from exc
    rows = []
    for lineno, line in enumerate(lines, 1):
        parts = line.split()
        if not parts:
            continue
        where = f"{path}:{lineno}"
        if len(parts) != 5:
            raise DatasetError(f"{where}: expected 'cls cx cy w h', got {len(parts)} fields")
        try:
            cls, cx, cy, bw, bh = (float(v) for v in parts)
        except ValueError:
            raise DatasetError(f"{where}: non-numeric field in {line!r}") from None
        if not all(math.isfinite(v) for v in (cls, cx, cy, bw, bh)):
            raise DatasetError(f"{where}: non-finite number in {line!r}")
        if not cls.is_integer() or cls < 0:
            raise DatasetError(f"{where}: class id {parts[0]!r} is not a non-negative integer")
        if int(cls) >= n_classes:
            raise DatasetError(f"{where}: class id {int(cls)} but the spec has {n_classes} classes")
        if bw <= 0 or bh <= 0:
            raise DatasetError(f"{where}: box has no area ({bw} x {bh})")
        rows.append((int(cls), cx, cy, bw, bh))
    return rows


def to_pixels(
    rows: Iterable[tuple[int, float, float, float, float]], w: int, h: int
) -> list[tuple[int, list[float]]]:
    """Normalized ``(cls, cx, cy, bw, bh)`` rows -> ``(cls, [x1, y1, x2, y2])`` in pixels."""
    return [
        (cls, [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        for cls, cx, cy, bw, bh in rows
    ]


def load_yolo_labels(
    txt: str | Path, w: int, h: int, n_classes: int
) -> list[tuple[int, list[float]]]:
    """YOLO ``cls cx cy bw bh`` (normalized) rows -> ``(cls, [x1, y1, x2, y2])`` in pixels."""
    return to_pixels(parse_label_file(txt, n_classes), w, h)


def pair_labels(images: Sequence[Path], labels_dir: str | Path) -> dict[Path, Path]:
    """Map each image to its ``<stem>.txt`` in ``labels_dir``; the two sets must match exactly.

    Raises ``DatasetError`` for a labels directory that does not exist, two images with one stem,
    images without a label file, and label files without an image (a partial copy of the split,
    or the labels of another split).
    """
    labels_dir = Path(labels_dir)
    if not labels_dir.is_dir():
        raise DatasetError(f"labels directory not found: {labels_dir}")
    stems = Counter(p.stem for p in images)
    clash = sorted(s for s, n in stems.items() if n > 1)
    if clash:
        raise DatasetError(
            f"images share a file stem (one label would serve both): {_names(clash)}"
        )
    pairs = {img: labels_dir / f"{img.stem}.txt" for img in images}
    problems = []
    missing = [img.name for img, lab in pairs.items() if not lab.is_file()]
    if missing:
        problems.append(
            f"{len(missing)} of {len(images)} images have no label file in {labels_dir} "
            f"(first: {_names(missing)}); an image without objects needs an empty label file"
        )
    known = {img.stem for img in images}
    orphans = sorted(
        p.name
        for p in labels_dir.glob("*.txt")
        if p.stem not in known and p.stem not in NON_LABEL_FILES
    )
    if orphans:
        problems.append(
            f"{len(orphans)} label files have no image (first: {_names(orphans)}): "
            "the images and labels are not the same split"
        )
    if problems:
        raise DatasetError(problems)
    return pairs


def count_ground_truth(
    rows_by_image: Mapping[Path, Sequence[Sequence]], class_names: Sequence[str]
) -> dict[str, int]:
    """Ground-truth boxes per class name (every name present, zero included)."""
    counts = Counter(row[0] for rows in rows_by_image.values() for row in rows)
    return {name: counts.get(i, 0) for i, name in enumerate(class_names)}


def evidence_problems(
    class_names: Sequence[str],
    gt: Mapping[str, int],
    n_images: int,
    *,
    gate: bool,
    min_gt: int = GATE_MIN_GT,
    min_images: int = MIN_IMAGES,
) -> list[str]:
    """Why this dataset cannot support a gate or baseline verdict (empty list: it can).

    Any verdict needs ``min_images`` images. The gate also needs the spec's class names to be
    ``workshop.classes.NAMES`` in that order (the tiers and the label ids are defined on it) and
    ``min_gt`` ground-truth boxes for every tier-1 class.
    """
    problems = []
    if n_images < min_images:
        problems.append(f"{n_images} images; a verdict needs at least {min_images}")
    if gate:
        if list(class_names) != list(NAMES):
            problems.append(
                f"the spec's {len(class_names)} class names are not workshop.classes.NAMES "
                f"({len(NAMES)} names, same order): the gate tiers and the label ids do not line up"
            )
        short = [
            f"{c.name} {gt.get(c.name, 0)}"
            for c in CLASSES
            if c.tier == 1 and gt.get(c.name, 0) < min_gt
        ]
        if short:
            problems.append(
                f"tier-1 classes need at least {min_gt} ground-truth boxes each, "
                f"found: {_names(short)}"
            )
    return problems


def gate_status(per_class: Mapping[str, Mapping]) -> dict:
    """Per-tier recall gate (``workshop.classes.RECALL_GATE``) over a ``summarize`` result.

    ``failures``: classes with ground truth whose recall is below their tier's gate.
    ``not_evaluated``: classes with no ground truth at all, so no verdict was possible for them.
    """
    tiers = {c.name: c.tier for c in CLASSES}
    failures, not_evaluated = [], []
    for name, m in per_class.items():
        if name not in tiers:
            continue
        if m["gt"] == 0:
            not_evaluated.append({"class": name, "tier": tiers[name], "reason": "no ground truth"})
            continue
        need = RECALL_GATE[tiers[name]]
        if m["recall"] < need:
            failures.append({"class": name, "recall": m["recall"], "gate": need})
    return {"failures": failures, "not_evaluated": not_evaluated}


def finite_number(value: object) -> bool:
    """A real number (not a bool) that is neither NaN nor infinite."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _reject_constant(name: str):
    raise ValueError(f"non-finite number literal {name}")


def read_json(path: str | Path, error: type[EvidenceError]) -> object:
    """Parse a JSON file strictly: unreadable, unparsable and NaN/Infinity files raise ``error``."""
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, ValueError) as exc:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        raise error(f"cannot read {path}: {exc}") from exc


def load_baseline(path: str | Path) -> object:
    """The ``--baseline`` report as parsed JSON; it is checked by ``validate_baseline``."""
    return read_json(path, BaselineError)


def validate_baseline(
    baseline: object,
    *,
    class_names: Sequence[str],
    gt: Mapping[str, int],
    images: int,
    iou: float,
    conf: float,
    min_gt: int,
) -> dict[str, dict]:
    """The baseline's ``per_class`` table, once it is shown to describe this evaluation.

    The baseline must be a detector-eval report with the same class names, the same ground truth
    (image count and boxes per class) and the same IoU and confidence threshold, otherwise a
    difference in recall says nothing about the model. Needs only the ground-truth counts, so it
    runs before any inference. Raises ``BaselineError`` also when no class has ``min_gt`` boxes:
    the comparison would be empty.
    """
    if not isinstance(baseline, dict):
        raise BaselineError("baseline is not a JSON object")
    schema = baseline.get("schema")
    if schema not in BASELINE_SCHEMAS:
        raise BaselineError(
            f"baseline is not a detector-eval report (schema {schema!r}, expected one of "
            f"{', '.join(BASELINE_SCHEMAS)})"
        )
    if baseline.get("status", "ok") != "ok":
        raise BaselineError(f"baseline is not a valid evaluation (status {baseline['status']!r})")
    table = baseline.get("per_class")
    if not isinstance(table, dict):
        raise BaselineError("baseline has no 'per_class' table")
    if set(table) != set(class_names):
        missing = sorted(set(class_names) - set(table))
        extra = sorted(set(table) - set(class_names))
        raise BaselineError(
            f"baseline has other classes than this run (missing: {_names(missing) or 'none'}; "
            f"unexpected: {_names(extra) or 'none'})"
        )
    for name, b in table.items():
        n, recall = (b.get("gt"), b.get("recall")) if isinstance(b, dict) else (None, None)
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise BaselineError(f"baseline class {name!r}: gt {n!r} is not a count")
        if (recall is None) != (n == 0) or (
            recall is not None and not (finite_number(recall) and 0 <= recall <= 1)
        ):
            raise BaselineError(f"baseline class {name!r}: recall {recall!r} with gt {n}")
    for key, mine in (("iou", iou), ("conf", conf)):
        theirs = baseline.get(key)
        if not finite_number(theirs) or not math.isclose(theirs, mine, rel_tol=1e-9, abs_tol=1e-12):
            raise BaselineError(f"baseline was measured at {key} {theirs!r}, this run at {mine}")
    if baseline.get("images") != images:
        raise BaselineError(f"baseline covers {baseline.get('images')!r} images, this run {images}")
    other = [f"{n} {table[n]['gt']} vs {gt[n]}" for n in class_names if table[n]["gt"] != gt[n]]
    if other:
        raise BaselineError(f"baseline was measured on other ground truth: {_names(other)}")
    if not any(gt[n] >= min_gt for n in class_names):
        raise BaselineError(
            f"no class has {min_gt} ground-truth boxes: nothing to compare with the baseline"
        )
    return table


def baseline_status(
    per_class: Mapping[str, Mapping], table: Mapping[str, Mapping], *, max_drop: float, min_gt: int
) -> dict:
    """Recall drop against a validated baseline table.

    ``failures``: classes whose recall fell by more than ``max_drop``; ``compared``: the classes
    checked; ``skipped``: classes with fewer than ``min_gt`` ground-truth boxes, not checked.
    """
    failures, compared, skipped = [], [], []
    for name, m in per_class.items():
        if m["gt"] < min_gt:
            skipped.append(
                {"class": name, "gt": m["gt"], "reason": f"fewer than {min_gt} ground-truth boxes"}
            )
            continue
        compared.append(name)
        drop = table[name]["recall"] - m["recall"]
        if drop > max_drop:
            failures.append(
                {
                    "class": name,
                    "recall": m["recall"],
                    "baseline": table[name]["recall"],
                    "drop": drop,
                }
            )
    return {"failures": failures, "compared": compared, "skipped": skipped}


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _bucket(counts: Counts, cls: int) -> dict[str, int]:
    return counts.setdefault(cls, {"gt": 0, "tp": 0, "fp": 0, "fn": 0})


def match(preds: Iterable[Mapping], gts: Iterable[tuple[int, Box]], iou_thr: float = 0.5) -> Counts:
    """Greedy per-class matching by descending confidence.

    Each GT box matches at most one prediction (IoU >= ``iou_thr``); unmatched or
    duplicate predictions are FP, unmatched GT are FN.
    """
    counts: Counts = {}
    gt_by_cls: dict[int, list[Box]] = {}
    for cls, box in gts:
        gt_by_cls.setdefault(int(cls), []).append(box)
        _bucket(counts, int(cls))["gt"] += 1
    used: dict[int, set[int]] = {}
    for p in sorted(preds, key=lambda d: d["confidence"], reverse=True):
        cls = int(p["class_id"])
        cands = gt_by_cls.get(cls, [])
        taken = used.setdefault(cls, set())
        best_i, best_iou = -1, iou_thr
        for i, g in enumerate(cands):
            if i in taken:
                continue
            v = iou(p["bbox"], g)
            if v >= best_iou:
                best_i, best_iou = i, v
        if best_i >= 0:
            taken.add(best_i)
            _bucket(counts, cls)["tp"] += 1
        else:
            _bucket(counts, cls)["fp"] += 1
    for cls, boxes in gt_by_cls.items():
        _bucket(counts, cls)["fn"] += len(boxes) - len(used.get(cls, ()))
    return counts


def merge(total: Counts, counts: Counts) -> Counts:
    for cls, c in counts.items():
        b = _bucket(total, cls)
        for k in ("gt", "tp", "fp", "fn"):
            b[k] += c[k]
    return total


def summarize(counts: Counts, names: Sequence[str]) -> dict[str, dict]:
    out = {}
    for cls, name in enumerate(names):
        c = counts.get(cls, {"gt": 0, "tp": 0, "fp": 0, "fn": 0})
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        out[name] = {
            **c,
            "recall": tp / (tp + fn) if c["gt"] > 0 else None,
            "precision": tp / (tp + fp) if tp + fp > 0 else None,
        }
    return out
