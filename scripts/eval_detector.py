#!/usr/bin/env python3
"""Evaluate a YOLO26 detector: HEF on the Hailo, ONNX on CPU, or DFC emulation of the HAR.

Run with the repo on PYTHONPATH:

    # one image, optionally checked against reference boxes (exit 1 on mismatch)
    eval_detector.py --backend hef --model x.hef --predict bus.jpg \
        [--expect ref.json --min-iou 0.8]

    # per-class recall/precision over a YOLO-format split
    eval_detector.py --backend hef|onnx|har --model P [--spec J] --images DIR --labels DIR \
        [--conf 0.25] [--iou 0.5] [--gate [--gate-min-gt 150]] \
        [--baseline REPORT.json --max-recall-drop 0.10 --min-gt 20] [--min-images 100] \
        --out REPORT.json

``hef`` needs /dev/hailo0 free (``pi_detector_ctl.py release`` first); its spec
defaults to the HEF sidecar. ``onnx`` and ``har`` need ``--spec`` for class names. All use
the same letterbox, so their boxes are directly comparable. ``har`` runs the Dataflow Compiler's
emulation of the quantized model in the DFC env (``hailo_sdk_client``; pick the GPU with
``CUDA_VISIBLE_DEVICES``). ``--har-context quantized`` (the default) is the DFC's fast emulation
of the quantized network; only ``bit-exact`` reproduces the device's integer arithmetic. Neither
is a device measurement (the report says ``"kind": "dfc-emulation"`` and records the context in
``har_context``): a HAR report is evidence, not the HEF gate.

``--gate`` and ``--baseline`` decide whether a compiled model may replace the live detector, so
they refuse to decide on thin evidence instead of passing it. Exit codes:

    0  every requested check passed (a run without --gate/--baseline only measures)
    1  the model failed: a tier gate, the baseline recall drop, or an --expect box
    2  setup or usage error: model, spec, backend or a backend crash, and an argument out of range
       (``--iou`` not in (0, 1], NaN/inf, ``--gate`` with ``--predict``)
    3  the images, labels or ``--expect`` boxes cannot support a verdict: a missing images or labels
       directory, an image without a label file or a label without an image, a malformed label
       row, a class id the spec does not have, fewer than ``--min-images`` images, a tier-1 class
       with fewer than ``--gate-min-gt`` ground-truth boxes, or a spec whose classes are not
       workshop.classes.NAMES
    4  the ``--baseline`` report cannot be compared: unreadable or unparsable, not a detector-eval
       report, other classes, other ground truth, IoU or confidence, or no class with ``--min-gt``
       boxes to compare

The report (``detector-eval/2``) has a ``verdict`` (passed, failed, invalid, or not_applied when
neither check was requested) and, for each check, its status, what it compared and what it could
not evaluate; an evaluation that exits 3 or 4 still writes a report with ``verdict: invalid`` and
the reasons, so a stale passing report is never left in its place.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from backend.src.drivers.ai.yolo26 import (
    DetectorSpec,
    Letterbox,
    decode_yolo26,
    letterbox,
    load_detector_spec,
    load_spec_file,
)
from workshop.classes import HARD_STOP
from workshop.detector_eval import (
    GATE_MIN_GT,
    MIN_IMAGES,
    REPORT_SCHEMA,
    BaselineError,
    DatasetError,
    EvidenceError,
    baseline_status,
    count_ground_truth,
    evidence_problems,
    finite_number,
    gate_status,
    iou,
    load_baseline,
    match,
    merge,
    pair_labels,
    parse_label_file,
    read_json,
    summarize,
    to_pixels,
    validate_baseline,
)

IMG_EXTS = (".jpg", ".jpeg", ".png")
EXIT_FAILED, EXIT_SETUP, EXIT_DATASET, EXIT_BASELINE = 1, 2, 3, 4


class SetupError(Exception):
    """Backend/model/usage problem: exit 2."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_rgb(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SetupError(f"cannot read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class HefBackend:
    def __init__(self, model: Path, spec: DetectorSpec):
        self.model = model
        self.spec = spec
        self.driver = None

    async def __aenter__(self):
        from backend.src.drivers.ai.hailo_driver import HailoDriver

        self.driver = HailoDriver({"hef_path": str(self.model), "input_format": "uint8"})
        await self.driver.initialize()
        if not self.driver.hardware_available:
            raise SetupError(f"Hailo unavailable: {self.driver.hardware_error}")
        await self.driver.start()
        return self

    async def __aexit__(self, *exc):
        if self.driver is not None:
            await self.driver.stop()

    async def predict(self, rgb: np.ndarray) -> tuple[list[dict], float]:
        tensor, lb = letterbox(rgb, self.spec.input_size)
        t0 = time.perf_counter()
        result = await self.driver.infer(tensor)
        ms = (time.perf_counter() - t0) * 1000
        return decode_yolo26(result.outputs, self.spec, lb), ms


class HarBackend:
    """DFC emulation of the quantized (or compiled) HAR; not the device."""

    CONTEXTS = {"quantized": "SDK_QUANTIZED", "bit-exact": "SDK_BIT_EXACT"}

    def __init__(self, model: Path, spec: DetectorSpec, context: str = "quantized"):
        self.model = model
        self.spec = spec
        self.context = context
        self.runner = None
        self._cm = None
        self._ctx = None

    async def __aenter__(self):
        try:
            from hailo_sdk_client import ClientRunner, InferenceContext
        except ImportError as exc:
            raise SetupError(f"--backend har needs the DFC env (hailo_sdk_client): {exc}") from exc
        self.runner = ClientRunner(har=str(self.model))
        self._cm = self.runner.infer_context(getattr(InferenceContext, self.CONTEXTS[self.context]))
        self._ctx = self._cm.__enter__()
        return self

    async def __aexit__(self, *exc):
        if self._cm is not None:
            self._cm.__exit__(*exc)

    async def predict(self, rgb: np.ndarray) -> tuple[list[dict], float]:
        tensor, lb = letterbox(rgb, self.spec.input_size)
        t0 = time.perf_counter()
        raw = self.runner.infer(self._ctx, tensor.astype(np.float32))
        ms = (time.perf_counter() - t0) * 1000
        items = raw.items() if isinstance(raw, dict) else enumerate(raw)
        outputs = {f"out{k}": np.asarray(v) for k, v in items}
        return decode_yolo26(outputs, self.spec, lb), ms


class OnnxBackend:
    def __init__(self, model: Path, spec: DetectorSpec):
        import onnxruntime as ort

        self.spec = spec
        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def predict(self, rgb: np.ndarray) -> tuple[list[dict], float]:
        tensor, lb = letterbox(rgb, self.spec.input_size)
        x = np.ascontiguousarray(tensor.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
        t0 = time.perf_counter()
        out = self.session.run(None, {self.input_name: x})[0]
        ms = (time.perf_counter() - t0) * 1000
        if out.ndim != 3 or out.shape[0] != 1 or out.shape[2] != 6:
            raise SetupError(f"ONNX output shape {out.shape} is not end2end (1, N, 6)")
        return self._decode(out[0], lb), ms

    def _decode(self, rows: np.ndarray, lb: Letterbox) -> list[dict]:
        dets = []
        for x1, y1, x2, y2, score, cls in rows.tolist():
            if score < self.spec.conf_threshold:
                continue
            box = [
                min(max((x1 - lb.pad_x) / lb.scale, 0.0), lb.src_w),
                min(max((y1 - lb.pad_y) / lb.scale, 0.0), lb.src_h),
                min(max((x2 - lb.pad_x) / lb.scale, 0.0), lb.src_w),
                min(max((y2 - lb.pad_y) / lb.scale, 0.0), lb.src_h),
            ]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            c = int(cls)
            if not 0 <= c < len(self.spec.class_names):
                raise SetupError(
                    f"the model predicts class id {c} but the spec has "
                    f"{len(self.spec.class_names)} classes: wrong spec for this model"
                )
            dets.append(
                {
                    "class": self.spec.class_names[c],
                    "class_id": c,
                    "confidence": float(score),
                    "bbox": box,
                }
            )
        dets.sort(key=lambda d: d["confidence"], reverse=True)
        return dets


def _load_spec(a) -> DetectorSpec:
    if a.backend in ("onnx", "har") and not a.spec:
        raise SetupError(f"--backend {a.backend} needs --spec (class names come from it)")
    try:
        spec = load_spec_file(a.spec) if a.spec else load_detector_spec(a.model)
    except (FileNotFoundError, ValueError) as exc:
        raise SetupError(str(exc)) from exc
    if a.conf is not None:
        spec = dataclasses.replace(spec, conf_threshold=a.conf)
    return spec


def _backend(a, spec: DetectorSpec):
    model = Path(a.model)
    if not model.exists():
        raise SetupError(f"model not found: {model}")
    if a.backend == "hef":
        return HefBackend(model, spec)
    if a.backend == "har":
        return HarBackend(model, spec, a.har_context)
    return OnnxBackend(model, spec)


def _write_report(path: Path, report: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise SetupError(f"cannot write the report {path}: {exc}") from exc


def _load_reference(path: str) -> list[dict]:
    """The ``--expect`` boxes. An empty or malformed file would pass the check vacuously."""
    ref = read_json(path, DatasetError)
    if not isinstance(ref, list) or not ref:
        raise DatasetError(f"{path} lists no reference boxes: nothing to compare the model with")
    for i, r in enumerate(ref):
        box = r.get("bbox") if isinstance(r, dict) else None
        ok = isinstance(box, list) and len(box) == 4 and all(finite_number(v) for v in box)
        if not ok or not isinstance(r.get("class"), str):
            raise DatasetError(f"{path}: entry {i} needs a 'class' and four finite 'bbox' numbers")
    return ref


async def run_predict(a, spec: DetectorSpec) -> int:
    ref = _load_reference(a.expect) if a.expect else None  # before the model is opened
    rgb = _read_rgb(Path(a.predict))
    async with _backend(a, spec) as be:
        dets, ms = await be.predict(rgb)
    print(json.dumps(dets, indent=2))
    print(f"[eval] {len(dets)} detections in {ms:.1f} ms", file=sys.stderr)
    if ref is None:
        return 0
    failures = []
    for r in ref:
        best = max(
            (iou(r["bbox"], d["bbox"]) for d in dets if d["class"] == r["class"]), default=0.0
        )
        status = "ok" if best >= a.min_iou else "MISSING"
        conf = r.get("confidence")
        print(
            f"[eval] ref {r['class']:<12} conf {f'{conf:.2f}' if finite_number(conf) else '-'} "
            f"best IoU {best:.3f} {status}",
            file=sys.stderr,
        )
        if best < a.min_iou:
            failures.append(r)
    matched = len(ref) - len(failures)
    print(f"[eval] expect: {matched}/{len(ref)} reference boxes matched", file=sys.stderr)
    return EXIT_FAILED if failures else 0


def _check_args(a) -> None:
    """Refuse arguments that would make a check pass without measuring anything."""

    def need(ok: bool, message: str) -> None:
        if not ok:
            raise SetupError(message)

    fin = math.isfinite
    need(fin(a.iou) and 0 < a.iou <= 1, f"--iou must be in (0, 1], got {a.iou}")
    need(fin(a.min_iou) and 0 < a.min_iou <= 1, f"--min-iou must be in (0, 1], got {a.min_iou}")
    need(
        a.conf is None or (fin(a.conf) and 0 < a.conf < 1),
        f"--conf must be in (0, 1), got {a.conf}",
    )
    need(
        fin(a.max_recall_drop) and 0 <= a.max_recall_drop < 1,
        f"--max-recall-drop must be in [0, 1), got {a.max_recall_drop}",
    )
    for flag, value in (
        ("--min-gt", a.min_gt),
        ("--gate-min-gt", a.gate_min_gt),
        ("--min-images", a.min_images),
    ):
        need(value >= 1, f"{flag} must be at least 1, got {value}")
    if a.predict:
        need(
            not (a.gate or a.baseline or a.labels or a.out),
            "--gate, --baseline, --labels and --out apply to --images; --predict only has --expect",
        )
    else:
        need(not a.expect, "--expect applies to --predict; --images has --gate and --baseline")


def _list_images(directory: str) -> list[Path]:
    d = Path(directory)
    if not d.is_dir():
        raise DatasetError(f"images directory not found: {d}")
    images = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise DatasetError(f"no images in {d}")
    return images


def _gather_evidence(a, spec: DetectorSpec):
    """Everything the verdict rests on that does not need the model: images, labels, baseline.

    Raises ``DatasetError`` / ``BaselineError`` before any inference when it cannot support the
    requested verdict. Returns ``(images, rows_by_image, gt_by_class, baseline_table)``.
    """
    images = _list_images(a.images)
    pairs = pair_labels(images, a.labels)
    n = len(spec.class_names)
    rows = {img: parse_label_file(lab, n) for img, lab in pairs.items()}
    gt = count_ground_truth(rows, spec.class_names)
    if a.gate or a.baseline:
        problems = evidence_problems(
            spec.class_names,
            gt,
            len(images),
            gate=bool(a.gate),
            min_gt=a.gate_min_gt,
            min_images=a.min_images,
        )
        if problems:
            raise DatasetError(problems)
    table = None
    if a.baseline:
        table = validate_baseline(
            load_baseline(a.baseline),
            class_names=spec.class_names,
            gt=gt,
            images=len(images),
            iou=a.iou,
            conf=spec.conf_threshold,
            min_gt=a.min_gt,
        )
    return images, rows, gt, table


def _kind(backend: str) -> str:
    return {"hef": "device", "har": "dfc-emulation", "onnx": "float-cpu"}[backend]


def _context(a, spec: DetectorSpec) -> dict:
    """The report fields that say what was evaluated, shared by valid and invalid reports."""
    model = Path(a.model)
    return {
        "schema": REPORT_SCHEMA,
        "backend": a.backend,
        "kind": _kind(a.backend),
        # the DFC's emulation context decides what the numbers mean; None for other backends
        "har_context": a.har_context if a.backend == "har" else None,
        "model": str(a.model),
        "model_sha256": _sha256(model) if model.is_file() else None,
        "images_dir": str(a.images),
        "labels_dir": str(a.labels),
        "conf": spec.conf_threshold,
        "iou": a.iou,
        "min_images": a.min_images,
    }


def _invalid_report(a, spec: DetectorSpec, exc: EvidenceError) -> dict:
    return {
        **_context(a, spec),
        "status": "invalid",
        "verdict": "invalid",
        "error": "baseline" if isinstance(exc, BaselineError) else "dataset",
        "problems": exc.problems,
        "gate": {"applied": bool(a.gate), "status": "invalid" if a.gate else "not_applied"},
        "baseline": {
            "applied": bool(a.baseline),
            "path": a.baseline,
            "status": "invalid" if a.baseline else "not_applied",
        },
    }


async def run_eval(a, spec: DetectorSpec) -> int:
    out = Path(a.out)
    try:
        images, rows, gt, table = _gather_evidence(a, spec)
    except EvidenceError as exc:
        _write_report(out, _invalid_report(a, spec, exc))
        raise
    total: dict = {}
    times = []
    async with _backend(a, spec) as be:
        for img in images:
            rgb = _read_rgb(img)
            h, w = rgb.shape[:2]
            try:
                dets, ms = await be.predict(rgb)
            except ValueError as exc:  # decode_yolo26: the head tensors do not fit the spec
                raise SetupError(
                    f"{img.name}: the model output does not fit the spec: {exc}"
                ) from exc
            times.append(ms)
            merge(total, match(dets, to_pixels(rows[img], w, h), a.iou))
    per_class = summarize(total, spec.class_names)

    gate = {"applied": False, "status": "not_applied", "failures": []}
    if a.gate:
        verdict = gate_status(per_class)
        gate = {
            "applied": True,
            "status": "failed" if verdict["failures"] else "passed",
            "failures": verdict["failures"],
            "min_gt": a.gate_min_gt,
            "tier1_gt": {n: gt[n] for n in HARD_STOP},
            "not_evaluated": verdict["not_evaluated"],
        }
    base = {"applied": False, "status": "not_applied", "failures": []}
    if a.baseline:
        verdict = baseline_status(per_class, table, max_drop=a.max_recall_drop, min_gt=a.min_gt)
        base = {
            "applied": True,
            "status": "failed" if verdict["failures"] else "passed",
            "path": str(a.baseline),
            "failures": verdict["failures"],
            "compared": verdict["compared"],
            "skipped": verdict["skipped"],
            "min_gt": a.min_gt,
            "max_recall_drop": a.max_recall_drop,
        }
    checks = [c["status"] for c in (gate, base) if c["applied"]]
    result = "not_applied" if not checks else "failed" if "failed" in checks else "passed"
    report = {
        **_context(a, spec),
        "status": "ok",
        "verdict": result,
        "images": len(images),
        "label_files": len(images),
        "per_class": per_class,
        "gate": gate,
        "baseline": base,
        "mean_infer_ms": round(sum(times) / len(times), 2),
    }
    _write_report(out, report)

    def fmt(v):
        return "-" if v is None else f"{v:.3f}"

    print(f"{'class':18} {'gt':>5} {'tp':>5} {'fp':>5} {'fn':>5} {'recall':>7} {'prec':>7}")
    for name, m in per_class.items():
        print(
            f"{name:18} {m['gt']:>5} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} "
            f"{fmt(m['recall']):>7} {fmt(m['precision']):>7}"
        )
    print(f"[eval] {len(images)} images, mean infer {report['mean_infer_ms']} ms -> {out}")
    if a.gate:
        tier1 = ", ".join(f"{n} {c}" for n, c in gate["tier1_gt"].items())
        print(f"[eval] gate evidence: tier-1 ground truth {tier1} (>= {a.gate_min_gt} each)")
        for f in gate["failures"]:
            print(f"[eval] GATE FAIL {f['class']}: recall {f['recall']:.3f} < {f['gate']}")
        if gate["not_evaluated"]:
            names = ", ".join(f"{c['class']} (tier {c['tier']})" for c in gate["not_evaluated"])
            print(f"[eval] gate NOT evaluated, no ground truth: {names}")
        print(f"[eval] gate {gate['status'].upper()}")
    if a.baseline:
        print(
            f"[eval] baseline compared {len(base['compared'])} classes, "
            f"skipped {len(base['skipped'])} with fewer than {a.min_gt} boxes"
        )
        for f in base["failures"]:
            print(
                f"[eval] BASELINE FAIL {f['class']}: recall {f['recall']:.3f} "
                f"vs {f['baseline']:.3f} (drop {f['drop']:.3f})"
            )
        print(f"[eval] baseline {base['status'].upper()}")
    return EXIT_FAILED if result == "failed" else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=("hef", "onnx", "har"), required=True)
    ap.add_argument("--har-context", choices=("quantized", "bit-exact"), default="quantized")
    ap.add_argument("--model", required=True)
    ap.add_argument("--spec")
    ap.add_argument("--conf", type=float, help="override the spec confidence threshold")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predict", metavar="IMG")
    mode.add_argument("--images", metavar="DIR")
    ap.add_argument("--expect", metavar="REF.json")
    ap.add_argument("--min-iou", type=float, default=0.8)
    ap.add_argument("--labels", metavar="DIR")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument(
        "--gate-min-gt",
        type=int,
        default=GATE_MIN_GT,
        help="ground-truth boxes every tier-1 class needs for --gate (default %(default)s: zero "
        "misses in that many shows recall >= 0.98 at 95 %% confidence)",
    )
    ap.add_argument("--baseline", metavar="REPORT.json")
    ap.add_argument("--max-recall-drop", type=float, default=0.10)
    ap.add_argument(
        "--min-gt",
        type=int,
        default=20,
        help="ground-truth boxes a class needs to take part in the --baseline check",
    )
    ap.add_argument(
        "--min-images",
        type=int,
        default=MIN_IMAGES,
        help="images --gate / --baseline need to decide (default %(default)s)",
    )
    ap.add_argument("--out", metavar="REPORT.json")
    a = ap.parse_args(argv)
    if a.images and not (a.labels and a.out):
        ap.error("--images needs --labels and --out")
    try:
        _check_args(a)
        spec = _load_spec(a)
        return asyncio.run(run_predict(a, spec) if a.predict else run_eval(a, spec))
    except SetupError as exc:
        print(f"[eval] error: {exc}", file=sys.stderr)
        return EXIT_SETUP
    except DatasetError as exc:
        _print_invalid("the images, labels or reference boxes cannot support a verdict", exc)
        return EXIT_DATASET
    except BaselineError as exc:
        _print_invalid("the baseline cannot be compared with this run", exc)
        return EXIT_BASELINE
    except Exception as exc:  # a crash is not a verdict: exit 1 must only mean "the model failed"
        traceback.print_exc()
        print(f"[eval] error: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_SETUP


def _print_invalid(headline: str, exc: EvidenceError) -> None:
    print(f"[eval] INVALID, no verdict: {headline}", file=sys.stderr)
    for problem in exc.problems:
        print(f"[eval]   - {problem}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
